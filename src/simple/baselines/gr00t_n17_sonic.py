"""GR00T N1.7 78D action -> SONIC decoder -> G1 joint targets."""

from __future__ import annotations

from collections import deque
import os
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy.spatial.transform import Rotation as R

from simple.agents.primitive_agent import PrimitiveAgent
from simple.agents.sonic_decoupled_wbc_agent import SonicDecoupledWbcAgent
from simple.baselines.client import HttpActionClient
from simple.baselines.textop_tracker_adapter import (
    DEFAULT_ANGLES,
    ISAACLAB_TO_MUJOCO,
    MUJOCO_JOINT_NAMES,
    MUJOCO_TO_ISAACLAB,
)
from simple.core.action import ActionCmd


SONIC_ACTION_SCALE = np.asarray(
    [
        0.3506614664, 0.3506614664, 0.5475464652, 0.3506614664,
        0.4385773139, 0.4385773139, 0.3506614664, 0.3506614664,
        0.5475464652, 0.3506614664, 0.4385773139, 0.4385773139,
        0.5475464652, 0.4385773139, 0.4385773139, 0.4385773139,
        0.4385773139, 0.4385773139, 0.4385773139, 0.4385773139,
        0.0745008703, 0.0745008703, 0.4385773139, 0.4385773139,
        0.4385773139, 0.4385773139, 0.4385773139, 0.0745008703,
        0.0745008703,
    ],
    dtype=np.float32,
)


INITIAL_MOTION_TOKEN = np.asarray(
    [
        -0.0625, 0.0, -0.0625, -0.125, -0.1875, -0.0625, 0.1875, 0.25,
        0.1875, -0.125, 0.0625, -0.0625, -0.25, -0.25, -0.3125, -0.0625,
        0.0, -0.0625, -0.125, -0.1875, 0.0, -0.25, 0.0, -0.25,
        -0.0625, 0.0625, 0.125, -0.125, 0.25, 0.1875, 0.25, -0.125,
        0.125, 0.1875, -0.0625, 0.0, -0.1875, -0.1875, 0.25, 0.0,
        0.0, -0.125, 0.0625, 0.0, -0.0625, -0.0625, 0.1875, -0.0625,
        0.0, 0.0625, 0.125, 0.0625, 0.125, 0.0625, 0.125, 0.0,
        0.125, 0.1875, 0.0, 0.0, 0.0625, 0.0625, 0.1875, 0.0625,
    ],
    dtype=np.float32,
)


class Gr00tN17SonicAgent(PrimitiveAgent):
    """Run the official SONIC decoder locally while GR00T runs as a server."""

    reset_before_stabilize = True

    def __init__(self, robot, host: str, port: int, **kwargs):
        super().__init__(robot)
        sonic_config = kwargs.get("sonic_config")
        if sonic_config is None:
            raise ValueError("gr00t_n17_sonic requires sonic_config")
        # The released SONIC decoder is the runtime controller after policy
        # engagement.  The proven Decoupled-WBC controller is retained only
        # for the reset/stabilization phase so the floating-base robot can
        # actively balance before the first VLA token arrives.
        self._stabilizer = SonicDecoupledWbcAgent(robot, sonic_config)
        self._wbc_policy = self._stabilizer._wbc_policy
        self.client = HttpActionClient(host, port)
        decoder = os.environ.get(
            "SONIC_DECODER_ONNX",
            "/pfs/pfs-ilWc5D/yzh/SONIC_my/gear_sonic_deploy/"
            "policy/release/model_decoder.onnx",
        )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(decoder, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        if self.session.get_inputs()[0].shape[-1] != 994:
            raise ValueError(f"Expected SONIC decoder input 994D: {decoder}")

        self.execution_horizon = int(os.environ.get("GR00T_EXECUTION_HORIZON", "20"))
        self.initial_pose_steps = int(
            os.environ.get("GR00T_INITIAL_POSE_STEPS", "50")
        )
        self.history_len = 10
        self._pending = deque()
        self._history = deque(maxlen=self.history_len)
        self._last_action_isaac = np.zeros(29, dtype=np.float32)
        self._stabilize_target = None
        self._decode_count = 0
        self._initial_pose_step = 0
        self._initial_pose_start_q = None
        self._reset_history = True

        self.body_qpos_adrs = None
        self.body_qvel_adrs = None
        self.debug_dir = os.environ.get("GR00T_SONIC_DEBUG_DIR")
        self._debug_query_count = 0

    def _ensure_mujoco_indices(self) -> None:
        if self.body_qpos_adrs is not None:
            return
        if not hasattr(self.robot, "mjModel") or not hasattr(self.robot, "mjData"):
            raise RuntimeError(
                "G1Sonic MuJoCo control is not initialized yet; "
                "mjModel/mjData are unavailable"
            )
        model = self.robot.mjModel
        self.body_qpos_adrs = np.asarray(
            [model.joint(name).qposadr[0] for name in MUJOCO_JOINT_NAMES],
            dtype=np.int32,
        )
        self.body_qvel_adrs = np.asarray(
            [model.joint(name).dofadr[0] for name in MUJOCO_JOINT_NAMES],
            dtype=np.int32,
        )

    def _projected_gravity(self) -> np.ndarray:
        self._ensure_mujoco_indices()
        quat = np.asarray(self.robot.mjData.qpos[3:7], dtype=np.float64)
        return R.from_quat(quat, scalar_first=True).inv().apply([0.0, 0.0, -1.0]).astype(
            np.float32
        )

    def _body_state_isaac(self) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_mujoco_indices()
        q = np.asarray(self.robot.mjData.qpos[self.body_qpos_adrs], dtype=np.float32)
        dq = np.asarray(self.robot.mjData.qvel[self.body_qvel_adrs], dtype=np.float32)
        return q[MUJOCO_TO_ISAACLAB], dq[MUJOCO_TO_ISAACLAB]

    def _history_frame(self) -> dict[str, np.ndarray]:
        q, dq = self._body_state_isaac()
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.robot.mjModel,
            self.robot.mjData,
            mujoco.mjtObj.mjOBJ_BODY,
            self.robot.root_body_id,
            velocity,
            1,
        )
        # MuJoCo object velocity is [angular, linear].
        base_ang_vel = velocity[:3].astype(np.float32)
        return {
            "gravity": self._projected_gravity(),
            "base_ang_vel": base_ang_vel,
            # C++ GatherRobotStateToLogger stores body_q in IsaacLab order
            # after subtracting the MuJoCo-order default joint angles.
            "joint_pos": q - DEFAULT_ANGLES[MUJOCO_TO_ISAACLAB],
            "joint_vel": dq,
            "actions": self._last_action_isaac.copy(),
        }

    def _append_history(self) -> None:
        frame = self._history_frame()
        if not self._history:
            for _ in range(self.history_len):
                self._history.append(
                    {name: value.copy() for name, value in frame.items()}
                )
        else:
            self._history.append(frame)

    def _actor_history(self) -> np.ndarray:
        # Follow gear_sonic_deploy/policy/release/observation_config.yaml
        # exactly. Its decoder input is token64 followed by these five
        # term-major 10-frame blocks (oldest -> newest).
        term_order = ("base_ang_vel", "joint_pos", "joint_vel", "actions", "gravity")
        return np.concatenate(
            [
                np.concatenate([frame[name] for frame in self._history], axis=0)
                for name in term_order
            ],
            axis=0,
        ).astype(np.float32)

    def _sonic_state46(self, observation) -> np.ndarray:
        q = np.asarray(observation["joint_qpos"], dtype=np.float32)
        if q.shape != (43,):
            raise ValueError(f"Expected G1 joint_qpos43, got {q.shape}")
        # SIMPLE joint_qpos is body29 + left_hand7 + right_hand7, with each
        # hand exposed as thumb,index,middle.  The training dataset's SONIC
        # state is left_arm,left_hand,right_arm,right_hand, and state hands use
        # RobotModel order index,middle,thumb.
        left_hand = q[[32, 33, 34, 35, 29, 30, 31]]
        right_hand = q[[39, 40, 41, 42, 36, 37, 38]]
        return np.concatenate(
            [
                q[0:15],
                q[15:22],
                left_hand,
                q[22:29],
                right_hand,
                self._projected_gravity(),
            ]
        ).astype(np.float32)

    @staticmethod
    def _hand_to_mujoco(hand: np.ndarray) -> np.ndarray:
        # GR00T action: thumb,index,middle -> MuJoCo: thumb,middle,index.
        return np.asarray(hand, dtype=np.float32)[[0, 1, 2, 5, 6, 3, 4]]

    def _decode(
        self, action78: np.ndarray, transition_alpha: float | None = None
    ) -> ActionCmd:
        token = np.asarray(action78[:64], dtype=np.float32)
        actor_obs = self._actor_history()
        decoder_input = np.concatenate([token, actor_obs])[None]
        if decoder_input.shape != (1, 994):
            raise RuntimeError(f"Bad SONIC decoder input shape: {decoder_input.shape}")
        action_isaac = self.session.run(
            None, {self.input_name: decoder_input.astype(np.float32)}
        )[0].reshape(29)
        # Match SONIC C++ CreatePolicyCommand:
        # target = default_angles + decoder_action * g1_action_scale.
        body_target = (
            action_isaac[ISAACLAB_TO_MUJOCO] * SONIC_ACTION_SCALE + DEFAULT_ANGLES
        ).astype(np.float32)
        if transition_alpha is not None:
            if self._initial_pose_start_q is None:
                self._initial_pose_start_q = np.asarray(
                    self.robot.mjData.qpos[self.body_qpos_adrs], dtype=np.float32
                ).copy()
            alpha = float(np.clip(transition_alpha, 0.0, 1.0))
            body_target = (
                (1.0 - alpha) * self._initial_pose_start_q + alpha * body_target
            ).astype(np.float32)
            # SONIC history must contain the action actually sent to the robot.
            action_mujoco = (body_target - DEFAULT_ANGLES) / SONIC_ACTION_SCALE
            self._last_action_isaac = action_mujoco[MUJOCO_TO_ISAACLAB].astype(
                np.float32
            )
        else:
            self._last_action_isaac = action_isaac.astype(np.float32)
        if self._decode_count < 3 or 48 <= self._decode_count < 55:
            q_now = np.asarray(
                self.robot.mjData.qpos[self.body_qpos_adrs], dtype=np.float32
            )
            jumps = np.abs(body_target - q_now)
            top = np.argsort(jumps)[-5:][::-1]
            top_text = ",".join(
                f"{MUJOCO_JOINT_NAMES[index]}:{jumps[index]:.3f}" for index in top
            )
            print(
                "[GR00T-N1.7/SONIC] decode "
                f"step={self._decode_count} "
                f"token=[{token.min():.3f},{token.max():.3f}] "
                f"action=[{action_isaac.min():.3f},{action_isaac.max():.3f}] "
                f"max_target_jump={jumps.max():.3f} top=[{top_text}]"
            )
        self._decode_count += 1
        return ActionCmd(
            "sonic_decoder",
            target_q=body_target,
            left_hand_q=self._hand_to_mujoco(action78[64:71]),
            right_hand_q=self._hand_to_mujoco(action78[71:78]),
        )

    def get_stabilize_action(self, observation) -> ActionCmd:
        # Keep SONIC's proprioceptive history alive while Decoupled-WBC is
        # balancing. The history frame observes the action applied on the
        # preceding control step, matching Isaac Lab's last_action term.
        self._append_history()
        action = self._stabilizer.get_stabilize_action(observation)
        target_mujoco = np.asarray(action["target_q"], dtype=np.float32)
        applied_mujoco = (
            target_mujoco - DEFAULT_ANGLES
        ) / SONIC_ACTION_SCALE
        self._last_action_isaac = applied_mujoco[MUJOCO_TO_ISAACLAB].astype(
            np.float32
        )
        return action

    def get_action(self, observation, instruction=None, **kwargs):
        self._append_history()
        if self._initial_pose_step < self.initial_pose_steps:
            self._initial_pose_step += 1
            alpha = self._initial_pose_step / max(1, self.initial_pose_steps)
            initial_action = np.concatenate(
                [INITIAL_MOTION_TOKEN, np.zeros(14, dtype=np.float32)]
            )
            return self._decode(initial_action, transition_alpha=alpha)
        if not self._pending:
            sonic_state = self._sonic_state46(observation)
            if self.debug_dir and self._debug_query_count == 0:
                debug_dir = Path(self.debug_dir)
                debug_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(
                    np.asarray(observation["head_stereo_left"], dtype=np.uint8)
                ).save(debug_dir / "first_policy_image.png")
                np.savez(
                    debug_dir / "first_policy_input.npz",
                    sonic_state=sonic_state,
                    actor_history=self._actor_history(),
                    robot_root_qpos=np.asarray(
                        self.robot.mjData.qpos[:7], dtype=np.float32
                    ),
                )
            history = {"reset": True} if self._reset_history else {}
            self._reset_history = False
            action, *_ = self.client.query_action(
                {"ego_view": observation["head_stereo_left"]},
                instruction or "move forward to pick up the cylinder",
                {"sonic_state": sonic_state},
                {},
                history=history,
                dataset="unitree_g1_sonic",
            )
            action = np.asarray(action, dtype=np.float32)
            if action.ndim != 2 or action.shape[1] != 78:
                raise ValueError(f"Expected GR00T action (T,78), got {action.shape}")
            count = min(len(action), self.execution_horizon)
            self._pending.extend(action[:count])
            if self.debug_dir:
                debug_dir = Path(self.debug_dir)
                np.save(
                    debug_dir / f"predicted_action78_query_{self._debug_query_count:04d}.npy",
                    action,
                )
                np.savez(
                    debug_dir / f"policy_state_query_{self._debug_query_count:04d}.npz",
                    sonic_state=sonic_state,
                    robot_root_qpos=np.asarray(
                        self.robot.mjData.qpos[:7], dtype=np.float32
                    ),
                    body_qpos_isaac=self._body_state_isaac()[0],
                )
                if self._debug_query_count == 0:
                    np.save(debug_dir / "first_predicted_action78.npy", action)
                Image.fromarray(
                    np.asarray(observation["head_stereo_left"], dtype=np.uint8)
                ).save(
                    debug_dir
                    / f"policy_image_query_{self._debug_query_count:04d}.jpg",
                    quality=90,
                )
            left = action[:, 64:71]
            right = action[:, 71:78]
            print(
                "[GR00T-N1.7/SONIC] query "
                f"{self._debug_query_count} "
                f"left_hand=[{left.min():.3f},{left.max():.3f}] "
                f"right_hand=[{right.min():.3f},{right.max():.3f}] "
                f"right_delta={np.max(np.abs(right[-1] - right[0])):.3f}"
            )
            self._debug_query_count += 1
            print(f"[GR00T-N1.7/SONIC] received {len(action)} frames, queueing {count}")
        self._last_pred_action = self._decode(self._pending.popleft())
        return self._last_pred_action

    def reset(self, **kwargs):
        self._pending.clear()
        self._history.clear()
        self._last_action_isaac[:] = 0.0
        self._stabilize_target = None
        self._decode_count = 0
        self._initial_pose_step = 0
        self._initial_pose_start_q = None
        self._reset_history = True
        self._stabilizer._cached_target_q = None
        self._stabilizer._cached_left_hand_q = None
        self._stabilizer._cached_right_hand_q = None
        self._last_pred_action = None
        self._debug_query_count = 0
