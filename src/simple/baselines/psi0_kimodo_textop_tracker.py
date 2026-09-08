from __future__ import annotations

import os

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

from simple.agents.sonic_decoupled_wbc_agent import SonicDecoupledWbcAgent
from simple.baselines.psi0_kimodo_decoupled_wbc import (
    Psi0KimodoDecoupledWbcAgent,
    _quat_wxyz_to_rpy,
)
from simple.baselines.textop_tracker_adapter import TextOpTrackerAdapter
from simple.core.action import ActionCmd


class Psi0KimodoTextOpTrackerAgent(Psi0KimodoDecoupledWbcAgent):
    """Psi0 policy action -> Kimodo qpos36 -> TextOp tracker body29 + Psi0 hand14.

    Supported action layouts:
    - 44D: hand14 + root6(xyz,rpy) + 4 * EE6(xyz,rpy)
    - 59D: hand14 + root9(xyz,rot6d) + 4 * EE9(xyz,rot6d)
    """

    def __init__(self, robot, host: str, port: int, upsample_factor=1, **kwargs):
        super().__init__(robot, host, port, upsample_factor=upsample_factor, **kwargs)
        self._textop_adapter = None
        self._direct_queue: list[ActionCmd | tuple[dict[str, np.ndarray], int, np.ndarray]] = []
        self._rate_limit = os.environ.get("TEXTOP_TARGET_RATE_LIMIT", "0") == "1"
        self._leg_max_delta = float(os.environ.get("TEXTOP_LEG_MAX_DELTA", "0.03"))
        self._torso_max_delta = float(os.environ.get("TEXTOP_TORSO_MAX_DELTA", "0.04"))
        self._arm_max_delta = float(os.environ.get("TEXTOP_ARM_MAX_DELTA", "0.08"))
        self._last_body_target: np.ndarray | None = None
        self._debug = os.environ.get("TEXTOP_DEBUG", "0") == "1"
        self._debug_dir = os.environ.get("TEXTOP_DEBUG_DIR", "/pfs/pfs-ilWc5D/yzh/Psi0/outputs/textop_debug")
        self._debug_trace = []
        self._debug_chunk_trace = []
        self._init_to_ref = os.environ.get("TEXTOP_INIT_TO_REF", "0") == "1"
        self._init_hold_action = os.environ.get("TEXTOP_INIT_HOLD_ACTION", "1") == "1"
        self._did_init_to_ref = False
        self._use_policy_root_ee = os.environ.get("TEXTOP_POLICY_ROOT_EE", "0") == "1"
        self._rot6d59_convention = os.environ.get("ROT6D59_CONVENTION", "cols_rowmajor")
        self._printed_rot6d59_convention = False
        self._policy_execution_horizon = int(os.environ.get("POLICY_EXECUTION_HORIZON", "0"))
        self._vla_proprio_source = os.environ.get(
            "VLA_PROPRIO_SOURCE",
            os.environ.get("BRIDGE_VLA_PROPRIO_SOURCE", "actual"),
        ).strip().lower()
        if self._vla_proprio_source in {"", "dds", "sim"}:
            self._vla_proprio_source = "actual"
        if self._vla_proprio_source not in {"actual", "tracker_target"}:
            raise ValueError(
                f"Unsupported VLA_PROPRIO_SOURCE/BRIDGE_VLA_PROPRIO_SOURCE={self._vla_proprio_source!r}; "
                "use actual or tracker_target."
            )
        self._last_tracker_target_proprio: dict[str, np.ndarray | int] | None = None
        print(f"[TextOpTracker] VLA proprio source: {self._vla_proprio_source}")

    @staticmethod
    def _root_summary(name: str, arr: np.ndarray, horizon: int | None = None) -> str:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return f"{name}=shape{arr.shape}"
        n = arr.shape[0] if horizon is None else min(arr.shape[0], max(1, int(horizon)))
        seg = arr[:n]
        return (
            f"{name}[0:{n}] "
            f"x {seg[0,0]:+.4f}->{seg[-1,0]:+.4f} d={seg[-1,0]-seg[0,0]:+.4f}, "
            f"y {seg[0,1]:+.4f}->{seg[-1,1]:+.4f} d={seg[-1,1]-seg[0,1]:+.4f}, "
            f"z {seg[0,2]:+.4f}->{seg[-1,2]:+.4f} d={seg[-1,2]-seg[0,2]:+.4f}"
        )

    @staticmethod
    def _rot6d_to_rpy(rot6d: np.ndarray, *, convention: str = "cols_rowmajor") -> np.ndarray:
        """Convert rot6d to xyz Euler angles.

        ``cols_rowmajor`` is the Psi0/TextOp deploy convention:
        ``mat[:, :2].reshape(-1) == [r00, r01, r10, r11, r20, r21]``.
        """
        rot6d = np.asarray(rot6d, dtype=np.float64)
        if rot6d.shape[-1] != 6:
            raise ValueError(f"Expected rot6d last dim 6, got {rot6d.shape}")

        if convention == "cols_rowmajor":
            a1 = rot6d[..., [0, 2, 4]]
            a2 = rot6d[..., [1, 3, 5]]
            build_matrix_from_basis = "columns"
        elif convention == "cols":
            a1 = rot6d[..., :3]
            a2 = rot6d[..., 3:6]
            build_matrix_from_basis = "columns"
        elif convention == "rows":
            a1 = rot6d[..., :3]
            a2 = rot6d[..., 3:6]
            build_matrix_from_basis = "rows"
        else:
            raise ValueError(
                f"Unsupported ROT6D59_CONVENTION={convention!r}; "
                "use cols_rowmajor, cols, or rows."
            )

        b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
        a2_orth = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
        b2 = a2_orth / np.maximum(np.linalg.norm(a2_orth, axis=-1, keepdims=True), 1e-8)
        b3 = np.cross(b1, b2)

        mats = np.empty(rot6d.shape[:-1] + (3, 3), dtype=np.float64)
        if build_matrix_from_basis == "columns":
            mats[..., :, 0] = b1
            mats[..., :, 1] = b2
            mats[..., :, 2] = b3
        else:
            mats[..., 0, :] = b1
            mats[..., 1, :] = b2
            mats[..., 2, :] = b3
        return R.from_matrix(mats.reshape(-1, 3, 3)).as_euler("xyz").reshape(rot6d.shape[:-1] + (3,))

    def _policy59_to_policy44(self, policy_action: np.ndarray) -> np.ndarray:
        policy_action = np.asarray(policy_action, dtype=np.float32)
        if policy_action.ndim != 2 or policy_action.shape[1] != 59:
            raise ValueError(f"Expected rot6d59 policy action shape (T, 59), got {policy_action.shape}")
        if not self._printed_rot6d59_convention:
            print(f"[Rot6D59Adapter] ROT6D59_CONVENTION={self._rot6d59_convention}")
            self._printed_rot6d59_convention = True

        hand14 = policy_action[:, :14]
        root9 = policy_action[:, 14:23]
        ee36 = policy_action[:, 23:59].reshape(policy_action.shape[0], 4, 9)

        root6 = np.concatenate(
            [
                root9[:, :3],
                self._rot6d_to_rpy(root9[:, 3:9], convention=self._rot6d59_convention).astype(np.float32),
            ],
            axis=1,
        )
        ee_chunks = []
        for i in range(4):
            pose9 = ee36[:, i]
            ee_chunks.append(
                np.concatenate(
                    [
                        pose9[:, :3],
                        self._rot6d_to_rpy(pose9[:, 3:9], convention=self._rot6d59_convention).astype(np.float32),
                    ],
                    axis=1,
                )
            )
        return np.concatenate([hand14, root6.astype(np.float32), *ee_chunks], axis=1).astype(np.float32)

    def _normalize_policy_action(self, policy_action: np.ndarray) -> np.ndarray:
        policy_action = np.asarray(policy_action, dtype=np.float32)
        if policy_action.ndim != 2:
            raise ValueError(f"Expected policy action shape (T, D), got {policy_action.shape}")
        if policy_action.shape[1] == 44:
            return policy_action
        if policy_action.shape[1] == 59:
            converted = self._policy59_to_policy44(policy_action)
            if self._debug:
                print("[Rot6D59Adapter] converted policy action 59D -> 44D for Kimodo/TextOp")
            return converted
        raise ValueError(f"Expected policy action dim 44 or 59, got {policy_action.shape}")

    @staticmethod
    def _state49_from_qpos36_hand(qpos36: np.ndarray, hand14: np.ndarray) -> np.ndarray:
        qpos36 = np.asarray(qpos36, dtype=np.float32).reshape(36)
        hand14 = np.asarray(hand14, dtype=np.float32).reshape(14)
        body29 = qpos36[7:36].reshape(1, 29)
        root_xyz = qpos36[:3].reshape(1, 3)
        root_rpy = _quat_wxyz_to_rpy(qpos36[3:7]).reshape(1, 3)
        return np.concatenate([hand14.reshape(1, 14), body29, root_xyz, root_rpy], axis=1).astype(np.float32)

    def _select_vla_proprio_inputs(self, observation) -> tuple[np.ndarray, np.ndarray, str]:
        actual_state49 = self._build_policy_state49(observation)
        actual_qpos36 = self._current_mujoco_qpos36()
        if self._vla_proprio_source == "tracker_target":
            if self._last_tracker_target_proprio is not None:
                qpos36 = np.asarray(self._last_tracker_target_proprio["qpos36"], dtype=np.float32).reshape(36)
                state49 = np.asarray(self._last_tracker_target_proprio["state49"], dtype=np.float32).reshape(1, 49)
                ref_t = int(self._last_tracker_target_proprio.get("ref_t", -1))
                print(f"[TextOpTracker] VLA proprio source=tracker_target ref_t={ref_t}")
                return state49, qpos36, "tracker_target"
            print("[TextOpTracker] VLA proprio source=tracker_target requested but unavailable; fallback=actual")
            return actual_state49, actual_qpos36, "fallback_actual"
        return actual_state49, actual_qpos36, "actual"

    def _body_pose_6d_from_data(self, data, candidates: tuple[str, ...]) -> np.ndarray:
        for name in candidates:
            body_id = mujoco.mj_name2id(self.robot.mjModel, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                xyz = np.asarray(data.xpos[body_id], dtype=np.float32)
                quat = np.asarray(data.xquat[body_id], dtype=np.float32)
                rpy = R.from_quat(quat, scalar_first=True).as_euler("xyz").astype(np.float32)
                return np.concatenate([xyz, rpy], axis=0).astype(np.float32)
        raise ValueError(f"Could not find any MuJoCo body from candidates: {candidates}")

    def _constraints28_from_qpos36(self, qpos36: np.ndarray) -> np.ndarray:
        qpos36 = np.asarray(qpos36, dtype=np.float32).reshape(36)
        root_yaw = R.from_quat(qpos36[3:7], scalar_first=True).as_euler("xyz")[2]
        root_xyzyaw = np.asarray([qpos36[0], qpos36[1], qpos36[2], root_yaw], dtype=np.float32)
        data = mujoco.MjData(self.robot.mjModel)
        data.qpos[:] = 0.0
        data.qpos[3] = 1.0
        data.qpos[:7] = qpos36[:7]
        if self._textop_adapter is not None:
            body_qpos_adrs = self._textop_adapter.body_qpos_adrs
        else:
            # Lazily construct the adapter so the body qpos address order exactly
            # matches TextOp's 29D MuJoCo joint order.
            self._textop_adapter = TextOpTrackerAdapter(self.robot.mjModel)
            body_qpos_adrs = self._textop_adapter.body_qpos_adrs
        data.qpos[body_qpos_adrs] = qpos36[7:36]
        mujoco.mj_forward(self.robot.mjModel, data)
        left_hand = self._body_pose_6d_from_data(data, ("left_wrist_yaw_link", "left_wrist_pitch_link", "left_wrist_roll_link"))
        right_hand = self._body_pose_6d_from_data(data, ("right_wrist_yaw_link", "right_wrist_pitch_link", "right_wrist_roll_link"))
        left_foot = self._body_pose_6d_from_data(data, ("left_ankle_roll_link", "left_ankle_pitch_link"))
        right_foot = self._body_pose_6d_from_data(data, ("right_ankle_roll_link", "right_ankle_pitch_link"))
        return np.concatenate([root_xyzyaw, left_hand, right_hand, left_foot, right_foot], axis=0).astype(np.float32)

    def _policy44_to_textop_actions(
        self,
        policy_action: np.ndarray,
        instruction: str | None = None,
        current_qpos_mujoco: np.ndarray | None = None,
    ) -> list[ActionCmd]:
        policy_action = self._normalize_policy_action(policy_action)
        if self._textop_adapter is None:
            self._textop_adapter = TextOpTrackerAdapter(self.robot.mjModel)

        if current_qpos_mujoco is None:
            current_qpos_mujoco = self._current_mujoco_qpos36()
        _, qpos50 = self._kimodo_adapter.policy44_to_simple36(
            policy_action,
            prev_qpos_mujoco=current_qpos_mujoco,
            current_constraints28_mujoco=self._constraints28_from_qpos36(current_qpos_mujoco),
            instruction=instruction,
        )
        if self._debug:
            exec_h = self._policy_execution_horizon if self._policy_execution_horizon > 0 else policy_action.shape[0]
            hand14 = np.asarray(policy_action[:, :14], dtype=np.float32)
            rh = hand14[:, 7:14]
            rh_from0 = np.linalg.norm(rh - rh[0:1], axis=1)
            onset = int(np.argmax(rh_from0 > 0.5)) if np.any(rh_from0 > 0.5) else -1
            print(
                "[TextOpDebugChunk] "
                + self._root_summary("policy_root", policy_action[:, 14:17], exec_h)
                + " | "
                + self._root_summary("kimodo_qpos", qpos50[:, :3], exec_h)
                + " | "
                + self._root_summary("actual_root", current_qpos_mujoco[:3].reshape(1, 3), 1)
                + " | "
                + f"right_hand_from0 max={float(rh_from0.max()):.3f} "
                + f"@19={float(rh_from0[min(19, len(rh_from0)-1)]):.3f} "
                + f"@33={float(rh_from0[min(33, len(rh_from0)-1)]):.3f} "
                + f"@last={float(rh_from0[-1]):.3f} onset>0.5@{onset}"
            )
            self._debug_chunk_trace.append(
                [
                    float(self._global_step_idx),
                    float(policy_action[0, 14]),
                    float(policy_action[min(policy_action.shape[0] - 1, exec_h - 1), 14]),
                    float(qpos50[0, 0]),
                    float(qpos50[min(qpos50.shape[0] - 1, exec_h - 1), 0]),
                    float(current_qpos_mujoco[0]),
                ]
            )
            os.makedirs(self._debug_dir, exist_ok=True)
            np.save(
                os.path.join(self._debug_dir, "chunk_root_trace.npy"),
                np.asarray(self._debug_chunk_trace, dtype=np.float32),
            )
            chunk_i = len(self._debug_chunk_trace) - 1
            np.save(
                os.path.join(self._debug_dir, f"policy_hand14_chunk_{chunk_i:04d}.npy"),
                hand14,
            )
            np.save(
                os.path.join(self._debug_dir, f"policy_action_chunk_{chunk_i:04d}.npy"),
                np.asarray(policy_action, dtype=np.float32),
            )
        self._last_kimodo_qpos_mujoco = qpos50[-1].copy()
        init_hold_action = None
        if self._init_to_ref and not self._did_init_to_ref:
            self.robot.mjData.qpos[:7] = qpos50[0, :7]
            self.robot.mjData.qpos[self._textop_adapter.body_qpos_adrs] = qpos50[0, 7:36]
            self.robot.mjData.qvel[:] = 0.0
            mujoco.mj_forward(self.robot.mjModel, self.robot.mjData)
            self._did_init_to_ref = True
            print("[TextOpInitToRef] set root and body29 qpos to Kimodo qpos50[0]")

        hand14 = np.asarray(policy_action[:, :14], dtype=np.float32)
        if self._init_to_ref and self._init_hold_action:
            init_hold_action = ActionCmd(
                "textop_tracker",
                target_q=np.asarray(qpos50[0, 7:36], dtype=np.float32),
                left_hand_q=np.asarray(hand14[0, :7], dtype=np.float32),
                right_hand_q=np.asarray(hand14[0, 7:14], dtype=np.float32),
            )
        if self._use_policy_root_ee:
            root6 = np.asarray(policy_action[:, 14:20], dtype=np.float32)
            ee24 = np.asarray(policy_action[:, 20:44], dtype=np.float32)
            ref = self._textop_adapter.prepare_reference_external(qpos50, root6=root6, ee24=ee24)
        else:
            ref = self._textop_adapter.prepare_reference(qpos50)
        actions = []
        if init_hold_action is not None:
            actions.append(init_hold_action)
        for t, hand_q in enumerate(hand14):
            actions.append((ref, t, np.asarray(hand_q, dtype=np.float32)))
        return actions

    def _rate_limit_body_target(self, body_target: np.ndarray) -> np.ndarray:
        if not self._rate_limit:
            return body_target
        out = np.asarray(body_target, dtype=np.float32).copy()
        prev = (
            np.asarray(self._last_body_target, dtype=np.float32)
            if self._last_body_target is not None
            else np.asarray(self.robot.mjData.qpos[self._textop_adapter.body_qpos_adrs], dtype=np.float32)
        )
        max_delta = np.empty(29, dtype=np.float32)
        max_delta[:12] = self._leg_max_delta
        max_delta[12:15] = self._torso_max_delta
        max_delta[15:29] = self._arm_max_delta
        out = prev + np.clip(out - prev, -max_delta, max_delta)
        self._last_body_target = out.copy()
        print(
            "[TextOpRateLimit] enabled "
            f"leg={self._leg_max_delta}, torso={self._torso_max_delta}, arm={self._arm_max_delta}"
        )
        return out

    def _make_textop_action(self, item) -> ActionCmd:
        if isinstance(item, ActionCmd):
            return item
        ref, t, hand_q = item
        raw_body_q = self._textop_adapter.target_from_reference(ref, self.robot.mjData, t)
        body_q = self._rate_limit_body_target(raw_body_q)
        if self._debug and t == 0:
            self._dump_final_target_debug(raw_body_q[None], body_q[None])
        qpos36 = np.asarray(ref["qpos36"], dtype=np.float32)
        ref_t = min(max(0, int(t)), qpos36.shape[0] - 1)
        target_qpos36 = np.concatenate([qpos36[ref_t, :7], np.asarray(body_q, dtype=np.float32)], axis=0)
        target_hand14 = np.asarray(hand_q, dtype=np.float32).reshape(14)
        self._last_tracker_target_proprio = {
            "qpos36": target_qpos36.astype(np.float32),
            "hand14": target_hand14.copy(),
            "state49": self._state49_from_qpos36_hand(target_qpos36, target_hand14),
            "ref_t": int(ref_t),
        }
        return ActionCmd(
            "textop_tracker",
            target_q=np.asarray(body_q, dtype=np.float32),
            left_hand_q=np.asarray(hand_q[:7], dtype=np.float32),
            right_hand_q=np.asarray(hand_q[7:14], dtype=np.float32),
        )

    def _dump_final_target_debug(self, raw_targets: np.ndarray, final_targets: np.ndarray) -> None:
        if not self._debug:
            return
        os.makedirs(self._debug_dir, exist_ok=True)
        current_q = np.asarray(self.robot.mjData.qpos[self._textop_adapter.body_qpos_adrs], dtype=np.float32)
        root = np.asarray(self.robot.mjData.qpos[:7], dtype=np.float32)
        root_rpy = R.from_quat(root[3:7], scalar_first=True).as_euler("xyz").astype(np.float32)
        np.savez(
            os.path.join(self._debug_dir, "final_targets_debug.npz"),
            current_q=current_q,
            root=root,
            root_rpy=root_rpy,
            raw_target0=np.asarray(raw_targets[0], dtype=np.float32),
            final_target0=np.asarray(final_targets[0], dtype=np.float32),
            raw_minus_current=np.asarray(raw_targets[0] - current_q, dtype=np.float32),
            final_minus_current=np.asarray(final_targets[0] - current_q, dtype=np.float32),
            raw_targets=np.asarray(raw_targets, dtype=np.float32),
            final_targets=np.asarray(final_targets, dtype=np.float32),
        )
        print(
            "[TextOpDebugFinal] "
            f"raw max|t0-current|={np.max(np.abs(raw_targets[0] - current_q)):.4f}, "
            f"final max|t0-current|={np.max(np.abs(final_targets[0] - current_q)):.4f}, "
            f"root_rpy={root_rpy}"
        )

    def get_action(self, observation, instruction=None, info=None, conditions=None, **kwargs):
        self._last_observation = observation
        self._last_qpos = observation["joint_qpos"]

        if not self._direct_queue:
            # The GR00T checkpoint still expects its training-time modality
            # name.  Football supplies the Arena-aligned mono frame, while
            # legacy SIMPLE tasks retain the left stereo image.
            policy_camera_key = (
                "front_camera" if "front_camera" in observation else "head_stereo_left"
            )
            observations = {"rgb_head_stereo_left": observation[policy_camera_key]}
            state49, current_qpos36, proprio_source = self._select_vla_proprio_inputs(observation)
            state_dict = {"states": state49}
            history = {"reset": True} if self._reset_history else {}
            self._reset_history = False

            policy_action, *_ = self.client.query_action(
                observations,
                instruction or "bend to pick up the object",
                state_dict,
                {},
                history=history,
                dataset="simple",
            )
            print(
                f"Received {policy_action.shape[0]} Kimodo-policy actions for TextOp tracker "
                f"(proprio_source={proprio_source})."
            )
            textop_actions = self._policy44_to_textop_actions(
                policy_action,
                instruction=instruction,
                current_qpos_mujoco=current_qpos36,
            )
            if self._policy_execution_horizon > 0:
                # Keep the full policy chunk for Kimodo/TextOp reference construction,
                # but only execute a receding-horizon prefix before replanning.
                # If TEXTOP_INIT_TO_REF inserted an ActionCmd hold at the front, keep it
                # in addition to the requested number of policy-timed tracker steps.
                keep = self._policy_execution_horizon
                if textop_actions and isinstance(textop_actions[0], ActionCmd):
                    keep += 1
                textop_actions = textop_actions[:keep]
                print(
                    "[TextOpTracker] queueing "
                    f"{len(textop_actions)} actions with POLICY_EXECUTION_HORIZON="
                    f"{self._policy_execution_horizon}"
                )
            self._direct_queue.extend(textop_actions)

        self._last_pred_action = self._make_textop_action(self._direct_queue.pop(0))
        self._record_debug_trace()
        self._global_step_idx += 1
        return self._last_pred_action

    def _record_debug_trace(self) -> None:
        if not self._debug:
            return
        root = np.asarray(self.robot.mjData.qpos[:7], dtype=np.float32)
        rpy = R.from_quat(root[3:7], scalar_first=True).as_euler("xyz").astype(np.float32)
        self._debug_trace.append([self._global_step_idx, root[2], rpy[0], rpy[1], rpy[2]])
        os.makedirs(self._debug_dir, exist_ok=True)
        np.save(os.path.join(self._debug_dir, "root_trace.npy"), np.asarray(self._debug_trace, dtype=np.float32))

    def reset(self, **kwargs):
        SonicDecoupledWbcAgent.reset(self, **kwargs)
        self._kimodo_adapter.begin_episode()
        self._global_step_idx = 0
        self._last_qpos = None
        self._last_observation = None
        self._last_pred_action = None
        self._reset_history = True
        self._last_target_yaw = 0.0
        self._last_kimodo_qpos_mujoco = None
        self._last_simple_actions = None
        self._last_body_target = None
        self._debug_trace = []
        self._debug_chunk_trace = []
        self._did_init_to_ref = False
        self._direct_queue.clear()
        self._last_tracker_target_proprio = None
        if self._textop_adapter is not None:
            self._textop_adapter.reset()


Psi0KimodoTextopTrackerAgent = Psi0KimodoTextOpTrackerAgent
