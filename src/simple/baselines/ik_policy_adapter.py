"""Convert Psi0 44D wrist-EE actions directly to SIMPLE 36D via MuJoCo IK.

This adapter is intentionally independent from the Kimodo adapter.  It is a
debug path for checking whether Kimodo's generated whole-body motion is the
source of grasping error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R


LEFT_ARM_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def _finite_diff(values: np.ndarray, fps: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size <= 1:
        return np.zeros_like(values)
    out = np.empty_like(values)
    out[:-1] = (values[1:] - values[:-1]) * fps
    out[-1] = out[-2]
    return out


def _finite_diff_angle(values: np.ndarray, fps: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size <= 1:
        return np.zeros_like(values)
    delta = np.arctan2(np.sin(values[1:] - values[:-1]), np.cos(values[1:] - values[:-1]))
    out = np.empty_like(values)
    out[:-1] = delta * fps
    out[-1] = out[-2]
    return out


def _smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    window = int(window)
    if window <= 1 or values.size < 3:
        return values
    window = min(window, values.size)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _quat_error(target_wxyz: np.ndarray, current_wxyz: np.ndarray) -> np.ndarray:
    target = R.from_quat(target_wxyz, scalar_first=True)
    current = R.from_quat(current_wxyz, scalar_first=True)
    return (target * current.inv()).as_rotvec().astype(np.float32)


@dataclass
class IkPolicyAdapterConfig:
    output_fps: float = 50.0
    max_iters: int = 30
    damping: float = 1e-3
    step_scale: float = 0.75
    pos_weight: float = 1.0
    rot_weight: float = 0.25
    continuity_weight: float = 2e-3
    success_pos_tol: float = 0.04
    success_rot_tol: float = 0.7
    base_cmd_smooth_window: int = 5
    max_abs_vx: float = 0.8
    max_abs_vy: float = 0.8
    max_abs_vyaw: float = 2.5


class MujocoWristIkPolicyAdapter:
    """Map ``hand14 + root6 + ee24`` to SIMPLE's 36D WBC action."""

    def __init__(self, model, cfg: IkPolicyAdapterConfig | None = None):
        import mujoco

        self._mujoco = mujoco
        self.model = model
        self.data = mujoco.MjData(model)
        self.cfg = cfg or IkPolicyAdapterConfig()
        self.arm_joint_names = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
        self.arm_qpos_adrs = np.asarray([self._joint_qpos_adr(name) for name in self.arm_joint_names], dtype=np.int32)
        self.arm_dof_adrs = np.asarray([self._joint_dof_adr(name) for name in self.arm_joint_names], dtype=np.int32)
        self.left_body_id = self._body_id(("left_wrist_yaw_link", "left_wrist_pitch_link", "left_wrist_roll_link"))
        self.right_body_id = self._body_id(("right_wrist_yaw_link", "right_wrist_pitch_link", "right_wrist_roll_link"))
        self._last_arm14: np.ndarray | None = None

    def reset(self) -> None:
        self._last_arm14 = None

    def policy44_to_simple36(
        self,
        policy_action: np.ndarray,
        *,
        current_qpos36: np.ndarray,
    ) -> np.ndarray:
        policy_action = np.asarray(policy_action, dtype=np.float32)
        if policy_action.ndim != 2 or policy_action.shape[1] != 44:
            raise ValueError(f"Expected policy action shape (T, 44), got {policy_action.shape}")

        hand14 = policy_action[:, :14]
        root6 = policy_action[:, 14:20]
        ee24 = policy_action[:, 20:44]

        root_rpy = root6[:, 3:6].astype(np.float32)
        root_yaw = root_rpy[:, 2]
        arm14 = self._solve_arm_sequence(root6, ee24, current_qpos36)

        vx_world = _finite_diff(root6[:, 0], self.cfg.output_fps)
        vy_world = _finite_diff(root6[:, 1], self.cfg.output_fps)
        cos_yaw = np.cos(root_yaw).astype(np.float32)
        sin_yaw = np.sin(root_yaw).astype(np.float32)
        vx = cos_yaw * vx_world + sin_yaw * vy_world
        vy = -sin_yaw * vx_world + cos_yaw * vy_world
        vyaw = _finite_diff_angle(root_yaw, self.cfg.output_fps)

        vx = np.clip(_smooth_1d(vx, self.cfg.base_cmd_smooth_window), -self.cfg.max_abs_vx, self.cfg.max_abs_vx)
        vy = np.clip(_smooth_1d(vy, self.cfg.base_cmd_smooth_window), -self.cfg.max_abs_vy, self.cfg.max_abs_vy)
        vyaw = np.clip(
            _smooth_1d(vyaw, self.cfg.base_cmd_smooth_window),
            -self.cfg.max_abs_vyaw,
            self.cfg.max_abs_vyaw,
        )

        return np.concatenate(
            [
                hand14,
                arm14,
                root_rpy,
                root6[:, 2:3],
                vx[:, None],
                vy[:, None],
                vyaw[:, None],
                root_yaw[:, None],
            ],
            axis=1,
        ).astype(np.float32)

    def _solve_arm_sequence(self, root6: np.ndarray, ee24: np.ndarray, current_qpos36: np.ndarray) -> np.ndarray:
        current_qpos36 = np.asarray(current_qpos36, dtype=np.float32).reshape(-1)
        if current_qpos36.size != 36:
            raise ValueError(f"Expected current qpos36, got {current_qpos36.shape}")
        qpos = np.zeros(self.model.nq, dtype=np.float64)
        qpos[:36] = current_qpos36
        seed = qpos[self.arm_qpos_adrs].astype(np.float64)
        if self._last_arm14 is not None:
            seed = self._last_arm14.astype(np.float64)

        solved = []
        for i in range(root6.shape[0]):
            qpos[:3] = root6[i, :3]
            qpos[3:7] = R.from_euler("xyz", root6[i, 3:6]).as_quat(scalar_first=True)
            left_pose = ee24[i, 0:6]
            right_pose = ee24[i, 6:12]
            seed = self._solve_frame(qpos, seed, left_pose, right_pose)
            solved.append(seed.astype(np.float32))

        arm14 = np.asarray(solved, dtype=np.float32)
        self._last_arm14 = arm14[-1].copy()
        return arm14

    def _solve_frame(
        self,
        base_qpos: np.ndarray,
        seed_arm14: np.ndarray,
        left_pose6: np.ndarray,
        right_pose6: np.ndarray,
    ) -> np.ndarray:
        qpos = base_qpos.copy()
        qpos[self.arm_qpos_adrs] = seed_arm14
        target = (
            (self.left_body_id, left_pose6[:3], R.from_euler("xyz", left_pose6[3:6]).as_quat(scalar_first=True)),
            (self.right_body_id, right_pose6[:3], R.from_euler("xyz", right_pose6[3:6]).as_quat(scalar_first=True)),
        )
        seed = seed_arm14.astype(np.float64)

        for _ in range(self.cfg.max_iters):
            self.data.qpos[:] = qpos
            self._mujoco.mj_forward(self.model, self.data)

            residuals = []
            jac_rows = []
            max_pos_err = 0.0
            max_rot_err = 0.0
            for body_id, pos_target, quat_target in target:
                pos_err = np.asarray(pos_target, dtype=np.float64) - self.data.xpos[body_id]
                rot_err = _quat_error(quat_target, self.data.xquat[body_id]).astype(np.float64)
                max_pos_err = max(max_pos_err, float(np.linalg.norm(pos_err)))
                max_rot_err = max(max_rot_err, float(np.linalg.norm(rot_err)))

                jacp = np.zeros((3, self.model.nv), dtype=np.float64)
                jacr = np.zeros((3, self.model.nv), dtype=np.float64)
                self._mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
                residuals.append(self.cfg.pos_weight * pos_err)
                residuals.append(self.cfg.rot_weight * rot_err)
                jac_rows.append(self.cfg.pos_weight * jacp[:, self.arm_dof_adrs])
                jac_rows.append(self.cfg.rot_weight * jacr[:, self.arm_dof_adrs])

            if max_pos_err < self.cfg.success_pos_tol and max_rot_err < self.cfg.success_rot_tol:
                break

            residuals.append(np.sqrt(self.cfg.continuity_weight) * (seed - qpos[self.arm_qpos_adrs]))
            jac_rows.append(np.sqrt(self.cfg.continuity_weight) * np.eye(len(self.arm_qpos_adrs)))
            err = np.concatenate(residuals, axis=0)
            jac = np.concatenate(jac_rows, axis=0)
            lhs = jac.T @ jac + self.cfg.damping * np.eye(jac.shape[1])
            rhs = jac.T @ err
            try:
                delta = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                break
            qpos[self.arm_qpos_adrs] += self.cfg.step_scale * delta
            self._clip_qpos_inplace(qpos)

        return qpos[self.arm_qpos_adrs].astype(np.float32)

    def _joint_qpos_adr(self, name: str) -> int:
        mujoco = self._mujoco
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"MuJoCo joint not found: {name}")
        return int(self.model.jnt_qposadr[jid])

    def _joint_dof_adr(self, name: str) -> int:
        mujoco = self._mujoco
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"MuJoCo joint not found: {name}")
        return int(self.model.jnt_dofadr[jid])

    def _body_id(self, candidates: tuple[str, ...]) -> int:
        for name in candidates:
            mujoco = self._mujoco
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                return int(bid)
        raise ValueError(f"Could not find any MuJoCo body from candidates: {candidates}")

    def _clip_qpos_inplace(self, qpos: np.ndarray) -> None:
        for adr, name in zip(self.arm_qpos_adrs, self.arm_joint_names):
            mujoco = self._mujoco
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if self.model.jnt_limited[jid]:
                lo, hi = self.model.jnt_range[jid]
                qpos[adr] = np.clip(qpos[adr], lo, hi)
