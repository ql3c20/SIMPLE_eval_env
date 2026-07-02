from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from simple.baselines.textop_tracker_adapter import MUJOCO_JOINT_NAMES


def _quat_error(target_wxyz: np.ndarray, current_wxyz: np.ndarray) -> np.ndarray:
    target = R.from_quat(target_wxyz, scalar_first=True)
    current = R.from_quat(current_wxyz, scalar_first=True)
    return (target * current.inv()).as_rotvec().astype(np.float64)


@dataclass
class FullBodyIkTextOpConfig:
    max_iters: int = 35
    damping: float = 1e-3
    step_scale: float = 0.7
    pos_weight: float = 1.0
    rot_weight: float = 0.18
    continuity_weight: float = 5e-3
    posture_weight: float = 2e-4
    success_pos_tol: float = 0.035
    success_rot_tol: float = 0.75


class FullBodyIkTextOpAdapter:
    """Solve 44D sparse EE/root actions into qpos36 for TextOp reference."""

    def __init__(self, model, cfg: FullBodyIkTextOpConfig | None = None):
        import mujoco

        self._mujoco = mujoco
        self.model = model
        self.data = mujoco.MjData(model)
        self.cfg = cfg or FullBodyIkTextOpConfig()
        self.joint_names = tuple(MUJOCO_JOINT_NAMES)
        self.qpos_adrs = np.asarray([self._joint_qpos_adr(name) for name in self.joint_names], dtype=np.int32)
        self.dof_adrs = np.asarray([self._joint_dof_adr(name) for name in self.joint_names], dtype=np.int32)
        self.target_body_ids = (
            self._body_id(("left_wrist_yaw_link", "left_wrist_pitch_link", "left_wrist_roll_link")),
            self._body_id(("right_wrist_yaw_link", "right_wrist_pitch_link", "right_wrist_roll_link")),
            self._body_id(("left_ankle_roll_link", "left_ankle_pitch_link")),
            self._body_id(("right_ankle_roll_link", "right_ankle_pitch_link")),
        )
        self._last_body29: np.ndarray | None = None

    def reset(self) -> None:
        self._last_body29 = None

    def policy44_to_qpos36(self, policy_action: np.ndarray, *, current_qpos36: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        policy_action = np.asarray(policy_action, dtype=np.float32)
        if policy_action.ndim != 2 or policy_action.shape[1] != 44:
            raise ValueError(f"Expected policy action shape (T, 44), got {policy_action.shape}")
        current_qpos36 = np.asarray(current_qpos36, dtype=np.float32).reshape(-1)
        if current_qpos36.size != 36:
            raise ValueError(f"Expected current qpos36, got {current_qpos36.shape}")

        hand14 = policy_action[:, :14].astype(np.float32)
        root6 = policy_action[:, 14:20].astype(np.float32)
        ee24 = policy_action[:, 20:44].astype(np.float32)

        qpos = np.zeros(self.model.nq, dtype=np.float64)
        qpos[:7] = current_qpos36[:7]
        qpos[self.qpos_adrs] = current_qpos36[7:36]
        seed = qpos[self.qpos_adrs].copy()
        if self._last_body29 is not None:
            seed = self._last_body29.astype(np.float64)
        nominal = current_qpos36[7:36].astype(np.float64)

        qpos36 = []
        for i in range(policy_action.shape[0]):
            qpos[:3] = root6[i, :3]
            qpos[3:7] = R.from_euler("xyz", root6[i, 3:6]).as_quat(scalar_first=True)
            seed = self._solve_frame(qpos, seed, nominal, ee24[i])
            qpos36.append(np.concatenate([qpos[:7], seed], axis=0).astype(np.float32))

        out = np.asarray(qpos36, dtype=np.float32)
        self._last_body29 = out[-1, 7:36].copy()
        return hand14, out

    def _solve_frame(
        self,
        base_qpos: np.ndarray,
        seed_body29: np.ndarray,
        nominal_body29: np.ndarray,
        ee24: np.ndarray,
    ) -> np.ndarray:
        qpos = base_qpos.copy()
        qpos[self.qpos_adrs] = seed_body29
        targets = []
        for idx, body_id in enumerate(self.target_body_ids):
            pose6 = ee24[idx * 6 : (idx + 1) * 6]
            targets.append(
                (
                    body_id,
                    pose6[:3].astype(np.float64),
                    R.from_euler("xyz", pose6[3:6]).as_quat(scalar_first=True),
                )
            )

        seed = seed_body29.astype(np.float64)
        nominal = nominal_body29.astype(np.float64)
        for _ in range(self.cfg.max_iters):
            self.data.qpos[:] = qpos
            self._mujoco.mj_forward(self.model, self.data)

            residuals = []
            jac_rows = []
            max_pos_err = 0.0
            max_rot_err = 0.0
            for body_id, pos_target, quat_target in targets:
                pos_err = pos_target - self.data.xpos[body_id]
                rot_err = _quat_error(quat_target, self.data.xquat[body_id])
                max_pos_err = max(max_pos_err, float(np.linalg.norm(pos_err)))
                max_rot_err = max(max_rot_err, float(np.linalg.norm(rot_err)))

                jacp = np.zeros((3, self.model.nv), dtype=np.float64)
                jacr = np.zeros((3, self.model.nv), dtype=np.float64)
                self._mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
                residuals.append(self.cfg.pos_weight * pos_err)
                residuals.append(self.cfg.rot_weight * rot_err)
                jac_rows.append(self.cfg.pos_weight * jacp[:, self.dof_adrs])
                jac_rows.append(self.cfg.rot_weight * jacr[:, self.dof_adrs])

            if max_pos_err < self.cfg.success_pos_tol and max_rot_err < self.cfg.success_rot_tol:
                break

            residuals.append(np.sqrt(self.cfg.continuity_weight) * (seed - qpos[self.qpos_adrs]))
            jac_rows.append(np.sqrt(self.cfg.continuity_weight) * np.eye(len(self.qpos_adrs)))
            residuals.append(np.sqrt(self.cfg.posture_weight) * (nominal - qpos[self.qpos_adrs]))
            jac_rows.append(np.sqrt(self.cfg.posture_weight) * np.eye(len(self.qpos_adrs)))

            err = np.concatenate(residuals, axis=0)
            jac = np.concatenate(jac_rows, axis=0)
            lhs = jac.T @ jac + self.cfg.damping * np.eye(jac.shape[1])
            rhs = jac.T @ err
            try:
                delta = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                break
            qpos[self.qpos_adrs] += self.cfg.step_scale * delta
            self._clip_qpos_inplace(qpos)

        return qpos[self.qpos_adrs].astype(np.float32)

    def _joint_qpos_adr(self, name: str) -> int:
        jid = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"MuJoCo joint not found: {name}")
        return int(self.model.jnt_qposadr[jid])

    def _joint_dof_adr(self, name: str) -> int:
        jid = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"MuJoCo joint not found: {name}")
        return int(self.model.jnt_dofadr[jid])

    def _body_id(self, candidates: tuple[str, ...]) -> int:
        for name in candidates:
            bid = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                return int(bid)
        raise ValueError(f"Could not find any MuJoCo body from candidates: {candidates}")

    def _clip_qpos_inplace(self, qpos: np.ndarray) -> None:
        for adr, name in zip(self.qpos_adrs, self.joint_names):
            jid = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, name)
            if self.model.jnt_limited[jid]:
                lo, hi = self.model.jnt_range[jid]
                qpos[adr] = np.clip(qpos[adr], lo, hi)
