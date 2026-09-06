from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


NUM_ACTIONS = 29
FUTURE_STEPS = int(os.environ.get("TEXTOP_FUTURE_STEPS", "10"))
VAE_WINDOW_STEPS = int(os.environ.get("TEXTOP_VAE_WINDOW_STEPS", str(FUTURE_STEPS)))
FPS = float(os.environ.get("TEXTOP_FPS", "50"))
TASK_PROJ_GRAV_ANCHOR_EE_OBS_TRANSFORMER_VAE = "Tracking-Flat-G1-ProjGravAnchorEEObs-TransformerVAE-NMMLP-v0"
TASK_PROJ_GRAV_ANCHOR_EE_OBS_ONESTEP_TRANSFORMER_VAE = (
    "Tracking-Flat-G1-ProjGravAnchorEEObsOneStep-TransformerVAE-NMMLP-v0"
)
TEXTOP_TASK = os.environ.get("TEXTOP_TASK", TASK_PROJ_GRAV_ANCHOR_EE_OBS_TRANSFORMER_VAE)
TEXTOP_VAE_PROVIDER = os.environ.get("TEXTOP_VAE_PROVIDER", "cuda").strip().lower()
TEXTOP_POLICY_PROVIDER = os.environ.get("TEXTOP_POLICY_PROVIDER", "cpu").strip().lower()

TEXTOP_ROOT = Path(os.environ.get("TEXTOP_ROOT", "/pfs/pfs-ilWc5D/yzh"))
TRACKER_RUN = Path(
    os.environ.get(
        "TEXTOP_TRACKER_RUN",
        str(TEXTOP_ROOT / "textop/2026-05-18_19-55-31_transformer_vae_eeobs_g1_before_2023"),
    )
)
VAE_RUN = Path(
    os.environ.get(
        "TEXTOP_VAE_RUN",
        str(TEXTOP_ROOT / "textop/2026-05-12_16-34-08_optitrack_npz_soma_before_2023"),
    )
)

DEFAULT_POLICY_ONNX = Path(os.environ.get("TEXTOP_POLICY_ONNX", str(TRACKER_RUN / "exported/policy.onnx")))
DEFAULT_VAE_ONNX = Path(
    os.environ.get("TEXTOP_VAE_ONNX", str(VAE_RUN / "artifacts/motion_transformer_vae_encoder_z_c.onnx"))
)
DEFAULT_VAE_STATS = Path(os.environ.get("TEXTOP_VAE_STATS", str(VAE_RUN / "artifacts/stats.npz")))
DEBUG = os.environ.get("TEXTOP_DEBUG", "0") == "1"
DEBUG_DIR = Path(os.environ.get("TEXTOP_DEBUG_DIR", "/pfs/pfs-ilWc5D/yzh/Psi0/outputs/textop_debug"))
TEXTOP_EEOBS_TERMS = (
    "command",
    "motion_anchor_pos_b",
    "motion_anchor_ori_b",
    "robot_anchor_pos_w",
    "robot_anchor_ori_w",
    "projected_gravity",
    "base_lin_vel",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
    "motion_ee_pos_b",
    "motion_ee_ori_b",
)
TEXTOP_EEOBS_FIXED_DIMS = {
    "command": 128,
    "robot_anchor_pos_w": 3,
    "robot_anchor_ori_w": 6,
    "projected_gravity": 3,
    "base_lin_vel": 3,
    "base_ang_vel": 3,
    "joint_pos": NUM_ACTIONS,
    "joint_vel": NUM_ACTIONS,
    "actions": NUM_ACTIONS,
}
TEXTOP_EEOBS_FUTURE_DIMS = {
    "motion_anchor_pos_b": 3,
    "motion_anchor_ori_b": 6,
    "motion_ee_pos_b": 4 * 3,
    "motion_ee_ori_b": 4 * 6,
}


def _textop_future_steps_for_motion_terms(task: str, future_steps: int) -> int:
    if task == TASK_PROJ_GRAV_ANCHOR_EE_OBS_ONESTEP_TRANSFORMER_VAE:
        return 1
    return int(future_steps)


def _textop_expected_obs_dim(task: str, future_steps: int) -> int:
    motion_future_steps = _textop_future_steps_for_motion_terms(task, future_steps)
    dim = 0
    for term in TEXTOP_EEOBS_TERMS:
        if term in TEXTOP_EEOBS_FIXED_DIMS:
            dim += TEXTOP_EEOBS_FIXED_DIMS[term]
        elif term in TEXTOP_EEOBS_FUTURE_DIMS:
            dim += motion_future_steps * TEXTOP_EEOBS_FUTURE_DIMS[term]
        else:
            raise ValueError(f"Unsupported TextOp observation term: {term}")
    return dim


def _textop_obs_dims(task: str, future_steps: int) -> np.ndarray:
    motion_future_steps = _textop_future_steps_for_motion_terms(task, future_steps)
    dims = []
    for term in TEXTOP_EEOBS_TERMS:
        if term in TEXTOP_EEOBS_FIXED_DIMS:
            dims.append(TEXTOP_EEOBS_FIXED_DIMS[term])
        else:
            dims.append(motion_future_steps * TEXTOP_EEOBS_FUTURE_DIMS[term])
    return np.asarray(dims, dtype=np.int32)


def _ort_providers(ort, requested: str, component: str) -> list[str]:
    if requested == "cpu":
        return ["CPUExecutionProvider"]
    if requested != "cuda":
        raise ValueError(
            f"Unsupported {component} provider {requested!r}; use 'cuda' or 'cpu'."
        )

    preload_dlls = getattr(ort, "preload_dlls", None)
    if callable(preload_dlls):
        preload_dlls(directory="")
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            f"{component} requested CUDAExecutionProvider, but ONNX Runtime only "
            f"provides {available}. Install onnxruntime-gpu in the active environment."
        )
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]

DEFAULT_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]
DEFAULT_EE_BODY_NAMES = [
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]

ISAACLAB_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint",
    "right_wrist_roll_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
MUJOCO_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
ISAACLAB_TO_MUJOCO = [ISAACLAB_JOINT_NAMES.index(name) for name in MUJOCO_JOINT_NAMES]
MUJOCO_TO_ISAACLAB = [MUJOCO_JOINT_NAMES.index(name) for name in ISAACLAB_JOINT_NAMES]


def _joint_default(name: str) -> float:
    if name.endswith("_hip_pitch_joint"):
        return -0.312
    if name.endswith("_knee_joint"):
        return 0.669
    if name.endswith("_ankle_pitch_joint"):
        return -0.363
    if name.endswith("_elbow_joint"):
        return 0.6
    if name == "left_shoulder_roll_joint":
        return 0.2
    if name == "left_shoulder_pitch_joint":
        return 0.2
    if name == "right_shoulder_roll_joint":
        return -0.2
    if name == "right_shoulder_pitch_joint":
        return 0.2
    return 0.0


def _action_scale(name: str) -> float:
    suffix_scale = {
        "_hip_yaw_joint": 0.5475464652142303,
        "_hip_roll_joint": 0.3506614663788243,
        "_hip_pitch_joint": 0.5475464652142303,
        "_knee_joint": 0.3506614663788243,
        "_ankle_pitch_joint": 0.43857731392336724,
        "_ankle_roll_joint": 0.43857731392336724,
        "_shoulder_pitch_joint": 0.43857731392336724,
        "_shoulder_roll_joint": 0.43857731392336724,
        "_shoulder_yaw_joint": 0.43857731392336724,
        "_elbow_joint": 0.43857731392336724,
        "_wrist_roll_joint": 0.43857731392336724,
        "_wrist_pitch_joint": 0.07450087032950714,
        "_wrist_yaw_joint": 0.07450087032950714,
    }
    if name in {"waist_roll_joint", "waist_pitch_joint"}:
        return 0.43857731392336724
    if name == "waist_yaw_joint":
        return 0.5475464652142303
    for suffix, scale in suffix_scale.items():
        if name.endswith(suffix):
            return scale
    raise ValueError(f"No action scale for {name}")


DEFAULT_ANGLES = np.asarray([_joint_default(n) for n in MUJOCO_JOINT_NAMES], dtype=np.float32)
ACTION_SCALE = np.asarray([_action_scale(n) for n in MUJOCO_JOINT_NAMES], dtype=np.float32)


def _quat_inv(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out / np.sum(q * q, axis=-1, keepdims=True).clip(1e-9)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def _quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    v = np.asarray(v)
    shape = v.shape
    return R.from_quat(q.reshape(-1, 4), scalar_first=True).apply(v.reshape(-1, 3)).reshape(shape)


def _rot6d_rows_from_quat(q: np.ndarray) -> np.ndarray:
    """TextOp deploy convention: mat[..., :2].reshape(-1)."""
    q = np.asarray(q)
    return (
        R.from_quat(q.reshape(-1, 4), scalar_first=True)
        .as_matrix()[..., :2]
        .reshape(*q.shape[:-1], 6)
    )


def _rot6d_cols_from_quat(q: np.ndarray) -> np.ndarray:
    """Motion Transformer VAE feature convention: first two matrix columns."""
    q = np.asarray(q)
    mat = R.from_quat(q.reshape(-1, 4), scalar_first=True).as_matrix()
    col0 = mat[..., :, 0]
    col1 = mat[..., :, 1]
    return np.concatenate([col0, col1], axis=-1).reshape(*q.shape[:-1], 6)


def _yaw_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    mag = np.sqrt(q[..., 0] * q[..., 0] + q[..., 3] * q[..., 3] + 1e-12)
    out = np.zeros_like(q)
    out[..., 0] = q[..., 0] / mag
    out[..., 3] = q[..., 3] / mag
    return out


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def _quat_apply_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_xyz = q[..., 1:]
    t = 2.0 * np.cross(q_xyz, v)
    return v + q[..., 0:1] * t + np.cross(q_xyz, t)


def _matrix_rows6_from_quat(q: np.ndarray) -> np.ndarray:
    return _rot6d_rows_from_quat(q)


def _finite_diff(x: np.ndarray, fps: float) -> np.ndarray:
    if len(x) <= 1:
        return np.zeros_like(x, dtype=np.float32)
    out = np.empty_like(x, dtype=np.float64)
    out[:-1] = x[1:] - x[:-1]
    out[-1] = out[-2]
    return (out * fps).astype(np.float32)


def _quat_ang_vel(q: np.ndarray, fps: float) -> np.ndarray:
    if len(q) <= 1:
        return np.zeros((len(q), 3), dtype=np.float32)
    dq = _quat_mul(q[1:], _quat_inv(q[:-1]))
    rotvec = R.from_quat(dq, scalar_first=True).as_rotvec() * fps
    out = np.empty((len(q), 3), dtype=np.float32)
    out[:-1] = rotvec.astype(np.float32)
    out[-1] = out[-2]
    return out


def _subtract_frame(pos_a: np.ndarray, quat_a: np.ndarray, pos_b: np.ndarray, quat_b: np.ndarray):
    q_inv = _quat_inv(quat_a)
    return _quat_apply(q_inv, pos_b - pos_a), _quat_mul(q_inv, quat_b)


class TextOpTrackerAdapter:
    def __init__(self, mj_model, *, policy_onnx: Path | None = None, vae_onnx: Path | None = None):
        import mujoco
        import onnxruntime as ort

        self.mujoco = mujoco
        self.model = mj_model
        self.data = mujoco.MjData(mj_model)
        self.task = TEXTOP_TASK
        self.future_steps = FUTURE_STEPS
        self.vae_window_steps = VAE_WINDOW_STEPS
        self.motion_future_steps = _textop_future_steps_for_motion_terms(self.task, self.future_steps)
        self.expected_obs_dim = _textop_expected_obs_dim(self.task, self.future_steps)
        self.last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        if self.task not in {
            TASK_PROJ_GRAV_ANCHOR_EE_OBS_TRANSFORMER_VAE,
            TASK_PROJ_GRAV_ANCHOR_EE_OBS_ONESTEP_TRANSFORMER_VAE,
        }:
            raise ValueError(
                f"Unsupported TEXTOP_TASK={self.task!r}; supported tasks are "
                f"{TASK_PROJ_GRAV_ANCHOR_EE_OBS_TRANSFORMER_VAE!r} and "
                f"{TASK_PROJ_GRAV_ANCHOR_EE_OBS_ONESTEP_TRANSFORMER_VAE!r}."
            )
        print(
            "[TextOpTrackerAdapter] "
            f"task={self.task}, future_steps={self.future_steps}, "
            f"motion_future_steps={self.motion_future_steps}, vae_window_steps={self.vae_window_steps}, "
            f"expected_obs_dim={self.expected_obs_dim}"
        )

        self.body_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, n) for n in DEFAULT_BODY_NAMES]
        self.ee_body_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, n) for n in DEFAULT_EE_BODY_NAMES]
        if min(self.body_ids + self.ee_body_ids) < 0:
            raise ValueError("Missing TextOp body name in MuJoCo model")
        self.body_qpos_adrs = []
        self.body_qvel_adrs = []
        for name in MUJOCO_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Missing TextOp joint name in MuJoCo model: {name}")
            self.body_qpos_adrs.append(int(mj_model.jnt_qposadr[joint_id]))
            self.body_qvel_adrs.append(int(mj_model.jnt_dofadr[joint_id]))
        self.body_qpos_adrs = np.asarray(self.body_qpos_adrs, dtype=np.int32)
        self.body_qvel_adrs = np.asarray(self.body_qvel_adrs, dtype=np.int32)

        vae_providers = _ort_providers(ort, TEXTOP_VAE_PROVIDER, "TextOp VAE")
        policy_providers = _ort_providers(ort, TEXTOP_POLICY_PROVIDER, "TextOp policy")
        self.vae_session = ort.InferenceSession(
            str(vae_onnx or DEFAULT_VAE_ONNX),
            providers=vae_providers,
        )
        if (
            TEXTOP_VAE_PROVIDER == "cuda"
            and self.vae_session.get_providers()[0] != "CUDAExecutionProvider"
        ):
            raise RuntimeError(
                "TextOp VAE CUDA provider initialization failed; active providers are "
                f"{self.vae_session.get_providers()}."
            )
        self.vae_input = self.vae_session.get_inputs()[0].name
        vae_shape = self.vae_session.get_inputs()[0].shape
        if len(vae_shape) >= 3 and isinstance(vae_shape[1], int) and vae_shape[1] != self.vae_window_steps:
            raise ValueError(
                f"TextOp VAE encoder window {vae_shape[1]} does not match TEXTOP_VAE_WINDOW_STEPS="
                f"{self.vae_window_steps}. Check TEXTOP_VAE_ONNX and TEXTOP_VAE_WINDOW_STEPS."
            )
        self.vae_output = "z_c" if any(o.name == "z_c" for o in self.vae_session.get_outputs()) else self.vae_session.get_outputs()[0].name
        self.policy_session = ort.InferenceSession(
            str(policy_onnx or DEFAULT_POLICY_ONNX),
            providers=policy_providers,
        )
        self.policy_input = self.policy_session.get_inputs()[0].name
        policy_shape = self.policy_session.get_inputs()[0].shape
        if len(policy_shape) >= 2 and isinstance(policy_shape[1], int) and policy_shape[1] != self.expected_obs_dim:
            raise ValueError(
                f"TextOp policy input dim {policy_shape[1]} does not match task={self.task} "
                f"expected_obs_dim={self.expected_obs_dim}. Check TEXTOP_TASK and TEXTOP_FUTURE_STEPS."
            )
        print(
            "[TextOpTrackerAdapter] "
            f"VAE providers={self.vae_session.get_providers()}, "
            f"policy providers={self.policy_session.get_providers()}"
        )

        stats = np.load(DEFAULT_VAE_STATS)
        self.mean = stats["mean"].astype(np.float32)
        self.std = stats["std"].astype(np.float32)

    def qpos36_to_body_targets(self, qpos36: np.ndarray, sim_data) -> np.ndarray:
        ref = self.prepare_reference(qpos36)
        targets = []
        for t in range(ref["qpos36"].shape[0]):
            targets.append(self.target_from_reference(ref, sim_data, t))
        return np.stack(targets, axis=0)

    def prepare_reference(self, qpos36: np.ndarray) -> dict[str, np.ndarray]:
        qpos36 = np.asarray(qpos36, dtype=np.float32)
        body_pos, body_quat = self._fk_chunk(qpos36)
        features = self._features70(qpos36)
        windows = self._future_windows((features - self.mean) / self.std)
        latents = self.vae_session.run([self.vae_output], {self.vae_input: windows.astype(np.float32)})[0]
        return {"qpos36": qpos36, "body_pos": body_pos, "body_quat": body_quat, "latents": latents}

    def prepare_reference_external(
        self,
        qpos36: np.ndarray,
        *,
        root6: np.ndarray,
        ee24: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Prepare TextOp reference using external root/EE poses.

        ``qpos36`` is still used for the 70D VAE motion feature.  The future
        anchor and EE observations are taken directly from ``root6`` and
        ``ee24`` instead of MuJoCo FK.
        """
        qpos36 = np.asarray(qpos36, dtype=np.float32)
        root6 = np.asarray(root6, dtype=np.float32)
        ee24 = np.asarray(ee24, dtype=np.float32)
        if root6.shape != (qpos36.shape[0], 6):
            raise ValueError(f"Expected root6 shape {(qpos36.shape[0], 6)}, got {root6.shape}")
        if ee24.shape != (qpos36.shape[0], 24):
            raise ValueError(f"Expected ee24 shape {(qpos36.shape[0], 24)}, got {ee24.shape}")

        features = self._features70(qpos36)
        windows = self._future_windows((features - self.mean) / self.std)
        latents = self.vae_session.run([self.vae_output], {self.vae_input: windows.astype(np.float32)})[0]
        anchor_pos = root6[:, :3].astype(np.float32)
        anchor_quat = R.from_euler("xyz", root6[:, 3:6]).as_quat(scalar_first=True).astype(np.float32)
        ee_pose = ee24.reshape(qpos36.shape[0], 4, 6)
        ee_pos = ee_pose[..., :3].astype(np.float32)
        ee_quat = R.from_euler("xyz", ee_pose[..., 3:6].reshape(-1, 3)).as_quat(scalar_first=True)
        ee_quat = ee_quat.reshape(qpos36.shape[0], 4, 4).astype(np.float32)
        return {
            "qpos36": qpos36,
            "body_pos": None,
            "body_quat": None,
            "latents": latents,
            "anchor_pos": anchor_pos,
            "anchor_quat": anchor_quat,
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
        }

    def target_from_reference(self, ref: dict[str, np.ndarray], sim_data, t: int) -> np.ndarray:
        qpos36 = ref["qpos36"]
        obs = self._obs(
            sim_data,
            qpos36,
            ref["body_pos"],
            ref["body_quat"],
            ref["latents"],
            t,
            anchor_pos=ref.get("anchor_pos"),
            anchor_quat=ref.get("anchor_quat"),
            ee_pos=ref.get("ee_pos"),
            ee_quat=ref.get("ee_quat"),
        )
        action = self.policy_session.run(None, {self.policy_input: obs.reshape(1, -1).astype(np.float32)})[0].reshape(-1)
        target = action[ISAACLAB_TO_MUJOCO] * ACTION_SCALE + DEFAULT_ANGLES
        if DEBUG and t == 0:
            self._dump_step0_debug(sim_data, qpos36[0], action, target, obs)
        self.last_action = action.astype(np.float32)
        return target.astype(np.float32)

    def reset(self):
        self.last_action[:] = 0.0

    def _fk_chunk(self, qpos36: np.ndarray):
        pos = np.empty((len(qpos36), len(DEFAULT_BODY_NAMES), 3), dtype=np.float32)
        quat = np.empty((len(qpos36), len(DEFAULT_BODY_NAMES), 4), dtype=np.float32)
        for i, qpos in enumerate(qpos36):
            self.data.qpos[:] = 0.0
            self.data.qpos[3] = 1.0
            self.data.qpos[:7] = qpos[:7]
            self.data.qpos[self.body_qpos_adrs] = qpos[7:36]
            self.data.qvel[:] = 0.0
            self.mujoco.mj_forward(self.model, self.data)
            pos[i] = self.data.xpos[self.body_ids]
            quat[i] = self.data.xquat[self.body_ids]
        return pos, quat

    def _features70(self, qpos36: np.ndarray) -> np.ndarray:
        joint_pos_mujoco = qpos36[:, 7:36].astype(np.float32)
        joint_vel_mujoco = _finite_diff(joint_pos_mujoco, FPS)
        joint_pos = joint_pos_mujoco[:, MUJOCO_TO_ISAACLAB]
        joint_vel = joint_vel_mujoco[:, MUJOCO_TO_ISAACLAB]
        root_quat = qpos36[:, 3:7]
        yaw_inv = _quat_conjugate(_yaw_quat(root_quat))
        pelvis_quat_b = _quat_mul(yaw_inv, root_quat)
        pelvis_rot6d_b = _rot6d_cols_from_quat(pelvis_quat_b).astype(np.float32)
        lin_vel_w = _finite_diff(qpos36[:, :3], FPS)
        ang_vel_w = _quat_ang_vel(root_quat, FPS)
        pelvis_lin_vel_b = _quat_apply_np(yaw_inv, lin_vel_w).astype(np.float32)
        pelvis_ang_vel_b = _quat_apply_np(yaw_inv, ang_vel_w).astype(np.float32)
        return np.concatenate([joint_pos, joint_vel, pelvis_rot6d_b, pelvis_lin_vel_b, pelvis_ang_vel_b], axis=1)

    def _future_windows(self, features: np.ndarray) -> np.ndarray:
        ids = np.arange(features.shape[0])[:, None] + np.arange(self.vae_window_steps)[None, :]
        ids = np.clip(ids, 0, features.shape[0] - 1)
        return features[ids].astype(np.float32)

    def _obs(
        self,
        sim_data,
        qpos36,
        body_pos,
        body_quat,
        latents,
        t: int,
        *,
        anchor_pos=None,
        anchor_quat=None,
        ee_pos=None,
        ee_quat=None,
    ) -> np.ndarray:
        robot_pos = sim_data.body("pelvis").xpos.copy()
        robot_quat = sim_data.body("pelvis").xquat.copy()
        fut = np.clip(t + np.arange(self.motion_future_steps), 0, len(qpos36) - 1)

        if anchor_pos is None:
            anchor_pos_fut = body_pos[fut, 0]
            anchor_quat_fut = body_quat[fut, 0]
        else:
            anchor_pos_fut = anchor_pos[fut]
            anchor_quat_fut = anchor_quat[fut]
        anchor_pos_b, anchor_quat_b = _subtract_frame(
            np.repeat(robot_pos[None], self.motion_future_steps, axis=0),
            np.repeat(robot_quat[None], self.motion_future_steps, axis=0),
            anchor_pos_fut,
            anchor_quat_fut,
        )
        if ee_pos is None:
            ee_idx = [DEFAULT_BODY_NAMES.index(n) for n in DEFAULT_EE_BODY_NAMES]
            ee_pos_fut = body_pos[fut][:, ee_idx]
            ee_quat_fut = body_quat[fut][:, ee_idx]
        else:
            ee_pos_fut = ee_pos[fut]
            ee_quat_fut = ee_quat[fut]
        ee_pos_b, ee_quat_b = _subtract_frame(
            np.repeat(robot_pos[None, None], self.motion_future_steps, axis=0).repeat(4, axis=1),
            np.repeat(robot_quat[None, None], self.motion_future_steps, axis=0).repeat(4, axis=1),
            ee_pos_fut,
            ee_quat_fut,
        )

        pieces = [
            latents[t].reshape(-1),
            anchor_pos_b.reshape(-1),
            _matrix_rows6_from_quat(anchor_quat_b).reshape(-1),
            robot_pos.astype(np.float32),
            _matrix_rows6_from_quat(robot_quat[None]).reshape(-1),
            self._projected_gravity(sim_data),
            _quat_apply(_quat_inv(sim_data.qpos[3:7]), sim_data.qvel[0:3]).astype(np.float32),
            sim_data.qvel[3:6].astype(np.float32),
            (sim_data.qpos[self.body_qpos_adrs] - DEFAULT_ANGLES)[MUJOCO_TO_ISAACLAB].astype(np.float32),
            sim_data.qvel[self.body_qvel_adrs][MUJOCO_TO_ISAACLAB].astype(np.float32),
            self.last_action.astype(np.float32),
            ee_pos_b.reshape(-1),
            _matrix_rows6_from_quat(ee_quat_b).reshape(-1),
        ]
        obs = np.concatenate(pieces).astype(np.float32)
        if obs.shape[0] != self.expected_obs_dim:
            raise ValueError(f"Expected TextOp obs {self.expected_obs_dim}D for task={self.task}, got {obs.shape}")
        return obs

    @staticmethod
    def _projected_gravity(sim_data) -> np.ndarray:
        qw, qx, qy, qz = sim_data.qpos[3:7]
        return np.asarray([
            2 * (-qz * qx + qw * qy),
            -2 * (qz * qy + qw * qx),
            1 - 2 * (qw * qw + qz * qz),
        ], dtype=np.float32)

    def _dump_step0_debug(self, sim_data, kimodo_qpos0: np.ndarray, action: np.ndarray, target_q: np.ndarray, obs: np.ndarray):
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        current_q = np.asarray(sim_data.qpos[self.body_qpos_adrs], dtype=np.float32)
        current_root = np.asarray(sim_data.qpos[:7], dtype=np.float32)
        kimodo_qpos0 = np.asarray(kimodo_qpos0, dtype=np.float32)
        obs_names = np.asarray(TEXTOP_EEOBS_TERMS)
        obs_dims = _textop_obs_dims(self.task, self.future_steps)
        starts = np.concatenate([[0], np.cumsum(obs_dims)[:-1]])
        obs_abs_max = np.asarray([np.max(np.abs(obs[s:s + d])) for s, d in zip(starts, obs_dims)], dtype=np.float32)
        obs_abs_mean = np.asarray([np.mean(np.abs(obs[s:s + d])) for s, d in zip(starts, obs_dims)], dtype=np.float32)
        np.savez(
            DEBUG_DIR / "step0_debug.npz",
            current_q=current_q,
            target_q=np.asarray(target_q, dtype=np.float32),
            target_minus_current=np.asarray(target_q - current_q, dtype=np.float32),
            current_root=current_root,
            kimodo_root0=kimodo_qpos0[:7],
            root_delta_xyz=kimodo_qpos0[:3] - current_root[:3],
            kimodo_q0=kimodo_qpos0[7:36],
            kimodo_q0_minus_current=kimodo_qpos0[7:36] - current_q,
            raw_policy_action=np.asarray(action, dtype=np.float32),
            obs=np.asarray(obs, dtype=np.float32),
            textop_task=np.asarray(self.task),
            textop_future_steps=np.asarray(self.future_steps),
            textop_motion_future_steps=np.asarray(self.motion_future_steps),
            textop_vae_window_steps=np.asarray(self.vae_window_steps),
            textop_expected_obs_dim=np.asarray(self.expected_obs_dim),
            obs_names=obs_names,
            obs_dims=obs_dims,
            obs_abs_max=obs_abs_max,
            obs_abs_mean=obs_abs_mean,
        )
        print(f"[TextOpDebug] wrote {DEBUG_DIR / 'step0_debug.npz'}")
        print(
            "[TextOpDebug] max|target-current|="
            f"{np.max(np.abs(target_q - current_q)):.4f}, "
            f"root_delta_xyz={kimodo_qpos0[:3] - current_root[:3]}"
        )
        print(
            "[TextOpDebug] obs max "
            + ", ".join(f"{name}={val:.3f}" for name, val in zip(obs_names, obs_abs_max))
        )
