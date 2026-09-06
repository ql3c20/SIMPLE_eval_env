from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


MUJOCO_TO_KIMODO = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

EE_FIELDS = (
    "left_hand_pose",
    "right_hand_pose",
    "left_foot_pose",
    "right_foot_pose",
)


@dataclass
class KimodoAdapterConfig:
    kimodo_root: Path = Path(os.environ.get("KIMODO_ROOT", "/pfs/pfs-ilWc5D/yzh/kimodo_my"))
    distill_config: Path = Path(
        os.environ.get(
            "KIMODO_DISTILL_CONFIG",
            "/pfs/pfs-ilWc5D/yzh/kimodo_my/outputs/"
            "g1_distill_16to8_100to20_dagger_teacher_gt03_100k_bs4x4_k3_cosine_selfacc50_"
            "rootacc10_headingacc10_gtbranch/resolved_config.yaml",
        )
    )
    distill_ckpt: Path = Path(
        os.environ.get(
            "KIMODO_DISTILL_CKPT",
            "/pfs/pfs-ilWc5D/yzh/kimodo_my/outputs/"
            "g1_distill_16to8_100to20_dagger_teacher_gt03_100k_bs4x4_k3_cosine_selfacc50_"
            "rootacc10_headingacc10_gtbranch/ema_final.pt",
        )
    )
    python: str = os.environ.get("KIMODO_PYTHON", "uv run python")
    prompt: str = os.environ.get("KIMODO_PROMPT", "a humanoid bends and picks up an object")
    source_fps: float = float(os.environ.get("KIMODO_SOURCE_FPS", "50"))
    kimodo_fps: float = float(os.environ.get("KIMODO_FPS", "30"))
    output_fps: float = float(os.environ.get("KIMODO_OUTPUT_FPS", "50"))
    keyframe_step: int = int(os.environ.get("KIMODO_KEYFRAME_STEP", "10"))
    diffusion_steps: int = int(os.environ.get("KIMODO_DIFFUSION_STEPS", "20"))
    seed: int | None = int(os.environ["KIMODO_SEED"]) if "KIMODO_SEED" in os.environ else None
    cfg_type: str = os.environ.get("KIMODO_CFG_TYPE", "separated")
    cfg_weight_text: float = float(os.environ.get("KIMODO_CFG_WEIGHT_TEXT", "2.0"))
    cfg_weight_constraint: float = float(os.environ.get("KIMODO_CFG_WEIGHT_CONSTRAINT", "2.0"))
    work_dir: Path = Path(os.environ.get("KIMODO_WORK_DIR", "/tmp/psi0_kimodo_eval"))
    keep_work: bool = os.environ.get("KIMODO_KEEP_WORK", "0") == "1"
    server_url: str | None = os.environ.get("KIMODO_SERVER_URL")
    base_cmd_smooth_window: int = int(os.environ.get("KIMODO_BASE_CMD_SMOOTH_WINDOW", "5"))
    max_abs_vx: float = float(os.environ.get("KIMODO_MAX_ABS_VX", "0.5"))
    max_abs_vy: float = float(os.environ.get("KIMODO_MAX_ABS_VY", "0.08"))
    max_abs_vyaw: float = float(os.environ.get("KIMODO_MAX_ABS_VYAW", "0.25"))
    anchor_mode: str = os.environ.get("KIMODO_ANCHOR_MODE", "policy_only")
    policy_only_initial_qpos: bool = (
        os.environ.get("KIMODO_POLICY_ONLY_INITIAL_QPOS", "0") == "1"
    )
    rtc_prefix_frames: int = int(os.environ.get("KIMODO_RTC_PREFIX_FRAMES", "0"))
    rtc_prefix_blend_frames: int = int(os.environ.get("KIMODO_RTC_PREFIX_BLEND_FRAMES", "0"))
    episode_subdir: bool = os.environ.get("KIMODO_EPISODE_SUBDIR", "1") == "1"


def _rot_mujoco_to_kimodo(rot_m: np.ndarray) -> np.ndarray:
    return MUJOCO_TO_KIMODO @ rot_m @ MUJOCO_TO_KIMODO.T


def _xyz_mujoco_to_kimodo(xyz_m: np.ndarray) -> np.ndarray:
    return np.asarray([xyz_m[1], xyz_m[2], xyz_m[0]], dtype=np.float64)


def _yaw_from_rot_kimodo(rot_k: np.ndarray) -> float:
    forward = rot_k @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return float(np.arctan2(forward[0], forward[2]))


def _yaw_from_quat_mujoco(quat_wxyz: np.ndarray) -> float:
    return float(R.from_quat(quat_wxyz, scalar_first=True).as_euler("xyz")[2])


def _select_keyframes(num_frames: int, step: int) -> np.ndarray:
    idx = np.arange(0, num_frames, max(1, step), dtype=np.int64)
    if len(idx) == 0 or idx[-1] != num_frames - 1:
        idx = np.append(idx, num_frames - 1)
    return idx


def _resample_qpos(qpos: np.ndarray, input_fps: float, output_fps: float, target_frames: int) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"Expected qpos shape (T, 36), got {qpos.shape}")
    if qpos.shape[0] == target_frames and abs(input_fps - output_fps) < 1e-6:
        return qpos.astype(np.float32)

    src_t = np.arange(qpos.shape[0], dtype=np.float64) / float(input_fps)
    dst_t = np.arange(target_frames, dtype=np.float64) / float(output_fps)
    dst_t = np.clip(dst_t, src_t[0], src_t[-1])

    out = np.empty((target_frames, 36), dtype=np.float64)
    for j in (0, 1, 2):
        out[:, j] = np.interp(dst_t, src_t, qpos[:, j])
    quat = R.from_quat(qpos[:, 3:7], scalar_first=True)
    out[:, 3:7] = Slerp(src_t, quat)(dst_t).as_quat(scalar_first=True)
    for j in range(7, 36):
        out[:, j] = np.interp(dst_t, src_t, qpos[:, j])
    return out.astype(np.float32)


def _finite_diff(values: np.ndarray, fps: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] <= 1:
        return np.zeros_like(values, dtype=np.float32)
    diff = np.empty_like(values, dtype=np.float64)
    diff[:-1] = values[1:] - values[:-1]
    diff[-1] = diff[-2]
    return (diff * float(fps)).astype(np.float32)


def _finite_diff_angle(angles: np.ndarray, fps: float) -> np.ndarray:
    angles = np.asarray(angles, dtype=np.float64)
    if angles.shape[0] <= 1:
        return np.zeros_like(angles, dtype=np.float32)
    unwrapped = np.unwrap(angles)
    return _finite_diff(unwrapped, fps)


def _smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if window <= 1 or values.shape[0] <= 2:
        return values
    window = min(int(window), int(values.shape[0]))
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.full(window, 1.0 / window, dtype=np.float32)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _inherit_qpos_prefix(
    generated: np.ndarray,
    prefix: np.ndarray,
    blend_frames: int = 0,
) -> np.ndarray:
    out = np.asarray(generated, dtype=np.float32).copy()
    inherited = np.asarray(prefix, dtype=np.float32)
    if out.ndim != 2 or out.shape[1] != 36:
        raise ValueError(f"Expected generated qpos shape (T, 36), got {out.shape}")
    if inherited.ndim != 2 or inherited.shape[1] != 36:
        raise ValueError(f"Expected prefix qpos shape (T, 36), got {inherited.shape}")
    prefix_frames = min(int(inherited.shape[0]), int(out.shape[0]))
    if prefix_frames <= 0:
        return out

    out[:prefix_frames] = inherited[-prefix_frames:]
    if blend_frames <= 0 or prefix_frames >= out.shape[0]:
        return out
    if blend_frames == 1:
        raise ValueError("KIMODO_RTC_PREFIX_BLEND_FRAMES must be 0 or >= 2")

    blend_count = min(int(blend_frames), int(out.shape[0] - prefix_frames))
    blend_u = np.linspace(0.0, 1.0, blend_count, dtype=np.float64)
    weights = 1.0 - blend_u * blend_u * (3.0 - 2.0 * blend_u)

    generated64 = np.asarray(generated, dtype=np.float64)
    linear_indices = np.r_[0:3, 7:36]
    linear_offset = inherited[-1, linear_indices].astype(np.float64) - generated64[
        prefix_frames - 1, linear_indices
    ]
    out[prefix_frames : prefix_frames + blend_count, linear_indices] = (
        generated64[prefix_frames : prefix_frames + blend_count, linear_indices]
        + weights[:, None] * linear_offset[None, :]
    ).astype(np.float32)

    prefix_rotation = R.from_quat(inherited[-1, 3:7], scalar_first=True)
    generated_boundary = R.from_quat(generated64[prefix_frames - 1, 3:7], scalar_first=True)
    rotation_offset = prefix_rotation * generated_boundary.inv()
    rotation_blend = Slerp(
        [0.0, 1.0],
        R.concatenate([R.identity(), rotation_offset]),
    )(weights)
    generated_rotations = R.from_quat(
        generated64[prefix_frames : prefix_frames + blend_count, 3:7],
        scalar_first=True,
    )
    out[prefix_frames : prefix_frames + blend_count, 3:7] = (
        rotation_blend * generated_rotations
    ).as_quat(scalar_first=True).astype(np.float32)
    return out


class KimodoPolicyAdapter:
    def __init__(self, cfg: KimodoAdapterConfig | None = None):
        self.cfg = cfg or KimodoAdapterConfig()
        self._call_index = 0
        self._episode_index = -1
        self._episode_chunk_index = 0
        self._server_config: dict[str, object] | None = None
        self._previous_qpos_tail: np.ndarray | None = None
        if self.cfg.rtc_prefix_frames < 0:
            raise ValueError("KIMODO_RTC_PREFIX_FRAMES must be >= 0")
        if self.cfg.rtc_prefix_blend_frames < 0 or self.cfg.rtc_prefix_blend_frames == 1:
            raise ValueError("KIMODO_RTC_PREFIX_BLEND_FRAMES must be 0 or >= 2")

    def begin_episode(self) -> None:
        self._episode_index += 1
        self._call_index = 0
        self._episode_chunk_index = 0
        self._previous_qpos_tail = None

    def policy44_to_simple36(
        self,
        policy_action: np.ndarray,
        *,
        prev_qpos_mujoco: np.ndarray | None,
        current_constraints28_mujoco: np.ndarray | None = None,
        instruction: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        policy_action = np.asarray(policy_action, dtype=np.float32)
        if policy_action.ndim != 2 or policy_action.shape[1] != 44:
            raise ValueError(f"Expected policy_action shape (T, 44), got {policy_action.shape}")

        root6 = policy_action[:, 14:20].astype(np.float64)
        ee24 = policy_action[:, 20:44].astype(np.float64)
        root_xyzyaw = np.concatenate([root6[:, :3], root6[:, 5:6]], axis=1)
        constraints28 = np.concatenate([root_xyzyaw, ee24], axis=1).astype(np.float64)
        policy_constraints28 = constraints28.copy()
        hand14 = policy_action[:, :14].astype(np.float32)
        target_yaw = root6[:, 5:6].astype(np.float32)
        prefix_qpos_world = None
        if self.cfg.rtc_prefix_frames > 0 and self._previous_qpos_tail is not None:
            prefix_qpos_world = self._previous_qpos_tail[-self.cfg.rtc_prefix_frames :].copy()

        root_xy = policy_constraints28[:, :2].copy()
        policy_anchor_xy = root_xy[0].copy()
        anchor_mode = self._effective_anchor_mode()
        use_current_start = anchor_mode in {"current", "policy_delta"}
        if anchor_mode not in {"current", "policy_delta", "policy_only"}:
            raise ValueError(
                f"Unsupported KIMODO_ANCHOR_MODE={anchor_mode!r}; "
                "use policy_only, policy_delta, or current."
            )
        constrain_episode_start = (
            self.cfg.policy_only_initial_qpos
            and anchor_mode == "policy_only"
            and self._episode_chunk_index == 0
            and prev_qpos_mujoco is not None
        )

        if prev_qpos_mujoco is not None and use_current_start:
            prev_qpos_mujoco = np.asarray(prev_qpos_mujoco, dtype=np.float64).reshape(-1)
            anchor_xy = prev_qpos_mujoco[:2].copy()
            constraints28[0, 0:3] = prev_qpos_mujoco[:3]
            constraints28[0, 3] = _yaw_from_quat_mujoco(prev_qpos_mujoco[3:7])
        else:
            anchor_xy = policy_anchor_xy.copy()
            if prev_qpos_mujoco is not None:
                prev_qpos_mujoco = np.asarray(prev_qpos_mujoco, dtype=np.float64).reshape(-1)

        if current_constraints28_mujoco is not None and use_current_start:
            current_constraints28_mujoco = np.asarray(current_constraints28_mujoco, dtype=np.float64).reshape(-1)
            if current_constraints28_mujoco.size != 28:
                raise ValueError(f"Expected current constraints to be 28D, got {current_constraints28_mujoco.shape}")
            constraints28[0, 4:28] = current_constraints28_mujoco[4:28]

        constraints_local = constraints28.copy()
        recenter_xy = anchor_xy
        if prev_qpos_mujoco is not None and anchor_mode == "policy_delta":
            # Future root/EE constraints come from the policy's own predicted coordinate frame.
            # Use only their displacement from the policy's first predicted root, then attach that
            # local plan to the current simulated root.  This avoids pulling the robot back toward
            # any absolute-coordinate bias in the policy output.
            recenter_xy = policy_anchor_xy
        elif anchor_mode == "policy_only":
            recenter_xy = policy_anchor_xy

        constraints_local[:, 0] -= recenter_xy[0]
        constraints_local[:, 1] -= recenter_xy[1]
        for start in (4, 10, 16, 22):
            constraints_local[:, start] -= recenter_xy[0]
            constraints_local[:, start + 1] -= recenter_xy[1]

        if prev_qpos_mujoco is not None and anchor_mode == "policy_delta":
            constraints_local[0, 0] = 0.0
            constraints_local[0, 1] = 0.0
            if current_constraints28_mujoco is not None:
                for start in (4, 10, 16, 22):
                    constraints_local[0, start] = current_constraints28_mujoco[start] - anchor_xy[0]
                    constraints_local[0, start + 1] = current_constraints28_mujoco[start + 1] - anchor_xy[1]

        with self._workspace() as work:
            constraints_path = work / "constraints.json"
            heading_path = work / "heading_source.npz"
            output_stem = work / "g1_generated"
            start_qpos_local = self._heading_qpos(prev_qpos_mujoco, constraints28)
            start_qpos_local = np.asarray(start_qpos_local, dtype=np.float32).reshape(-1, 36)
            start_qpos_local[:, 0] -= anchor_xy[0]
            start_qpos_local[:, 1] -= anchor_xy[1]
            prefix_qpos_local = None
            if prefix_qpos_world is not None:
                prefix_model_frames = max(
                    1,
                    int(round((prefix_qpos_world.shape[0] - 1) * self.cfg.kimodo_fps / self.cfg.output_fps))
                    + 1,
                )
                prefix_qpos_local = _resample_qpos(
                    prefix_qpos_world,
                    input_fps=self.cfg.output_fps,
                    output_fps=self.cfg.kimodo_fps,
                    target_frames=prefix_model_frames,
                )
                prefix_qpos_local[:, 0] -= anchor_xy[0]
                prefix_qpos_local[:, 1] -= anchor_xy[1]
            np.savez(
                heading_path,
                qpos=start_qpos_local,
                anchor_xy=np.asarray(anchor_xy, dtype=np.float32),
                recenter_xy=np.asarray(recenter_xy, dtype=np.float32),
                anchor_mode=np.asarray(anchor_mode),
            )
            self._write_constraints_json(
                constraints_local,
                constraints_path,
                keyframe_step=self._effective_keyframe_step(),
                skip_first_frame=constrain_episode_start,
            )
            request_start_qpos = (
                start_qpos_local[0]
                if prev_qpos_mujoco is not None and (use_current_start or constrain_episode_start)
                else None
            )
            if constrain_episode_start:
                print(
                    "[KimodoAdapter] policy_only episode frame-0 qpos constraint enabled",
                    flush=True,
                )
            self._run_kimodo(
                constraints_path=constraints_path,
                heading_path=heading_path,
                output_stem=output_stem,
                duration_sec=policy_action.shape[0] / self.cfg.source_fps,
                prompt=instruction or self.cfg.prompt,
                start_qpos_mujoco=request_start_qpos,
                prefix_qpos_mujoco=prefix_qpos_local,
            )
            qpos = np.loadtxt(str(output_stem.with_suffix(".csv")), delimiter=",").astype(np.float32)
            if qpos.ndim == 1:
                qpos = qpos[None, :]

        qpos50 = _resample_qpos(
            qpos,
            input_fps=self.cfg.kimodo_fps,
            output_fps=self.cfg.output_fps,
            target_frames=policy_action.shape[0],
        ).astype(np.float32)
        qpos50[:, 0] += anchor_xy[0]
        qpos50[:, 1] += anchor_xy[1]
        if prev_qpos_mujoco is not None and use_current_start:
            qpos50[0] = prev_qpos_mujoco.astype(np.float32)
        if constrain_episode_start:
            # The reset state is authoritative.  The dense Kimodo constraint
            # should already reproduce it, but explicitly pin the resampled
            # output so CSV row 0, Isaac request 0 and Kimodo qpos50[0] are
            # numerically identical before TextOp executes its first target.
            qpos50[0] = prev_qpos_mujoco.astype(np.float32)
            if not np.array_equal(qpos50[0], prev_qpos_mujoco.astype(np.float32)):
                raise RuntimeError("failed to pin Kimodo frame 0 to the reset qpos")
            print(
                "[KimodoAdapter] aligned qpos50[0] exactly to current reset qpos",
                flush=True,
            )
        if prefix_qpos_world is not None:
            qpos50 = _inherit_qpos_prefix(
                qpos50,
                prefix_qpos_world,
                blend_frames=self.cfg.rtc_prefix_blend_frames,
            )
            print(
                "[KimodoAdapter] inherited qpos prefix: "
                f"frames={prefix_qpos_world.shape[0]} "
                f"blend_frames={self.cfg.rtc_prefix_blend_frames}",
                flush=True,
            )
        if self.cfg.rtc_prefix_frames > 0:
            self._previous_qpos_tail = qpos50[-self.cfg.rtc_prefix_frames :].copy()

        simple36 = self._qpos_to_simple36(qpos50, hand14=hand14, target_yaw=target_yaw)
        self._episode_chunk_index += 1
        return simple36, qpos50

    def _heading_qpos(self, prev_qpos_mujoco: np.ndarray | None, constraints28: np.ndarray) -> np.ndarray:
        if prev_qpos_mujoco is not None:
            return np.asarray(prev_qpos_mujoco, dtype=np.float32).reshape(1, 36)
        qpos = np.zeros((1, 36), dtype=np.float32)
        qpos[0, :3] = constraints28[0, :3]
        qpos[0, 3:7] = R.from_euler("xyz", [0.0, 0.0, constraints28[0, 3]]).as_quat(
            scalar_first=True
        ).astype(np.float32)
        return qpos

    def _get_server_config(self) -> dict[str, object]:
        if not self.cfg.server_url:
            return {}
        if self._server_config is None:
            import requests

            response = requests.get(
                self.cfg.server_url.rstrip("/") + "/config",
                timeout=10,
            )
            if response.status_code == 404:
                self._server_config = {}
                print(
                    "[KimodoAdapter] Server /config unavailable; using local "
                    f"keyframe_step={self.cfg.keyframe_step}, anchor_mode={self.cfg.anchor_mode}",
                    flush=True,
                )
                return self._server_config
            response.raise_for_status()
            self._server_config = dict(response.json())
            print(
                "[KimodoAdapter] Using server config: "
                f"keyframe_step={self._server_config.get('keyframe_step')}, "
                f"anchor_mode={self._server_config.get('anchor_mode')}",
                flush=True,
            )
        return self._server_config

    def _effective_keyframe_step(self) -> int:
        config = self._get_server_config()
        return max(1, int(config.get("keyframe_step", self.cfg.keyframe_step)))

    def _effective_anchor_mode(self) -> str:
        config = self._get_server_config()
        return str(config.get("anchor_mode", self.cfg.anchor_mode))

    def _write_constraints_json(
        self,
        constraints28: np.ndarray,
        path: Path,
        *,
        keyframe_step: int | None = None,
        skip_first_frame: bool = False,
    ) -> None:
        step = self.cfg.keyframe_step if keyframe_step is None else keyframe_step
        src_idx = _select_keyframes(constraints28.shape[0], step)
        if skip_first_frame:
            src_idx = src_idx[src_idx != 0]
            if len(src_idx) == 0:
                raise ValueError("Cannot skip the only policy constraint frame")
        elif 0 not in src_idx:
            src_idx = np.insert(src_idx, 0, 0)
        dst_idx = np.round(src_idx * self.cfg.kimodo_fps / self.cfg.source_fps).astype(np.int64)
        keep = np.ones(len(dst_idx), dtype=bool)
        keep[1:] = dst_idx[1:] != dst_idx[:-1]
        src_idx = src_idx[keep]
        dst_idx = dst_idx[keep]

        item: dict[str, object] = {
            "type": "ee-pose",
            "frame_indices": dst_idx.tolist(),
            "root_xyzyaw": [],
        }
        for field in EE_FIELDS:
            item[field] = []

        for t in src_idx:
            root_xyz_m = constraints28[t, 0:3]
            root_yaw_m = constraints28[t, 3]
            root_rot_m = R.from_euler("xyz", [0.0, 0.0, root_yaw_m]).as_matrix()
            root_xyz_k = _xyz_mujoco_to_kimodo(root_xyz_m)
            root_yaw_k = _yaw_from_rot_kimodo(_rot_mujoco_to_kimodo(root_rot_m))
            item["root_xyzyaw"].append([float(root_xyz_k[0]), float(root_xyz_k[1]), float(root_xyz_k[2]), root_yaw_k])

            for field_i, field in enumerate(EE_FIELDS):
                start = 4 + field_i * 6
                xyz_m = constraints28[t, start : start + 3]
                rpy_m = constraints28[t, start + 3 : start + 6]
                rot_k = _rot_mujoco_to_kimodo(R.from_euler("xyz", rpy_m).as_matrix())
                xyz_k = _xyz_mujoco_to_kimodo(xyz_m)
                rpy_k = R.from_matrix(rot_k).as_euler("xyz")
                item[field].append(
                    [
                        float(xyz_k[0]),
                        float(xyz_k[1]),
                        float(xyz_k[2]),
                        float(rpy_k[0]),
                        float(rpy_k[1]),
                        float(rpy_k[2]),
                    ]
                )

        path.write_text(json.dumps([item], indent=2), encoding="utf-8")

    def _run_kimodo(
        self,
        *,
        constraints_path: Path,
        heading_path: Path,
        output_stem: Path,
        duration_sec: float,
        prompt: str,
        start_qpos_mujoco: np.ndarray | None = None,
        prefix_qpos_mujoco: np.ndarray | None = None,
    ) -> None:
        if self.cfg.server_url:
            self._run_kimodo_server(
                constraints_path=constraints_path,
                heading_path=heading_path,
                output_stem=output_stem,
                duration_sec=duration_sec,
                prompt=prompt,
                start_qpos_mujoco=start_qpos_mujoco,
                prefix_qpos_mujoco=prefix_qpos_mujoco,
            )
            return

        if prefix_qpos_mujoco is not None:
            raise RuntimeError("Kimodo qpos prefix inheritance requires KIMODO_SERVER_URL")

        script = self.cfg.kimodo_root / "scripts" / "generate_g1_with_first_heading.py"
        cmd = [
            *shlex.split(self.cfg.python),
            str(script),
            prompt,
            "--duration",
            f"{duration_sec:.6f}",
            "--constraints",
            str(constraints_path),
            "--heading_source_npz",
            str(heading_path),
            "--output",
            str(output_stem),
            "--distill_config",
            str(self.cfg.distill_config),
            "--distill_ckpt",
            str(self.cfg.distill_ckpt),
            "--diffusion_steps",
            str(self.cfg.diffusion_steps),
            "--cfg_type",
            self.cfg.cfg_type,
            "--cfg_weight",
            str(self.cfg.cfg_weight_text),
            str(self.cfg.cfg_weight_constraint),
            "--hard_project_observed_motion",
        ]
        if self.cfg.seed is not None:
            cmd += ["--seed", str(self.cfg.seed + self._call_index)]
        self._call_index += 1
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.cfg.kimodo_root) + os.pathsep + env.get("PYTHONPATH", "")
        kimodo_hf_home = self.cfg.kimodo_root / "huggingface"
        env["HF_HOME"] = str(kimodo_hf_home)
        env["HUGGINGFACE_HUB_CACHE"] = str(kimodo_hf_home / "hub")
        env["HUGGINGFACE_CACHE_DIR"] = str(kimodo_hf_home / "hub")
        env["TRANSFORMERS_OFFLINE"] = env.get("TRANSFORMERS_OFFLINE", "1")
        env["HF_HUB_OFFLINE"] = env.get("HF_HUB_OFFLINE", "1")
        subprocess.run(cmd, cwd=str(self.cfg.kimodo_root), env=env, check=True)
        csv_path = output_stem.with_suffix(".csv")
        if not csv_path.exists():
            raise FileNotFoundError(f"Kimodo did not produce expected CSV: {csv_path}")

    def _run_kimodo_server(
        self,
        *,
        constraints_path: Path,
        heading_path: Path,
        output_stem: Path,
        duration_sec: float,
        prompt: str,
        start_qpos_mujoco: np.ndarray | None = None,
        prefix_qpos_mujoco: np.ndarray | None = None,
    ) -> None:
        import requests

        assert self.cfg.server_url is not None
        payload = {
            "prompt": prompt,
            "duration": float(duration_sec),
            "constraints": str(constraints_path),
            "heading_source_npz": str(heading_path),
            "output": str(output_stem),
            "diffusion_steps": int(self.cfg.diffusion_steps),
            "sampler": "auto",
            "num_samples": 1,
            "num_transition_frames": 5,
            "hard_project_observed_motion": True,
            "cfg_type": self.cfg.cfg_type,
            "cfg_weight": [float(self.cfg.cfg_weight_text), float(self.cfg.cfg_weight_constraint)],
        }
        if start_qpos_mujoco is not None:
            payload["start_qpos_mujoco"] = np.asarray(start_qpos_mujoco, dtype=np.float32).reshape(36).tolist()
        if prefix_qpos_mujoco is not None:
            payload["prefix_qpos_mujoco"] = np.asarray(
                prefix_qpos_mujoco, dtype=np.float32
            ).reshape(-1, 36).tolist()
        if self.cfg.seed is not None:
            payload["seed"] = int(self.cfg.seed + self._call_index)
        self._call_index += 1

        response = requests.post(
            self.cfg.server_url.rstrip("/") + "/generate",
            json=payload,
            timeout=None,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Kimodo server failed ({response.status_code}): {response.text}")
        data = response.json()
        if data.get("status") != "ok":
            raise RuntimeError(f"Kimodo server returned non-ok response: {data}")
        csv_path = output_stem.with_suffix(".csv")
        if not csv_path.exists():
            raise FileNotFoundError(f"Kimodo server did not produce expected CSV: {csv_path}")

    def _qpos_to_simple36(self, qpos: np.ndarray, *, hand14: np.ndarray, target_yaw: np.ndarray) -> np.ndarray:
        if qpos.shape[1] != 36:
            raise ValueError(f"Expected qpos36, got {qpos.shape}")
        n = qpos.shape[0]
        rpy = R.from_quat(qpos[:, 3:7], scalar_first=True).as_euler("xyz").astype(np.float32)
        yaw = rpy[:, 2]
        vx_world = _finite_diff(qpos[:, 0], self.cfg.output_fps)
        vy_world = _finite_diff(qpos[:, 1], self.cfg.output_fps)
        cos_yaw = np.cos(yaw).astype(np.float32)
        sin_yaw = np.sin(yaw).astype(np.float32)

        # SIMPLE/WBC navigation commands use robot-local planar velocity.
        # Kimodo qpos is a world-frame root trajectory, so invert the yaw
        # integration used when building the Kimodo training constraints.
        vx = cos_yaw * vx_world + sin_yaw * vy_world
        vy = -sin_yaw * vx_world + cos_yaw * vy_world
        vyaw = _finite_diff_angle(yaw, self.cfg.output_fps)

        vx = _smooth_1d(vx, self.cfg.base_cmd_smooth_window)
        vy = _smooth_1d(vy, self.cfg.base_cmd_smooth_window)
        vyaw = _smooth_1d(vyaw, self.cfg.base_cmd_smooth_window)
        vx = np.clip(vx, -self.cfg.max_abs_vx, self.cfg.max_abs_vx).reshape(n, 1)
        vy = np.clip(vy, -self.cfg.max_abs_vy, self.cfg.max_abs_vy).reshape(n, 1)
        vyaw = np.clip(vyaw, -self.cfg.max_abs_vyaw, self.cfg.max_abs_vyaw).reshape(n, 1)
        joints29 = qpos[:, 7:36].astype(np.float32)
        arm14 = joints29[:, 15:29]
        height = qpos[:, 2:3].astype(np.float32)
        return np.concatenate([hand14, arm14, rpy, height, vx, vy, vyaw, target_yaw], axis=1).astype(np.float32)

    def _workspace(self):
        parent = self.cfg.work_dir
        if self.cfg.episode_subdir:
            episode_index = max(self._episode_index, 0)
            parent = parent / f"ep{episode_index:04d}"
        if self.cfg.keep_work:
            path = parent / f"chunk_{self._call_index:06d}"
            path.mkdir(parents=True, exist_ok=True)
            return _PersistentWorkspace(path)
        parent.mkdir(parents=True, exist_ok=True)
        return _TemporaryWorkspace(parent)


class _PersistentWorkspace:
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self) -> Path:
        return self.path

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _TemporaryWorkspace:
    def __init__(self, parent: Path):
        self.parent = parent
        self._tmp: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> Path:
        self._tmp = tempfile.TemporaryDirectory(prefix="chunk_", dir=str(self.parent))
        return Path(self._tmp.__enter__())

    def __exit__(self, exc_type, exc, tb) -> bool:
        assert self._tmp is not None
        return bool(self._tmp.__exit__(exc_type, exc, tb))
