"""Pure transform helpers for read-only task render proxies."""

from __future__ import annotations

import numpy as np
import transforms3d as t3d


def world_pose_to_scaled_root_local(
    body_position,
    body_orientation,
    root_position,
    root_orientation,
    root_scale,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a body world pose to a non-uniformly scaled root's local pose.

    USD render assets often author a scale on their root prim. Writing a
    descendant's world pose through that non-uniform scale requires a sheared
    transform and can visually shrink articulated parts. This conversion
    instead writes the descendant in the asset's authored, pre-scale frame.
    Quaternions use the MuJoCo/Isaac ``wxyz`` convention.
    """
    body_position = np.asarray(body_position, dtype=np.float64).reshape(3)
    root_position = np.asarray(root_position, dtype=np.float64).reshape(3)
    body_orientation = np.asarray(body_orientation, dtype=np.float64).reshape(4)
    root_orientation = np.asarray(root_orientation, dtype=np.float64).reshape(4)
    root_scale = np.asarray(root_scale, dtype=np.float64).reshape(3)

    if not all(
        np.isfinite(value).all()
        for value in (
            body_position,
            root_position,
            body_orientation,
            root_orientation,
            root_scale,
        )
    ):
        raise ValueError("Task visual poses and scale must be finite")
    if np.any(np.abs(root_scale) <= 1.0e-12):
        raise ValueError(f"Task visual root scale must be non-zero; got {root_scale}")
    body_norm = np.linalg.norm(body_orientation)
    root_norm = np.linalg.norm(root_orientation)
    if body_norm <= 1.0e-12 or root_norm <= 1.0e-12:
        raise ValueError("Task visual orientations must be non-zero quaternions")

    body_orientation = body_orientation / body_norm
    root_orientation = root_orientation / root_norm
    root_rotation = t3d.quaternions.quat2mat(root_orientation)
    local_position = root_rotation.T @ (body_position - root_position)
    local_position = local_position / root_scale
    local_orientation = t3d.quaternions.qmult(
        t3d.quaternions.qinverse(root_orientation), body_orientation
    )
    local_orientation = local_orientation / np.linalg.norm(local_orientation)
    return local_position, local_orientation


def world_pose_to_scaled_root_local_matrix(
    body_position,
    body_orientation,
    root_position,
    root_orientation,
    root_scale,
) -> np.ndarray:
    """Return an exact USD row-vector local matrix for a scaled rigid body.

    A child rotation below a non-uniformly scaled USD root cannot generally be
    represented by translation, quaternion and scale alone: the local matrix
    needs a shear term that cancels the parent's scale before applying the
    body's rigid rotation.  This constructs the desired body world transform
    (asset scale in the body's rotating frame) and solves ``local @ root =
    body`` using USD/Gf's row-vector matrix convention.
    """
    body_position = np.asarray(body_position, dtype=np.float64).reshape(3)
    root_position = np.asarray(root_position, dtype=np.float64).reshape(3)
    body_orientation = np.asarray(body_orientation, dtype=np.float64).reshape(4)
    root_orientation = np.asarray(root_orientation, dtype=np.float64).reshape(4)
    root_scale = np.asarray(root_scale, dtype=np.float64).reshape(3)

    if not all(
        np.isfinite(value).all()
        for value in (
            body_position,
            root_position,
            body_orientation,
            root_orientation,
            root_scale,
        )
    ):
        raise ValueError("Task visual poses and scale must be finite")
    if np.any(np.abs(root_scale) <= 1.0e-12):
        raise ValueError(f"Task visual root scale must be non-zero; got {root_scale}")

    body_norm = float(np.linalg.norm(body_orientation))
    root_norm = float(np.linalg.norm(root_orientation))
    if body_norm <= 1.0e-12 or root_norm <= 1.0e-12:
        raise ValueError("Task visual orientations must be non-zero quaternions")

    def _scaled_world_matrix(position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        rotation = t3d.quaternions.quat2mat(orientation / np.linalg.norm(orientation))
        matrix = np.eye(4, dtype=np.float64)
        # transforms3d uses column vectors; USD/Gf matrices use row vectors.
        matrix[:3, :3] = np.diag(root_scale) @ rotation.T
        matrix[3, :3] = position
        return matrix

    body_world = _scaled_world_matrix(body_position, body_orientation)
    root_world = _scaled_world_matrix(root_position, root_orientation)
    return body_world @ np.linalg.inv(root_world)
