#!/usr/bin/env python3
"""Open a complete local HSSD room in Isaac Sim and optionally save a preview."""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path


SIMPLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = (
    SIMPLE_ROOT
    / "data"
    / "scenes"
    / "hssd"
    / "102344250"
    / "102344250_local.usd"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview an HSSD room with its walls and referenced furniture."
    )
    parser.add_argument("--usd-path", type=Path, default=DEFAULT_SCENE)
    parser.add_argument(
        "--output",
        type=Path,
        default=SIMPLE_ROOT / "data" / "output" / "hssd_scene13_preview.png",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--hide-ceilings", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--keep-open-seconds",
        type=float,
        default=0.0,
        help="Keep a non-headless preview open for this many seconds; 0 waits until closed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    usd_path = args.usd_path.expanduser().resolve()
    if not usd_path.is_file():
        raise FileNotFoundError(f"HSSD USD does not exist: {usd_path}")

    # Isaac/Omniverse modules must be imported after SimulationApp starts.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": args.headless,
            "width": args.width,
            "height": args.height,
            "renderer": "RayTracedLighting",
            "multi_gpu": False,
        }
    )

    try:
        import numpy as np
        import omni.usd
        from pxr import Gf, Usd, UsdGeom, UsdLux, UsdUtils

        context = omni.usd.get_context()
        if not context.open_stage(str(usd_path)):
            raise RuntimeError(f"Isaac Sim could not open: {usd_path}")
        while context.get_stage_loading_status()[2] > 0:
            simulation_app.update()
        print("[step] USD stage opened", flush=True)

        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("Isaac Sim returned no USD stage")

        # Keep preview-only orientation, lighting, and visibility changes out
        # of the reusable local scene asset.
        stage.SetEditTarget(stage.GetSessionLayer())

        # HSSD rooms are authored Y-up. SIMPLE references them into a Z-up
        # simulation stage and rotates the room by +90 degrees around X.
        source_up_axis = UsdGeom.GetStageUpAxis(stage)
        if source_up_axis == UsdGeom.Tokens.y:
            scene_root = UsdGeom.Xformable(stage.GetDefaultPrim())
            preview_op_name = "xformOp:rotateXYZ:simplePreview"
            preview_op = next(
                (
                    op
                    for op in scene_root.GetOrderedXformOps()
                    if op.GetOpName() == preview_op_name
                ),
                None,
            )
            if preview_op is None:
                preview_op = scene_root.AddRotateXYZOp(opSuffix="simplePreview")
            preview_op.Set(Gf.Vec3f(90, 0, 0))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        print(
            f"[step] stage_up_axis={source_up_axis} -> {UsdGeom.Tokens.z}",
            flush=True,
        )

        if args.hide_ceilings:
            ceilings = stage.GetPrimAtPath("/World/ceilings")
            if ceilings.IsValid():
                UsdGeom.Imageable(ceilings).MakeInvisible()

        # The downloaded room contains geometry and materials but no useful
        # standalone lighting. These lights live only in the unsaved preview.
        dome = UsdLux.DomeLight.Define(stage, "/World/PreviewDomeLight")
        dome.CreateIntensityAttr(650.0)
        distant = UsdLux.DistantLight.Define(stage, "/World/PreviewDistantLight")
        distant.CreateIntensityAttr(1800.0)
        distant.CreateAngleAttr(1.0)
        UsdGeom.Xformable(distant).AddRotateXYZOp().Set(Gf.Vec3f(315, 0, 35))

        prims = list(stage.Traverse())
        meshes = sum(prim.GetTypeName() == "Mesh" for prim in prims)
        xforms = sum(prim.GetTypeName() == "Xform" for prim in prims)
        print("[step] USD prims traversed", flush=True)
        _, _, unresolved = UsdUtils.ComputeAllDependencies(str(usd_path))
        print("[step] USD dependencies checked", flush=True)

        purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)
        bounds = bbox_cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
        print("[step] USD bounds computed", flush=True)
        bbox_min = np.asarray(bounds.GetMin(), dtype=np.float64)
        bbox_max = np.asarray(bounds.GetMax(), dtype=np.float64)
        center = (bbox_min + bbox_max) * 0.5
        size = bbox_max - bbox_min
        diagonal = float(np.linalg.norm(size))

        print(f"[ok] USD: {usd_path}", flush=True)
        print(f"[ok] prims={len(prims)} meshes={meshes} xforms={xforms}", flush=True)
        print(f"[ok] unresolved_dependencies={len(unresolved)}", flush=True)
        print(
            f"[ok] bounds_center={center.tolist()} bounds_size={size.tolist()}",
            flush=True,
        )
        if unresolved:
            for missing in unresolved[:10]:
                print(f"[missing] {missing}")
            raise RuntimeError("The HSSD stage still has unresolved dependencies")

        distance = max(diagonal * 0.72, 3.0)
        eye = center + np.array([distance * 0.72, distance * 0.72, distance * 0.55])
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

        # Build the camera transform explicitly. ViewportCameraState can retain
        # the source Y-up convention when the stage up-axis changes at runtime.
        camera_path = "/SimplePreviewCamera"
        camera = UsdGeom.Camera.Define(stage, camera_path)
        camera.CreateFocalLengthAttr(24.0)
        view_matrix = Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*eye), Gf.Vec3d(*center), Gf.Vec3d(0, 0, 1)
        )
        UsdGeom.Xformable(camera).AddTransformOp().Set(view_matrix.GetInverse())
        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("Isaac Sim has no active viewport to capture")
        viewport.set_texture_resolution((args.width, args.height))
        viewport.camera_path = camera_path

        for _ in range(16):
            simulation_app.update()

        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        capture = capture_viewport_to_file(viewport, file_path=str(output))
        import asyncio

        capture_task = asyncio.ensure_future(capture.wait_for_result(completion_frames=30))
        while not capture_task.done():
            simulation_app.update()
        # Isaac Sim 6 may report a false result even after the asynchronous
        # writer has produced a valid PNG, so validate the artifact itself.
        capture_task.result()
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError("Isaac Sim viewport capture did not produce an image")
        print(f"[ok] preview={output}", flush=True)

        if not args.headless:
            started = time.monotonic()
            while simulation_app.is_running():
                simulation_app.update()
                if args.keep_open_seconds > 0:
                    if time.monotonic() - started >= args.keep_open_seconds:
                        break
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
