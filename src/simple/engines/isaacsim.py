"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import importlib.util
import os
from typing import Dict, Tuple

import carb
import carb.settings
import numpy as np
import omni.isaac.core.utils.prims as isaacsim_prims
import omni.isaac.core.utils.stage as isaacsim_stage
import omni.kit.commands
import omni.kit.primitive.mesh
import omni.replicator.core as rep
import omni.usd
import transforms3d as t3d
from curobo.types.state import JointState
from isaacsim.util.debug_draw import _debug_draw
from omni.isaac.core.objects import GroundPlane, cuboid, sphere
from omni.isaac.core.prims import GeometryPrim, RigidPrim, XFormPrim
from omni.isaac.core.robots.robot import Robot as IsaacRobot

from isaacsim.core.prims import SingleArticulation
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.world.world import World
# from omni.isaac.core.materials import VisualMaterial
from omni.isaac.sensor import Camera as IssacCamera
from omni.kit.material.library import create_mdl_material
from pxr import Gf, Sdf, Semantics, Tf, Usd, UsdGeom, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation

import simple
import simple.scenes
# from simple.core import sensors
import simple.sensors
from simple.core.actor import ObjectActor, VisualFrame, VisualGrasp, ArticulatedObjectActor
from simple.core.robot import Robot
from simple.core.simulator import Simulator
from simple.core.task import Task
from simple.scenes.hssd import HssdSuite
from simple.utils import env_flag, resolve_data_path
from simple.task_visuals import world_pose_to_scaled_root_local_matrix

# import isaacsim

# from isaacsim.core.api.world.world import World
# from isaacsim.core.api.robots.robot import Robot
# from isaacsim.core.prims import RigidPrim, GeometryPrim, XFormPrim
# from isaacsim.sensors.camera import Camera
# import omni.replicator.core as rep
# import carb.settings
# import omni.usd
# import omni.kit.commands

# import isaacsim.core.utils.prims as isaacsim_prims
# import isaacsim.core.utils.stage as isaacsim_stage


     

# import importlib.resources as res
# from importlib.resources import as_file



def euler_angles_to_quats(euler_angles: np.ndarray, degrees: bool = False, extrinsic: bool = True, device=None
) -> np.ndarray:
    if extrinsic:
        order = "xyz"
    else:
        order = "XYZ"
    rot = Rotation.from_euler(order, euler_angles, degrees=degrees)
    result = rot.as_quat()
    if len(result.shape) == 1:
        result = result[[3, 0, 1, 2]]
    else:
        result = result[:, [3, 0, 1, 2]]
    return result

def compute_obb(bbox_cache: UsdGeom.BBoxCache, prim_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the Oriented Bounding Box (OBB) of a prim

    .. note::

        * The OBB does not guarantee the smallest possible bounding box, it rotates and scales the default AABB.
        * The rotation matrix incorporates any scale factors applied to the object.
        * The `half_extent` values do not include these scaling effects.

    Args:
        bbox_cache (UsdGeom.BBoxCache): USD Bounding Box Cache object to use for computation
        prim_path (str): Prim path to compute OBB for

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the following OBB information:
            - The centroid of the OBB as a NumPy array.
            - The axes of the OBB as a 2D NumPy array, where each row represents a different axis.
            - The half extent of the OBB as a NumPy array.

    Example:

    .. code-block:: python

        >>> import omni.isaac.core.utils.bounds as bounds_utils
        >>>
        >>> # 1 stage unit length cube centered at (0.0, 0.0, 0.0)
        >>> cache = bounds_utils.create_bbox_cache()
        >>> centroid, axes, half_extent = bounds_utils.compute_obb(cache, prim_path="/World/Cube")
        >>> centroid
        [0. 0. 0.]
        >>> axes
        [[1. 0. 0.]
         [0. 1. 0.]
         [0. 0. 1.]]
        >>> half_extent
        [0.5 0.5 0.5]
        >>>
        >>> # the same cube rotated 45 degrees around the z-axis
        >>> cache = bounds_utils.create_bbox_cache()
        >>> centroid, axes, half_extent = bounds_utils.compute_obb(cache, prim_path="/World/Cube")
        >>> centroid
        [0. 0. 0.]
        >>> axes
        [[ 0.70710678  0.70710678  0.        ]
         [-0.70710678  0.70710678  0.        ]
         [ 0.          0.          1.        ]]
        >>> half_extent
        [0.5 0.5 0.5]
    """
    # Compute the BBox3d for the prim
    prim = get_prim_at_path(prim_path)
    bound = bbox_cache.ComputeWorldBound(prim)

    # Compute the translated centroid of the world bound
    centroid = bound.ComputeCentroid()

    # Compute the axis vectors of the OBB
    # NOTE: The rotation matrix incorporates the scale factors applied to the object
    rotation_matrix = bound.GetMatrix().ExtractRotationMatrix()
    x_axis = rotation_matrix.GetRow(0)
    y_axis = rotation_matrix.GetRow(1)
    z_axis = rotation_matrix.GetRow(2)

    # Compute the half-lengths of the OBB along each axis
    # NOTE the size/extent values do not include any scaling effects
    half_extent = bound.GetRange().GetSize() * 0.5

    return np.array([*centroid]), np.array([[*x_axis], [*y_axis], [*z_axis]]), np.array(half_extent)

def get_obb_corners(centroid: np.ndarray, axes: np.ndarray, half_extent: np.ndarray) -> np.ndarray:
    """Computes the corners of the Oriented Bounding Box (OBB) from the given OBB information

    Args:
        centroid (np.ndarray): The centroid of the OBB as a NumPy array.
        axes (np.ndarray): The axes of the OBB as a 2D NumPy array, where each row represents a different axis.
        half_extent (np.ndarray): The half extent of the OBB as a NumPy array.

    Returns:
        np.ndarray: NumPy array of shape (8, 3) containing each corner location of the OBB

        :math:`c_0 = (x_{min}, y_{min}, z_{min})`
        |br| :math:`c_1 = (x_{min}, y_{min}, z_{max})`
        |br| :math:`c_2 = (x_{min}, y_{max}, z_{min})`
        |br| :math:`c_3 = (x_{min}, y_{max}, z_{max})`
        |br| :math:`c_4 = (x_{max}, y_{min}, z_{min})`
        |br| :math:`c_5 = (x_{max}, y_{min}, z_{max})`
        |br| :math:`c_6 = (x_{max}, y_{max}, z_{min})`
        |br| :math:`c_7 = (x_{max}, y_{max}, z_{max})`

    Example:

    .. code-block:: python

        >>> import omni.isaac.core.utils.bounds as bounds_utils
        >>>
        >>> cache = bounds_utils.create_bbox_cache()
        >>> centroid, axes, half_extent = bounds_utils.compute_obb(cache, prim_path="/World/Cube")
        >>> bounds_utils.get_obb_corners(centroid, axes, half_extent)
        [[-0.5 -0.5 -0.5]
        [-0.5 -0.5  0.5]
        [-0.5  0.5 -0.5]
        [-0.5  0.5  0.5]
        [ 0.5 -0.5 -0.5]
        [ 0.5 -0.5  0.5]
        [ 0.5  0.5 -0.5]
        [ 0.5  0.5  0.5]]
    """
    corners = [
        centroid - axes[0] * half_extent[0] - axes[1] * half_extent[1] - axes[2] * half_extent[2],
        centroid - axes[0] * half_extent[0] - axes[1] * half_extent[1] + axes[2] * half_extent[2],
        centroid - axes[0] * half_extent[0] + axes[1] * half_extent[1] - axes[2] * half_extent[2],
        centroid - axes[0] * half_extent[0] + axes[1] * half_extent[1] + axes[2] * half_extent[2],
        centroid + axes[0] * half_extent[0] - axes[1] * half_extent[1] - axes[2] * half_extent[2],
        centroid + axes[0] * half_extent[0] - axes[1] * half_extent[1] + axes[2] * half_extent[2],
        centroid + axes[0] * half_extent[0] + axes[1] * half_extent[1] - axes[2] * half_extent[2],
        centroid + axes[0] * half_extent[0] + axes[1] * half_extent[1] + axes[2] * half_extent[2],
    ]
    return np.array(corners)


class IsaacSimSimulator(Simulator):

    SCENE_PRIM_PATH = "/World/scene"

    def __init__(self, task:Task, render_hz: int=30, headless: bool=False) -> None:
        self._config_isaac()
        # self.render_hz = render_hz # @deprecated

        # self.layout = layout
        self.task = task
        self.render_hz = task.metadata["render_hz"] if "render_hz" in self.task.metadata else render_hz

        self.step_id = 0
        self.is_scene_create = False
        self.is_isaac_reset = False
        self.headless = headless

        self.visulize_spheres = self.task.robot.visulize_spheres

        self.need_gravity = self.task.metadata.get("need_gravity", False)
        

    def reset(self) -> None:
        self._config_isaac()
        self.step_id = 0
        if hasattr(self, "cameras"):
            for camera in self.cameras.values():
                camera.initialize()

        self.is_isaac_reset = False
        self._last_policy_frame_id = None
        self._mirrored_frame_id = None
        self._mirrored_mujoco_sim_time = None

    def _config_isaac(self):
        success, result = omni.kit.commands.execute('ChangeSetting',
            path='/rtx/directLighting/sampledLighting/autoEnable', value=False
        )
        if not success:
            print("warning: set autoEnable sampleed lighting falied", result)
        zero_delay = env_flag("SIMPLE_ISAAC_ZERO_DELAY", default=True)
        disable_throttling_async = env_flag("SIMPLE_ISAAC_DISABLE_THROTTLING_ASYNC", default=True)

        # Disable capture on play. Delay/throttling settings honor env flags so
        # sync diagnostics can intentionally test async behavior.
        carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)
        carb.settings.get_settings().set("/omni/replicator/asyncRendering", False)
        carb.settings.get_settings().set("/app/asyncRendering", False)
        carb.settings.get_settings().set("/app/hydraEngine/waitIdle", bool(zero_delay))
        carb.settings.get_settings().set(
            "/app/updateOrder/checkForHydraRenderComplete",
            1000 if zero_delay else 0,
        )
        carb.settings.get_settings().set(
            "/exts/isaacsim.core.throttling/enable_async",
            not disable_throttling_async,
        )
        rep.settings.set_render_rtx_realtime()
        # rep.orchestrator.set_capture_on_play(False) # Data will be captured manually using step
        # carb.settings.get_settings().set_bool("/omni/replicator/captureMotionBlur", 0)
        # carb.settings.get_settings().set_bool("/rtx/post/motionblur/enabled", 0)   

    def update_layout(self) -> None:
        if not self.is_scene_create:
            self._setup_scene()
            self.is_scene_create = True

        self.__update_scene()
        self.__update_robot()
        self.__update_cameras()
        self.__update_lights()
        self.__update_task_visuals()
        self.update_visuals()
        
        # 5. add objects
        self.__reset_objects()
        self.current_visible_objects = []
        # object_shader_params = copy.deepcopy(self.task.layout.material_info["object_shader_params"])
        for obj_name, obj_info in self.task.layout.actors.items():
            if not isinstance(obj_info, ObjectActor) and not isinstance(obj_info, ArticulatedObjectActor):
                continue
            
            if isinstance(obj_info, ArticulatedObjectActor):
                self.add_articulated_object(obj_name, obj_info)
                
            else:
                if obj_info.asset.label not in self.objects:
                    self.__create_object(obj_name, obj_info)

                self.__update_object(obj_name, obj_info)
                """ if obj_info["bTarget"]:
                    self.target_obj_id = obj_info["id"]
                    self.target_obj = self.objects[obj_name] """
                self.current_visible_objects.append(self.objects[obj_info.asset.label]) # obj_name

        self.step_id = 0
        """ since isaac is already reset, we reset init robot qpos here 
        """
        if self.is_isaac_reset and self.is_scene_create:
            joint_state = self.task.robot.init_joint_states
            joint_indices = []
            qpos = []
            # print("syncing isaac to mujoco qpos:", joint_state)
            for jname, jpos in joint_state.items():
                isaac_jname = self.task.robot.jname_mujoco_to_isaac(jname)
                joint_indices.append(self.robot.get_dof_index(isaac_jname))
                qpos.append(jpos)
            self.robot.set_joint_positions(qpos, joint_indices=np.arange(len(joint_state)))

    def _setup_scene(self, move_surface_to_origin=True):
        assert not self.is_scene_create
        # NOTE: set correct physics_dt, rendering_dt to only affects isaacsim rendering quality
        
        self.world = World(physics_dt=1/(self.render_hz*10), rendering_dt=1/(self.render_hz), backend="numpy")
        self.workspace_prim_path = f"/World/workspace"
        self.workspace = XFormPrim(prim_path=self.workspace_prim_path)

        self.objects = {}
        self.current_visible_objects = []
        self.active_tableground_materials = []
        self.active_tableground_shaders = []
        self.scenes = {}
        self.cameras = {}
        self.lights = []
        self.articulated_objects = {}
        self.task_visuals = {}
        self._missing_task_visual_paths = set()

        self.add_robot()
        if getattr(self.task, "isaac_reproduction_spec", None) is not None:
            self._recording_mujoco_model = self.task.isaac_mujoco_model
            self.__add_recording_mjcf_visual()
            # The recording-derived USD is the visible source of robot and
            # task geometry.  Keep SIMPLE's generic articulation only as an
            # internal joint-state adapter for existing observation code.
            XFormPrim(prim_path=self.robot.prim_path).set_visibility(False)
        self.add_cameras()
        self.add_lights()

        self.__pre_add_objects()
        self.spheres = None

    def __update_scene(self, move_surface_to_origin=True):
        scene = getattr(self.task.layout, "scene", None)
        if scene is None:
            return

        scene_uid = scene.uid

        for uid, (_scene,_) in self.scenes.items():
            if scene_uid != uid:
                _scene.GetAttribute("visibility").Set("invisible")
                # scene.GetAttribute("xformOp:translate").Set((0, 0, 0))

        assert isinstance(scene, HssdSuite), "not supported scene type yet"

        scene_prim_path = f"{self.SCENE_PRIM_PATH}/s_{scene.uid.replace(':', '_')}"
        surface_prim_path = scene.conf["surface"]["prim_path"].replace("/World", scene_prim_path)
        
        has_surface2 = "surface2" in scene.conf and scene.conf["surface2"] is not None
        surface2_prim_path = None
        if has_surface2:
            surface2_prim_path = scene.conf["surface2"]["prim_path"].replace("/World", scene_prim_path)

        if scene_uid not in self.scenes:
            # self.add_scene(move_surface_to_origin)
            if isinstance(scene, simple.scenes.ShowHouse):
                raise NotImplementedError("ShowHouse scene is not implemented yet.")
            
            #### START adding HSSD scenes ####
            from pxr import UsdGeom, UsdPhysics

            reproduction_spec = getattr(self.task, "isaac_reproduction_spec", None)
            if reproduction_spec is not None:
                # Scene13 deliberately uses 102344250_local.usd; deriving this
                # filename from scene.name would load the wrong frozen asset.
                env_url = os.path.abspath(
                    resolve_data_path(
                        reproduction_spec.hssd.usd_relative_path,
                        auto_download=False,
                    )
                )
            else:
                # data_dir = resolve_data_path
                try:
                    data_dir = resolve_data_path(scene.data_dir, auto_download=True)
                except FileNotFoundError:
                    # put download logic into SceneManager
                    from simple.scenes import SceneManager
                    SceneManager.get(scene.uid.split(":")[0]).load(scene.uid)
                    data_dir = resolve_data_path(scene.data_dir)
                env_url = os.path.abspath(f"{data_dir}/{scene.name}.usd")
            isaacsim_stage.add_reference_to_stage(usd_path=env_url, prim_path=scene_prim_path)
            scene_prim = self.world.stage.GetPrimAtPath(scene_prim_path)
            if reproduction_spec is not None:
                self.__disable_prim_collisions(scene_prim)

            surface_prim = self.world.stage.GetPrimAtPath(surface_prim_path)
            surface_prim.GetAttribute("visibility").Set("invisible")

            if has_surface2:
                surface2_prim = self.world.stage.GetPrimAtPath(surface2_prim_path)
                surface2_prim.GetAttribute("visibility").Set("invisible")

            if not scene_prim.GetAttribute("xformOp:translate"):
                UsdGeom.Xformable(scene_prim).AddTranslateOp() # type: ignore
            if not scene_prim.GetAttribute("xformOp:rotateXYZ"):
                UsdGeom.Xformable(scene_prim).AddRotateXYZOp() # type: ignore
            if not scene_prim.GetAttribute("xformOp:scale"):
                UsdGeom.Xformable(scene_prim).AddScaleOp() # type: ignore

            # scene: HssdSuite = self.task.layout.scene
            scene_prim.GetAttribute("xformOp:rotateXYZ").Set(tuple(scene.center_orientation))

            scale = scene.conf["scale"]
            scene_prim.GetAttribute("xformOp:scale").Set((scale, scale, scale))

            surface_obb = self.calc_surface_center(surface_prim)
            self.scenes[scene.uid] = [scene_prim, surface_obb] # hssd_env # store it
        else:
            scene_prim = self.world.stage.GetPrimAtPath(scene_prim_path)
            surface_prim = self.world.stage.GetPrimAtPath(surface_prim_path)
            surface_prim.GetAttribute("visibility").Set("invisible")
            self.scenes[scene.uid][0] = scene_prim # hssd_env # store it
            surface_obb = self.scenes[scene.uid][1]

            scale = scene.conf["scale"]
            scene_prim.GetAttribute("xformOp:scale").Set((scale, scale, scale))
            scene_prim.GetAttribute("xformOp:translate").Set((0,0,0))
            scene_prim.GetAttribute("xformOp:rotateXYZ").Set(tuple(scene.center_orientation))
            surface_obb = self.calc_surface_center(surface_prim)

            if has_surface2:
                surface2_prim = self.world.stage.GetPrimAtPath(surface2_prim_path)
                surface2_prim.GetAttribute("visibility").Set("invisible")

        ceiling = scene_prim.GetPrimAtPath(f"{scene_prim_path}/ceilings")
        if ceiling.IsValid() and ceiling.GetAttribute("visibility"):
            ceiling.GetAttribute("visibility").Set("invisible")

        reproduction_spec = getattr(self.task, "isaac_reproduction_spec", None)
        if move_surface_to_origin and reproduction_spec is not None:
            from omni.isaac.core.utils.bounds import (
                compute_combined_aabb,
                create_bbox_cache,
            )

            target = np.asarray(
                reproduction_spec.hssd.target_surface_center, dtype=np.float64
            )
            surface_bounds_cache = create_bbox_cache()
            surface_bounds = np.asarray(
                compute_combined_aabb(surface_bounds_cache, surface_prim_path),
                dtype=np.float64,
            )
            surface_center = 0.5 * (
                surface_bounds[:3] + surface_bounds[3:]
            )
            room_offset = target - surface_center
            scene_prim.GetAttribute("xformOp:translate").Set(tuple(room_offset))

            # Recompute after surface alignment, then ground the visible room.
            bounds_cache = create_bbox_cache()
            room_bounds = np.asarray(
                compute_combined_aabb(bounds_cache, scene_prim_path),
                dtype=np.float64,
            )
            room_offset[2] -= float(room_bounds[2])
            scene_prim.GetAttribute("xformOp:translate").Set(tuple(room_offset))

            hidden_paths = (
                reproduction_spec.hssd.hidden_prims
                + ("floors",)
            )
            for relative_path in hidden_paths:
                hidden = self.world.stage.GetPrimAtPath(
                    f"{scene_prim_path}/{relative_path}"
                )
                if hidden.IsValid() and hidden.GetAttribute("visibility"):
                    hidden.GetAttribute("visibility").Set("invisible")
            print(
                f"[Task{reproduction_spec.task_number}Reproduction] "
                f"HSSD={env_url} room_offset={room_offset.tolist()}"
            )
        elif move_surface_to_origin:
            surface_center_position = - surface_obb["position"] + \
                np.array(scene.center_offset, dtype=np.float32) #self._config.hssd.center_offset
            if self.task.robot.uid == "g1_sonic":
                # FIXME backward compatibility: robot touches the ground
                surface_center_position[2] = 0.0 
            scene_prim.GetAttribute("xformOp:translate").Set(tuple(surface_center_position))

        scene_prim.GetAttribute("visibility").Set("visible")

        table_box = self.task.layout.actors.get("table")
        if table_box is not None:
            self._setup_table(f"{self.workspace_prim_path}/table_cuboid", table_box)

        table2_box = self.task.layout.actors.get("table2")
        if table2_box is not None:
            self._setup_table(f"{self.workspace_prim_path}/table2_cuboid", table2_box)

    def __update_cameras(self):
        for cname, cameraEntity in self.task.layout.cameras.items():
            p, q = cameraEntity.pose.position, cameraEntity.pose.quaternion
            isaacsim_camera = self.cameras[cname]
            # Isaac Sim's Camera API treats the quaternion as a world-camera
            # frame by default and converts it to USD camera axes internally.
            # Tasks that already provide an authored USD/OpenGL camera pose
            # must opt out of that second conversion.
            camera_axes = getattr(self.task, "isaac_camera_axes", "world")
            isaacsim_camera.set_local_pose(p, q, camera_axes=camera_axes)
            isaacsim_camera.set_clipping_range(
                cameraEntity.cam_cfg.near, cameraEntity.cam_cfg.far
            )
            isaacsim_camera.set_focal_length(cameraEntity.focal_length)

            horizontal_aperture = cameraEntity.focal_length * cameraEntity.resolution[0] / cameraEntity.fx
            vertical_aperture = cameraEntity.focal_length * cameraEntity.resolution[1] / cameraEntity.fy
            isaacsim_camera.set_horizontal_aperture(horizontal_aperture)
            isaacsim_camera.set_vertical_aperture(vertical_aperture)

    def __update_robot(self):
        robot_prim_path = self.robot.prim_path
        for shader_path in isaacsim_prims.find_matching_prim_paths(f'{robot_prim_path}/Looks/material_*/Shader'): # FIXME
            shader = UsdShade.Shader(isaacsim_prims.get_prim_at_path(shader_path))
            for key in ['reflection_roughness_constant', 'metallic_constant', 'specular_level']:
                shader.CreateInput(key, Sdf.ValueTypeNames.Float).Set(
                    self.task.layout.robot.shaders[key]
                ) # type: ignore

    def __update_lights(self):
        assert len(self.lights) == len(self.task.layout.lights)

        for i, light_prim in enumerate(self.lights): # self.lights:
            # light_prim = light[0]
            light_info = self.task.layout.lights[i]

            if i == 0:
                self.light_center.set_local_pose(light_info.center_light_postion, light_info.center_light_orientation)

            light_prim.GetAttribute("xformOp:translate").Set(tuple(light_info.pose.position))
            light_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(1., 0., 0., 0.))
            light_prim.GetAttribute("inputs:radius").Set(light_info.light_radius)
            light_prim.GetAttribute("inputs:length").Set(light_info.light_length)
            light_prim.GetAttribute("inputs:intensity").Set(light_info.light_intensity)
            light_prim.GetAttribute("inputs:enableColorTemperature").Set(True)
            light_prim.GetAttribute("inputs:colorTemperature").Set(light_info.light_color_temperature)

    def __resolve_task_usd_path(self, raw_path: str | os.PathLike) -> str:
        raw_path = os.fspath(raw_path)
        if os.path.isabs(raw_path):
            if not os.path.exists(raw_path):
                raise FileNotFoundError(raw_path)
            return raw_path

        if os.path.exists(raw_path):
            return os.path.abspath(raw_path)

        return os.path.abspath(resolve_data_path(raw_path.removeprefix("data/"), auto_download=True))

    def __set_prim_pose_scale(
        self,
        prim_path: str,
        position,
        orientation=None,
        scale=None,
        visible: bool = True,
    ) -> None:
        xform = XFormPrim(prim_path=prim_path)
        quat = [1.0, 0.0, 0.0, 0.0] if orientation is None else orientation
        xform.set_local_pose(position, quat)
        xform.set_visibility(visible)

        if scale is not None:
            prim = self.world.stage.GetPrimAtPath(prim_path)
            if not prim.GetAttribute("xformOp:scale"):
                UsdGeom.Xformable(prim).AddScaleOp()
            prim.GetAttribute("xformOp:scale").Set(Gf.Vec3d(*[float(x) for x in scale]))

    def __set_prim_display_color(self, prim, color) -> None:
        if color is None:
            return
        rgba = list(color)
        if len(rgba) == 3:
            rgba.append(1.0)
        try:
            gprim = UsdGeom.Gprim(prim)
            gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(*[float(x) for x in rgba[:3]])])
            gprim.CreateDisplayOpacityAttr().Set([float(rgba[3])])
        except Exception:
            pass

    def __disable_prim_collisions(self, root_prim) -> None:
        for prim in Usd.PrimRange(root_prim):
            attr = prim.GetAttribute("physics:collisionEnabled")
            if attr:
                attr.Set(False)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI(prim)
                enabled_attr = collision_api.GetCollisionEnabledAttr()
                if enabled_attr:
                    enabled_attr.Set(False)
            rigid_enabled = prim.GetAttribute("physics:rigidBodyEnabled")
            if rigid_enabled:
                rigid_enabled.Set(False)

    def __add_recording_mjcf_visual(self) -> None:
        """Import the exact episode MJCF as a hash-isolated read-only USD."""
        from simple.evals.task234_reproduction import mjcf_bundle_hash

        visual_mjcf_path = getattr(
            self.task, "isaac_mujoco_visual_mjcf_path", None
        )
        if visual_mjcf_path is None:
            visual_mjcf_path = self.task.mujoco_full_scene_mjcf_path
        mjcf_path = os.path.abspath(os.fspath(visual_mjcf_path))
        bundle_hash = mjcf_bundle_hash(mjcf_path)
        cache_root = os.environ.get(
            "SIMPLE_ISAAC_MJCF_CACHE", "/tmp/simple_isaac_mjcf_cache"
        )
        cache_dir = os.path.join(cache_root, bundle_hash)
        os.makedirs(cache_dir, exist_ok=True)
        usd_path = os.path.join(cache_dir, "scene.usd")
        root_path = "/World/Task3"

        if os.path.isfile(usd_path):
            isaacsim_stage.add_reference_to_stage(usd_path, root_path)
        else:
            status, import_config = omni.kit.commands.execute(
                "MJCFCreateImportConfig"
            )
            if not status:
                raise RuntimeError("Isaac MJCF importer is not enabled")
            import_config.set_fix_base(False)
            import_config.set_make_default_prim(False)
            import_config.set_create_physics_scene(False)
            import_config.set_import_inertia_tensor(True)
            import_config.set_visualize_collision_geoms(False)
            status, _ = omni.kit.commands.execute(
                "MJCFCreateAsset",
                mjcf_path=mjcf_path,
                import_config=import_config,
                prim_path=root_path,
                dest_path=usd_path,
            )
            if not status or not os.path.isfile(usd_path):
                raise RuntimeError(
                    f"Failed to derive Isaac USD from recording MJCF: {mjcf_path}"
                )
            if not self.world.stage.GetPrimAtPath(root_path).IsValid():
                isaacsim_stage.add_reference_to_stage(usd_path, root_path)

        root_prim = self.world.stage.GetPrimAtPath(root_path)
        if not root_prim.IsValid():
            raise RuntimeError(f"Derived recording USD has no root {root_path}")
        self.__disable_prim_collisions(root_prim)

        frozen_camera_parent = self.task.isaac_policy_camera_parent_prim
        if not self.world.stage.GetPrimAtPath(frozen_camera_parent).IsValid():
            raise RuntimeError(
                "Derived recording USD does not preserve the frozen torso path: "
                f"{frozen_camera_parent}"
            )

        # Match imported Xforms back to MuJoCo bodies by authored body name.
        prims_by_name: dict[str, list] = {}
        for prim in Usd.PrimRange(root_prim):
            prims_by_name.setdefault(prim.GetName(), []).append(prim)
            if prim.GetName().lower() in {"floor", "ground", "groundplane"}:
                visibility = prim.GetAttribute("visibility")
                if visibility:
                    visibility.Set("invisible")

        model = self._recording_mujoco_model
        root_joint_id = next(
            joint_id
            for joint_id in range(model.njnt)
            if int(model.jnt_qposadr[joint_id]) == 0
        )
        robot_root_body = int(model.jnt_bodyid[root_joint_id])

        def is_descendant(body_id: int, ancestor: int) -> bool:
            current = body_id
            while current > 0:
                if current == ancestor:
                    return True
                current = int(model.body_parentid[current])
            return False

        self._recording_body_visuals = []
        missing_bodies = []
        for body_id in range(1, model.nbody):
            body_name = str(model.body(body_id).name or "")
            matches = prims_by_name.get(body_name, [])
            if len(matches) != 1:
                missing_bodies.append((body_name, len(matches)))
                continue
            self._recording_body_visuals.append(
                (
                    body_id,
                    body_name,
                    is_descendant(body_id, robot_root_body),
                    XFormPrim(prim_path=str(matches[0].GetPath())),
                )
            )
        if missing_bodies:
            raise RuntimeError(
                "Cannot map every MJCF body to exactly one derived USD prim: "
                f"{missing_bodies[:12]}"
            )
        self._recording_mjcf_hash = bundle_hash
        print(
            f"[Task{self.task.isaac_reproduction_spec.task_number}Reproduction] "
            f"derived_usd={usd_path} mjcf_bundle_sha256={bundle_hash}"
        )

    def __sync_recording_mjcf_visual(self, mujoco_env) -> None:
        spec = self.task.isaac_reproduction_spec
        translation = np.asarray(spec.task_translate, dtype=np.float64)
        for body_id, body_name, is_robot, visual in self._recording_body_visuals:
            position = np.asarray(mujoco_env.mjData.xpos[body_id], dtype=np.float64)
            position = position + translation
            if is_robot:
                position[2] += spec.robot_visual_z_offset
            elif spec.trash_visual_z_offset is not None:
                lower_name = body_name.lower()
                if "trash" in lower_name or "can" in lower_name:
                    position[2] += spec.trash_visual_z_offset
            orientation = np.asarray(
                mujoco_env.mjData.xquat[body_id], dtype=np.float64
            )
            visual.set_world_pose(position=position, orientation=orientation)

    def __task_visual_specs(self, attr_name: str) -> list[dict]:
        specs = getattr(self.task, attr_name, [])
        if callable(specs):
            specs = specs()
        return list(specs or [])

    def __create_task_usd_visual(self, spec: dict) -> None:
        name = str(spec["name"])
        prim_path = spec.get("prim_path") or f"{self.workspace_prim_path}/{name}"
        usd_path = self.__resolve_task_usd_path(spec["usd_path"])

        if not self.world.stage.GetPrimAtPath(prim_path).IsValid():
            isaacsim_stage.add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            prim = self.world.stage.GetPrimAtPath(prim_path)
            if spec.get("disable_collision", True):
                self.__disable_prim_collisions(prim)
        self.task_visuals[name] = {
            "prim_path": prim_path,
            "sync_body": spec.get("sync_body"),
            "sync_subprims": dict(spec.get("sync_subprims", {})),
            "sync_subprims_pose_space": spec.get(
                "sync_subprims_pose_space", "world"
            ),
            "sync_root_scale": spec.get("sync_root_scale"),
        }

    def __create_task_primitive_visual(self, spec: dict) -> None:
        name = str(spec["name"])
        prim_path = spec.get("prim_path") or f"{self.workspace_prim_path}/{name}"
        primitive_type = str(spec.get("type", "cube")).lower()

        if not self.world.stage.GetPrimAtPath(prim_path).IsValid():
            if primitive_type in {"dome_light", "domelight"}:
                isaacsim_prims.create_prim(prim_path, "DomeLight")
            elif primitive_type in {"sphere", "ball"}:
                sphere.VisualSphere(
                    prim_path=prim_path,
                    position=np.asarray(spec.get("pos", [0.0, 0.0, 0.0]), dtype=np.float32),
                    radius=float(spec.get("radius", 1.0)),
                    color=np.asarray(spec.get("color", [1.0, 1.0, 1.0]), dtype=np.float32),
                )
            else:
                omni.kit.commands.execute(
                    "CreateMeshPrimWithDefaultXform",
                    prim_type="Cube",
                    prim_path=prim_path,
                )
                prim = self.world.stage.GetPrimAtPath(prim_path)
                self.__set_prim_display_color(prim, spec.get("color"))
            self.__apply_task_primitive_material(prim_path, spec)
        self.__apply_task_primitive_attrs(prim_path, spec)

        self.task_visuals[name] = {
            "prim_path": prim_path,
            "sync_body": spec.get("sync_body"),
            "sync_subprims": dict(spec.get("sync_subprims", {})),
            "sync_subprims_pose_space": spec.get(
                "sync_subprims_pose_space", "world"
            ),
            "sync_root_scale": spec.get("sync_root_scale"),
        }

    def __apply_task_primitive_material(self, prim_path: str, spec: dict) -> None:
        textures_dir = spec.get("grass_textures_dir")
        material_tool = spec.get("grass_material_tool")
        if not textures_dir or not material_tool:
            return

        key = f"grass_material:{prim_path}"
        try:
            module_spec = importlib.util.spec_from_file_location(
                "_simple_arena_grass_ground_material",
                os.fspath(material_tool),
            )
            if module_spec is None or module_spec.loader is None:
                raise ImportError(f"Cannot load {material_tool}")
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            module.apply_grass_pbr_to_ground(
                prim_path=prim_path,
                textures_dir=os.fspath(textures_dir),
                uv_scale=tuple(spec.get("uv_scale", (150.0, 150.0))),
            )
        except Exception as exc:
            if key not in self._missing_task_visual_paths:
                print(f"[IsaacTaskVisual] grass material skipped for {prim_path}: {exc}")
                self._missing_task_visual_paths.add(key)

    def __apply_task_primitive_attrs(self, prim_path: str, spec: dict) -> None:
        primitive_type = str(spec.get("type", "cube")).lower()
        if primitive_type not in {"dome_light", "domelight"}:
            return

        prim = self.world.stage.GetPrimAtPath(prim_path)
        if spec.get("intensity") is not None and prim.GetAttribute("inputs:intensity"):
            prim.GetAttribute("inputs:intensity").Set(float(spec["intensity"]))
        if spec.get("color") is not None and prim.GetAttribute("inputs:color"):
            prim.GetAttribute("inputs:color").Set(
                Gf.Vec3f(*[float(x) for x in spec["color"][:3]])
            )

    def __update_task_visual_spec(self, spec: dict, visual_type: str) -> None:
        name = str(spec["name"])
        if name not in self.task_visuals:
            if visual_type == "usd":
                try:
                    self.__create_task_usd_visual(spec)
                except FileNotFoundError as exc:
                    required = bool(spec.get("required", False))
                    key = os.fspath(spec.get("usd_path", exc.filename))
                    if key not in self._missing_task_visual_paths:
                        print(f"[IsaacTaskVisual] missing USD, skipped: {key}")
                        self._missing_task_visual_paths.add(key)
                    if required:
                        raise
                    return
            else:
                self.__create_task_primitive_visual(spec)

        visual = self.task_visuals[name]
        self.__set_prim_pose_scale(
            visual["prim_path"],
            spec.get("pos", [0.0, 0.0, 0.0]),
            spec.get("quat", [1.0, 0.0, 0.0, 0.0]),
            spec.get("scale"),
            bool(spec.get("visible", True)),
        )
        self.__apply_task_primitive_attrs(visual["prim_path"], spec)

    def __update_task_visuals(self) -> None:
        if not hasattr(self, "task_visuals"):
            return

        for visual in self.task_visuals.values():
            XFormPrim(prim_path=visual["prim_path"]).set_visibility(False)

        for spec in self.__task_visual_specs("isaac_extra_usd_references"):
            self.__update_task_visual_spec(spec, "usd")
        for spec in self.__task_visual_specs("isaac_extra_primitives"):
            self.__update_task_visual_spec(spec, "primitive")

    def __sync_task_visuals(self, synced_pose_by_name: dict[str, tuple]) -> None:
        for visual in getattr(self, "task_visuals", {}).values():
            sync_body = visual.get("sync_body")
            root_pose = synced_pose_by_name.get(sync_body)
            if sync_body in synced_pose_by_name:
                obj_pos, obj_ori = root_pose
                XFormPrim(prim_path=visual["prim_path"]).set_local_pose(
                    translation=obj_pos,
                    orientation=obj_ori,
                )
            for relative_path, body_name in visual.get("sync_subprims", {}).items():
                if body_name not in synced_pose_by_name:
                    continue
                obj_pos, obj_ori = synced_pose_by_name[body_name]
                subprim_path = f"{visual['prim_path'].rstrip('/')}/{relative_path.lstrip('/')}"
                prim = self.world.stage.GetPrimAtPath(subprim_path)
                if not prim.IsValid():
                    key = f"sync_subprim:{subprim_path}"
                    if key not in self._missing_task_visual_paths:
                        print(f"[IsaacTaskVisual] missing sync subprim: {subprim_path}")
                        self._missing_task_visual_paths.add(key)
                    continue
                pose_space = visual.get("sync_subprims_pose_space", "world")
                if pose_space == "root_local_scaled":
                    if root_pose is None or visual.get("sync_root_scale") is None:
                        raise ValueError(
                            "root_local_scaled task visual sync requires a published "
                            "sync_body pose and sync_root_scale"
                        )
                    root_pos, root_ori = root_pose
                    local_matrix = world_pose_to_scaled_root_local_matrix(
                        obj_pos,
                        obj_ori,
                        root_pos,
                        root_ori,
                        visual["sync_root_scale"],
                    )
                    # Translation/quaternion/scale decomposition is not exact
                    # below a non-uniformly scaled parent once the child
                    # rotates.  Author the compensating matrix directly so
                    # door leaves and handles remain rigid instead of shearing
                    # or sinking into one another.
                    matrix_op = UsdGeom.Xformable(prim).MakeMatrixXform()
                    matrix_op.Set(
                        Gf.Matrix4d(*local_matrix.reshape(-1).tolist())
                    )
                elif pose_space == "world":
                    XFormPrim(prim_path=subprim_path).set_world_pose(
                        position=obj_pos,
                        orientation=obj_ori,
                    )
                else:
                    raise ValueError(
                        f"Unsupported task visual sync pose space: {pose_space!r}"
                    )

    def __reset_objects(self):
        # from omni.isaac.core.utils.prims import delete_prim
        for obj_name in self.objects:
            obj = self.objects[obj_name]
            obj_xform = XFormPrim(prim_path=obj["object_prim_path"])
            obj_xform.set_visibility(False) # obj["geom"]
            obj_xform.set_local_pose([0.0, 0.0, -1.0]) # obj["xform"]
            obj["bActive"] = False

    def __pre_add_objects(self):
        for obj in self.task.preload_objects():
            self.__create_object(obj.asset.label, obj)

    def __create_object(self, obj_key: str, object_info: ObjectActor):
        obj_id = object_info.uid # object_info["id"]
        object_prim_path = f'{self.workspace_prim_path}/{object_info.asset.label}' # {object_info["name"]}

        # if obj_id == 100:
        #     cube = self.__create_cuboid(object_prim_path, 
        #                                 object_info["position"], 
        #                                 object_info["dims"], 
        #                                 object_info["orientation"],
        #                                 color=np.array([1., 0., 0.]))
        #     obj_xform = XFormPrim(prim_path=object_prim_path)
        #     obj_geom = cube
        #     obj_rigid = RigidPrim(prim_path=object_prim_path)
        # else:

        object_usd_path = os.path.abspath(resolve_data_path(object_info.asset.usd_path,auto_download=True))
        
        isaacsim_stage.add_reference_to_stage(usd_path=object_usd_path, prim_path=object_prim_path)

        obj_xform = XFormPrim(prim_path=object_prim_path)
        geom_prim_path = f'{object_prim_path}/Meshes'
        obj_geom = GeometryPrim(prim_path=geom_prim_path)
        obj_rigid = RigidPrim(prim_path=geom_prim_path)
        obj_rigid.disable_rigid_body_physics()
        obj_collision_geom = GeometryPrim(f"{geom_prim_path}/collision")
        obj_collision_geom.set_collision_enabled(False)

        usd_prim = isaacsim_prims.get_prim_at_path(object_prim_path)
        semantics=[("prim", f"{obj_id}")]
        for semantic_type, semantic_value in semantics:
            instance_name = Tf.MakeValidIdentifier(f"{semantic_type}_{semantic_value}")
            sem = Semantics.SemanticsAPI.Apply(usd_prim, instance_name)
            sem.CreateSemanticTypeAttr()
            sem.CreateSemanticDataAttr()
            sem.GetSemanticTypeAttr().Set(semantic_type)
            sem.GetSemanticDataAttr().Set(semantic_value)
    
        obj = {
            "id": obj_id,
            "object_prim_path": object_prim_path,
            "bActive": False,
            "bTarget": obj_key == "target" #object_info["bTarget"],
        }

        self.objects[object_info.asset.label] = obj

    def __update_object(self, obj_name, obj_info: ObjectActor):
        obj = self.objects[obj_info.asset.label] # obj_info["name"]
        # obj_geom = GeometryPrim(prim_path=obj["object_prim_path"])
        # obj_geom.set_visibility(True)

        # # enable physics for now
        # geom_prim_path = f'{obj["object_prim_path"]}/Meshes'
        # obj_rigid = RigidPrim(prim_path=geom_prim_path)
        # # obj_rigid.set_kinematic_enabled(True)
        # obj_rigid.set_linear_velocity((0, 0, 0))
        # obj_rigid.set_angular_velocity((0, 0, 0))

        # obj_rigid.enable_rigid_body_physics()
        # obj_collision_geom = GeometryPrim(f"{geom_prim_path}/collision")
        # obj_collision_geom.set_collision_enabled(True)

        # num_subframes = 1
        # self.world.step(render=False)
        # for _ in range(2):# BUG: extra render to avoid empty frame
        #     rep.orchestrator.step(rt_subframes=num_subframes, pause_timeline=False)

        # change object pose
        obj_xform = XFormPrim(prim_path=obj["object_prim_path"])
        obj_xform.set_local_pose(obj_info.pose.position, obj_info.pose.quaternion)
        obj_xform.set_visibility(True)
        # geom_prim_path = f'{obj["object_prim_path"]}/Meshes'
        # obj_geom = GeometryPrim(prim_path=geom_prim_path)
        # obj_geom.set_local_pose([0,0,0], [1,0,0,0]) # reset mesh local pose
        obj["bActive"] = True

        # # disable physics for now
        # obj_rigid = RigidPrim(prim_path=geom_prim_path)
        # obj_rigid.disable_rigid_body_physics()
        # obj_collision_geom = GeometryPrim(f"{geom_prim_path}/collision")
        # obj_collision_geom.set_collision_enabled(False)

        # object_prim_path = obj["xform"].prim_path
        # object_shader_param = object_shader_params.pop()
        for shader_path in isaacsim_prims.find_matching_prim_paths(f'{obj["object_prim_path"]}/Looks/material_*/material_*'):
            shader = UsdShade.Shader(isaacsim_prims.get_prim_at_path(shader_path))
            # shader.SetSourceAsset('OmniPBR.mdl')
            for key in ['reflection_roughness_constant', 'metallic_constant', 'specular_level']:
                shader.CreateInput(key, Sdf.ValueTypeNames.Float).Set(obj_info.material[key]) # type:ignore object_shader_param[key]

    def step(self, mujoco_env = None):
        read_only_mirror = bool(
            self.task.metadata.get("isaac_read_only_mirror", False)
        )
        if not self.is_isaac_reset:
            self.world.reset()
            # self.__update_object()
            self.robot.initialize()

            # update articulated objects if articulated objects is not empty
            if self.articulated_objects:
                for obj_name in self.articulated_objects.keys():
                    self.articulated_objects[obj_name].initialize()
                    self.articulated_objects[obj_name].disable_gravity()

            #TODO
            if not self.need_gravity:
                self.robot.disable_gravity()
            else:
                self.robot.disable_gravity()
            
            # set robot to initial pose
            robo_actor = self.task.layout.actors["robot"]
            robo_position = robo_actor.pose.position.copy()
            # robo_position[2] += 0.055 # HACK fix for veag 1 base height
            self.robot.set_world_pose(robo_position, robo_actor.pose.quaternion)
            # self.world.physics_sim_view.flush()


            # set articulated objects to initial pose if articulated objects is not empty
            if self.articulated_objects:
                ##TODO support multiple articulated objects
                articulated = self.task.layout.actors["articulated"]
                for obj_name in self.articulated_objects.keys():
                    self.articulated_objects[obj_name].set_world_pose(articulated.pose.position, articulated.pose.quaternion)
                    # self.world.physics_sim_view.flush()
                    
            # Disable motors
            num_dofs = self.robot._articulation_view.num_dof
            zeros = np.zeros(num_dofs, dtype=np.float32)
            ones = np.ones(num_dofs, dtype=np.float32) # fix joint oscillation
            self.robot._articulation_view.set_gains(kps=ones*800, kds=ones*20)

            # disable motors if articulated objects is not empty
            if self.articulated_objects:
                for obj_name in self.articulated_objects.keys():
                    num_dofs_articulated = self.articulated_objects[obj_name]._articulation_view.num_dof
                    zero = np.zeros(num_dofs_articulated, dtype=np.float32)
                    one = np.ones(num_dofs_articulated, dtype=np.float32) # fix joint oscillation
                    self.articulated_objects[obj_name]._articulation_view.set_gains(kps=zero, kds=zero)

            joint_state = self.task.robot.init_joint_states #self.layout.robot_info["init_robot_qpos"]
            joint_indices = []
            qpos = []
            # print("syncing isaac to mujoco qpos:", joint_state)
            for jname, jpos in joint_state.items():
                isaac_jname = self.task.robot.jname_mujoco_to_isaac(jname)
                joint_indices.append(self.robot.get_dof_index(isaac_jname))
                qpos.append(jpos)
            self.robot.set_joint_positions(qpos, joint_indices=joint_indices)
            # self.world.physics_sim_view.flush()


            # set articulated joint state if articulated objects is not empty
            if self.articulated_objects:
                for obj_name in self.articulated_objects.keys():
                    articulate_joint_qpos = self.task.layout.actors["articulated"].asset.articulate_init_joint_qpos
                    if articulate_joint_qpos is not None:
                        qpos = []
                        for articulate_joint, articulate_jpos in articulate_joint_qpos.items():
                            qpos.append(articulate_jpos)
                        self.articulated_objects[obj_name].set_joint_positions(qpos)
                    else:
                        init_joint_states = np.zeros(self.articulated_objects[obj_name]._articulation_view.num_dof, dtype=np.float32)
                        self.articulated_objects[obj_name].set_joint_positions(list(init_joint_states))


            # Stop all motion
            self.robot._articulation_view.set_joint_velocities(zeros)
            self.is_isaac_reset = True
            self._update_collision_spheres()

            # stop articulated objects joint motion if articulated objects is not empty
            if self.articulated_objects:
                for obj_name in self.articulated_objects.keys():
                    num_dofs_articulated = self.articulated_objects[obj_name]._articulation_view.num_dof
                    zero = np.zeros(num_dofs_articulated, dtype=np.float32)
                    self.articulated_objects[obj_name]._articulation_view.set_joint_velocities(zero)

        if mujoco_env is not None:
            self.sync_states(mujoco_env)

        # Strict reproduction never advances a second physics clock.  USD
        # transforms are authored directly from MuJoCo, then Replicator is
        # asked to render while the Isaac timeline remains paused.
        if not read_only_mirror:
            self.world.step(render=False)
        self.update_visuals()
        rep.orchestrator.step(
            rt_subframes=1,
            pause_timeline=read_only_mirror,
        )
        self._update_collision_spheres()
        self.step_id += 1

    def _update_collision_spheres(self):
        """Update collision spheres visualization for the robot."""
        if not self.visulize_spheres:
            return

        from curobo.types.base import TensorDeviceType
        tensor_args = TensorDeviceType()

        sim_js = self.robot.get_joints_state()
        sim_js_names = self.robot.dof_names
        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities),
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(
            [self.task.robot.jname_mujoco_to_isaac(j) for j in self.task.robot.joint_names]
        )
        sph_list = self.task.robot.kin_model.get_robot_as_spheres(cu_js.position)

        T_world_robot = np.eye(4)
        T_world_robot[:3, 3] = self.robot.get_world_pose()[0]
        T_world_robot[:3, :3] = t3d.quaternions.quat2mat(self.robot.get_world_pose()[1])

        if self.spheres is None:
            self.spheres = []
            for si, s in enumerate(sph_list[0]):
                T_robot_sphere = np.eye(4)
                T_robot_sphere[:3, 3] = s.pose[:3]
                T_robot_sphere[:3, :3] = t3d.quaternions.quat2mat(s.pose[3:])

                T_world_sphere = T_world_robot @ T_robot_sphere
                s.position = T_world_sphere[:3, 3]
                sp = sphere.VisualSphere(
                    prim_path="/curobo/robot_sphere_" + str(si),
                    position=np.ravel(s.position),
                    radius=float(s.radius),
                    color=np.array([0, 0.8, 0.2]),
                )
                self.spheres.append(sp)
        else:
            for si, s in enumerate(sph_list[0]):
                T_robot_sphere = np.eye(4)
                T_robot_sphere[:3, 3] = s.pose[:3]
                T_robot_sphere[:3, :3] = t3d.quaternions.quat2mat(s.pose[3:])

                T_world_sphere = T_world_robot @ T_robot_sphere
                s.position = T_world_sphere[:3, 3]
                self.spheres[si].set_world_pose(position=np.ravel(s.position))
                self.spheres[si].set_radius(float(s.radius))

    def sync_states(self, mujoco_env):
        states = mujoco_env.get_states()
        if self.articulated_objects:
            step_id, joint_state, obj_names, obj_positions, obj_orientations, robot_position, articulated_joints_state, articulate_object_position = states[:]
        else:
            step_id, joint_state, obj_names, obj_positions, obj_orientations, robot_position = states[:]

        if getattr(self.task, "isaac_reproduction_spec", None) is not None:
            self.__sync_recording_mjcf_visual(mujoco_env)
        
        synced_pose_by_name = {}
        for obj_name, obj_pos, obj_ori in zip(obj_names, obj_positions, obj_orientations):
            synced_pose_by_name[obj_name] = (obj_pos, obj_ori)
            obj = self.objects.get(obj_name)
            if obj is None:
                continue
            obj_xfrom = XFormPrim(prim_path=obj["object_prim_path"])
            obj_xfrom.set_local_pose(translation=obj_pos, orientation=obj_ori) # obj["xform"]
        self.__sync_task_visuals(synced_pose_by_name)
        
        joint_indices = []
        qpos = []
        # print("syncing isaac to mujoco qpos:", joint_state)
        for jname, jpos in joint_state.items():
            isaac_jname = self.task.robot.jname_mujoco_to_isaac(jname)
            joint_indices.append(self.robot.get_dof_index(isaac_jname))
            qpos.append(round(jpos,6))
        self.robot.set_joint_positions(qpos, joint_indices=joint_indices)

        if self.articulated_objects:
            for obj_name in self.articulated_objects.keys():
                articulated_object = self.articulated_objects[obj_name]
                articulate_joint_indices= []
                articulate_joint_pos = []
                for articulate_joint, articulate_jpos in articulated_joints_state.items():
                    articulate_joint_indices.append(articulated_object.get_dof_index(articulate_joint))
                    articulate_joint_pos.append(round(articulate_jpos,6))
                articulated_object.set_joint_positions(articulate_joint_pos, joint_indices=articulate_joint_indices)

        if self.need_gravity:
            robot_visual_pose = np.asarray(robot_position, dtype=np.float64).copy()
            reproduction_spec = getattr(self.task, "isaac_reproduction_spec", None)
            if reproduction_spec is not None:
                robot_visual_pose[:3] += np.asarray(
                    reproduction_spec.task_translate, dtype=np.float64
                )
                robot_visual_pose[2] += reproduction_spec.robot_visual_z_offset
            self.robot.set_world_pose(
                position=robot_visual_pose[:3], orientation=robot_visual_pose[3:]
            )
            if self.articulated_objects:
                for articulated_object in self.articulated_objects.values():
                    articulated_object.set_world_pose(position=articulate_object_position[:3], orientation=articulate_object_position[3:])
        # self.world.physics_sim_view.flush()
        # Stop all motion
        num_dofs = self.robot._articulation_view.num_dof
        zeros = np.zeros(num_dofs, dtype=np.float32)
        self.robot._articulation_view.set_joint_velocities(zeros)

        if self.articulated_objects:
            for articulated_object in self.articulated_objects.values():
                num_dofs = articulated_object._articulation_view.num_dof
                zero = np.zeros(num_dofs, dtype=np.float32)
                articulated_object._articulation_view.set_joint_velocities(zero)
        self._mirrored_mujoco_sim_time = float(mujoco_env.mjData.time)
        self._mirrored_frame_id = int(step_id)
        return step_id

    def get_states(self) -> Dict[str, float]:
        raise NotImplementedError
    
    def set_states(self, states: Dict[str, float]) -> None:
        raise NotImplementedError
    
    def add_articulated_object(self, obj_name: str, obj_info: ObjectActor):
        obj_name = obj_info.asset.name
        uid = obj_info.asset.uid

        isaacsim_stage.add_reference_to_stage(
            usd_path=self.resolve_data_path(obj_info.asset.usd_path),
            prim_path=f"{self.workspace_prim_path}/articulated_objects",
        )

        self.articulated_objects[obj_name] = SingleArticulation(
            prim_path=f"{self.workspace_prim_path}/articulated_objects/{obj_name}",
            position = obj_info.pose.position,
            orientation = obj_info.pose.quaternion,
        )
        self.articulated_objects[obj_name].set_enabled_self_collisions(False)

        from omni.isaac.core.prims import GeometryPrim
        from pxr import Usd

        articulation_prim = self.articulated_objects[obj_name].prim.GetParent()


        
        for prim in Usd.PrimRange(articulation_prim):
            prim_name = prim.GetName()
            

            if prim_name.lower() == "collisions" or prim_name.lower() == "collision":
                prim_path = str(prim.GetPath())
                
                # unique_name = f"{obj_name}_{prim_path.split('/')[-2]}_col" 
                
                try:
        
                    obj_collision_geom = GeometryPrim(
                        prim_path=prim_path, 
                        # name=unique_name,
                        # translation=None,
                        # orientation=None  
                    )
                    
                    
                    obj_collision_geom.set_collision_enabled(False)

                    
                except Exception as e:
                    print(f"skip: {prim_path}")


    def add_robot(self):
        robot_ns = self.task.robot.robot_ns
        eef_prim_path = self.task.robot.eef_prim_path
        hand_prim_path = self.task.robot.hand_prim_path

        isaacsim_stage.add_reference_to_stage(
            usd_path=self.resolve_data_path(self.task.robot.usd_path),
            prim_path=f"{self.workspace_prim_path}/Robot",
        )

        robo_actor = self.task.layout.actors["robot"]
        self.robot = IsaacRobot(
            prim_path=f"{self.workspace_prim_path}/Robot/{robot_ns}",
            position=robo_actor.pose.position, # does not work
            orientation=robo_actor.pose.quaternion, # does not work 
        )
        # TODO assert self.task.robot.FRANKA_FINGER_LENGTH == self.layout.robot_info["robot_eef_offset"], "bug"

        robot_eef_xform = XFormPrim(
            f'{self.workspace_prim_path}/Robot/{robot_ns}/{eef_prim_path}',
            translation=[0, 0, self.task.robot.robot_eef_offset],
            orientation=[1., 0., 0., 0.],
            # visible=False,
        )
        robot_hand = XFormPrim(
            f'{self.workspace_prim_path}/Robot/{robot_ns}/{hand_prim_path}',
            translation=[0, 0, 0],
            orientation=[1., 0., 0., 0.],
            # visible=False,
        )
        self.robot_eef_xform = robot_eef_xform
        self.robot_hand = robot_hand

    def add_cameras(self):
        for cam_key, cam_info in self.task.layout.cameras.items():
            if cam_info.mount == "eye_in_hand":
                # FIXME
                if cam_key == "wrist_left":
                    wrist_cam_link = self.task.robot.wrist_cam_link.replace("right", "left")
                else:
                    wrist_cam_link = self.task.robot.wrist_cam_link
                cam_prim_path = f"{self.workspace_prim_path}/Robot/{self.task.robot.robot_ns}/{wrist_cam_link}"
            elif cam_info.mount == "eye_on_base":
                cam_prim_path = f"{self.workspace_prim_path}/Robot/{self.task.robot.robot_ns}"
            elif cam_info.mount == "eye_in_head":
                head_prim_link = self.task.robot.head_cam_link
                frozen_parent = getattr(
                    self.task, "isaac_policy_camera_parent_prim", None
                )
                if frozen_parent:
                    cam_prim_path = frozen_parent
                else:
                    cam_prim_path = f"{self.workspace_prim_path}/Robot/{self.task.robot.robot_ns}/{head_prim_link}"
            else:
                raise ValueError(f"Unsupported camera mount: {cam_info.mount}")

            camera = IssacCamera(
                name=cam_key,
                prim_path=f"{cam_prim_path}/{cam_key}",
                resolution=cam_info.resolution,
            )
            camera.initialize()
            self.cameras[cam_key] = camera

    def add_lights(self):
        light_center = XFormPrim(f'{self.workspace_prim_path}/LightCenter')

        for light in self.task.layout.lights:
            prim_path = f"{self.workspace_prim_path}/LightCenter/{light.uid}"
            light = isaacsim_prims.create_prim(
                prim_path,
                light.type,
            )
            self.lights.append(light)

        self.light_center = light_center

    def update_visuals(self):
        stage = omni.usd.get_context().get_stage()
        stage_unit = UsdGeom.GetStageMetersPerUnit(stage)
        # points = [(0, 0, 0.14 / stage_unit)]
        dbg = _debug_draw.acquire_debug_draw_interface()

        for _, visual in self.task.layout.visuals.items():
            if isinstance(visual, VisualFrame):
                axis_length = 0.1
                origin = np.array(visual.pose.position, dtype=np.float32)#np.array([0.02013437, 0.08273282, 0.05279581])
                rotation_mat = t3d.quaternions.quat2mat(visual.pose.quaternion)

                x_axis = origin + rotation_mat[:3, 0] * axis_length
                y_axis = origin + rotation_mat[:3, 1] * axis_length
                z_axis = origin + rotation_mat[:3, 2] * axis_length

                dbg.draw_lines(
                    [origin, origin, origin],  # start points
                    [x_axis, y_axis, z_axis],  # end points
                    [carb.ColorRgba(1, 0, 0, 1),  # X = red
                    carb.ColorRgba(0, 1, 0, 1),  # Y = green
                    carb.ColorRgba(0, 0, 1, 1)], # Z = blue
                    [2.0, 2.0, 2.0]  # line widths
                )

            elif isinstance(visual, VisualGrasp):
                lines = visual.plot_lines()
                dbg.draw_lines(
                    [l[0] for l in lines],  # start points
                    [l[1] for l in lines],  # end points
                    [carb.ColorRgba(l[2]) for l in lines],  # colors
                    [l[3] for l in lines]  # line widths
                )


    def render(self, *args, **kwargs) -> dict[str, np.ndarray]:
        render_products = {}
        for cam_name, camera in self.cameras.items():
            # frame = camera.get_current_frame()
            # raw_rgb = frame['rgba'][...,:3].astype(np.uint8)
            frame = camera.get_rgba()
            raw_rgb = frame[...,:3].astype(np.uint8)
            render_products[cam_name] = raw_rgb
        reproduction_spec = getattr(self.task, "isaac_reproduction_spec", None)
        if reproduction_spec is not None:
            from simple.evals.task234_reproduction import validate_policy_rgb

            policy_camera = getattr(self.task, "isaac_policy_camera_name")
            if set(render_products) != {policy_camera}:
                raise RuntimeError(
                    "Strict eval requires exactly one Isaac policy render product "
                    f"named {policy_camera!r}; got {sorted(render_products)}"
                )
            frame_id = getattr(self, "_mirrored_frame_id", None)
            sim_time = getattr(self, "_mirrored_mujoco_sim_time", None)
            if frame_id is None or sim_time is None:
                raise RuntimeError("Isaac rendered before receiving a MuJoCo state")
            validate_policy_rgb(
                render_products[policy_camera],
                image_sim_time=sim_time,
                mujoco_sim_time=sim_time,
                frame_id=frame_id,
                previous_frame_id=getattr(self, "_last_policy_frame_id", None),
            )
            self._last_policy_frame_id = frame_id
        return render_products
    
    def calc_surface_center(self, surface_prim):
        from omni.isaac.core.utils.bounds import (compute_combined_aabb,
                                                  create_bbox_cache)
        bb_cache = create_bbox_cache()
        centroid, axes, half_extent = compute_obb(bb_cache, surface_prim.GetPrimPath())
        larger_xy_extent = (half_extent[0], half_extent[1], half_extent[2])
        obb_corners = get_obb_corners(centroid, axes, larger_xy_extent)
        # TODO test
        top_corners = [
            # obb_corners[0].tolist(),
            # obb_corners[1].tolist(),
            obb_corners[2].tolist(),
            obb_corners[3].tolist(),
            # obb_corners[4].tolist(),
            # obb_corners[5].tolist(),
            obb_corners[6].tolist(),
            obb_corners[7].tolist(),
        ]

        position = np.mean(top_corners, axis=0)
    
        return {
            "axes": axes,
            "half_extent": half_extent,
            "position": position,
        }

    def _setup_table(self, prim_path: str, table_box):
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath(prim_path).IsValid():
            stage.RemovePrim(prim_path)

        omni.kit.commands.execute("CreateMeshPrimWithDefaultXform", prim_type="Cube", prim_path=prim_path)
        prim = stage.GetPrimAtPath(prim_path)

        p, q, s = table_box.pose.position, table_box.pose.quaternion, table_box.size
        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*[float(x) for x in p]))
        # q is wxyz [qw, qx, qy, qz]; Gf.Quatd(real, i, j, k) expects (w, x, y, z)
        prim.GetAttribute("xformOp:orient").Set(
            Gf.Quatd(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
        prim.GetAttribute("xformOp:scale").Set(Gf.Vec3d(*[float(x) for x in s]))

        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim)

        mat_info = getattr(table_box, 'material', None)
        if mat_info is None:
            return
        raw_path = mat_info['path']
        if not os.path.isabs(raw_path):
            raw_path = resolve_data_path(raw_path.removeprefix("data/"), auto_download=True)
        created = [None]
        create_mdl_material(stage, raw_path, mat_info['name'], lambda p: created.__setitem__(0, p))
        if created[0] is not None:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(created[0]))
