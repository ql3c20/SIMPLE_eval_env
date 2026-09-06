"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

import mujoco
import mujoco.viewer
import numpy as np
import os
from threading import Lock
from typing import Any

from simple.core.task import Task
from simple.envs.base_dual_env import BaseDualSim
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

class SonicLocoManipEnv(BaseDualSim):

    _success: bool 

    def __init__(
        self, 
        task : str | Task, #
        sonic_config: dict , 
        # config: Dict[str, Any],
        sim_mode="mujoco_isaac",  # =SIM_MODE.MUJOCO_ISAAC
        headless=True, 
        enable_image_publish=False,
        
        *args, 
        **kwargs
    ) -> None:
        super().__init__(task, sim_mode, headless, sonic_config=sonic_config, *args, **kwargs)
        self.sonic_config = sonic_config
        
        self.offscreen = sim_mode in {"isaac", "mujoco_isaac"} or headless
        self.onscreen = not self.offscreen
        self._tracker_ghost_enabled = os.environ.get("SIMPLE_TRACKER_GHOST", "0") == "1"
        self._tracker_ghost_qpos36 = None
        self._tracker_ghost_data = None
        self._tracker_ghost_option = None
        self._tracker_ghost_perturb = None
        self._tracker_ghost_body_qpos_adrs = None
        self._tracker_ghost_root_body_id = -1
        self._tracker_ghost_visual_group = -1
        self._tracker_ghost_reported = False

        self.reward_lock = Lock()

        self.last_reward = 0
        # if self.offscreen:
        #     self.init_renderers()

        try:
            if self.sonic_config.get("INTERFACE", None):
                ChannelFactoryInitialize(self.sonic_config["DOMAIN_ID"], self.sonic_config["INTERFACE"])
            else:
                ChannelFactoryInitialize(self.sonic_config["DOMAIN_ID"])
        except Exception as e:
            print(f"Note: Channel factory initialization attempt: {e}")

        self.sim_thread = None
        self.viewer = None


    def init_publisher(self):
        pass

    # adapted from base_sim.update_viewer
    def update_viewer(self): # FIXME 
        if self.viewer is not None:
            if self._tracker_ghost_enabled:
                self._update_tracker_ghost()
            self.viewer.sync() # where .sync is called?

    def _body_is_descendant(self, body_id: int, root_body_id: int) -> bool:
        current = int(body_id)
        while current > 0:
            if current == root_body_id:
                return True
            current = int(self.mjModel.body_parentid[current])
        return root_body_id == 0

    def _init_tracker_ghost(self) -> None:
        if not self._tracker_ghost_enabled or self.viewer is None:
            return

        from simple.baselines.textop_tracker_adapter import MUJOCO_JOINT_NAMES

        self._tracker_ghost_data = mujoco.MjData(self.mjModel)
        self._tracker_ghost_option = mujoco.MjvOption()
        self._tracker_ghost_option.geomgroup[:] = 0
        self._tracker_ghost_perturb = mujoco.MjvPerturb()
        self._tracker_ghost_body_qpos_adrs = np.asarray(
            [int(self.mjModel.joint(name).qposadr[0]) for name in MUJOCO_JOINT_NAMES],
            dtype=np.int32,
        )

        root_name = os.environ.get("SIMPLE_TRACKER_GHOST_ROOT_BODY", "pelvis")
        self._tracker_ghost_root_body_id = mujoco.mj_name2id(
            self.mjModel, mujoco.mjtObj.mjOBJ_BODY, root_name
        )
        if self._tracker_ghost_root_body_id < 0:
            raise RuntimeError(f"Tracker ghost root body not found: {root_name}")

        group_counts: dict[int, int] = {}
        for geom_id in range(self.mjModel.ngeom):
            body_id = int(self.mjModel.geom_bodyid[geom_id])
            if self._body_is_descendant(body_id, self._tracker_ghost_root_body_id):
                group = int(self.mjModel.geom_group[geom_id])
                group_counts[group] = group_counts.get(group, 0) + 1
        if not group_counts:
            raise RuntimeError(f"Tracker ghost found no robot geoms below {root_name}")

        requested_group = os.environ.get("SIMPLE_TRACKER_GHOST_GEOM_GROUP", "").strip()
        if requested_group:
            visual_group = int(requested_group)
        elif 1 in group_counts:
            # SIMPLE's G1 MJCF uses group 1 for render meshes and group 0 for
            # collision primitives.  Prefer the actual robot appearance even
            # when collision decomposition happens to contain more geoms.
            visual_group = 1
        elif 2 in group_counts:
            visual_group = 2
        else:
            visual_group = max(group_counts, key=group_counts.get)
        if not 0 <= visual_group < len(self._tracker_ghost_option.geomgroup):
            raise RuntimeError(f"Invalid tracker ghost geom group: {visual_group}")
        self._tracker_ghost_visual_group = visual_group
        self._tracker_ghost_option.geomgroup[visual_group] = 1
        self._tracker_ghost_reported = False
        print(
            "[TrackerGhost] enabled "
            f"root={root_name} robot_geom_groups={group_counts} "
            f"visual_group={visual_group}",
            flush=True,
        )

    def _set_tracker_ghost_from_action(self, action) -> None:
        if not self._tracker_ghost_enabled:
            return
        parameters = getattr(action, "parameters", None)
        reference = (
            parameters.get("debug_reference_qpos36")
            if isinstance(parameters, dict)
            else None
        )
        if reference is None:
            return
        reference = np.asarray(reference, dtype=np.float64).reshape(-1)
        if reference.size != 36 or not np.all(np.isfinite(reference)):
            raise RuntimeError(
                "Tracker ghost reference must contain 36 finite qpos values; "
                f"got shape={reference.shape}"
            )
        self._tracker_ghost_qpos36 = reference.copy()

    def _update_tracker_ghost(self) -> None:
        if self.viewer is None:
            return
        if self._tracker_ghost_data is None:
            self._init_tracker_ghost()

        with self.viewer.lock():
            scene = self.viewer.user_scn
            scene.ngeom = 0
            if self._tracker_ghost_qpos36 is None:
                return

            self._tracker_ghost_data.qpos[:] = self.mjData.qpos
            self._tracker_ghost_data.qvel[:] = 0.0
            self._tracker_ghost_data.qpos[:7] = self._tracker_ghost_qpos36[:7]
            self._tracker_ghost_data.qpos[self._tracker_ghost_body_qpos_adrs] = (
                self._tracker_ghost_qpos36[7:36]
            )
            mujoco.mj_forward(self.mjModel, self._tracker_ghost_data)
            mujoco.mjv_addGeoms(
                self.mjModel,
                self._tracker_ghost_data,
                self._tracker_ghost_option,
                self._tracker_ghost_perturb,
                mujoco.mjtCatBit.mjCAT_DYNAMIC,
                scene,
            )

            ghost_count = 0
            rgba = np.asarray([0.45, 1.0, 0.55, 0.28], dtype=np.float32)
            for index in range(scene.ngeom):
                geom = scene.geoms[index]
                geom_id = int(geom.objid)
                is_robot_visual = (
                    int(geom.objtype) == int(mujoco.mjtObj.mjOBJ_GEOM)
                    and 0 <= geom_id < self.mjModel.ngeom
                    and int(self.mjModel.geom_group[geom_id])
                    == self._tracker_ghost_visual_group
                    and self._body_is_descendant(
                        int(self.mjModel.geom_bodyid[geom_id]),
                        self._tracker_ghost_root_body_id,
                    )
                )
                if not is_robot_visual:
                    geom.rgba[3] = 0.0
                    continue
                geom.rgba[:] = rgba
                geom.category = mujoco.mjtCatBit.mjCAT_DECOR
                ghost_count += 1

            if not self._tracker_ghost_reported:
                if ghost_count == 0:
                    raise RuntimeError(
                        "Tracker ghost produced zero robot visual geoms; "
                        f"selected group={self._tracker_ghost_visual_group}"
                    )
                print(f"[TrackerGhost] ghost geoms drawn: {ghost_count}", flush=True)
                self._tracker_ghost_reported = True
    
    def update_reward(self):
        with self.reward_lock:
            self.last_reward = 0

    def get_reward(self):
        with self.reward_lock:
            return self.last_reward

    def _get_obs(self):
        qpos = np.asarray(list(self.mujoco.get_robot_qpos().values()), dtype=np.float32)
        if self.isaac:
            isaacsim_joint_indices = []
            for jname, _ in self.mujoco.get_robot_qpos().items():
                isaac_jname = self.task.robot.jname_mujoco_to_isaac(jname)
                isaacsim_joint_indices.append(self.isaac.robot.get_dof_index(isaac_jname))
            qpos = self.isaac.robot.get_joint_positions(joint_indices=isaacsim_joint_indices)

        return {
            "joint_qpos": qpos,
            ** self._render_frame()
        }
    
    def _get_info(self):
        info = {}
        for k, v in self.mujoco.mj_objects.items():
            info[str(k)] = np.concatenate([v.xpos, v.xquat])

        # get wholebody proprioception info 
        # currently, it is returned as part of info dict, 
        # but we may want to merge into obs in the future
        proprio = self.task.robot.prepare_obs()
        return {**info, "proprio": proprio}
    
    def reset(
        self, 
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:  # type: ignore
        super().reset(seed=seed, options=options)
        # reset task layout in mujoco with domain randomization
        self.task.reset(seed, options)
        # update mujoco layout with new task layout
        self.mujoco.update_layout(sonic_config=self.sonic_config)

        # FIXME
        self.mjSpec = self.mujoco.mjSpec
        self.mjModel = self.mujoco.mjModel
        self.mjData = self.mujoco.mjData
        
        # set up viewer
        if self.onscreen:
            if self.viewer is not None:
                self.viewer.close()
            self.viewer = mujoco.viewer.launch_passive(
                self.mjModel,
                self.mjData,
                key_callback=self.task.robot.elastic_band.MujuocoKeyCallback,
                show_left_ui=False,
                show_right_ui=False,
            )
        else:
            mujoco.mj_forward(self.mjModel, self.mjData)
            self.viewer = None

        if self.viewer:
            self.viewer.cam.azimuth = 100
            self.viewer.cam.elevation = -30
            self.viewer.cam.distance = 3
            self.viewer.cam.lookat = np.array([0, 0, 0.38])
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE # mjCAMERA_TRACKING

        self._tracker_ghost_qpos36 = None
        self._tracker_ghost_data = None
        if self.viewer is not None and self._tracker_ghost_enabled:
            self._init_tracker_ghost()

        # In external-Isaac mode the reset observation must represent the exact
        # CSV row-0 state installed by task.reset()/mj_forward above.  The
        # blocking _get_obs() below publishes and renders it before any physics
        # integration.  Other simulator modes retain their historical warm-up.
        if self.external_isaac is None:
            self.mujoco.step(render=False)

        if self.isaac is not None:
            self.isaac.reset()
            self.isaac.update_layout()
            self.isaac.step(self.mujoco)

        self.control_decimal = int((1/self.mujoco.physics_dt) / self.task.render_hz)
        self.step_count = 0
        self.close_cmd_received = False
        obs = self._get_obs()
        info = self._get_info()
        self._success = False
        return obs, info
    
    def step(self, action):
        self._set_tracker_ghost_from_action(action)
        for i in range(self.control_decimal):
            # self.check_fall()
            # with self._telemetry.timer("mujoco.apply_action"):
            self.mujoco.apply_action(action)

            # with self._telemetry.timer("mujoco.step"):
            self.mujoco.step(render=False) # FIXME  # only render on last step
            if self.isaac:
                self.isaac.step(self.mujoco)
        
        self.step_count += 1
        # with self._telemetry.timer("env._get_obs"):
        obs = self._get_obs()
        # with self._telemetry.timer("env._get_info"):
        info = self._get_info()

        # with self._telemetry.timer("task.compute_reward"):
        reward = self.task.compute_reward(info , mujoco_env=self.mujoco)
        
        # with self._telemetry.timer("task.check_success"):
        terminated = self.task.check_success(info, mujoco_env=self.mujoco)
        
        truncated = False
        self._success = terminated
        return obs, reward, terminated, truncated, info
    
    """ def check_fall(self):
        self.fall = False
        if self.mjData.qpos[2] < 0.2:
            self.fall = True
            print(f"Warning: Robot has fallen, height: {self.mjData.qpos[2]:.3f} m")

        if self.fall:
            self.__reset() """

    """ def __reset(self): # FIXME
        mujoco.mj_resetData(self.mjModel, self.mjData)
        # Re-arm the elastic band so the robot hangs safely again after a fall.
        if self.task.robot.elastic_band:
            self.task.robot.elastic_band.length = 0.0
            self.task.robot.elastic_band.enable = True """
    
    def render(self):
        if self.external_isaac is not None:
            return self.external_isaac.cached_frame()
        return self._render_frame()

    def _render_frame(self):
        if self.external_isaac is not None:
            return self.external_isaac.sync_and_render(self.mujoco)
        frame_mujoco = self.mujoco.render()

        if self.isaac:
            frame_isaac = self.isaac.render()
            if "debug" in self.task.metadata and self.task.metadata["debug"]:
                # tile frame_mujoco and frame isaac together
                frame_tiled = {}
                for key, isaac_img in frame_isaac.items():
                    mujoco_img = frame_mujoco[key]
                    width = mujoco_img.shape[1]
                    frame_tiled[key] = np.concatenate([
                        isaac_img[:,:width//2,:],mujoco_img[:,width//2:,:] 
                    ], axis=1)
                return frame_tiled
            else:
                return frame_isaac
        else:
            return frame_mujoco

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        super().close()
