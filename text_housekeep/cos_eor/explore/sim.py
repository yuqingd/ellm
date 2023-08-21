from text_housekeep.cos_eor.sim.sim import CosRearrangementSim
import gzip
import json
import math
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import quaternion

from text_housekeep.habitat_lab.habitat.core.registry import registry
from text_housekeep.habitat_lab.habitat.core.simulator import (
    Config,
    Observations,
)
from text_housekeep.habitat_lab.habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from text_housekeep.cos_eor.utils.geometry import (
    cartesian_to_polar,
    quaternion_rotate_vector,
    compute_egocentric_delta,
    compute_updated_pose,
)
from text_housekeep.cos_eor.explore.utils.geometry import truncated_normal_noise

from text_housekeep.cos_eor.explore.utils.visualization import get_topdown_map_v2


@registry.register_simulator(name="ExploreSim-v0")
class ExploreSim(CosRearrangementSim):
    r"""Simulator wrapper over HabitatSim that additionally builds an
    occupancy map of the environment as the agent is moving.

    Args:
        config: configuration for initializing the simulator.

    Acknowledgement: large parts of the occupancy generation code were
    borrowed from https://github.com/taochenshh/exp4nav with some
    modifications for faster processing.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.initialize_map(config)

    def initialize_map(self, config):
        r"""Initializes the map configurations and useful variables for map
        computation.
        """
        occ_cfg = config.OCCUPANCY_MAPS

        # Handling noisy odometer scenario
        self._estimated_position = None
        self._estimated_rotation = None
        self._enable_odometer_noise = config.ENABLE_ODOMETRY_NOISE
        self._odometer_noise_eta = config.ODOMETER_NOISE_SCALING

        # ======================= Store map configurations ====================
        occ_info = {
            "map_scale": occ_cfg.MAP_SCALE,
            "map_size": occ_cfg.MAP_SIZE,
            "max_depth": occ_cfg.MAX_DEPTH,
            "small_map_range": occ_cfg.SMALL_MAP_RANGE,
            "large_map_range": occ_cfg.LARGE_MAP_RANGE,
            "small_map_size": config.FINE_OCC_SENSOR.WIDTH,
            "large_map_size": config.COARSE_OCC_SENSOR.WIDTH,
            "height_threshold": (occ_cfg.HEIGHT_LOWER, occ_cfg.HEIGHT_UPPER),
            "get_proj_loc_map": occ_cfg.GET_PROJ_LOC_MAP,
            "use_gt_occ_map": occ_cfg.USE_GT_OCC_MAP,
            # NOTE: This assumes that there is only one agent
            "agent_height": config.AGENT_0.HEIGHT,
            "Lx_min": None,
            "Lx_max": None,
            "Lz_min": None,
            "Lz_max": None,
            # Coverage novelty reward
            "coverage_novelty_pooling": config.OCCUPANCY_MAPS.COVERAGE_NOVELTY_POOLING,
        }
        # High-resolution map options.
        occ_info["get_highres_loc_map"] = occ_cfg.GET_HIGHRES_LOC_MAP
        if occ_info["get_highres_loc_map"]:
            occ_info["highres_large_map_size"] = config.HIGHRES_COARSE_OCC_SENSOR.WIDTH
        # Measure noise-free area covered or noisy area covered?
        if self._enable_odometer_noise:
            occ_info["measure_noise_free_area"] = occ_cfg.MEASURE_NOISE_FREE_AREA
        else:
            occ_info["measure_noise_free_area"] = False
        # Camera intrinsics.
        hfov = math.radians(self._sensor_suite.sensors["depth"].config.HFOV)
        v1 = 1.0 / np.tan(hfov / 2.0)
        v2 = 1.0 / np.tan(hfov / 2.0)  # Assumes both FoVs are same.
        intrinsic_matrix = np.array(
            [
                [v1, 0.0, 0.0, 0.0],
                [0.0, v2, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        occ_info["intrinsic_matrix"] = intrinsic_matrix
        occ_info["inverse_intrinsic_matrix"] = np.linalg.inv(intrinsic_matrix)
        self.occupancy_info = occ_info
        # ======================= Object annotations ==========================
        self.has_object_annotations = config.OBJECT_ANNOTATIONS.IS_AVAILABLE
        self.object_annotations_dir = config.OBJECT_ANNOTATIONS.PATH
        # ========== Memory to be allocated at the start of an episode ========
        self.grids_mat = None
        self.count_grids_mat = None
        self.noise_free_grids_mat = None
        self.gt_grids_mat = None
        self.proj_grids_mat = None
        # ========================== GT topdown map ===========================
        self._gt_top_down_map = None
        # Maintain a cache to avoid redundant computation and to store
        # useful statistics.
        self._cache = {}
        W = config.DEPTH_SENSOR.WIDTH
        # Cache meshgrid for depth projection.
        # [1, -1] for y as array indexing is y-down while world is y-up.
        xs, ys = np.meshgrid(np.linspace(-1, 1, W), np.linspace(1, -1, W))
        self._cache["xs"] = xs
        self._cache["ys"] = ys

    def create_grid_memory(self):
        r"""Pre-assigns memory for global grids which are used to aggregate
        the per-frame occupancy maps.
        """
        grid_size = self.occupancy_info["map_scale"]
        min_x, min_z, max_x, max_z = self.get_environment_extents()
        # Compute map size conditioned on environment extents.
        # Add a 5m buffer to account for noise in extent estimates.
        Lx_min, Lx_max, Lz_min, Lz_max = min_x - 5, max_x + 5, min_z - 5, max_z + 5
        is_same_environment = (
            (Lx_min == self.occupancy_info["Lx_min"])
            and (Lx_max == self.occupancy_info["Lx_max"])
            and (Lz_min == self.occupancy_info["Lz_min"])
            and (Lz_max == self.occupancy_info["Lz_max"])
        )
        # Only if the environment changes, create new arrays.
        if not is_same_environment:
            # Update extents data
            self.occupancy_info["Lx_min"] = Lx_min
            self.occupancy_info["Lx_max"] = Lx_max
            self.occupancy_info["Lz_min"] = Lz_min
            self.occupancy_info["Lz_max"] = Lz_max
            grid_num = (
                int((Lx_max - Lx_min) / grid_size),
                int((Lz_max - Lz_min) / grid_size),
            )
            # Create new arrays
            self.grids_mat = np.zeros(grid_num, np.uint8)
            self.count_grids_mat = np.zeros(grid_num, dtype=np.float32)
            if self.occupancy_info["measure_noise_free_area"]:
                self.noise_free_grids_mat = np.zeros(grid_num, np.uint8)
            if self.occupancy_info["use_gt_occ_map"]:
                self.gt_grids_mat = np.zeros(grid_num, np.uint8)
            """
            The local projection has 3 channels.
            One each for occupied, free and unknown.
            """
            if self.occupancy_info["get_proj_loc_map"]:
                self.proj_grids_mat = np.zeros((*grid_num, 3), np.uint8)

    def reset(self):
        sim_obs = super().reset()

        agent_state = self.get_agent_state()

        # If noisy odometer is enabled, maintain an
        # estimated position and rotation for the agent.
        if self._enable_odometer_noise:
            # Initialize with the ground-truth position, rotation
            self._estimated_position = agent_state.position
            self._estimated_rotation = agent_state.rotation
        # Create map memory and reset stats
        self.create_grid_memory()
        self.reset_occupancy_stats()
        # Obtain ground-truth environment layout
        self.gt_map_creation_height = agent_state.position[1]
        if self.occupancy_info["use_gt_occ_map"]:
            self._gt_top_down_map = self.get_original_map()
        # Update map based on current observations
        sim_obs = self._update_map_observations(sim_obs)
        # Load object annotations if available
        if self.has_object_annotations:
            scene_id = self._current_scene.split("/")[-1]
            annot_path = f"{self.object_annotations_dir}/{scene_id}.json.gz"
            with gzip.open(annot_path, "rt") as fp:
                self.object_annotations = json.load(fp)
        else:
            self.object_annotations = []

        self._prev_sim_obs = sim_obs
        self._is_episode_active = True
        return self._sensor_suite.get_observations(sim_obs)

    @property
    def fine_occupancy_map(self):
        return self._prev_sim_obs.get("fine_occupancy")

    @property
    def coarse_occupancy_map(self):
        return self._prev_sim_obs.get("coarse_occupancy")

    @property
    def proj_occupancy_map(self):
        return self._prev_sim_obs.get("proj_occupancy")

    @property
    def highres_coarse_occupancy_map(self):
        return self._prev_sim_obs.get("highres_coarse_occupancy")

    def get_specific_sensor_observations_at(
        self, position: List[float], rotation: List[float], sensor_uuid: str,
    ) -> Optional[Observations]:

        current_state = self.get_agent_state()
        success = self.set_agent_state(position, rotation, reset_sensors=False)

        if success:
            # specific_sim_obs = self.get_sensor_observations().get(sensor_uuid)
            specific_sim_obs = self._sensors.get(sensor_uuid).get_observation()
            self.set_agent_state(
                current_state.position, current_state.rotation, reset_sensors=False,
            )
            return specific_sim_obs
        else:
            return None

    def step(self, action):
        assert self._is_episode_active, (
            "episode is not active, environment not RESET or "
            "STOP action called previously"
        )
        agent_state = self.get_agent_state()
        position_before_step = agent_state.position
        rotation_before_step = agent_state.rotation

        # if action == self.index_stop_action:
        #     self._is_episode_active = False
        #     sim_obs = self._sim.get_sensor_observations()
        # else:
        #     sim_obs = self._sim.step(action)

        super().step(action)
        sim_obs = self._prev_sim_obs

        agent_state = self.get_agent_state()
        position_after_step = agent_state.position
        rotation_after_step = agent_state.rotation

        # Compute the estimated position, rotation.
        if self._enable_odometer_noise and action != self.index_stop_action:
            # Measure ground-truth delta in egocentric coordinates.
            delta_rpt_gt = compute_egocentric_delta(
                position_before_step,
                rotation_before_step,
                position_after_step,
                rotation_after_step,
            )
            delta_y_gt = position_after_step[1] - position_before_step[1]
            # Add noise to the ground-truth delta.
            eta = self._odometer_noise_eta
            D_rho, D_phi, D_theta = delta_rpt_gt
            D_rho_n = D_rho + truncated_normal_noise(eta, 2 * eta) * D_rho
            D_phi_n = D_phi
            D_theta_n = D_theta + truncated_normal_noise(eta, 2 * eta) * D_theta
            delta_rpt_n = np.array((D_rho_n, D_phi_n, D_theta_n))
            delta_y_n = delta_y_gt
            # Update noisy pose estimates
            old_position = self._estimated_position
            old_rotation = self._estimated_rotation
            (new_position, new_rotation) = compute_updated_pose(
                old_position, old_rotation, delta_rpt_n, delta_y_n
            )
            self._estimated_position = new_position
            self._estimated_rotation = new_rotation
        # Update map based on current observations
        sim_obs = self._update_map_observations(sim_obs)

        self._prev_sim_obs = sim_obs
        return self._sensor_suite.get_observations(self._prev_sim_obs)

    def convert_to_pointcloud(
        self,
        rgb: np.array,
        depth: np.array,
        agent_position: np.array,
        agent_rotation: np.quaternion,
    ) -> Tuple[np.array, Optional[np.array]]:
        """Converts depth input into a sequence of points corresponding to
        the 3D projection of camera points by using both intrinsic and
        extrinsic parameters.

        Args:
            rgb - uint8 RGB images
            depth - normalized depth inputs with values lying in [0.0, 1.0]
            agent_position - pre-computed agent position for efficiency
            agent_rotation - pre-computed agent rotation for efficiency
        Returns:
            xyz_world - a sequence of (x, y, z) real-world coordinates,
                        may contain noise depending on the settings.
            xyz_world_nf - a sequence of (x, y, z) real-world coordinates,
                            strictly noise-free (Optional).
        """
        # =============== Unnormalize depth input if applicable ===============
        depth_sensor = self._sensor_suite.sensors["depth"]
        min_depth_value = depth_sensor.config.MIN_DEPTH
        max_depth_value = depth_sensor.config.MAX_DEPTH
        if depth_sensor.config.NORMALIZE_DEPTH:
            depth_float = depth.astype(np.float32) * max_depth_value + min_depth_value
        else:
            depth_float = depth.astype(np.float32)
        depth_float = depth_float[..., 0]
        # ========== Convert to camera coordinates using intrinsics ===========
        W = depth.shape[1]
        xs = np.copy(self._cache["xs"]).reshape(-1)
        ys = np.copy(self._cache["ys"]).reshape(-1)
        depth_float = depth_float.reshape(-1)
        # Filter out invalid depths.
        valid_depths = (depth_float != 0.0) & (
            depth_float <= self.occupancy_info["max_depth"]
        ) & (
            ~np.isclose(depth_float, min_depth_value)
        )
        xs = xs[valid_depths]
        ys = ys[valid_depths]
        depth_float = depth_float[valid_depths]
        # Project to 3D coordinates.
        # Negate depth as the camera looks along -Z.
        xys = np.vstack(
            (
                xs * depth_float,
                ys * depth_float,
                -depth_float,
                np.ones(depth_float.shape),
            )
        )
        inv_K = self.occupancy_info["inverse_intrinsic_matrix"]
        xyz_cam = np.matmul(inv_K, xys)
        ## Uncomment for visualizing point-clouds in camera coordinates.
        # colors = rgb.reshape(-1, 3)
        # colors = colors[valid_depths, :]
        # cv2.imshow('RGB', rgb[:, :, ::-1])
        # cv2.imshow('Depth', depth)
        # cv2.waitKey(0)
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # colors = colors.astype(np.float32)/255.0
        # ax.scatter(xyz_cam[0, :], xyz_cam[1, :], xyz_cam[2, :], c = colors)
        # ax.set_xlabel('X axis')
        # ax.set_ylabel('Y axis')
        # ax.set_zlabel('Z axis')
        # ax.view_init(elev=0.0, azim=-90.0)
        # plt.show()
        # =========== Convert to world coordinates using extrinsics ===========
        T_world = np.eye(4)
        T_world[:3, :3] = quaternion.as_rotation_matrix(agent_rotation)
        T_world[:3, 3] = agent_position
        xyz_world = np.matmul(T_world, xyz_cam).T
        # Convert to non-homogeneous coordinates
        xyz_world = xyz_world[:, :3] / xyz_world[:, 3][:, np.newaxis]
        # ============ Compute noise-free point-cloud if required =============
        xyz_world_nf = None
        if self.occupancy_info["measure_noise_free_area"]:
            agent_state = self.get_agent_state()
            agent_position = agent_state.position
            agent_rotation = agent_state.rotation
            T_world = np.eye(4)
            T_world[:3, :3] = quaternion.as_rotation_matrix(agent_rotation)
            T_world[:3, 3] = agent_position
            xyz_world_nf = np.matmul(T_world, xyz_cam).T
            # Convert to non-homogeneous coordinates
            xyz_world_nf = xyz_world_nf[:, :3] / xyz_world_nf[:, 3][:, np.newaxis]
        return xyz_world, xyz_world_nf

    def reconfigure(self, config: Config) -> None:
        super().reconfigure(config)
        self.initialize_map(config)

    def get_observations_at(
        self,
        position: List[float] = None,
        rotation: List[float] = None,
        keep_agent_at_new_pose: bool = False,
    ) -> Optional[Observations]:

        current_state = self.get_agent_state()

        if position is None or rotation is None:
            success = True
        else:
            success = self.set_agent_state(position, rotation, reset_sensors=False)
        if success:
            sim_obs = self.get_sensor_observations()
            if keep_agent_at_new_pose:
                sim_obs = self._update_map_observations(sim_obs)
            else:
                # Difference being that the global map will not be updated
                # using the current observation.
                (
                    fine_occupancy,
                    coarse_occupancy,
                    highres_coarse_occupancy,
                ) = self.get_local_maps()
                sim_obs["coarse_occupancy"] = coarse_occupancy
                sim_obs["fine_occupancy"] = fine_occupancy
                sim_obs["highres_coarse_occupancy"] = highres_coarse_occupancy
                if self.occupancy_info["get_proj_loc_map"]:
                    proj_occupancy = self.get_proj_loc_map()
                    sim_obs["proj_occupancy"] = proj_occupancy
            self._prev_sim_obs = sim_obs
            # Process observations using sensor_suite.
            observations = self._sensor_suite.get_observations(sim_obs)
            if not keep_agent_at_new_pose:
                self.set_agent_state(
                    current_state.position, current_state.rotation, reset_sensors=False,
                )
            return observations
        else:
            return None

    def get_original_map(self) -> np.array:
        r"""Returns the top-down environment layout in the current global
        map scale.
        """
        x_min = self.occupancy_info["Lx_min"]
        x_max = self.occupancy_info["Lx_max"]
        z_min = self.occupancy_info["Lz_min"]
        z_max = self.occupancy_info["Lz_max"]
        top_down_map = get_topdown_map_v2(
            self, (x_min, x_max, z_min, z_max), self.occupancy_info["map_scale"], 20000,
        )
        return top_down_map

    def reset_occupancy_stats(self):
        r"""Resets occupancy maps, area estimates.
        """
        self.occupancy_info["seen_area"] = 0
        self.occupancy_info["inc_area"] = 0
        self.grids_mat.fill(0)
        self.count_grids_mat.fill(0)
        if self.occupancy_info["measure_noise_free_area"]:
            self.noise_free_grids_mat.fill(0)
        if self.occupancy_info["use_gt_occ_map"]:
            self.gt_grids_mat.fill(0)
        if self.occupancy_info["get_proj_loc_map"]:
            self.proj_grids_mat.fill(0)

    def get_seen_area(
        self,
        rgb: np.array,
        depth: np.array,
        out_mat: np.array,
        count_out_mat: np.array,
        gt_out_mat: Optional[np.array],
        proj_out_mat: Optional[np.array],
        noise_free_out_mat: Optional[np.array],
    ) -> int:
        r"""Given new RGBD observations, it updates the global occupancy map
        and computes total area seen after the update.

        Args:
            rgb - uint8 RGB images.
            depth - normalized depth inputs with values lying in [0.0, 1.0].
            *out_mat - global map to aggregate current inputs in.
        Returns:
            Area seen in the environment after aggregating current inputs.
            Area is measured in gridcells. Multiply by map_scale**2 to
            get area in m^2.
        """
        sensor_state = self.get_agent_state().sensor_states["depth"]
        if self._enable_odometer_noise:
            sensor_position = self._estimated_position
            sensor_rotation = self._estimated_rotation
        else:
            sensor_position = sensor_state.position
            sensor_rotation = sensor_state.rotation
        # ====================== Compute the pointcloud =======================
        XYZ_ego, XYZ_ego_nf = self.convert_to_pointcloud(
            rgb, depth, sensor_position, sensor_rotation
        )
        # Normalizing the point cloud so that ground plane is Y=0
        if self._enable_odometer_noise:
            current_sensor_y = self._estimated_position[1]
        else:
            current_sensor_y = sensor_state.position[1]
        ground_plane_y = current_sensor_y - self.occupancy_info["agent_height"]
        XYZ_ego[:, 1] -= ground_plane_y
        # Measure pointcloud without ground-truth pose instead of estimated.
        if self.occupancy_info["measure_noise_free_area"]:
            ground_plane_y = (
                sensor_state.position[1] - self.occupancy_info["agent_height"]
            )
            XYZ_ego_nf[:, 1] -= ground_plane_y
        # ================== Compute local occupancy map ======================
        grids_mat = np.zeros(self.grids_mat.shape, dtype=np.uint8)
        Lx_min = self.occupancy_info["Lx_min"]
        Lz_min = self.occupancy_info["Lz_min"]
        grid_size = self.occupancy_info["map_scale"]
        height_thresh = self.occupancy_info["height_threshold"]
        points = XYZ_ego
        # Compute grid coordinates of points in pointcloud.
        grid_locs = (points[:, [0, 2]] - np.array([[Lx_min, Lz_min]])) / grid_size
        grid_locs = np.floor(grid_locs).astype(int)
        # Classify points in occupancy map as free/occupied/unknown
        # using height-based thresholds on the point-cloud.
        high_filter_idx = points[:, 1] < height_thresh[1]
        low_filter_idx = points[:, 1] > height_thresh[0]
        obstacle_idx = np.logical_and(low_filter_idx, high_filter_idx)
        # Assign known space as all free initially.
        self.safe_assign(
            grids_mat, grid_locs[high_filter_idx, 0], grid_locs[high_filter_idx, 1], 2,
        )
        kernel = np.ones((3, 3), np.uint8)
        grids_mat = cv2.morphologyEx(grids_mat, cv2.MORPH_CLOSE, kernel)
        # Assign occupied space based on presence of obstacles.
        obs_mat = np.zeros(self.grids_mat.shape, dtype=np.uint8)
        self.safe_assign(
            obs_mat, grid_locs[obstacle_idx, 0], grid_locs[obstacle_idx, 1], 1
        )
        kernel = np.ones((3, 3), np.uint8)
        obs_mat = cv2.morphologyEx(obs_mat, cv2.MORPH_CLOSE, kernel)
        # ================== Update global occupancy map ======================
        visible_mask = grids_mat == 2
        occupied_mask = obs_mat == 1
        np.putmask(out_mat, visible_mask, 2)
        np.putmask(out_mat, occupied_mask, 1)
        # Update counts to each grid location
        seen_mask = (visible_mask | occupied_mask).astype(np.float32)
        count_out_mat += seen_mask
        inv_count_out_mat = np.ma.array(
            1 / np.sqrt(np.clip(count_out_mat, 1.0, math.inf)), mask=1 - seen_mask
        )
        # Pick out counts for locations seen in this frame
        if self.occupancy_info["coverage_novelty_pooling"] == "mean":
            seen_count_reward = inv_count_out_mat.mean().item()
        elif self.occupancy_info["coverage_novelty_pooling"] == "median":
            seen_count_reward = np.ma.median(inv_count_out_mat).item()
        elif self.occupancy_info["coverage_novelty_pooling"] == "max":
            seen_count_reward = inv_count_out_mat.max().item()
        self.occupancy_info["seen_count_reward"] = seen_count_reward
        # If ground-truth navigability is required (and not height-based),
        # obtain the navigability values for valid locations in out_mat from
        # use self._gt_top_down_map.
        if self.occupancy_info["use_gt_occ_map"]:
            gt_visible_mask = visible_mask | occupied_mask
            # Dilate the visible mask
            dkernel = np.ones((9, 9), np.uint8)
            gt_visible_mask = cv2.dilate(
                gt_visible_mask.astype(np.uint8), dkernel, iterations=2
            )
            gt_visible_mask = gt_visible_mask != 0
            gt_occupied_mask = gt_visible_mask & (self._gt_top_down_map == 0)
            np.putmask(gt_out_mat, gt_visible_mask, 2)
            np.putmask(gt_out_mat, gt_occupied_mask, 1)
        # If noise-free measurement for area-seen is required, then compute a
        # global map that uses the ground-truth pose values.
        if self.occupancy_info["measure_noise_free_area"]:
            # --------------- Compute local occupancy map ---------------------
            nf_grids_mat = np.zeros(self.grids_mat.shape, dtype=np.uint8)
            points_nf = XYZ_ego_nf
            # Compute grid coordinates of points in pointcloud.
            grid_locs_nf = (
                points_nf[:, [0, 2]] - np.array([[Lx_min, Lz_min]])
            ) / grid_size
            grid_locs_nf = np.floor(grid_locs_nf).astype(int)
            # Classify points in occupancy map as free/occupied/unknown
            # using height-based thresholds on the point-cloud.
            high_filter_idx_nf = points_nf[:, 1] < height_thresh[1]
            low_filter_idx_nf = points_nf[:, 1] > height_thresh[0]
            obstacle_idx_nf = np.logical_and(low_filter_idx_nf, high_filter_idx_nf)
            # Assign known space as all free initially.
            self.safe_assign(
                nf_grids_mat,
                grid_locs_nf[high_filter_idx_nf, 0],
                grid_locs_nf[high_filter_idx_nf, 1],
                2,
            )
            kernel = np.ones((3, 3), np.uint8)
            nf_grids_mat = cv2.morphologyEx(nf_grids_mat, cv2.MORPH_CLOSE, kernel)
            # Assign occupied space based on presence of obstacles.
            nf_obs_mat = np.zeros(self.grids_mat.shape, dtype=np.uint8)
            self.safe_assign(
                nf_obs_mat,
                grid_locs_nf[obstacle_idx_nf, 0],
                grid_locs_nf[obstacle_idx_nf, 1],
                1,
            )
            kernel = np.ones((3, 3), np.uint8)
            nf_obs_mat = cv2.morphologyEx(nf_obs_mat, cv2.MORPH_CLOSE, kernel)
            np.putmask(nf_grids_mat, nf_obs_mat == 1, 1)
            # ---------------- Update global occupancy map --------------------
            visible_mask_nf = nf_grids_mat == 2
            occupied_mask_nf = nf_grids_mat == 1
            np.putmask(noise_free_out_mat, visible_mask_nf, 2)
            np.putmask(noise_free_out_mat, occupied_mask_nf, 1)
        # ================== Measure area seen (m^2) in the map =====================
        if self.occupancy_info["measure_noise_free_area"]:
            seen_area = (
                float(np.count_nonzero(noise_free_out_mat > 0)) * (grid_size) ** 2
            )
        else:
            seen_area = float(np.count_nonzero(out_mat > 0)) * (grid_size) ** 2
        # ================= Compute local depth projection ====================
        if self.occupancy_info["get_proj_loc_map"]:
            proj_out_mat.fill(0)
            # Set everything to unknown initially.
            proj_out_mat[..., 2] = 1
            # Set obstacles.
            np.putmask(proj_out_mat[..., 0], obs_mat == 1, 1)
            np.putmask(proj_out_mat[..., 2], obs_mat == 1, 0)
            # Set free space.
            free_space_mask = (obs_mat != 1) & (grids_mat == 2)
            np.putmask(proj_out_mat[..., 1], free_space_mask, 1)
            np.putmask(proj_out_mat[..., 2], free_space_mask, 0)
        return seen_area

    def safe_assign(self, im_map, x_idx, y_idx, value):
        try:
            im_map[x_idx, y_idx] = value
        except IndexError:
            valid_idx1 = np.logical_and(x_idx >= 0, x_idx < im_map.shape[0])
            valid_idx2 = np.logical_and(y_idx >= 0, y_idx < im_map.shape[1])
            valid_idx = np.logical_and(valid_idx1, valid_idx2)
            im_map[x_idx[valid_idx], y_idx[valid_idx]] = value

    def get_camera_grid_pos(self) -> Tuple[np.array, np.array]:
        """Returns the agent's current position in both the real world
        (X, Z, theta from -Z to X) and the grid world (Xg, Zg) coordinates.
        """
        if self._enable_odometer_noise:
            position = self._estimated_position
            rotation = self._estimated_rotation
        else:
            agent_state = self.get_agent_state()
            position = agent_state.position
            rotation = agent_state.rotation
        X, Z = position[0], position[2]
        grid_size = self.occupancy_info["map_scale"]
        Lx_min = self.occupancy_info["Lx_min"]
        Lx_max = self.occupancy_info["Lx_max"]
        Lz_min = self.occupancy_info["Lz_min"]
        Lz_max = self.occupancy_info["Lz_max"]
        # Clamp positions within range.
        X = min(max(X, Lx_min), Lx_max)
        Z = min(max(Z, Lz_min), Lz_max)
        # Compute grid world positions.
        Xg = (X - Lx_min) / grid_size
        Zg = (Z - Lz_min) / grid_size
        # Real world rotation.
        direction_vector = np.array([0, 0, -1])
        heading_vector = quaternion_rotate_vector(rotation.inverse(), direction_vector)
        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        phi = -phi  # (rotation from -Z to X)
        return np.array((X, Z, phi)), np.array((Xg, Zg))

    def get_local_maps(self) -> Tuple[np.array, np.array, np.array]:
        r"""Generates egocentric crops of the global occupancy map.
        Returns:
            The occupancy images display free, occupied and unknown space.
            The color conventions are:
                free-space - (0, 255, 0)
                occupied-space - (0, 0, 255)
                unknown-space - (255, 255, 255)
            The outputs are:
                fine_ego_map_color - (H, W, 3) occupancy image
                coarse_ego_map_color - (H, W, 3) occupancy image
                highres_coarse_ego_map_color - (H, W, 3) occupancy image
        """
        # ================ The global occupancy map ===========================
        if self.occupancy_info["use_gt_occ_map"]:
            top_down_map = self.gt_grids_mat.copy()  # (map_size, map_size)
        else:
            top_down_map = self.grids_mat.copy()  # (map_size, map_size)
        # =========== Obtain local crop around the agent ======================
        # Agent's world and map positions.
        xzt_world, xz_map = self.get_camera_grid_pos()
        xz_map = (int(xz_map[0]), int(xz_map[1]))
        # Crop out only the essential parts of the global map.
        # This saves computation cost for the subsequent operations.
        # *_range - #grid-cells of the map on either sides of the center
        large_map_range = self.occupancy_info["large_map_range"]
        small_map_range = self.occupancy_info["small_map_range"]
        # *_size - output image size
        large_map_size = self.occupancy_info["large_map_size"]
        small_map_size = self.occupancy_info["small_map_size"]
        min_range = int(1.5 * large_map_range)
        x_start = max(0, xz_map[0] - min_range)
        x_end = min(top_down_map.shape[0], xz_map[0] + min_range)
        y_start = max(0, xz_map[1] - min_range)
        y_end = min(top_down_map.shape[1], xz_map[1] + min_range)
        ego_map = top_down_map[x_start:x_end, y_start:y_end]
        # Pad the cropped map to account for out-of-bound indices
        top_pad = max(min_range - xz_map[0], 0)
        left_pad = max(min_range - xz_map[1], 0)
        bottom_pad = max(min_range - top_down_map.shape[0] + xz_map[0] + 1, 0)
        right_pad = max(min_range - top_down_map.shape[1] + xz_map[1] + 1, 0)
        ego_map = np.pad(
            ego_map,
            ((top_pad, bottom_pad), (left_pad, right_pad)),
            "constant",
            constant_values=((0, 0), (0, 0)),
        )
        # The global map is currently addressed as follows:
        # rows are -X to X top to bottom, cols are -Z to Z left to right
        # To get -Z top and X right, we need transpose the map.
        ego_map = ego_map.transpose(1, 0)
        # Rotate the global map to obtain egocentric top-down view.
        half_size = ego_map.shape[0] // 2
        center = (half_size, half_size)
        rot_angle = math.degrees(xzt_world[2])
        M = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
        ego_map = cv2.warpAffine(
            ego_map,
            M,
            (ego_map.shape[1], ego_map.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255,),
        )
        # =========== Obtain final maps at different resolutions ==============
        # Obtain the fine occupancy map.
        start = int(half_size - small_map_range)
        end = int(half_size + small_map_range)
        fine_ego_map = ego_map[start:end, start:end]
        fine_ego_map = cv2.resize(
            fine_ego_map,
            (small_map_size, small_map_size),
            interpolation=cv2.INTER_NEAREST,
        )
        fine_ego_map = np.clip(fine_ego_map, 0, 2)
        # Obtain the coarse occupancy map.
        start = int(half_size - large_map_range)
        end = int(half_size + large_map_range)
        coarse_ego_map_orig = ego_map[start:end, start:end]
        coarse_ego_map = cv2.resize(
            coarse_ego_map_orig,
            (large_map_size, large_map_size),
            interpolation=cv2.INTER_NEAREST,
        )
        coarse_ego_map = np.clip(coarse_ego_map, 0, 2)
        # Obtain a high-resolution coarse occupancy map.
        # This is primarily useful as an input to an A* path-planner.
        if self.occupancy_info["get_highres_loc_map"]:
            map_size = self.occupancy_info["highres_large_map_size"]
            highres_coarse_ego_map = cv2.resize(
                coarse_ego_map_orig,
                (map_size, map_size),
                interpolation=cv2.INTER_NEAREST,
            )
            highres_coarse_ego_map = np.clip(highres_coarse_ego_map, 0, 2)
        # Convert to RGB maps.
        # Fine occupancy map.
        map_shape = (*fine_ego_map.shape, 3)
        fine_ego_map_color = np.zeros(map_shape, dtype=np.uint8)
        fine_ego_map_color[fine_ego_map == 0] = np.array([255, 255, 255])
        fine_ego_map_color[fine_ego_map == 1] = np.array([0, 0, 255])
        fine_ego_map_color[fine_ego_map == 2] = np.array([0, 255, 0])
        # Coarse occupancy map.
        map_shape = (*coarse_ego_map.shape, 3)
        coarse_ego_map_color = np.zeros(map_shape, dtype=np.uint8)
        coarse_ego_map_color[coarse_ego_map == 0] = np.array([255, 255, 255])
        coarse_ego_map_color[coarse_ego_map == 1] = np.array([0, 0, 255])
        coarse_ego_map_color[coarse_ego_map == 2] = np.array([0, 255, 0])
        # High-resolution coarse occupancy map.
        if self.occupancy_info["get_highres_loc_map"]:
            map_shape = (*highres_coarse_ego_map.shape, 3)
            highres_coarse_ego_map_color = np.zeros(map_shape, dtype=np.uint8)
            highres_coarse_ego_map_color[highres_coarse_ego_map == 0] = np.array(
                [255, 255, 255]
            )
            highres_coarse_ego_map_color[highres_coarse_ego_map == 1] = np.array(
                [0, 0, 255]
            )
            highres_coarse_ego_map_color[highres_coarse_ego_map == 2] = np.array(
                [0, 255, 0]
            )
        else:
            highres_coarse_ego_map_color = None

        return fine_ego_map_color, coarse_ego_map_color, highres_coarse_ego_map_color

    def get_proj_loc_map(self):
        """Generates a fine egocentric projection of depth map.
        Returns:
            The occupancy map is binary and indicates free, occupied and
            unknown spaces. Channel 0 - occupied space, channel 1 - free space,
            channel 2 - unknown space.
            The outputs are:
                fine_ego_map - (H, W, 3) occupancy map
        """
        # ================ The global occupancy map ===========================
        top_down_map = self.proj_grids_mat.copy()  # (map_size, map_size)
        # =========== Obtain local crop around the agent ======================
        # Agent's world and map positions.
        xzt_world, xz_map = self.get_camera_grid_pos()
        xz_map = (int(xz_map[0]), int(xz_map[1]))
        # Crop out only the essential parts of the global map.
        # This saves computation cost for the subsequent opeartions.
        # *_range - #grid-cells of the map on either sides of the center
        small_map_range = self.occupancy_info["small_map_range"]
        # *_size - output image size
        small_map_size = self.occupancy_info["small_map_size"]
        min_range = int(1.5 * small_map_range)
        x_start = max(0, xz_map[0] - min_range)
        x_end = min(top_down_map.shape[0], xz_map[0] + min_range)
        y_start = max(0, xz_map[1] - min_range)
        y_end = min(top_down_map.shape[1], xz_map[1] + min_range)
        ego_map = top_down_map[x_start:x_end, y_start:y_end]
        # Pad the cropped map to account for out-of-bound indices
        top_pad = max(min_range - xz_map[0], 0)
        left_pad = max(min_range - xz_map[1], 0)
        bottom_pad = max(min_range - top_down_map.shape[0] + xz_map[0] + 1, 0)
        right_pad = max(min_range - top_down_map.shape[1] + xz_map[1] + 1, 0)
        ego_map = np.pad(
            ego_map,
            ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
            "constant",
            constant_values=0,
        )
        # The global map is currently addressed as follows:
        # rows are -X to X top to bottom, cols are -Z to Z left to right
        # To get -Z top and X right, we need transpose the map.
        ego_map = ego_map.transpose(1, 0, 2)
        # Rotate the global map to obtain egocentric top-down view.
        half_size = ego_map.shape[0] // 2
        center = (half_size, half_size)
        rot_angle = math.degrees(xzt_world[2])
        M = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
        ego_map = cv2.warpAffine(
            ego_map,
            M,
            (ego_map.shape[1], ego_map.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0,),
        )
        # =========== Obtain final maps at different resolutions ==============
        # Obtain the fine occupancy map
        start = int(half_size - small_map_range)
        end = int(half_size + small_map_range)
        fine_ego_map = ego_map[start:end, start:end]
        fine_ego_map = cv2.resize(
            fine_ego_map,
            (small_map_size, small_map_size),
            interpolation=cv2.INTER_NEAREST,
        )
        # Note: There is no conversion to RGB here.
        fine_ego_map = np.clip(fine_ego_map, 0, 1)  # (H, W, 1)

        return fine_ego_map

    def get_environment_extents(self) -> Tuple[float, float, float, float]:
        """Returns the minimum and maximum X, Z coordinates navigable on
        the current floor.
        """
        num_samples = 20000
        start_height = self.get_agent_state().position[1]
        min_x, max_x = (math.inf, -math.inf)
        min_z, max_z = (math.inf, -math.inf)
        for _ in range(num_samples):
            point = self.sample_navigable_point()
            # Check if on same level as original
            if np.abs(start_height - point[1]) > 0.5:
                continue
            min_x = min(point[0], min_x)
            max_x = max(point[0], max_x)
            min_z = min(point[2], min_z)
            max_z = max(point[2], max_z)

        return (min_x, min_z, max_x, max_z)

    def _update_map_observations(self, sim_obs):
        r"""Given the default simulator observations, update it by adding the
        occupancy maps.

        Args:
            sim_obs - a dictionary containing observations from self._sim.
        Returns:
            sim_obs with occupancy maps added as keys to it.
        """
        sensors = self._sensor_suite.sensors
        proc_rgb = sensors["rgb"].get_observation(sim_obs)
        proc_depth = sensors["depth"].get_observation(sim_obs)
        # If the agent went to a new floor, update the GT map
        if self.occupancy_info["use_gt_occ_map"]:
            agent_height = self.get_agent_state().position[1]
            if abs(agent_height - self.gt_map_creation_height) >= 0.5:
                self._gt_top_down_map = self.get_original_map()
                self.gt_map_creation_height = agent_height
        # Update the map with new observations
        seen_area = self.get_seen_area(
            proc_rgb,
            proc_depth,
            self.grids_mat,
            self.count_grids_mat,
            self.gt_grids_mat,
            self.proj_grids_mat,
            self.noise_free_grids_mat,
        )
        inc_area = seen_area - self.occupancy_info["seen_area"]
        # Crop out new egocentric maps
        (
            fine_occupancy,
            coarse_occupancy,
            highres_coarse_occupancy,
        ) = self.get_local_maps()
        # Update stats, observations
        self.occupancy_info["seen_area"] = seen_area
        self.occupancy_info["inc_area"] = inc_area
        sim_obs["coarse_occupancy"] = coarse_occupancy
        sim_obs["fine_occupancy"] = fine_occupancy
        if self.occupancy_info["get_highres_loc_map"]:
            sim_obs["highres_coarse_occupancy"] = highres_coarse_occupancy
        if self.occupancy_info["get_proj_loc_map"]:
            proj_occupancy = self.get_proj_loc_map()
            sim_obs["proj_occupancy"] = proj_occupancy
        return sim_obs