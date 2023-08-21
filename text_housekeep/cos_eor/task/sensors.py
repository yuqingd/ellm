from typing import Any, Dict

import numpy as np
from gym import spaces
from text_housekeep.habitat_lab.habitat.utils.visualizations.utils import images_to_video

from text_housekeep.habitat_lab.habitat.config import Config
from text_housekeep.habitat_lab.habitat.config.default import _C, CN
from text_housekeep.habitat_lab.habitat.core.dataset import Episode
from text_housekeep.habitat_lab.habitat.core.embodied_task import EmbodiedTask
from text_housekeep.habitat_lab.habitat.core.registry import registry
from text_housekeep.habitat_lab.habitat.core.simulator import Observations, Sensor, SensorTypes

from text_housekeep.habitat_lab.habitat.tasks.nav.nav import (
    PointGoalSensor
)
from text_housekeep.cos_eor.dataset.dataset import CosRearrangementEpisode

from text_housekeep.cos_eor.task.task import CosRearrangementTask

from text_housekeep.habitat_lab.habitat.utils.visualizations import maps, fog_of_war
from text_housekeep.cos_eor.utils.geometry import (
    geodesic_distance, get_polar_angle
)
from text_housekeep.cos_eor.utils.planner import (
    find_shortest_path_for_multiple_objects,
    compute_distance_mat_using_fmm,
    compute_distance_using_fmm,
    find_dist_from_map
)
from scipy.spatial.transform import Rotation as R

from text_housekeep.cos_eor.sim.sim import CosRearrangementSim
from text_housekeep.habitat_lab.habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat_sim.utils.common import quat_to_angle_axis, quat_to_coeffs
from text_housekeep.habitat_lab.habitat.tasks.utils import cartesian_to_polar


@registry.register_sensor
class CosEorSensor(Sensor):
    def __init__(
        self,
        *args: Any,
        sim: CosRearrangementSim,
        config: Config,
        task: CosRearrangementTask,
        **kwargs: Any
    ):
        self._sim = sim
        self._task = task
        self.sensor_cache = {}
        self.debug = registry.mapping["debug"]
        self.debug_sensor = None
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(67000,),
            dtype=np.uint8,
        )

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "cos_eor"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def add_gps_compass(self, task_data, episode):
        agent_state = self._sim.get_agent_state()
        task_data["agent_rot"] = quat_to_coeffs(agent_state.rotation).tolist()
        task_data["agent_pos"] = agent_state.position.tolist()
        task_data["agent_start_pos"] = episode.start_position
        direction_vector = agent_state.position - episode.start_position
        rho, phi = cartesian_to_polar(-direction_vector[2], direction_vector[0])
        task_data["gps_compass"] = np.array([rho, -phi], dtype=np.float32)
        return task_data

    def add_camera_center_ray(self, task_data):
        render_camera = self._sim._sensors["rgb"]._sensor_object.render_camera
        center_ray = render_camera.unproject(self._sim.screen_center)
        task_data["camera_center_ray"] = {
            "origin": list(center_ray.origin),
            "direction": list(center_ray.direction)
        }
        return task_data

    def get_observation(
        self,
        observations: Dict[str, Observations],
        episode: Episode,
        task: CosRearrangementTask,
        *args: Any,
        **kwargs: Any
    ):
        gripped_obj_id = self._sim.gripped_object_id
        task_data = {"gripped_object_id": gripped_obj_id}

        # add start and current agent position
        task_data = self.add_gps_compass(task_data, episode)
        task_trackers = self._task.trackers(only_return=True)

        # add iid and sid of gripped object
        task_data["gripped_iid"] = -1 if gripped_obj_id == -1 else task_trackers["sim_obj_id_to_iid"][gripped_obj_id]
        task_data["gripped_sid"] = -1 if gripped_obj_id == -1 else task_trackers["iid_to_sid"][task_data["gripped_iid"]]

        # add info to registry (will be used in building semantic class framex)
        max_iids = 200
        iid_to_sid = task_trackers["iid_to_sid"]
        iids = np.array(list(iid_to_sid.keys()))
        sids = np.array(list(iid_to_sid.values()))
        iid_sid_map = np.zeros(iids.max() + 1, dtype=sids.dtype)  # k,v from approach #1
        iid_sid_map[iids] = sids
        task_data["semantic_class"] = iid_sid_map[observations["semantic"]]

        # add visible iids and sids
        visible_iids = np.unique(observations["semantic"]).tolist()
        obj_iids, rec_iids = [], []
        obj_sids, rec_sids = [], []

        for iid in visible_iids:
            if iid == 0:
                continue
            oid = task_trackers["iid_to_sim_obj_id"][iid]
            typ = task_trackers["sim_obj_id_to_type"][oid]
            sid = task_trackers["iid_to_sid"][iid]

            if typ == "rec":
                rec_iids.append(iid)
                rec_sids.append(sid)
            else:
                obj_iids.append(iid)
                obj_sids.append(sid)

        # crop
        obj_iids, obj_sids = obj_iids[:max_iids], obj_sids[:max_iids]
        rec_iids, rec_sids = rec_iids[:max_iids], rec_sids[:max_iids]
        task_data["num_visible_objs"] = len(obj_iids)
        task_data["num_visible_recs"] = len(rec_iids)

        # pad
        obj_iids, obj_sids = obj_iids + [0] * (-len(obj_iids) + max_iids), obj_sids + [0] * (-len(obj_sids) + max_iids)
        rec_iids, rec_sids = rec_iids + [0] * (-len(rec_iids) + max_iids), rec_sids + [0] * (-len(rec_sids) + max_iids)

        task_data["visible_obj_iids"] = obj_iids
        task_data["visible_obj_sids"] = obj_sids

        task_data["visible_rec_iids"] = rec_iids
        task_data["visible_rec_sids"] = rec_sids

        # don't need other sensors for training E2E policy
        if "HIE" not in self.config.CONTENT and not self.debug:
            self.debug_sensor = task_data
            return task_data

        task_data.update(task_trackers)
        # episode data lists all the objects that are present in the scene
        # (regardless of not they need to be rearranged / not)
        episode_data = {
            "objs_keys": episode.objs_keys,
            "recs_keys": episode.recs_keys,
            "recs_pos": episode.recs_pos,
            "objs_pos": episode.get_current_object_positions(self._task),
            "current_matrix": episode.state_matrix,
            "start_matrix": episode.start_matrix,
            "correct_mapping": episode.get_correct_mapping(),
            "current_mapping": episode.get_mapping("current"),
            "start_mapping": episode.get_mapping("start"),
        }
        task_data.update(episode_data)

        if "HIE" not in self.config.CONTENT and self.debug:
            task_data.update({
            "agent_obj_dists": episode.get_agent_object_distance(self._task, types=["l2"]),  # expensive
            "agent_rec_dists": episode.get_agent_receptacle_distance(self._task, types=["l2"]),  # expensive
            })
            return task_data

        task_data["is_collided"] = self._sim.previous_step_collided

        self.add_camera_center_ray(task_data)

        return task_data


@registry.register_sensor
class AllObjectPositions(PointGoalSensor):
    """Positions of only objects and not receptacles. Returned in observations """
    def __init__(
        self,
        sim: CosRearrangementSim,
        config: Config,
        task: EmbodiedTask,
        *args: Any,
        **kwargs: Any
    ):
        super().__init__(sim, config, *args, **kwargs)
        self._task = task

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "all_object_positions"

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (
            5,
            self._dimensionality,
        )

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self, observations, episode, task, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation
        sensor_data = np.zeros((5, 2))
        sim_object_ids = episode.get_objects_ids(self._task.obj_key_to_sim_obj_id)
        for idx, sim_obj_id in zip(range(len(sim_object_ids)), sim_object_ids):
            if sim_obj_id == self._task.agent_object_id or sim_obj_id in registry.mapping["ignore_object_ids"]:
                continue
            object_position = task.get_translation(sim_obj_id)
            sensor_data[idx] = self._compute_pointgoal(agent_position, rotation_world_agent, object_position)
        return sensor_data


@registry.register_sensor
class AllObjectGoals(PointGoalSensor):
    def __init__(
        self,
        sim: CosRearrangementSim,
        config: Config,
        task: EmbodiedTask,
        *args: Any,
        **kwargs: Any
    ):
        super().__init__(sim, config, *args, **kwargs)
        self._task = task

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "all_object_goals"

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (
            5,
            self._dimensionality,
        )

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        """ returned in observations // this maybe wrong we are only considering starting positions"""
        sensor_data = np.zeros((5, 2))
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation

        sim_object_ids = episode.get_objects_ids(self._task.obj_key_to_sim_obj_id)
        receptacles = episode.get_receptacles()

        # filter object-ids of the only the objects
        for idx, sim_obj_id, rec in zip(range(len(receptacles)), sim_object_ids, receptacles):
            if sim_obj_id == self._task.agent_object_id or sim_obj_id in registry.mapping["ignore_object_ids"]:
                continue
            goal_position = np.array(rec.position, dtype=np.float32)
            sensor_data[idx] = self._compute_pointgoal(agent_position, rotation_world_agent, goal_position)
        return sensor_data


@registry.register_sensor(name="OracleNextObjectSensor")
class OracleNextObjectSensor(PointGoalSensor):
    def __init__(self, sim: CosRearrangementSim, config: Config, **kwargs):
        self.uuid = "oracle_next_object"
        self._sim = sim
        self._config = config
        super().__init__(sim=sim, config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (5,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def _reset_sensor(
            self,
            observations: Dict[str, Observations],
            episode: CosRearrangementEpisode,
            task: CosRearrangementTask,
            **kwargs
    ):
        self.pickup_order = [id-1 for id in episode.gd_pickup_order]

    def get_pointgoal_for_object(self, episode, task, ep_obj_id, ret_id):
        """given ep_obj_id, returns agent-obj, obj-goal pointgoals"""
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        agent_rotation = agent_state.rotation

        try:
            sim_obj_id = task.ep_obj_id_to_sim_obj_id[(ep_obj_id, "o")]
            object_position = task.get_translation(sim_obj_id)
            rec = episode.get_receptacle(ep_obj_id)
            ep_rec_id = rec.object_id
            sim_rec_id = task.ep_obj_id_to_sim_obj_id[(ep_rec_id, "r")]
            rec_position = task.get_translation(sim_rec_id)
        except:
            import pdb
            pdb.set_trace()


        object_position[1] = rec_position[1]
        dist = geodesic_distance(task._simple_pathfinder, object_position, rec_position)

        agent_rec_pointgoal = self._compute_pointgoal(agent_position, agent_rotation, rec_position)
        agent_obj_pointgoal = self._compute_pointgoal(agent_position, agent_rotation, object_position)

        if ret_id == "r":
            return dist, np.concatenate([[ep_rec_id], agent_obj_pointgoal, agent_rec_pointgoal], axis=0)
        else:
            return dist, np.concatenate([[ep_obj_id], agent_obj_pointgoal, agent_rec_pointgoal], axis=0)


    def _find_next_pointgoal(self, task, episode):
        gripped_sim_obj_id = self._sim.gripped_object_id

        # holding an object
        if gripped_sim_obj_id != -1:
            gripped_ep_obj_id = task.sim_obj_id_to_ep_obj_id[gripped_sim_obj_id][0]
            dist, pointgoal = self.get_pointgoal_for_object(
                episode, task, gripped_ep_obj_id, "r"
            )
            return pointgoal

        # not holding an object
        for i, ep_obj_id in enumerate(self.pickup_order):
            dist, pointgoal = self.get_pointgoal_for_object(
                episode, task, ep_obj_id, "o"
            )
            # does this object need to be rearranged?
            if dist > task._config.SUCCESS.SUCCESS_DISTANCE:
                return pointgoal

        # no need to do anything, episode finished.
        return (pointgoal * 0)

    def _get_observation(
            self,
            observations: Dict[str, Observations],
            episode: CosRearrangementEpisode,
            task: CosRearrangementTask,
            **kwargs
    ):
        # if the episode was reset; reset the sensor!
        if task.did_episode_reset():
            self._reset_sensor(observations, episode, task, **kwargs)

        pointgoal = self._find_next_pointgoal(task, episode)
        return pointgoal

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)


@registry.register_sensor(name="L2DistObjectSensor")
class L2DistObjectSensor(OracleNextObjectSensor):
    def __init__(self, sim: CosRearrangementSim, config: Config, **kwargs):
        super().__init__(sim=sim, config=config)
        self.uuid = "l2dist_object"

    def _reset_sensor(
            self,
            observations: Dict[str, Observations],
            episode: CosRearrangementEpisode,
            task: CosRearrangementTask,
            **kwargs
    ):
        self.current_obj_id = -1
        self.prev_episode_id = episode.episode_id
        self.pickup_order = episode.l2_pickup_order
        if self.pickup_order is None or len(self.pickup_order) == 0:
            # FIXME: Pickup order starts from 1 :(
            self.pickup_order = np.arange(1, len(episode.receptacles) + 1)

        # print("L2DistObjectSensor reset!", self.pickup_order)


@registry.register_sensor(name="RandomObjectSensor")
class RandomObjectSensor(OracleNextObjectSensor):
    def __init__(self, sim: CosRearrangementSim, config: Config, **kwargs):
        super().__init__(sim=sim, config=config)
        self.uuid = "random_object"

    def _reset_sensor(
            self,
            observations: Dict[str, Observations],
            episode: CosRearrangementEpisode,
            task: CosRearrangementTask,
            **kwargs
    ):
        self.current_obj_id = -1
        self.prev_episode_id = episode.episode_id
        # FIXME: Pickup order starts from 1 :(
        self.pickup_order = np.arange(1, len(episode.receptacles) + 1)
        np.random.shuffle(self.pickup_order)


@registry.register_sensor(name="ClosestObjectSensor")
class ClosestObjectSensor(OracleNextObjectSensor):
    def __init__(self, sim: CosRearrangementSim, config: Config, **kwargs):
        super().__init__(sim=sim, config=config)
        self.uuid = "closest_object"

    def _build_greedy_plan(self, task, episode):
        min_dist = 10000
        min_obj_id = -1

        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        object_positions = [x.position for x in episode.objects]
        goal_positions = [x.position for x in episode.receptacles]

        # dist_mat = np.zeros((2 * len(object_positions) + 1, 2 * len(goal_positions) + 1))
        # print(f"Shape: {dist_mat.shape}")

        aor_len = 1 + len(object_positions) + len(goal_positions)
        dist_mat = np.zeros((aor_len, aor_len))

        for i, p1 in enumerate([agent_position] + object_positions + goal_positions):
            for j, p2 in enumerate([agent_position] + object_positions + goal_positions):
                dist_mat[i, j] = self._euclidean_distance(p1, p2)
        # print(dist_mat)

        prev_idx = 0
        pickup_order = []
        while len(pickup_order) != len(object_positions):
            min_dist = 10000
            min_idx = -1
            for i in range(1, len(object_positions) + 1):
                l2dist = dist_mat[prev_idx, i]
                if l2dist < min_dist and (i not in pickup_order):
                    min_dist = l2dist
                    min_idx = i
            pickup_order.append(min_idx)
            # prev_idx = min_idx + len(object_positions)
            # receptacle-index
            prev_idx = episode.gd_recs_ind[min_idx-1]


            # print(pickup_order, min_idx, min_idx + len(object_positions))

        return pickup_order

    def _reset_sensor(
            self,
            observations: Dict[str, Observations],
            episode: CosRearrangementEpisode,
            task: CosRearrangementTask,
            **kwargs
    ):
        self.current_obj_id = -1
        self.prev_episode_id = episode.episode_id

        self.pickup_order = self._build_greedy_plan(task, episode)
        # print("Greedy Plan: {}".format(self.pickup_order))


@registry.register_sensor(name="MapNextObjectSensor")
class MapNextObjectSensor(OracleNextObjectSensor):
    def __init__(self, sim: CosRearrangementSim, config: Config, **kwargs):
        super().__init__(sim=sim, config=config)
        self.uuid = "map_object"
        self.visible_mask = None
        self.top_down_map = None
        self.steps = 0

    def _reset_sensor(
            self,
            observations: Dict[str, Observations],
            episode: CosRearrangementEpisode,
            task: CosRearrangementTask,
            **kwargs
    ):
        self.current_obj_id = -1
        self.prev_episode_id = episode.episode_id
        self.pickup_order = list(range(len(episode.objects)))

        self.prev_route_idxs = None
        self.prev_pickup_idxs = None

        self.steps = 0

        self.top_down_map = maps.get_topdown_map(
            task._simple_pathfinder,
            self._sim.get_agent(0).state.position[1],
            256
        )
        self.visible_mask = np.zeros_like(self.top_down_map)
        self.obstacles_mask = np.ones_like(self.visible_mask)

        goal_positions = [obj.position for obj in episode.receptacles]
        self.grid_goal_positions = []

        # draw the objectgoal positions.
        for i, goal_pos in enumerate(goal_positions):
            tdm_pos = maps.to_grid(
                goal_pos[2],
                goal_pos[0],
                self.top_down_map.shape[0:2],
                sim=self._sim,
            )

            self.grid_goal_positions.append(tdm_pos)

        self._agent_episode_distance = 0
        self._best_route_cost = 10000

    def reveal_map(self, task, episode):
        agent_state = self._sim.get_agent(0).get_state()
        agent_position = agent_state.position
        ref_rotation = agent_state.rotation

        agent_rotation = get_polar_angle(ref_rotation)

        a_y, a_x = maps.to_grid(
            agent_position[2],
            agent_position[0],
            self.top_down_map.shape[0:2],
            sim=self._sim,
        )

        self.visible_mask = fog_of_war.reveal_fog_of_war(
            self.top_down_map,
            self.visible_mask,
            np.array([a_y, a_x]),
            agent_rotation,
            fov=90,
            max_line_len=499
        )

        self.obstacles_mask = np.copy(self.visible_mask)
        self.obstacles_mask[self.visible_mask <= 1] = 1
        self.obstacles_mask[self.visible_mask > 1] = 0

        task.misc_dict.update({
            'obstacles_mask': self.obstacles_mask,
            'partial_top_down_map': self.top_down_map,
        })

    def compute_plan(self, task, episode):
        self.grid_object_positions = []
        object_positions = [obj.position for obj in episode.objects]

        unplaced_grid_obj_pos = []
        unplaced_grid_goal_pos = []
        unplaced_obj_idxs = []

        # Find those object which have not been placed!
        for i, obj_pos in enumerate(object_positions):
            tdm_pos = maps.to_grid(
                obj_pos[2],
                obj_pos[0],
                self.top_down_map.shape[0:2],
                sim=self._sim,
            )
            self.grid_object_positions.append(tdm_pos)

            dist, pointgoal = self.get_pointgoal_wrt_agent_pos(
                episode, task, i
            )
            if dist > task._config.SUCCESS.SUCCESS_DISTANCE:
                unplaced_grid_obj_pos.append(tdm_pos)
                unplaced_grid_goal_pos.append(self.grid_goal_positions[i])
                unplaced_obj_idxs.append(i)

        agent_position = self._sim.get_agent_state().position
        a_y, a_x = maps.to_grid(
            agent_position[2],
            agent_position[0],
            self.top_down_map.shape[0:2],
            sim=self._sim,
        )

        dmat = compute_distance_mat_using_fmm(self.obstacles_mask, [a_y, a_x], unplaced_grid_obj_pos,
                                              unplaced_grid_goal_pos)

        max_value = np.max(dmat[dmat != np.inf])
        dmat[dmat == np.inf] = max_value + 100
        # dmat = dmat / (max_value + 100)
        self.dmat = dmat
        route_idxs, pickup_idxs = find_shortest_path_for_multiple_objects(dmat)

        if route_idxs is None:
            print("Could not come up with a plan.")
            if len(self.pickup_order) == 0:
                self.pickup_order = [i + 1 for i in unplaced_obj_idxs]
            return

        route_cost = 0
        for i in range(len(route_idxs) - 1):
            route_cost += dmat[route_idxs[i], route_idxs[i + 1]]

        if self.prev_pickup_idxs != pickup_idxs and self.prev_route_idxs and len(self.prev_pickup_idxs) == len(
                pickup_idxs):
            prev_route_cost = 0
            for i in range(len(self.prev_route_idxs) - 1):
                prev_route_cost += dmat[self.prev_route_idxs[i], self.prev_route_idxs[i + 1]]

            # print(f'Prev Cost: {prev_route_cost}, Current Cost: {route_cost}')
            # print(f'Previous Pickup Order: {self.prev_pickup_idxs}; Updated Order: {pickup_idxs}')
        elif not self.prev_pickup_idxs or (len(self.prev_pickup_idxs) != len(pickup_idxs)):
            prev_route_cost = route_cost + 10000
        else:
            prev_route_cost = route_cost

        if (prev_route_cost - route_cost) >= 9.0:
            self.prev_pickup_idxs = pickup_idxs
            self.prev_route_idxs = route_idxs

            # self.pickup_order[5 - len(self.prev_pickup_idxs): ] = [unplaced_obj_idxs[i-1]+1 for i in self.prev_pickup_idxs]
            self.pickup_order = [unplaced_obj_idxs[i - 1] + 1 for i in self.prev_pickup_idxs]
            # print(f"Updated Pickup Order: {self.pickup_order}")

    def compute_agent_distances(self, agent_pos):
        dmap = compute_distance_using_fmm(self.top_down_map, agent_pos)
        return find_dist_from_map(dmap, self._previous_position)

    def _get_observation(
            self,
            observations: Dict[str, Observations],
            episode: CosRearrangementEpisode,
            task: CosRearrangementTask,
            **kwargs
    ):

        # if the episode was reset; reset the sensor!
        if task.did_episode_reset():
            self._reset_sensor(observations, episode, task, **kwargs)

        agent_position = self._sim.get_agent_state().position
        a_y, a_x = maps.to_grid(
            agent_position[2],
            agent_position[0],
            self.top_down_map.shape[0:2],
            sim=self._sim,
        )

        if task.did_episode_reset():
            self._previous_position = [a_y, a_x]

        self.reveal_map(task, episode)

        # additional_distance = self.compute_agent_distances(
        #     [a_y, a_x]
        # )
        # print(additional_distance, self._agent_episode_distance)
        # self._agent_episode_distance += additional_distance
        # self._previous_position = [a_y, a_x]

        if self.steps % 5 == 0:
            self.compute_plan(task, episode)

        self.steps += 1

        pointgoal = self._find_next_pointgoal(task, episode)
        return pointgoal


@registry.register_sensor
class VisibleObjectMaskSensor(Sensor):
    def __init__(
            self, *args: Any, sim: CosRearrangementSim, config: Config, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(5,),
            dtype=np.float32,
        )

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "visible_object_mask"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def get_observation(
            self,
            observations: Dict[str, Observations],
            episode: Episode,
            *args: Any,
            **kwargs: Any
    ):
        habitat_sim = self._sim._sim
        mask = np.ones((5), dtype=np.bool)
        for i, obj_id in enumerate(habitat_sim.get_existing_object_ids()[1:]):
            mask[i] = 0
        return mask


# Set defaults in the config system.
# -----------------------------------------------------------------------------
# # REARRANGEMENT TASK NEXT OBJECT POS AND GOAL SENSOR
# -----------------------------------------------------------------------------
_C.TASK.ORACLE_NEXT_OBJECT_SENSOR = CN()
_C.TASK.ORACLE_NEXT_OBJECT_SENSOR.TYPE = "OracleNextObjectSensor"
_C.TASK.ORACLE_NEXT_OBJECT_SENSOR.GOAL_FORMAT = "POLAR"
_C.TASK.ORACLE_NEXT_OBJECT_SENSOR.DIMENSIONALITY = 2
# -----------------------------------------------------------------------------
# # REARRANGEMENT TASK L2DIST OBJECT POS AND GOAL SENSOR
# -----------------------------------------------------------------------------
_C.TASK.L2DIST_OBJECT_SENSOR = CN()
_C.TASK.L2DIST_OBJECT_SENSOR.TYPE = "L2DistObjectSensor"
_C.TASK.L2DIST_OBJECT_SENSOR.GOAL_FORMAT = "POLAR"
_C.TASK.L2DIST_OBJECT_SENSOR.DIMENSIONALITY = 2
# -----------------------------------------------------------------------------
# # REARRANGEMENT TASK CLOSEST OBJECT POS AND GOAL SENSOR
# -----------------------------------------------------------------------------
_C.TASK.CLOSEST_OBJECT_SENSOR = CN()
_C.TASK.CLOSEST_OBJECT_SENSOR.TYPE = "ClosestObjectSensor"
_C.TASK.CLOSEST_OBJECT_SENSOR.GOAL_FORMAT = "POLAR"
_C.TASK.CLOSEST_OBJECT_SENSOR.DIMENSIONALITY = 2
# -----------------------------------------------------------------------------
# # REARRANGEMENT TASK RANDOM OBJECT POS AND GOAL SENSOR
# -----------------------------------------------------------------------------
_C.TASK.RANDOM_OBJECT_SENSOR = CN()
_C.TASK.RANDOM_OBJECT_SENSOR.TYPE = "RandomObjectSensor"
_C.TASK.RANDOM_OBJECT_SENSOR.GOAL_FORMAT = "POLAR"
_C.TASK.RANDOM_OBJECT_SENSOR.DIMENSIONALITY = 2
# -----------------------------------------------------------------------------
# # REARRANGEMENT TASK MAP OBJECT SENSOR
# -----------------------------------------------------------------------------
_C.TASK.MAP_NEXT_OBJECT_SENSOR = CN()
_C.TASK.MAP_NEXT_OBJECT_SENSOR.TYPE = "MapNextObjectSensor"
_C.TASK.MAP_NEXT_OBJECT_SENSOR.GOAL_FORMAT = "POLAR"
_C.TASK.MAP_NEXT_OBJECT_SENSOR.DIMENSIONALITY = 2
# -----------------------------------------------------------------------------
# # REARRANGEMENT TASK SENSOR
# -----------------------------------------------------------------------------
_C.TASK.COS_EOR_SENSOR = CN()
_C.TASK.COS_EOR_SENSOR.TYPE = "CosEorSensor"
# -----------------------------------------------------------------------------
# # REARRANGEMENT TASK ALL OBJECT POSITIONS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.OBJECT_POSITION = CN()
_C.TASK.OBJECT_POSITION.TYPE = "ObjectPosition"
_C.TASK.OBJECT_POSITION.GOAL_FORMAT = "POLAR"
_C.TASK.OBJECT_POSITION.DIMENSIONALITY = 2
# -----------------------------------------------------------------------------
# # REARRANGEMENT TASK ALL OBJECT GOALS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.OBJECT_GOAL = CN()
_C.TASK.OBJECT_GOAL.TYPE = "ObjectGoal"
_C.TASK.OBJECT_GOAL.GOAL_FORMAT = "POLAR"
_C.TASK.OBJECT_GOAL.DIMENSIONALITY = 2
# -----------------------------------------------------------------------------
# # REARRANGEMENT TASK ALL OBJECT POSITIONS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.ALL_OBJECT_POSITIONS = CN()
_C.TASK.ALL_OBJECT_POSITIONS.TYPE = "AllObjectPositions"
_C.TASK.ALL_OBJECT_POSITIONS.GOAL_FORMAT = "POLAR"
_C.TASK.ALL_OBJECT_POSITIONS.DIMENSIONALITY = 2
# -----------------------------------------------------------------------------
# # REARRANGEMENT TASK ALL OBJECT GOALS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.ALL_OBJECT_GOALS = CN()
_C.TASK.ALL_OBJECT_GOALS.TYPE = "AllObjectGoals"
_C.TASK.ALL_OBJECT_GOALS.GOAL_FORMAT = "POLAR"
_C.TASK.ALL_OBJECT_GOALS.DIMENSIONALITY = 2


_C.TASK.NEXT_OBJECT_SENSOR_UUID = "oracle_next_object"