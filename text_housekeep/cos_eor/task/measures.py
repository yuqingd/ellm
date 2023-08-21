from typing import Any

import numpy as np
from text_housekeep.habitat_lab.habitat.config.default import Config, _C, CN
from text_housekeep.habitat_lab.habitat.core.embodied_task import EmbodiedTask
from text_housekeep.habitat_lab.habitat.core.embodied_task import Measure
from text_housekeep.habitat_lab.habitat.core.registry import registry
from text_housekeep.habitat_lab.habitat.core.simulator import Simulator
from text_housekeep.cos_eor.task.utils import start_env_episode_distance
from text_housekeep.cos_eor.utils.geometry import geodesic_distance
from text_housekeep.habitat_lab.habitat.tasks.nav.nav import TopDownMap
from text_housekeep.habitat_lab.habitat.utils.visualizations import maps

"""
Todo: Remove code duplication from update_metric functions 
"""

@registry.register_measure
class ObjectToGoalDistance(Measure):
    """The measure calculates distance of object towards the goal."""

    cls_uuid: str = "object_to_goal_distance"

    def __init__(
        self,
        sim: Simulator,
        config: Config,
        task: EmbodiedTask,
        *args: Any,
        **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._task = task
        self._elapsed_steps = 0
        self._cache = {}
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return ObjectToGoalDistance.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._elapsed_steps = 0
        self._cache = {}
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, episode, *args: Any, **kwargs: Any):
        """todo: update for multiple goals."""
        self._elapsed_steps += 1
        self._metric = episode.get_object_goal_distance(self._task, self._cache)



@registry.register_measure
class AgentToObjectDistance(Measure):
    """The measure calculates the distance of objects from the agent"""

    cls_uuid: str = "agent_to_object_distance"

    def __init__(
        self,
        sim: Simulator,
        config: Config,
        task: EmbodiedTask,
        *args: Any,
        **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._task = task
        self._elapsed_steps = 0

        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return AgentToObjectDistance.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._elapsed_steps = 0
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, episode, *args: Any, **kwargs: Any):
        self._elapsed_steps += 1
        self._metric = episode.get_agent_object_distance(self._task)


@registry.register_measure
class RearrangementSPL(Measure):
    r"""SPL (Success weighted by Path Length) for the the rearrangement task
    """

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._episode_view_points = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "rearrangement_spl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = start_env_episode_distance(task, episode, episode.pickup_order)

        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        # ep_success = task.measurements.measures[Success.cls_uuid].get_metric()

        current_position = self._sim.get_agent_state().position
        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )


@registry.register_measure
class EpisodeDistance(Measure):
    r"""SPL (Success weighted by Path Length) for the the rearrangement task
    """

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._episode_view_points = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "episode_distance"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0

        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        # ep_success = task.measurements.measures[Success.cls_uuid].get_metric()

        current_position = self._sim.get_agent_state().position
        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position
        self._metric = self._agent_episode_distance


@registry.register_measure
class RearrangementTopDownMap(TopDownMap):
    r"""Top Down Map measure"""

    def __init__(
            self, sim: Simulator, config: Config, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        super().__init__(sim, config, task, *args, **kwargs)
        self._task = task

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "top_down_map"

    def get_original_map(self):
        agent_position = self._sim.get_agent(0).get_state().position
        top_down_map = maps.get_topdown_map(
            self._sim.pathfinder,
            agent_position[1],
            1024
        )

        import pdb
        pdb.set_trace()

        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = np.zeros_like(top_down_map)
        else:
            self._fog_of_war_mask = None

        return top_down_map

    def draw_object_info(self, episode):
        object_positions = [obj.position for obj in episode.objects]
        goal_positions = [obj.position for obj in episode.receptacles]

        self.grid_object_positions = []
        self.grid_goal_positions = []

        for i, obj_pos in enumerate(object_positions):
            tdm_pos = maps.to_grid(
                obj_pos[2],
                obj_pos[0],
                self._top_down_map.shape[0:2],
                sim=self._sim,
            )
            self.grid_object_positions.append(tdm_pos)

        for i, goal_pos in enumerate(goal_positions):
            tdm_pos = maps.to_grid(
                goal_pos[2],
                goal_pos[0],
                self._top_down_map.shape[0:2],
                sim=self._sim,
            )
            self.grid_goal_positions.append(tdm_pos)

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._step_count = 0
        self._metric = None
        self._top_down_map = self.get_original_map()
        agent_position = self._sim.get_agent_state().position
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            self._top_down_map.shape[0:2],
            sim=self._sim,
        )
        self._previous_xy_location = (a_y, a_x)

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        self.draw_object_info(episode)

        # self._draw_shortest_path(episode, agent_position)

        if self._config.DRAW_SOURCE:
            self._draw_point(
                episode.start_position, maps.MAP_SOURCE_POINT_INDICATOR
            )

    def update_object_positions(self, episode):
        current_obj_positions = [0] * len(episode.objects)
        sim_object_ids = episode.get_objects_ids(self._task.obj_key_to_sim_obj_id)

        for sim_obj_id in sim_object_ids:
            if sim_obj_id == self._task.agent_object_id or sim_obj_id in registry.mapping["ignore_object_ids"]:
                continue
            obj_id = self._task.sim_obj_id_to_ep_obj_id[sim_obj_id]
            obj_pos = np.array(
                self._task.get_translation(sim_obj_id)
            ).tolist()

            tdm_pos = maps.to_grid(
                obj_pos[2],
                obj_pos[0],
                self._top_down_map.shape[0:2],
                sim=self._sim,
            )
            current_obj_positions[obj_id] = tdm_pos
        return current_obj_positions

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        self._step_count += 1
        house_map, map_agent_x, map_agent_y = self.update_map(
            self._sim.get_agent_state().position
        )

        current_obj_positions = self.update_object_positions(episode)

        self._metric = {
            "map": house_map,
            "fog_of_war_mask": self._fog_of_war_mask,
            "agent_map_coord": (map_agent_x, map_agent_y),
            "agent_angle": self.get_polar_angle(),
            "object_positions": self.grid_object_positions,
            "goal_positions": self.grid_goal_positions,
            "current_positions": current_obj_positions
        }

# -----------------------------------------------------------------------------
# # OBJECT_DISTANCE_TO_GOAL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.OBJECT_TO_GOAL_DISTANCE = CN()
_C.TASK.OBJECT_TO_GOAL_DISTANCE.TYPE = "ObjectToGoalDistance"
# -----------------------------------------------------------------------------
# # OBJECT_DISTANCE_FROM_AGENT MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.AGENT_TO_OBJECT_DISTANCE = CN()
_C.TASK.AGENT_TO_OBJECT_DISTANCE.TYPE = "AgentToObjectDistance"
# -----------------------------------------------------------------------------
# REARRANGEMENT SPL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.REARRANGEMENT_SPL = CN()
_C.TASK.REARRANGEMENT_SPL.TYPE = "RearrangementSPL"
# -----------------------------------------------------------------------------
# EPISODE DISTANCE MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.EPISODE_DISTANCE = CN()
_C.TASK.EPISODE_DISTANCE.TYPE = "EpisodeDistance"
