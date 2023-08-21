from typing import Any

import copy
import math

import numpy as np
from gym import spaces
from text_housekeep.habitat_lab.habitat.config.default import _C, CN, Config
from text_housekeep.habitat_lab.habitat.core.registry import registry
from text_housekeep.habitat_lab.habitat.core.simulator import Simulator, Sensor, SensorTypes
from text_housekeep.cos_eor.explore.sim import ExploreSim

from text_housekeep.cos_eor.sim.sim import CosRearrangementSim
from text_housekeep.cos_eor.utils.geometry import compute_heading_from_quaternion

RGBSENSOR_DIMENSION = 3

def check_sim_obs(obs: np.ndarray, sensor: Sensor) -> None:
    assert obs is not None, (
        "Observation corresponding to {} not present in "
        "simulator's observations".format(sensor.uuid)
    )

@registry.register_sensor
class FineOccSensor(Sensor):
    def __init__(self, sim: CosRearrangementSim, config, **kwargs: Any):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "fine_occupancy"

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.HEIGHT, self.config.WIDTH, RGBSENSOR_DIMENSION),
            dtype=np.uint8,
        )

    def get_observation(self, observations, **kwargs: Any):
        obs = self._sim.fine_occupancy_map
        check_sim_obs(obs, self)

        # remove alpha channel
        obs = obs[:, :, :RGBSENSOR_DIMENSION]
        return obs


@registry.register_sensor
class CoarseOccSensor(Sensor):
    def __init__(self, sim: CosRearrangementSim, config, **kwargs: Any):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "coarse_occupancy"

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.HEIGHT, self.config.WIDTH, RGBSENSOR_DIMENSION),
            dtype=np.uint8,
        )

    def get_observation(self, observations, **kwargs: Any):
        obs = self._sim.coarse_occupancy_map
        check_sim_obs(obs, self)

        # remove alpha channel
        obs = obs[:, :, :RGBSENSOR_DIMENSION]
        return obs


@registry.register_sensor
class HighResCoarseOccSensor(Sensor):
    def __init__(self, sim: CosRearrangementSim, config, **kwargs: Any):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "highres_coarse_occupancy"

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.HEIGHT, self.config.WIDTH, RGBSENSOR_DIMENSION),
            dtype=np.uint8,
        )

    def get_observation(self, observations, **kwargs: Any):
        obs = self._sim.highres_coarse_occupancy_map
        check_sim_obs(obs, self)

        # remove alpha channel
        obs = obs[:, :, :RGBSENSOR_DIMENSION]
        return obs


@registry.register_sensor
class ProjOccSensor(Sensor):
    def __init__(self, sim: CosRearrangementSim, config, **kwargs: Any):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "proj_occupancy"

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.HEIGHT, self.config.WIDTH, RGBSENSOR_DIMENSION),
            dtype=np.uint8,
        )

    def get_observation(self, observations, **kwargs: Any):
        obs = self._sim.proj_occupancy_map
        check_sim_obs(obs, self)

        # remove alpha channel
        obs = obs[:, :, :RGBSENSOR_DIMENSION]
        return obs


@registry.register_sensor
class DeltaSensor(Sensor):
    r"""Sensor that returns the odometer readings from the previous action.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PoseEstimation sensor.

    Attributes:
        _nRef: number of pose references
    """

    def __init__(self, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

        self.current_episode_id = None
        self.prev_position = None
        self.prev_rotation = None
        self.start_position = None
        self.start_rotation = None
        self._enable_odometer_noise = self._sim._enable_odometer_noise

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "delta"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=-1000000.0, high=1000000.0, shape=(4,), dtype=np.float32,)

    def get_observation(self, observations, episode, **kwargs: Any):
        episode_id = (episode.episode_id, episode.scene_id)
        if self._enable_odometer_noise:
            agent_position = self._sim._estimated_position
            agent_rotation = self._sim._estimated_rotation
        else:
            agent_state = self._sim.get_agent_state()
            agent_position = agent_state.position
            agent_rotation = agent_state.rotation

        if self.current_episode_id != episode_id:
            # A new episode has started
            self.current_episode_id = episode_id
            delta = np.array([0.0, 0.0, 0.0, 0.0])
            self.start_position = copy.deepcopy(agent_position)
            self.start_rotation = copy.deepcopy(agent_rotation)
        else:
            current_position = agent_position
            current_rotation = agent_rotation
            # For the purposes of this sensor, forward is X and rightward is Y.
            # The heading is measured positively from X to Y.
            curr_x, curr_y = -current_position[2], current_position[0]
            curr_heading = compute_heading_from_quaternion(current_rotation)
            prev_x, prev_y = -self.prev_position[2], self.prev_position[0]
            prev_heading = compute_heading_from_quaternion(self.prev_rotation)
            dr = math.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
            dphi = math.atan2(curr_y - prev_y, curr_x - prev_x)
            dhead = curr_heading - prev_heading
            # Convert these to the starting point's coordinate system.
            start_heading = compute_heading_from_quaternion(self.start_rotation)
            dphi = dphi - start_heading
            delta = np.array([dr, dphi, dhead, 0.0])
        self.prev_position = copy.deepcopy(agent_position)
        self.prev_rotation = copy.deepcopy(agent_rotation)

        return delta


@registry.register_sensor
class NewEpisodeSensor(Sensor):
    def __init__(self, sim, config, **kwargs: Any):
        self._sim = sim
        self.current_episode_id = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "new_episode"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Discrete(2)

    def get_observation(self, observations, episode, **kwargs: Any):
        current_episode_id = (episode.episode_id, episode.scene_id)
        is_new_episode = self.current_episode_id != current_episode_id
        self.current_episode_id = current_episode_id
        return int(is_new_episode)


@registry.register_sensor
class CollisionSensor(Sensor):
    """Returns 1 if a collision occured in the previous action, otherwise
    it returns 0.
    """

    def __init__(self, sim, config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "collision"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32,)

    def get_observation(self, observations, episode, **kwargs: Any):
        if self._sim.previous_step_collided:
            return np.array([1.0])
        else:
            return np.array([0.0])


@registry.register_sensor
class SeenAreaSensor(Sensor):
    """Returns the total area seen while exploring the map
    """

    def __init__(self, sim: ExploreSim, config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "seen_area"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=-1.e10, high=1.e10, shape=(1,), dtype=np.float32,)

    def get_observation(self, observations, episode, **kwargs: Any):
        return np.array([self._sim.occupancy_info["seen_area"]])

# -----------------------------------------------------------------------------
# # COMMON SENSOR CONFIG
# -----------------------------------------------------------------------------
SENSOR = CN()
SENSOR.HEIGHT = 84
SENSOR.WIDTH = 84
SENSOR.HFOV = 90  # horizontal field of view in degrees
SENSOR.POSITION = [0, 1.25, 0]
# -----------------------------------------------------------------------------
# # FINE-OCCUPANCY SENSOR
# -----------------------------------------------------------------------------
_C.TASK.FINE_OCC_SENSOR = SENSOR.clone()
_C.TASK.FINE_OCC_SENSOR.TYPE = "FineOccSensor"
# -----------------------------------------------------------------------------
# # COARSE-OCCUPANCY SENSOR
# -----------------------------------------------------------------------------
_C.TASK.COARSE_OCC_SENSOR = SENSOR.clone()
_C.TASK.COARSE_OCC_SENSOR.TYPE = "CoarseOccSensor"
# -----------------------------------------------------------------------------
# # PROJ-OCCUPANCY SENSOR
# -----------------------------------------------------------------------------
_C.TASK.PROJ_OCC_SENSOR = SENSOR.clone()
_C.TASK.PROJ_OCC_SENSOR.TYPE = "ProjOccSensor"
# -----------------------------------------------------------------------------
# # HIGHRES-COARSE-OCCUPANCY SENSOR
# -----------------------------------------------------------------------------
_C.TASK.HIGHRES_COARSE_OCC_SENSOR = SENSOR.clone()
_C.TASK.HIGHRES_COARSE_OCC_SENSOR.TYPE = "HighResCoarseOccSensor"
# -----------------------------------------------------------------------------
# # DELTA SENSOR
# -----------------------------------------------------------------------------
_C.TASK.DELTA_SENSOR = CN()
_C.TASK.DELTA_SENSOR.TYPE = "DeltaSensor"
# -----------------------------------------------------------------------------
# # NEW EPISODE SENSOR
# -----------------------------------------------------------------------------
_C.TASK.NEW_EPISODE_SENSOR = CN()
_C.TASK.NEW_EPISODE_SENSOR.TYPE = "NewEpisodeSensor"
# -----------------------------------------------------------------------------
# # COLLISION SENSOR
# -----------------------------------------------------------------------------
_C.TASK.COLLISION_SENSOR = CN()
_C.TASK.COLLISION_SENSOR.TYPE = "CollisionSensor"
# -----------------------------------------------------------------------------
# # SEEN AREA SENSOR
# -----------------------------------------------------------------------------
_C.TASK.SEEN_AREA_SENSOR = CN()
_C.TASK.SEEN_AREA_SENSOR.TYPE = "SeenAreaSensor"
# -----------------------------------------------------------------------------
