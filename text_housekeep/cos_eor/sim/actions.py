from typing import List, Any

import habitat_sim
import numpy as np
from tqdm import tqdm

from text_housekeep.habitat_lab.habitat.config.default import _C, CN
from text_housekeep.habitat_lab.habitat.core.embodied_task import SimulatorTaskAction
from text_housekeep.habitat_lab.habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)
from habitat_sim.agent.controls.controls import ActuationSpec
import attr
import magnum as mn
from text_housekeep.habitat_lab.habitat.core.registry import registry

from text_housekeep.cos_eor.utils.geometry import geodesic_distance, add_object_on_receptacle
from text_housekeep.cos_eor.task.task import CosRearrangementTask
from text_housekeep.cos_eor.dataset.dataset import CosRearrangementEpisode
from habitat_sim.physics import MotionType


def raycast(sim, crosshair_pos=[128, 128], sensor_name="rgb", max_distance=1.0, return_point=False):
    r"""Cast a ray in the direction of crosshair and check if it collides
    with another object within a certain distance threshold
    :param sim: Simulator object
    :param sensor_name: name of the visual sensor to be used for raycasting
    :param crosshair_pos: 2D coordiante in the viewport towards which the
        ray will be cast
    :param max_distance: distance threshold beyond which objects won't
        be considered
    """

    render_camera = sim._sensors[sensor_name]._sensor_object.render_camera
    center_ray = render_camera.unproject(mn.Vector2i(crosshair_pos))
    raycast_results = sim.cast_ray(center_ray, max_distance=max_distance)

    closest_object = -1
    closest_dist = 1000.0
    
    non_art_ids = sim.get_existing_object_ids(sim.get_rigid_object_manager())
    art_ids = sim.get_existing_articulated_object_ids()
    ids = non_art_ids + art_ids
    if raycast_results.has_hits():
        for hit in raycast_results.hits:
            # just return the first hit-point
            if return_point:
                return hit.point

            # don't count non-existent hits
            if (hit.ray_distance < closest_dist) \
                    and (hit.object_id in ids) \
                    and (hit.object_id not in registry.mapping["ignore_object_ids_ray"]):
                closest_dist = hit.ray_distance
                closest_object = hit.object_id

    return closest_object


@attr.s(auto_attribs=True, slots=True)
class GrabReleaseActuationSpec(ActuationSpec):
    visual_sensor_name: str = "rgb"
    crosshair_pos: List[int] = [128, 128]
    amount: float = 2.0


@habitat_sim.registry.register_move_fn(body_action=True)
class GrabOrReleaseObjectUnderCrosshair(habitat_sim.SceneNodeControl):
    def __call__(
        self,
        scene_node: habitat_sim.SceneNode,
        actuation_spec: GrabReleaseActuationSpec,
    ):
        pass



@registry.register_action_space_configuration(name="CosRearrangementActions-v0")
class RearrangementSimV0ActionSpaceConfiguration(
    HabitatSimV1ActionSpaceConfiguration
):
    def __init__(self, config):
        super().__init__(config)
        if not HabitatSimActions.has_action("GRAB_RELEASE"):
            HabitatSimActions.extend_action_space("GRAB_RELEASE")

    def get(self):
        config = super().get()
        new_config = {
            HabitatSimActions.GRAB_RELEASE: habitat_sim.ActionSpec(
                "grab_or_release_object_under_crosshair",
                GrabReleaseActuationSpec(
                    visual_sensor_name=self.config.VISUAL_SENSOR,
                    crosshair_pos=self.config.CROSSHAIR_POS,
                    amount=self.config.GRAB_DISTANCE,
                ),
            )
        }
        config.update(new_config)

        return config


@registry.register_task_action
class GrabOrReleaseActionIdBased(SimulatorTaskAction):
    def run_checks(self, task, kwargs, gripped_object_id, distance_threshold, episode):
        curr_observations = self._sim.get_sensor_observations()
        avail_iids = np.unique(curr_observations["semantic"]).tolist()
        agent_position = self._sim.get_agent_state().position
        fail_reason = "none"
        fail_action = False            

        if "iid" not in kwargs or kwargs["iid"] < 0:
            distance = -1
            distance_threshold = -1
            fail_action = True
            fail_reason = "No iid found!"
        elif kwargs["iid"] != -1:
            # ensure that the receptacle or object is within interact range
            try:
                sim_obj_id = task.iid_to_sim_obj_id[kwargs["iid"]]
            except:
                import pdb; pdb.set_trace()
            sim_obj_type = task.sim_obj_id_to_type[sim_obj_id]
            if sim_obj_type == "obj":
                obj_key = task.sim_obj_id_to_obj_key[sim_obj_id]
                rec_key = episode.get_rec(obj_key, episode.state_matrix)
                if rec_key == 'agent' and gripped_object_id == sim_obj_id:
                    distance = -1
                    distance_threshold = -1
                    fail_action = True
                    fail_reason = "Trying to pick up held object!"
                else:
                    rec_id = task.obj_key_to_sim_obj_id[rec_key]
                    rec_iid = task.sim_obj_id_to_iid[rec_id]
                    obj_distance = self._sim.get_or_dist(sim_obj_id, "l2")
            else:
                obj_distance = 1000
                rec_id = sim_obj_id
                rec_iid = kwargs["iid"]

            if not fail_action:
                rec_distance = self._sim.get_or_dist(rec_id, "l2")
                distance = min(rec_distance, obj_distance)

                # fail if it's too far
                fail_action = distance > distance_threshold
                if fail_action:
                    fail_reason = "Far from object"

                # fail if it's a receptacle being picked
                obj_type = task.sim_obj_id_to_type[sim_obj_id]
                if gripped_object_id == -1 and obj_type == "rec" and not fail_action:
                    fail_action = True
                    fail_reason = "pick on receptacle"

                # fail if placing on object
                if gripped_object_id != -1 and obj_type == "obj" and not fail_action:
                    fail_action = True
                    fail_reason = "place on object"

                # fail if semantic-id is not visible
                if kwargs["iid"] not in avail_iids and rec_iid not in avail_iids:
                    fail_action = True
                    fail_reason = f"iid not-visible/incorrect; chosen: {kwargs['iid']}, avail: {avail_iids}"
        else:
            raise AssertionError
        return distance, distance_threshold, fail_action, fail_reason, curr_observations

    def step(self, task: CosRearrangementTask, *args: Any, **kwargs: Any):
        r"""This method is called from ``Env`` on each ``step``."""
        assert "episode" in kwargs
        episode: CosRearrangementEpisode = kwargs["episode"]
        gripped_object_id = self._sim.gripped_object_id
        grab_type = task._config.ACTIONS.GRAB_RELEASE.GRAB_TYPE
        distance_threshold = task._config.ACTIONS.GRAB_RELEASE.GRAB_DISTANCE

        # check whether to use raycast or iid sent by the agent
        if grab_type == "crosshair":
            crosshair_pos = task._config.ACTIONS.GRAB_RELEASE.CROSSHAIR_POS
            obj_id = raycast(
                self._sim,
                crosshair_pos=crosshair_pos,
                max_distance=distance_threshold,
            )
            kwargs["iid"] = task.sim_obj_id_to_iid[obj_id] if obj_id >= 0 else obj_id

        distance, distance_threshold, fail_action, fail_reason, curr_observations\
            = self.run_checks(task, kwargs, gripped_object_id, distance_threshold, episode)

        if not fail_action:
            # If already holding an object
            if gripped_object_id != -1:
                # drop on floor -- unused
                if kwargs["iid"] == -1:
                    raise AssertionError
                else:
                    rec_id = task.iid_to_sim_obj_id[kwargs["iid"]]
                    success = add_object_on_receptacle(gripped_object_id, rec_id, self._sim, task.rec_packers)
                    if success:
                        # update episode-state
                        episode.update_mapping(gripped_object_id, "place", task, rec_id)
                        gripped_object_id = -1
                    else:
                        fail_action = True
                        fail_reason = "Not enough space on receptacle"

            # if not holding an object, then try to grab
            else:
                gripped_object_id = task.iid_to_sim_obj_id[kwargs["iid"]]
                try:
                    if gripped_object_id != -1:
                        obj_key = task.sim_obj_id_to_obj_key[gripped_object_id]
                        rec_key = episode.get_rec(obj_key, episode.state_matrix)
                        rec_id = task.obj_key_to_sim_obj_id[rec_key]
                        remove_result = task.rec_packers[rec_id].remove(gripped_object_id)
                        assert remove_result
                        # update episode-state
                        episode.update_mapping(gripped_object_id, "pick", task)
                    else:
                        import pdb
                        pdb.set_trace()
                except:
                    import pdb
                    pdb.set_trace()

            # remove the object from the scene
            if not fail_action:
                self._sim._sync_agent()
                self._sim._sync_gripped_object(gripped_object_id, invisible=True)

        # if fail_action:
        #     pass
        #     print(f"Pick/place action failed because {fail_reason} \t "
        #           f"distance: {round(distance, 2)} and threshold: {round(distance_threshold, 2)}")
        # else:
        #     print(f"Pick/place success")

        # obtain observations
        self._sim._prev_sim_obs.update(curr_observations)
        self._sim._prev_sim_obs["gripped_object_id"] = gripped_object_id

        observations = self._sim._sensor_suite.get_observations(
            self._sim._prev_sim_obs
        )
        observations['fail_action'] = fail_action
        return observations

# -----------------------------------------------------------------------------
# GRAB OR RELEASE ACTION
# -----------------------------------------------------------------------------
_C.TASK.ACTIONS.GRAB_RELEASE = CN()
_C.TASK.ACTIONS.GRAB_RELEASE.TYPE = "GrabOrReleaseActionIdBased"
_C.TASK.OBJECT_ANNOTATIONS = "cos_eor/scripts/dump/scale_rots_all.npy"
_C.TASK.ACTIONS.GRAB_RELEASE.GRAB_DISTANCE = 1.0
_C.TASK.ACTIONS.GRAB_RELEASE.GRAB_TYPE = "id_based"
# _C.TASK.ACTIONS.GRAB_RELEASE.GRAB_TYPE = "crosshair"
# -----------------------------------------------------------------------------
