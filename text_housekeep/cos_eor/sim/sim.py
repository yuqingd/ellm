import os
import math
from collections import Counter
from typing import Any, Dict, MutableMapping, Union, cast

import habitat_sim
import numpy as np
import yaml
import magnum as mn

from habitat_sim.nav import NavMeshSettings
from habitat_sim.physics import MotionType
from habitat_sim.utils.common import quat_from_angle_axis, quat_from_coeffs, quat_to_magnum
from text_housekeep.habitat_lab.habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from text_housekeep.habitat_lab.habitat.config.default import CN, Config
from text_housekeep.habitat_lab.habitat.config.default import _C
from text_housekeep.habitat_lab.habitat.core.registry import registry
from text_housekeep.habitat_lab.habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from text_housekeep.habitat_lab.habitat.utils.geometry_utils import (quaternion_from_coeff, quaternion_from_two_vectors)
from text_housekeep.habitat_lab.habitat.core.embodied_task import EmbodiedTask
from habitat_sim.utils import profiling_utils
from habitat_sim.agent import ActionSpec, ActuationSpec
from text_housekeep.habitat_lab.habitat.utils.visualizations import maps
from habitat_sim.geo import Ray

from text_housekeep.cos_eor.sim.actions import raycast
from text_housekeep.cos_eor.utils.geometry import geodesic_distance, euclidean_distance, \
    get_all_nav_points, get_random_nav_point, get_closest_nav_point, closest_point, get_bb, get_corners


@registry.register_simulator(name="CosRearrangementSim-v0")
class CosRearrangementSim(HabitatSim):
    r"""Simulator wrapper over habitat-sim with
    object rearrangement functionalities.
    """

    def __init__(self, config: Config) -> None:
        self.did_reset = False
        super().__init__(config=config)
        self.obj_load_type = self.habitat_config.OBJ_LOAD_TYPE
        self.agent_id = self.habitat_config.DEFAULT_AGENT_ID
        self.agent_config = self._get_agent_config(self.agent_id)
        agent_config = self._get_agent_config(self.agent_id)

        self.screen_center = mn.Vector2i(config.RGB_SENSOR.HEIGHT//2, config.RGB_SENSOR.WIDTH//2)

        self.navmesh_settings = NavMeshSettings()
        self.navmesh_settings.set_defaults()
        self.navmesh_settings.agent_radius = agent_config.RADIUS
        self.navmesh_settings.agent_height = agent_config.HEIGHT
        self.navmesh_settings.agent_max_climb = agent_config.MAX_CLIMB
        self.agent_object_id = -1
        self._prev_sim_obs = {}
        self._cache = {}
        self.same_scene = False

    def reconfigure(self, config: Config) -> None:
        self.same_scene = config.SCENE == self._current_scene
        super().reconfigure(config)

    def _rotate_agent_sensors(self):
        r"""Rotates the sensor to look down at the start of the episode.
        """
        agent = self.get_agent(self.agent_id)

        for _, v in agent._sensors.items():
            action = ActionSpec(
                name="look_down",
                actuation=ActuationSpec(
                    amount=self.habitat_config.INITIAL_LOOK_DOWN_ANGLE
                ),
            )

            agent.controls.action(
                v.object, action.name, action.actuation, apply_filter=False
            )

    def reset(self):
        obs = super().reset()
        self._prev_sim_obs["gripped_object_id"] = -1
        self.did_reset = True
        return obs

    def _sync_agent(self):
        if self.agent_object_id < 0:
            return
        self.set_translation(self._Simulator__last_state[self.agent_object_id].position, self.agent_object_id)
        self.set_rotation(quat_to_magnum(self._Simulator__last_state[self.agent_object_id].rotation), self.agent_object_id)

    def set_rotation(self, rot, obj_id, scene_id=0):
        if self.get_object_type(obj_id) == "non_art":
            super(CosRearrangementSim, self).set_rotation(rot, obj_id)
        else:
            pos = self.get_translation(obj_id)
            trans = mn.Matrix4.from_(rot.to_matrix(), mn.Vector3(*pos))
            self.set_articulated_object_root_state(obj_id, trans)

    def _sync_gripped_object(self, gripped_object_id, invisible=True):
        r"""
        Sync the gripped object with the object associated with the agent.
        """
        if gripped_object_id != -1:
            offset = [0, 0.3, 0] if not invisible else [0, 10, 0]
            agent_pos = self.get_agent_state().position
            object_pos = agent_pos + offset
            self.set_translation(object_pos, gripped_object_id)

    @property
    def gripped_object_id(self):
        return self._prev_sim_obs.get("gripped_object_id", -1)

    def step(self, action: int):
        self._num_total_frames += 1
        if isinstance(action, MutableMapping):
            return_single = False
        else:
            action = cast(Dict[int, Union[str, int]], {self._default_agent_id: action})
            return_single = True

        assert return_single

        collided_dict: Dict[int, bool] = {}
        for agent_id, agent_act in action.items():
            agent = self.get_agent(agent_id)
            collided_dict[agent_id] = agent.act(agent_act)
            self._Simulator__last_state[agent_id] = agent.get_state()

        self._previous_step_time = 0.0
        multi_observations = self.get_sensor_observations(agent_ids=list(action.keys()))

        for agent_id, agent_observation in multi_observations.items():
            agent_observation["collided"] = collided_dict[agent_id]

        if return_single:
            sim_obs = multi_observations[self._default_agent_id]
            self._prev_sim_obs.update(sim_obs)
            obs = self._sensor_suite.get_observations(sim_obs)
            return obs

        return multi_observations

    def debug_frame(self):
        import cv2
        obs = self.get_sensor_observations()
        rgb = cv2.cvtColor(obs["rgb"], cv2.COLOR_RGBA2RGB)
        cv2.imwrite("debug_frame.jpeg", rgb)

    def get_both_existing_object_ids(self):
        non_art_ids = self.get_existing_rigid_object_ids()
        art_ids = self.get_existing_articulated_object_ids()

        return {
            "art": art_ids,
            "non_art": non_art_ids
        }

    def get_both_existing_object_ids_with_positions(self):
        non_art_ids = self.get_existing_rigid_object_ids()
        non_art_pos = [self.get_translation(id) for id in non_art_ids]

        art_ids = self.get_existing_articulated_object_ids()
        art_pos = [np.array(self.get_articulated_object_root_state(id).translation) for id in art_ids]

        return {
            "art": art_ids,
            "art_pos": art_pos,
            "non_art": non_art_ids,
            "non_art_pos": non_art_pos,
        }

    def get_existing_rigid_object_ids(self):
        return self.get_existing_object_ids(self.get_rigid_object_manager())

    def get_existing_articulated_object_ids(self):
        return self.get_existing_object_ids(self.get_articulated_object_manager())

    def get_existing_object_ids(self, obj_manager):
        obj_handles = obj_manager.get_object_handles()
        obj_ids = [obj_manager.get_object_id_by_handle(h) for h in obj_handles]
        return obj_ids

    def get_object_type(self, object_id):
        ids_dict = self.get_both_existing_object_ids()
        if object_id in ids_dict["art"]:
            return "art"
        elif object_id in ids_dict["non_art"]:
            return "non_art"
        else:
            return "not_found"

    def set_object_semantic_id_art(self, semantic_id, object_id):
        nodes = self.get_object_visual_scene_nodes(object_id)
        for node in nodes:
            node.semantic_id = semantic_id

    def get_semantic_id(self, object_id):
        """0 is default for each object"""
        nodes = self.get_object_visual_scene_nodes(object_id)
        semantic_ids = [node.semantic_id for node in nodes]
        assert len(set(semantic_ids)) == 1
        return semantic_ids[0]

    def get_translation(self, object_id, scene_id=0):
        if self.get_object_type(object_id) == "non_art":
            trans = super(CosRearrangementSim, self).get_translation(object_id, scene_id)
        else:
            trans = self.get_articulated_object_root_state(object_id).translation
        return np.array(trans)

    def set_translation(self, translation, object_id, scene_id=0):
        try:
            if type(translation) != mn.Vector3:
                translation = mn.Vector3(*translation)
            if self.get_object_type(object_id) == "non_art":
                return super(CosRearrangementSim, self).set_translation(translation, object_id, scene_id)
            else:
                current_state = self.get_articulated_object_root_state(object_id)
                translated_state = getattr(mn.Matrix4, 'from_')(current_state.rotation(), mn.Vector3(*translation))
                return self.set_articulated_object_root_state(object_id, translated_state)
        except Exception as e:
            print(e)
            import pdb
            pdb.set_trace()

    def get_rotation(self, object_id, scene_id=0):
        if self.get_object_type(object_id) == "non_art":
            return super(CosRearrangementSim, self).get_rotation(object_id, scene_id)
        else:
            obj_state = self.get_articulated_object_root_state(object_id)
            return mn.Quaternion.from_matrix(obj_state.rotation())

    def add_articulated_object_from_urdf(self, urdf_file, fixed_base=False, global_scale=1.0, mass_scale=1.0, force_reload=False):
        raise NotImplementedError

    def add_art_object(self, file, transform, motion_type=0, frozen=False):
        obj_id = self.add_articulated_object_from_urdf(file, frozen)
        self.set_articulated_object_root_state(obj_id, transform)
        self.set_articulated_object_sleep(obj_id, False)
        self.set_articulated_object_motion_type(obj_id, MotionType(motion_type))
        return obj_id

    def remove_all_objects(self):
        object_ids = self.get_both_existing_object_ids()
        for type, obj_ids in object_ids.items():
            for id in obj_ids:
                if id == self.agent_object_id:
                    continue
                if type == "art":
                    self.remove_articulated_object(id)
                else:
                    self.remove_object(id)

    def remove_objects(self, ids):
        for id in ids:
            if self.get_object_type(id) == "art":
                self.remove_articulated_object(id)
            elif self.get_object_type(id) == "non_art":
                self.remove_object(id)
            else:
                import pdb; pdb.set_trace()
                assert self.get_object_type(id) == "not_found"
                print(f"Did not find {id} to delete")

    def set_objects_motion(self, fil_object_ids=None, motion_type=0):
        object_ids = self.get_both_existing_object_ids()
        for type, obj_ids in object_ids.items():
            for id in obj_ids:
                if (fil_object_ids is not None) and (id not in fil_object_ids):
                    continue
                if type == "art":
                    self.set_articulated_object_motion_type(id, MotionType(motion_type))
                else:
                    self.set_object_motion_type(MotionType(motion_type), id)

    def data_key_from_path(self, path):
        from text_housekeep.cos_eor.scripts.orm.utils import preprocess
        if "/" in path:
            path = path.split("/")[-1]
            assert "/" not in path
        if path.endswith("json"):
            key = path.split(".")[0]
        else:
            key = path.split("_")[:-2]
            key = "_".join(key)
        return preprocess(key).replace(" ", "_")

    def set_object_iid(self, sid, obj_id):
        if self.get_object_type(obj_id) == "art":
            self.set_object_semantic_id_art(sid, obj_id)
        else:
            self.set_object_semantic_id_art(sid, obj_id)

    def add_prim(self, obj_id):
        from text_housekeep.cos_eor.utils.geometry import get_bb
        # add wireframe cube around it
        prim_attr_mgr = self.get_asset_template_manager()
        obj_attr_mgr = self.get_object_template_manager()
        wire_handle = prim_attr_mgr.get_template_handles("cubeWire")[0]
        wire_template = obj_attr_mgr.create_template(wire_handle)
        obj_attr_mgr.register_template(wire_template)
        wire_id = self.add_object_by_handle(wire_handle)
        # find scale
        obj_bb = get_bb(self, obj_id)
        prim_bb = get_bb(self, wire_id)
        scale_x = (obj_bb.max[0] - obj_bb.min[0]) / (prim_bb.max[0] - prim_bb.min[0])
        scale_y = (obj_bb.max[1] - obj_bb.min[1]) / (prim_bb.max[1] - prim_bb.min[1])
        scale_z = (obj_bb.max[2] - obj_bb.min[2]) / (prim_bb.max[2] - prim_bb.min[2])
        # remove old object
        self.remove_objects([wire_id])
        scaled_wire_template = obj_attr_mgr.create_template(wire_handle)
        scaled_wire_template.handle = f"{obj_id}_{scaled_wire_template.handle}"
        scaled_wire_template.scale = np.array([scale_x, scale_y, scale_z]) * scaled_wire_template.scale
        obj_attr_mgr.register_template(scaled_wire_template)
        scaled_wire_id = self.add_object_by_handle(scaled_wire_template.handle)
        self.set_translation(self.get_translation(obj_id), scaled_wire_id)
        return scaled_wire_id

    def get_door_data(self, scene_name, metadata_dir):
        metadata_file = os.path.join(metadata_dir, f"assets/{scene_name}/metadata_v2_readjusted_with_outer_doors_v2.yaml")
        with open(metadata_file) as f:
            metadata = yaml.load(f)
        assert "urdfs" in metadata
        metadata = metadata["urdfs"]
        meta_keys = list(metadata.keys())
        door_data = {}
        for k in meta_keys:
            if "door" in k:
                door_data[k] = metadata[k]
        return door_data

    def insert_inarticulate_from_urdf(self, urdf_dir, file_name, pos, rot):
        obj_attr_mgr = self.get_object_template_manager()
        # add inarticulate assembled object
        file_path = os.path.join(urdf_dir.replace("assets", "assets_assemble"),
                                 file_name.split(".")[0], file_name.replace(".urdf", ".object_config.json"))

        try:
            assert os.path.exists(file_path)
        except:
            import pdb
            pdb.set_trace()

        obj_attr_mgr.load_configs(file_path)
        obj_id = self.add_object_by_handle(file_path)
        self.set_translation(pos, obj_id)
        if isinstance(rot, list):
            rot = quat_from_coeffs(rot)
        rot = quat_to_magnum(rot)
        self.set_rotation(rot, obj_id)
        self.set_object_motion_type(MotionType.STATIC, obj_id)
        return obj_id

    def get_ep_key_obj_type(self, room, file_name, episode):
        ep_key = f"{room}-{file_name}"
        if ep_key in episode.recs_keys or "door" in ep_key:
            obj_type = "rec"
        else:
            ep_key = file_name
            if ep_key in episode.objs_keys:
                obj_type = "obj"
            else:
                import pdb
                pdb.set_trace()
                raise AssertionError
        return ep_key, obj_type

    def load_metadata(self, metadata_dir, scene_name, metadata_file=None, urdf_dir=None, filter_objects=True):
        
        if metadata_dir.startswith('data'):
            metadata_dir = os.getcwd().split('exp_local')[0] + '/text_housekeep/' + metadata_dir

        if metadata_file is None:
            metadata_files = ["metadata_assembled.yaml", "metadata_v2.yaml", "metadata.yaml"]
            for file in metadata_files:
                metadata_file = os.path.join(metadata_dir, f"assets/{scene_name}/{file}")
                if os.path.exists(metadata_file):
                    break
        else:
            metadata_file = os.path.join(metadata_dir, f"assets/{scene_name}/{metadata_file}")

        if "readjusted" not in metadata_file:
            print("*"*40)
            print(f"Using metadata file: {metadata_file}")
            print("*"*40)

        if urdf_dir == None:
            urdf_dir = os.path.dirname(metadata_file)

        with open(metadata_file) as f:
            full_metadata = yaml.load(f)
            if "urdfs" in full_metadata:
                metadata = full_metadata["urdfs"]
            else:
                raise ValueError


        # remove objects on receptacles
        meta_keys = list(metadata.keys())
        prev_len = len(meta_keys)
        if filter_objects:
            for o, r in full_metadata["default_mapping"]:
                del metadata[o]
            full_metadata["default_mapping"] = []
        meta_keys = list(metadata.keys())
        new_len = len(meta_keys)
        print(f"Filtered {prev_len - new_len} objects during initialization")

        # add doors from arun's filtered data
        door_data = self.get_door_data(scene_name, metadata_dir)
        metadata.update(door_data)

        return metadata, full_metadata, urdf_dir

    def add_metadata_object(self, metadata, file_name, urdf_dir, add_prims, prim_ids, task, episode):
        rot = metadata[file_name]["rot"]
        pos = metadata[file_name]["pos"]
        room = metadata[file_name]["room"]

        if self.obj_load_type == "non-art":
            obj_id = self.insert_inarticulate_from_urdf(urdf_dir, file_name, pos, rot)
        else:
            # add articulate object
            motion_type = metadata[file_name]["obj_type"]
            orientation = mn.Quaternion(mn.Vector3(*rot[:3]), rot[3])
            trans = mn.Matrix4.from_(orientation.to_matrix(), mn.Vector3(*pos))
            file_path = os.path.join(urdf_dir, file_name)
            obj_id = self.add_art_object(file_path, trans, motion_type)

        if add_prims:
            prim_id = self.add_prim(obj_id)
            prim_ids.append(prim_id)

        # if running inside the task
        if task is not None:
            ep_key, obj_type = self.get_ep_key_obj_type(room, file_name, episode)
            assert episode is not None
            data_key = self.data_key_from_path(file_name)
            try:
                sid = task.class_sid_map[data_key]
            except:
                import pdb
                pdb.set_trace()

            iid = task.instance_id_count
            # add consistent instance ids and build dictionary of semantic-ids
            self.set_object_iid(iid, obj_id)

            room = "null" if room is None else room
            task.track(obj_id, sid, iid, ep_key, obj_type, room)
            task.instance_id_count += 1

        return obj_id

    def recompute_navmesh_with_prims(self, navmesh_file, object_ids, prim_ids, add_prims):
        # save or load navmesh
        self.set_objects_motion(object_ids, motion_type=2)  # igib objects are dynamic
        if add_prims:
            self.set_objects_motion(prim_ids, motion_type=0)
            self.recompute_navmesh(self.pathfinder, self.navmesh_settings, include_static_objects=True)
            # save navmesh and delete primitives
            self.pathfinder.save_nav_mesh(navmesh_file)
            print('Saved NavMesh to "' + navmesh_file + '"')
            self.remove_objects(prim_ids)
        else:
            self.pathfinder.load_nav_mesh(navmesh_file)
            print('Loaded NavMesh from "' + navmesh_file + '"')

        # debugging to check nav vertices in scene
        nav_vs = np.array(self.pathfinder.build_navmesh_vertices())
        nav_vs = nav_vs[nav_vs[:, 1] < 1.0]
        nav_vs_r = np.array([self.pathfinder.island_radius(nav_v) for nav_v in nav_vs])
        n_count_common = Counter(nav_vs_r).most_common()
        # print(f"Most common islands: {n_count_common}")

    def init_metadata_objects_same_scene(self, task, episode):
        prepend = os.getcwd().split('/exp_local')[0] + '/text_housekeep/'
        scene_name = os.path.basename(prepend + self.habitat_config.SCENE).split(".")[0]
        metadata_dir = os.path.dirname(prepend + self.habitat_config.SCENE)
        metadata_dir = os.path.dirname(metadata_dir)

        metadata, _, urdf_dir = self.load_metadata(metadata_dir, scene_name, filter_objects=True)

        fil_meta_keys = []
        object_ids = []
        prim_ids = []
        all_obj_keys = set()
        debug_count = 0

        navmesh_file = self.habitat_config.NAVMESH.format(scene=scene_name, num_objs=len(episode.recs_keys))
        add_prims = not os.path.exists(navmesh_file)
        meta_keys = list(metadata.keys())

        new_obj_keys = set(episode.recs_keys) | set(episode.objs_keys)
        ids_to_remove = []

        # add/remove iGibson objects
        for file_name in meta_keys:
            room = metadata[file_name]["room"]
            rec_key = f"{room}-{file_name}"
            old_rec = rec_key in task.obj_key_to_sim_obj_id
            old_obj = file_name in task.obj_key_to_sim_obj_id
            new_rec = rec_key in new_obj_keys
            new_obj = file_name in new_obj_keys
            if not (old_rec or old_obj or new_rec or new_obj):
                continue
            elif old_rec or new_rec:
                ep_key = rec_key
            elif old_obj or new_obj:
                ep_key = file_name
            else:
                raise ValueError

            obj_in_old_ep = old_rec or old_obj
            obj_in_new_ep = new_rec or new_obj
            if obj_in_old_ep and not obj_in_new_ep:
                ids_to_remove.append(task.obj_key_to_sim_obj_id[ep_key])
            elif obj_in_new_ep:
                if not obj_in_old_ep:
                    obj_id = self.add_metadata_object(metadata, file_name, urdf_dir, add_prims, prim_ids, task, episode)
                else:
                    obj_id = new_obj_keys

            if obj_in_new_ep:
                debug_count += 1
                object_ids.append(obj_id)
                fil_meta_keys.append(file_name)

        self.remove_objects(ids_to_remove)
        task.detrack_objects(ids_to_remove)

        self.recompute_navmesh_with_prims(navmesh_file, object_ids, prim_ids, add_prims)

    def init_metadata_objects(self, metadata_file=None, urdf_dir=None, navmesh_file=None,
                              filter_objects=True, task=None, episode=None):
        """maybe move to task.py?"""
        prepend = os.getcwd().split('/exp_local')[0] + '/text_housekeep/'
        scene_name = os.path.basename(prepend + self.habitat_config.SCENE).split(".")[0]
        metadata_dir = os.path.dirname(prepend + self.habitat_config.SCENE)
        metadata_dir = os.path.dirname(metadata_dir)

        metadata, full_metadata, urdf_dir = self.load_metadata(
            metadata_dir,
            scene_name,
            metadata_file=metadata_file,
            urdf_dir=urdf_dir,
            filter_objects=filter_objects
        )

        fil_meta_keys = []
        object_ids = []
        prim_ids = []
        debug_count = 0

        if navmesh_file is None:
            navmesh_file = self.habitat_config.NAVMESH.format(scene=scene_name, num_objs=len(episode.recs_keys))
        add_prims = not os.path.exists(navmesh_file) or True
        meta_keys = list(metadata.keys())

        # add iGibson objects
        for file_name in meta_keys:
            if task is not None:
                room = metadata[file_name]["room"]
                ep_key = f"{room}-{file_name}"
                if (
                    ep_key not in episode.recs_keys and
                    "door" not in ep_key and
                    file_name not in episode.objs_keys
                ):
                    continue
            obj_id = self.add_metadata_object(metadata, file_name, urdf_dir, add_prims, prim_ids, task, episode)

            debug_count += 1
            object_ids.append(obj_id)
            fil_meta_keys.append(file_name)

        self.recompute_navmesh_with_prims(navmesh_file, object_ids, prim_ids, add_prims)

        if full_metadata is not None:
            # if we do not have matching for any objects
            # turn them receptacles on the fly
            mapped_objects = [o for o, r in full_metadata["default_mapping"]]
            for file_name in fil_meta_keys:
                if metadata[file_name]["type"] == "object":
                    if file_name not in mapped_objects:
                        metadata[file_name]["type"] = "receptacle"

            return fil_meta_keys, object_ids, full_metadata, urdf_dir
        else:
            return fil_meta_keys, object_ids, metadata, urdf_dir

    # convert 3d points to 2d topdown coordinates
    def get_topdown_map(self, height=None, meters_per_pixel=0.01):
        if height is None:
            height = self.get_agent_state().position[1]
        hablab_topdown_map = maps.get_topdown_map(self.pathfinder, height, meters_per_pixel=meters_per_pixel)
        recolor_map = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
        hablab_topdown_map = recolor_map[hablab_topdown_map]
        return hablab_topdown_map

    def get_existing_semantic_ids(self):
        obj_ids = self.get_both_existing_object_ids()
        sem_ids = {}
        for k in obj_ids:
            sem_ids[k] = [self.get_semantic_id(obj_id) for obj_id in obj_ids[k]]
        return sem_ids

    def set_random_semantic_ids(self):
        "debugging function"
        obj_ids = self.get_both_existing_object_ids()
        for obj_id in obj_ids["art"] + obj_ids["non_art"]:
            if self.get_semantic_id(obj_id) <= 0:
                self.set_object_semantic_id_art(np.random.randint(1000), obj_id)

    def get_dist_pos(self, obj1_pos, obj2_pos, dist_type, ignore_y=True):
        obj1_pos, obj2_pos = np.array(obj1_pos, copy=True), np.array(obj2_pos, copy=True)
        if ignore_y:
            obj1_pos[1] = obj2_pos[1]
        if dist_type == "geo":
            return self.geodesic_distance(obj1_pos, obj2_pos)
        elif dist_type == "l2":
            return euclidean_distance(obj2_pos, obj1_pos)
        else:
            raise AssertionError

    def get_dist_id(self, obj1_id, obj2_id, dist_type, ignore_y=True):
        obj1_pos, obj2_pos = self.get_translation(obj1_id), self.get_translation(obj2_id)
        return self.get_dist_pos(obj1_pos, obj2_pos, dist_type, ignore_y)

    def get_or_dist(self, or_id, dist_type="l2"):
        """Get distance from object or receptacle from agent -- from closest corner or center"""
        agent_position = self.get_agent_state().position
        if or_id == self.gripped_object_id:
            return 0.0
        # get corners
        obj_node = self.get_object_scene_node(or_id)
        obj_bb = obj_node.cumulative_bb
        corners = get_corners(obj_bb, obj_node)
        corners.append(self.get_translation(or_id))
        dists = []
        for cor in corners:
            cor[1] = agent_position[1]
            dist = self.get_dist_pos(cor, agent_position, dist_type)
            dists.append(dist)
        return min(dists)

    def geodesic_distance(self, position_a, position_b, episode=None):
        nav_point_a = get_closest_nav_point(self, position_a)
        nav_point_b = get_closest_nav_point(self, position_b)
        geo_dist = super(CosRearrangementSim, self).geodesic_distance(nav_point_a, nav_point_b)
        l2_dist = euclidean_distance(position_a, position_b)

        if geo_dist > l2_dist:
            return geo_dist
        else:
            return l2_dist

    def get_camera_center(self, sensor="semantic"):
        render_camera = self._sensors[sensor]._sensor_object.render_camera
        center_ray = render_camera.unproject(self.screen_center)
        return center_ray

    def _angle_between_vectors(self, v1, v2):
        v1, v2 = v1.normalized(), v2.normalized()
        angle = float(mn.math.angle(v1, v2)) * 180/math.pi
        angle_dir = mn.math.cross(v1, v2)
        return math.copysign(angle, angle_dir)
        if angle_dir >= 0:
            return angle
        else:
            return 2 * math.pi - angle

    def _rotate_agent(self, action_name, angle):
        agent = self.get_agent(self.agent_id)
        actuation = ActuationSpec(angle)
        if agent.controls.is_body_action(action_name):
            agent.controls.action(
                agent.scene_node, action_name, actuation, apply_filter=True
            )
        else:
            for _, v in agent._sensors.items():
                habitat_sim.errors.assert_obj_valid(v)
                agent.controls.action(
                    v.object, action_name, actuation, apply_filter=False
                )

    def snap_id_to_navmesh(self, nav_obj_id, look_obj_id):
        nav_obj_pos = mn.Vector3(self.get_translation(nav_obj_id))
        look_obj_pos = mn.Vector3(self.get_translation(look_obj_id))
        look_obj_iid = self.get_semantic_id(look_obj_id)

        nav_obj_corners = get_corners(get_bb(self, nav_obj_id))
        nav_obj_pts = np.append(nav_obj_corners, [nav_obj_pos], axis=0)

        orig_agent_state = self.get_agent_state(self.agent_id)
        agent_state = self.get_agent_state(self.agent_id)

        nav_points = get_closest_nav_point(self, nav_obj_pos, return_all=True)
        best_nav_point = None
        for nav_idx, nav_point in enumerate(nav_points):
            if (np.linalg.norm(nav_point - nav_obj_pts, axis=1) > 1.5).all():
                print(
                    f"Snap to navmesh failed for nav id {nav_obj_id}, "
                    f"look id {look_obj_id} "
                    f"after {nav_idx} navmesh nodes"
                )
                break
            # assert self.set_agent_state(nav_point, agent_state.rotation)
            camera_dir = self.get_camera_center().direction
            camera_pos = mn.Vector3(nav_point)
            camera_pos.y = self.habitat_config.SEMANTIC_SENSOR.POSITION[1]
            obj_dir = look_obj_pos - camera_pos

            heading_diff = self._angle_between_vectors(obj_dir.xz, camera_dir.xz)
            up_vec = mn.Vector3(0, 1, 0)
            elevation_diff = float(
                mn.math.angle(obj_dir.normalized(), up_vec) - \
                mn.math.angle(camera_dir.normalized(), up_vec)
            ) * 180/math.pi

            agent_state = self.get_agent_state(self.agent_id)
            agent_state.position = nav_point
            self.set_agent_state(agent_state.position, agent_state.rotation)
            if heading_diff > 0:
                self._rotate_agent("turn_left", heading_diff)
            else:
                self._rotate_agent("turn_right", -heading_diff)
            if elevation_diff < 0:
                self._rotate_agent("look_up", -elevation_diff)
            else:
                self._rotate_agent("look_down", elevation_diff)

            multi_observations = self.get_sensor_observations(agent_ids=[self.agent_id])
            sim_obs = multi_observations[self._default_agent_id]
            obs = self._sensor_suite.get_observations(sim_obs)
            visible_iids = set(np.unique(obs["semantic"]))
            if look_obj_iid in visible_iids:
                best_nav_point = nav_point
                print(heading_diff, elevation_diff)
                break

        if best_nav_point is None:
            best_nav_point = np.array(nav_obj_pos)
            best_nav_point[1] = 0.2

        assert self.set_agent_state(orig_agent_state.position, orig_agent_state.rotation)

        return best_nav_point

    def get_shortest_path_next_action(self, obj_pos, snap_to_navmesh=True):
        if snap_to_navmesh:
            nav_pos = get_closest_nav_point(self, obj_pos)
        else:
            nav_pos = obj_pos
        goal_radius = 0.5
        follower = ShortestPathFollower(self, goal_radius=goal_radius, return_one_hot=False)
        follower.mode = "geodesic_path"
        return follower.get_next_action(nav_pos)

# Set defaults in the config system.
_C.SIMULATOR.CROSSHAIR_POS = [360, 540]  # (0.5xW, 0.75xH) from top-left
_C.SIMULATOR.GRAB_DISTANCE = 2.0
_C.SIMULATOR.VISUAL_SENSOR = "rgb"
