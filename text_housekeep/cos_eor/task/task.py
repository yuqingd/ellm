import glob
import os
from collections import defaultdict
from copy import deepcopy
from typing import Type, Union, Dict, Any
import json
import json_tricks
import numpy as np
import yaml
from tqdm import tqdm
import glob

from text_housekeep.habitat_lab.habitat.core.dataset import Episode
from text_housekeep.habitat_lab.habitat.tasks.nav.nav import NavigationTask
from text_housekeep.habitat_lab.habitat.core.registry import registry
from habitat_sim.physics import MotionType
from habitat_sim.utils.common import quat_from_coeffs, quat_to_magnum
from text_housekeep.cos_eor.dataset.dataset import CosRearrangementEpisode
import habitat_sim
import shortuuid
import magnum as mn
from text_housekeep.cos_eor.scripts.igib_assemble_obj import urdf_to_obj, URDF_OBJ_CACHE
from text_housekeep.cos_eor.utils.shelf_bin_packer import ShelfBinPacker
from text_housekeep.cos_eor.utils.geometry import get_bb, get_bb_base, euclidean_distance, get_corners
from text_housekeep.cos_eor.task.utils import get_packer_mapping
from text_housekeep.cos_eor.scripts.build_utils import aggregate_amt_annotations

@registry.register_task(name="CosRearrangementTask-v0")
class CosRearrangementTask(NavigationTask):
    r"""Embodied Rearrangement Task
    Goal: An agent must place objects at their corresponding goal position.
    """

    def __init__(self, **kwargs) -> None:
        """ep_obj_id: from dataset; sim_obj_id: from simulator"""
        super().__init__(**kwargs)
        self.reset_trackers(False)
        self.load_annotations()
        self.rec_packers: Dict[int, ShelfBinPacker] = {}
        self.original_start_matrix = None

    def get_translation(self, object_id):
        """Return actual position if not held, otherwise agent position"""
        if object_id != self._sim.gripped_object_id:
            obj_translation = self._sim.get_translation(object_id)
        else:
            obj_translation = self._sim.get_agent_state().position
        return obj_translation

    def get_dist(self, obj_id1, obj_id2, dist_type, ignore_y=True):
        obj1_pos = self.get_translation(obj_id1)
        obj2_pos = self.get_translation(obj_id2)
        if ignore_y:
            obj1_pos[1] = obj2_pos[1]

        if dist_type == "geo":
            return self._sim.geodesic_distance(obj1_pos, obj2_pos)
        elif dist_type == "l2":
            return euclidean_distance(obj2_pos, obj1_pos)
        else:
            raise AssertionError

    def get_iid_from_key(self, obj_key):
        obj_id = self.obj_key_to_sim_obj_id[obj_key]
        obj_iid = self.sim_obj_id_to_iid[obj_id]
        return obj_iid

    def reset_trackers(self, ep_reset):
        self._episode_was_reset = ep_reset
        self.misc_dict = {}
        self.replay = []

        self.instance_id_count = 1
        # iid are instance ids
        self.sim_obj_id_to_iid = {}
        self.iid_to_sim_obj_id = {}
        # sid are semantic ids
        self.iid_to_sid = {}
        self.sid_to_iid = {}
        self.sim_obj_id_to_obj_key = {}
        self.obj_key_to_sim_obj_id = {}
        self.sim_obj_id_to_type = {}
        # map objs/recs into rooms (start only, don't use)
        self.obj_id_to_room = {}
        # track episode only objects
        self.ep_obj_ids = []

    def detrack_objects(self, obj_ids):
        for obj_id in obj_ids:
            iid = self.sim_obj_id_to_iid[obj_id]
            sid = self.iid_to_sid[iid]
            ok = self.sim_obj_id_to_obj_key[obj_id]

            # detrack
            del self.sim_obj_id_to_iid[obj_id]
            del self.sim_obj_id_to_type[obj_id]
            del self.sim_obj_id_to_obj_key[obj_id]
            del self.obj_id_to_room[obj_id]

            del self.iid_to_sid[iid]
            del self.iid_to_sim_obj_id[iid]

            self.sid_to_iid[sid].remove(iid)
            if len(self.sid_to_iid) == 0:
                del self.sid_to_iid[sid]

            del self.obj_key_to_sim_obj_id[ok]

    def delete_episode_objects(self):
        # delete objects
        self._sim.remove_objects(self.ep_obj_ids)

        # detrack objects
        self.detrack_objects(self.ep_obj_ids)


        # reset episode
        self._episode_was_reset = True
        self.misc_dict = {}
        self.replay = []
        self.ep_obj_ids = []
        self.instance_id_count = max(list(self.iid_to_sid.keys())) + 1

        # check consistency
        self.assert_consistency()

    def track(self, obj_id=None, sid=None, iid=None, obj_key=None, obj_type=None, room=None, obj_level="scene"):
        if iid is not None:
            self.sim_obj_id_to_iid[obj_id] = iid
            self.iid_to_sim_obj_id[iid] = obj_id

            if sid is not None:
                self.iid_to_sid[iid] = sid
                if sid in self.sid_to_iid:
                    self.sid_to_iid[sid].append(iid)
                else:
                    self.sid_to_iid[sid] = [iid]
        if obj_key is not None:
            self.sim_obj_id_to_obj_key[obj_id] = obj_key
            self.obj_key_to_sim_obj_id[obj_key] = obj_id
        if obj_type is not None:
            assert obj_type in ["rec", "obj"]
            self.sim_obj_id_to_type[obj_id] = obj_type
        if room is not None:
            assert isinstance(room, str)
            self.obj_id_to_room[obj_id] = room

        if obj_level == "episode":
            self.ep_obj_ids.append(obj_id)

    def trackers(self, only_return=False):
        task_data = {}
        for attr in ["_episode_was_reset",
                     "misc_dict",
                     "replay",
                     "instance_id_count",
                     "sim_obj_id_to_iid",
                     "iid_to_sim_obj_id",
                     "sim_obj_id_to_obj_key",
                     "obj_key_to_sim_obj_id",
                     "iid_to_sid",
                     "sid_to_iid",
                     "sid_class_map",
                     "class_sid_map",
                     "obj_id_to_room",
                     "sim_obj_id_to_type"
                     ]:
            task_data[attr] = getattr(self, attr)
            if not only_return:
                print(f"{attr}: {task_data[attr]}")

        # asserts to ensure all trackers
        #  are consistent with each other
        self.assert_consistency()
        return task_data

    def assert_consistency(self, episode=None):
        for attrs in [
            ("sim_obj_id_to_iid",
             "iid_to_sim_obj_id",
             "sim_obj_id_to_obj_key",
             "obj_key_to_sim_obj_id",
             "iid_to_sid",
             "obj_id_to_room",
             "sim_obj_id_to_type"),
            ("sid_class_map",
             "class_sid_map")
                     ]:
            attr_lens = []
            for attr in attrs:
                attr_lens.append(len(getattr(self, attr)))
            assert len(set(attr_lens)) == 1

        if episode is not None:
            packer_mapping = get_packer_mapping(self.rec_packers, self)
            current_mapping = episode.get_mapping()
            assert current_mapping == packer_mapping

    def load_annotations(self):
        obj_attr_mgr = self._sim.get_object_template_manager()
        prepend_path = os.getcwd().split('/exp_local')[0] + '/text_housekeep/'

        # load cache handles -- igib objects
        cache_handles = [h for h in glob.glob(os.path.join(prepend_path + URDF_OBJ_CACHE, "**"), recursive=True) if h.endswith(".object_config.json")]
        for ch in cache_handles:
            obj_attr_mgr.load_configs(ch)

        # load ycb objects
        obj_attr_mgr.load_configs(prepend_path + "data/objects")

        # load other annotations
        self.object_annotations = np.load(self._config["OBJECT_ANNOTATIONS"], allow_pickle=True).item()
        skipped_templates = 0
        for k in tqdm(self.object_annotations, desc="Loading handles"):
            if "template" in self.object_annotations[k]:
                handle = self.object_annotations[k]['template']
                handle = prepend_path + handle
                if handle.endswith(".object_config.json") and os.path.exists(handle):
                    obj_attr_mgr.load_configs(handle)
                else:
                    skipped_templates += 1
        print(f"Skipped {skipped_templates}/ {len(self.object_annotations)} specified object templates!")

        # load semantic-class annotations
        sem_classes_path = os.getcwd().split('/exp_local')[0] + "/text_housekeep/cos_eor/scripts/dump/semantic_classes_amt.yaml"
        objects_data = yaml.load(open(sem_classes_path, "r"))
        self.sid_class_map = dict(objects_data["semantic_class_id_map"])
        self.class_sid_map = dict([tup[::-1] for tup in objects_data["semantic_class_id_map"]])

        # load amt annotations
        self.amt_data = aggregate_amt_annotations(only_amt=True)

    def register_object_templates(self):
        r"""
        Register object temmplates from the dataset into the simulator
        """
        obj_attr_mgr = self._sim.get_object_template_manager()
        object_templates = self._dataset.object_templates

        if isinstance(object_templates, dict):
            for name, template_info in object_templates.items():
                name = os.path.basename(name).split(".")[0]
                obj_handle = obj_attr_mgr.get_file_template_handles(name)[0]
                obj_template = obj_attr_mgr.get_template_by_handle(obj_handle)
                obj_template.scale = np.array(template_info["scale"])
                obj_attr_mgr.register_template(obj_template)

        elif isinstance(object_templates, list):
            raise AssertionError("Objects need to be scaled!")
            for name in object_templates:
                name = os.path.basename(name).split(".")[0]
                obj_handle = obj_attr_mgr.get_file_template_handles(name)[0]
                obj_template = obj_attr_mgr.get_template_by_handle(obj_handle)
                obj_template.scale = np.array([1.0, 1.0, 1.0])
                obj_attr_mgr.register_template(obj_template)

        # add sphere
        # obj_handle = obj_attr_mgr.get_file_template_handles("sphere")[0]
        # obj_template = obj_attr_mgr.get_template_by_handle(obj_handle)
        # obj_template.scale = np.array([0.8, 0.8, 0.8])
        # obj_attr_mgr.register_template(obj_template)

    def overwrite_sim_config(self, sim_config, episode):
        """Called within env.reconfigure() to add episode to simulator config."""
        sim_config = super().overwrite_sim_config(sim_config, episode)
        # self.register_object_templates()
        return sim_config

    def _initialize_receptacle_packers(self, episode: CosRearrangementEpisode, rec_obj_mapping: dict):
        self.rec_packers = {}

        rec_packers_matches = {} # store object key to 'matches' dict mapping
        for rec, objs_list in rec_obj_mapping.items():
            original_match = episode.recs_packers[rec]['matches']
            for obj_match, new_obj in zip(original_match, objs_list):
                obj_match['rect']['id'] = new_obj

        for rec_key in episode.recs_keys:
            if "agent" in rec_key:
                continue
            if rec_key not in self.obj_key_to_sim_obj_id:
                import pdb; pdb.set_trace()
            rec_id = self.obj_key_to_sim_obj_id[rec_key]
            self.rec_packers[rec_id] = ShelfBinPacker(get_bb_base(get_bb(self._sim, rec_id)))
            # remove any objects that weren't added/modified by the rec_obj_mapping
            mod_matches = []
            for matches in episode.recs_packers[rec_key]['matches']:
                if matches['rect']['id'] in self.obj_key_to_sim_obj_id:
                    mod_matches.append(matches)
            episode.recs_packers[rec_key]['matches'] = mod_matches
            try:
                self.rec_packers[rec_id].from_dict(episode.recs_packers[rec_key], self.obj_key_to_sim_obj_id)
            except:
                import pdb; pdb.set_trace()

        try:        
            self.assert_consistency(episode)
        except:
            import pdb; pdb.set_trace()

    def _initialize_episode_objects(self, episode):
        start_idx = episode.default_matrix_shape[-1]
        obj_attr_mgr = self._sim.get_object_template_manager()

        if self.original_start_matrix is None:
            self.original_start_matrix = episode.start_matrix
            self.original_recs = []
            for key in episode.objs_keys[start_idx:]:
                rec_key = episode.get_rec(key, self.original_start_matrix)
                self.original_recs.append(rec_key)

        # All receptacles this episode
        # only shuffle first three
        # shuffled_recs = deepcopy(self.original_recs[:2])
        # np.random.shuffle(shuffled_recs)
        # recs_list = shuffled_recs + self.original_recs[2:]
        recs_list = self.original_recs
        
        # set ~half to be correct already, so we learn to ignore already correctly placed objects
        # recs_list = []
        # for key, rec in zip(episode.objs_keys[start_idx:], shuffled_recs):
        #     if np.random.rand() < .5:
        #         # randomly pick correct receptacle
        #         possible_correct_recs = episode.get_correct_recs(key)
        #         recs_list.append(np.random.choice(possible_correct_recs))
        #     else:
        #         recs_list.append(rec)
        shuffled_recs = recs_list
    
        rec_obj_mapping = { k: [] for k in shuffled_recs}
        # add episode specific objects, we do not add/delete receptacles to/from scenes
        objs_files = []
        objs_keys = []
        objs_pos = []
        objs_rot = []
        objs_cats = []
        for i, (file, key, pos, rot, cat) in enumerate(zip(episode.objs_files[start_idx:], episode.objs_keys[start_idx:],
                                                    episode.objs_pos[start_idx:], episode.objs_rot[start_idx:],
                                                    episode.objs_cats[start_idx:])):
            orig_file = file
            prepend = os.getcwd().split('/exp_local')[0]
            file = prepend + '/text_housekeep/' + file
            # dynamically create obj files from urdfs, cache them, and replace in episode files as well
            if file.endswith(".urdf"):
                obj_file, already_exists = urdf_to_obj(file, URDF_OBJ_CACHE, return_exist=True)
                episode.objs_files = [obj_file if f == file else f for f in episode.objs_files]
                file = obj_file
                if not already_exists:
                    obj_attr_mgr.load_configs(file)

            # only add misplaced objects
            rec_key = episode.get_rec(key, episode.start_matrix)
            correct_rec_keys = episode.get_correct_recs(key)
            if rec_key in correct_rec_keys:
                # if rec_key in rec_obj_mapping and len(rec_obj_mapping[rec_key]) == 0:
                #     rec_obj_mapping.pop(rec_key)
                continue
            
            objs_files.append(orig_file)
            objs_keys.append(key)
            objs_pos.append(pos)
            objs_rot.append(rot)
            objs_cats.append(cat)

            sim_obj_id = self._sim.add_object_by_handle(file)
            if sim_obj_id == -1:
                import pdb; pdb.set_trace()
            self._sim.set_translation(pos, sim_obj_id)
            if isinstance(rot, list):
                rot = quat_from_coeffs(rot)
            rot = quat_to_magnum(rot)
            self._sim.set_rotation(rot, sim_obj_id)
            self._sim.set_object_motion_type(MotionType.DYNAMIC, sim_obj_id)
            sid = self.class_sid_map[cat]
            iid = self.instance_id_count
            self._sim.set_object_iid(iid, sim_obj_id)
            # figure out room and orm

            # randomize rec
            rec_key = shuffled_recs[i]
            rec_obj_mapping[rec_key].append(key)

            # update matrix with shuffled rec
            matrix = episode.start_matrix
            obj_ind = episode.objs_keys.index(key)
            shuffled_rec_ind = episode.recs_keys.index(rec_key)
            rec_ind = np.argwhere(matrix[:, obj_ind] == 1).squeeze(-1)
            matrix[rec_ind, obj_ind] = 0
            matrix[shuffled_rec_ind, obj_ind] = 1
            episode.start_matrix = matrix
            # rec_key = episode.get_rec(key, episode.start_matrix)
            rec_id = self.obj_key_to_sim_obj_id.get(rec_key, -1)
            rec_room = self.obj_id_to_room.get(rec_id, "null")
            # add to trackers
            self.track(sim_obj_id, sid, iid, key, "obj", rec_room, "episode")
            self.instance_id_count += 1

        # update stuff with shuffled recs
        episode.state_matrix = deepcopy(episode.start_matrix)
        state_mapping = episode.get_mapping('start')
        correct_mapping = episode.get_correct_mapping()
        episode.misplaced_count = 0
        for obj_key in episode.objs_keys:
            if state_mapping[obj_key] not in correct_mapping[obj_key]:
                episode.misplaced_count += 1

        episode.objs_files[start_idx:] = objs_files
        episode.objs_keys[start_idx:]  = objs_keys
        episode.objs_pos[start_idx:] = objs_pos
        episode.objs_rot[start_idx:] = objs_rot
        episode.objs_cats[start_idx:] = objs_cats

        self._initialize_receptacle_packers(episode, rec_obj_mapping)

        

    def _initialize_objects(self, episode: CosRearrangementEpisode):
        r"""
        Initialize the stage with the objects in the episode.
        """
        # TODO: change this back after we fix gpu memory leak.
        if self._sim.same_scene and self.instance_id_count > 1:
            # print(f"Only initializing episode objects")
            # delete and detrack last episodes objects
            
            self.delete_episode_objects()
            # reset trackers
            # self.reset_trackers(True)
            # # remove all existing objects
            # self._sim.remove_all_objects()
            # initialize scene default objects + receptacles
            # self._sim.init_metadata_objects_same_scene(task=self, episode=episode)
        else:
            # reset trackers
            self.reset_trackers(True)
            # remove all existing objects
            self._sim.remove_all_objects()
            # initialize scene default objects + receptacles
            self._sim.init_metadata_objects(task=self, episode=episode)

        # reset agent
        self._sim.set_agent_state(episode.start_position, quat_from_coeffs(episode.start_rotation))

        # initialize episode specific objects
        self._initialize_episode_objects(episode)

    def reset(self, episode: Episode):
        # self.print_trackers()
        episode.reset()
        self._initialize_objects(episode)
        self.assert_consistency(episode)
        return super().reset(episode)

    def step(self, action: Union[int, Dict[str, Any]], episode: Type[Episode]):
        self._episode_was_reset = False
        self.replay.append(action)
        return super().step(action, episode)

    def did_episode_reset(self, *args: Any, **kwargs: Any) -> bool:
        return self._episode_was_reset

    def save_replay(self, episode, info={}, path="", uuid=""):
        data = {
            'episode_id': episode.episode_id,
            'scene_id': episode.scene_id,
            'agent_pos': np.array(self._sim._last_state.position).tolist(),
            'current_position': {},
            'gripped_object_id': self._sim.gripped_object_id,
            'info': deepcopy(info)
        }
        task_data = {
            'sim_object_id_to_objid_mapping': self.sim_obj_id_to_ep_obj_id,
            'objid_to_sim_object_id_mapping': self.ep_obj_id_to_sim_obj_id,
            'actions': self.replay,
            'misc_dict': self.misc_dict,
            'sim_obj_id_to_iid': self.sim_obj_id_to_iid,
            'iid_to_sim_obj_id': self.iid_to_sim_obj_id,
            'obj_key_to_sim_obj_id': self.obj_key_to_sim_obj_id,
            'sim_obj_id_to_obj_key': self.sim_obj_id_to_obj_key
        }
        data.update(deepcopy(task_data))

        # save final positions
        object_positions = [np.array(obj.position).tolist() for obj in episode.objects]
        rec_positions = [np.array(rec.position).tolist() for rec in episode.get_receptacles()]
        data["object_positions"] = object_positions
        data["receptacle_positions"] = rec_positions

        keys = list(data["objid_to_sim_object_id_mapping"].keys())
        for k in keys:
            data["objid_to_sim_object_id_mapping"][str(k)] = data["objid_to_sim_object_id_mapping"].pop(k)

        with open(os.path.join(path, 'replays_{}_{}_{}.json'.format(uuid, episode.episode_id,
                                                                    episode.scene_id.split('/')[-1])), 'w') as f:
            for key in data:
                try:
                    json_tricks.dump(data[key], f)
                except:
                    import pdb
                    pdb.set_trace()

    def get_oid_from_sid(self, sid):
        return self.iid_to_sim_obj_id[sid]
