from typing import Dict
from dataclasses import dataclass

# CODES
SUC = "succeeded"
INP = "in-progress"
FAIL = "failed"
NAV = "navigating"
OBJ = "object"
REC = "receptacle"
PP = "pick-place"
EXP = "exploring"

@dataclass
class NavTargetInfo:
    obj_type: str
    idx: int

class NavModule:
    def __init__(self, envs, params=None, task_params=None):
        self.envs = envs
        self.params = params
        self.task_params = task_params
        self.dist_threshold = 1.0
        self.module = "oracle-nav"

        self.reset()

    def reset(self):
        self.snapped_navmesh_pts_cache = {}

    def update(self, obs):
        """this is called every single step"""
        pass

    def act(self, obs, targets: Dict[str, NavTargetInfo], hie_policy, num_nav_steps):
        """this is called only when the navigation module takes control"""
        value, action_log_probs, rnn_hidden_states = [None] * 3
        nav_target, look_target = targets["nav"], targets["look"]
        nav_obj_id = hie_policy.get_value(nav_target.idx, nav_target.obj_type, "obj_id")
        look_obj_id = hie_policy.get_value(look_target.idx, look_target.obj_type, "obj_id")
        if num_nav_steps == 0:
            obj_pos = self.envs.call_at(0, "snap_id_to_navmesh", {"nav_obj_id": nav_obj_id, "look_obj_id": look_obj_id})
            self.snapped_navmesh_pts_cache[(nav_obj_id, look_obj_id)] = obj_pos
        else:
            obj_pos = self.snapped_navmesh_pts_cache[(nav_obj_id, look_obj_id)]
        best_action = self.envs.call_at(0, "get_shortest_path_next_action", {"obj_pos": obj_pos, "snap_to_navmesh": False})
        status = "in-progress"
        return value, best_action, action_log_probs, rnn_hidden_states, self.module, status
