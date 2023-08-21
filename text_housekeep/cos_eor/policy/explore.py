import math
import random

from einops import asnumpy
import numpy as np
import torch

from text_housekeep.cos_eor.explore.frontier_agent import FrontierAgent
from text_housekeep.cos_eor.explore.utils.geometry import process_odometer, compute_egocentric_coors

import text_housekeep.cos_eor.explore.sensors
import text_housekeep.cos_eor.explore.sim

class ExploreModule:
    def __init__(self, params, num_envs):
        self.params = params
        actions = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "LOOK_UP", "LOOK_DOWN", "GRAB_RELEASE"]
        self.action_mapping = {action: idx for idx, action in enumerate(actions)}
        self.num_envs = num_envs

        large_map_range = 100.0
        self.occ_map_scale = 0.1 * (2 * large_map_range + 1) / params.highres_occ_map_size
        self.frontier_agent = FrontierAgent(
            {"forward": 1, "left": 2, "right": 3, "stop": 0},
            "habitat",
            self.occ_map_scale,
            show_animation=False,
            dilate_occupancy=True,
            max_time_per_target=30,
        )
        self.obs_odometer = torch.zeros(num_envs, 4)
        self.delta_ego = torch.zeros(num_envs, 4)
        self.seen_area = torch.zeros(num_envs)
        self._steps_since_new_area = torch.zeros(num_envs)

    def to(self, device):
        self.obs_odometer = self.obs_odometer.to(device)
        self.delta_ego = self.delta_ego.to(device)
        self.seen_area = self.seen_area.to(device)
        self._steps_since_new_area = self._steps_since_new_area.to(device)

    def reset(self):
        self.obs_odometer.fill_(0)
        self.delta_ego.fill_(0)
        self.seen_area.fill_(0)
        self._steps_since_new_area.fill_(0)

    @property
    def steps_since_new_area(self):
        return self._steps_since_new_area[0].item()

    def reset_steps_since_new_area(self):
        self._steps_since_new_area.fill_(0)

    def update(self, obs):
        """this is called every single step"""
        batch_size = self.delta_ego.shape[0]

        for i in range(batch_size):
            seen_area = obs["seen_area"][i][0]
            if math.isclose(self.seen_area[i], seen_area):
                self._steps_since_new_area[i] += 1
            else:
                self.seen_area[i] = seen_area
                self._steps_since_new_area[i] = 0

        if self.params.name == "frontier":
            obs_odometer_curr = process_odometer(obs["delta"])
            self.delta_ego = compute_egocentric_coors(
                obs_odometer_curr,
                self.obs_odometer,
                self.occ_map_scale
            )
            for i in range(batch_size):
                if obs["new_episode"][i] == 1:
                    self.obs_odometer[i] = obs_odometer_curr[i]
                else:
                    self.obs_odometer[i] += obs_odometer_curr[i]

    def _act_forward_right(self, obs):
        if obs["cos_eor"][0]["is_collided"]:
            action = self.action_mapping["TURN_RIGHT"]
        else:
            action = self.action_mapping["MOVE_FORWARD"]
        return action

    def _act_frontier(self, obs):
        occ_map = asnumpy(obs["coarse_occupancy"].cpu()).astype(np.uint8)
        collision = asnumpy(obs["collision"].cpu())
        delta_ego = asnumpy(self.delta_ego.cpu())
        batch_size = occ_map.shape[0]
        action = np.zeros(batch_size, dtype=int)
        for i in range(batch_size):
            action[i] = self.frontier_agent.act(occ_map[i], delta_ego[i], collision[i][0])
        return action[0]

    def act(self, obs):
        """this is called only when the exploration module takes control"""
        if self.params.name == "random":
            action = random.choice(list(self.action_mapping.values())[1:4])
        elif self.params.name == "forward_right":
            action = self._act_forward_right(obs)
        elif self.params.name == "frontier":
            action = self._act_frontier(obs)
        else:
            raise ValueError
        return {"action": action}

