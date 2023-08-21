#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import math
import random
import numpy as np
from collections import deque
from text_housekeep.cos_eor.explore.astar_pycpp import pyastar


class FastAStarPlanner:
    """AStarPlanner that uses an underlying C++ implementation for fast
    planning. Based on https://github.com/hjweide/a-star.
    """

    def __init__(self, obstacle_map, show_animation=False, max_planning_iters=math.inf):
        """
        Initialize grid map for a star planning
        """
        self.obmap = obstacle_map
        W = obstacle_map.shape[1]
        H = obstacle_map.shape[0]
        self.center = (W // 2, H // 2)
        self.minx = 0
        self.miny = 0
        self.maxx = W
        self.maxy = H
        self.xwidth = W
        self.ywidth = H
        self.show_animation = show_animation

    def planning(self, gx, gy):
        """
        A star path search
        NOTE - agent is always at the center
        input:
            gx: goal x grid position [m]
            gx: goal x grid position [m]
        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """
        start = self.center
        end = (gx, gy)
        rx, ry = pyastar.astar_planner(self.obmap, start, end, allow_diagonal=True)
        return rx, ry

    @property
    def planning_visualization(self):
        return None


class AStarActorAVD:
    """The A-star based planning policy for AVD uses a simple A-Star planner
    at each time-step to plan a shortest path, and selects an action to reach
    a nearby sub-goal along the shortest path.
    """

    def __init__(self, action_space, show_animation=False):
        self.action_space = action_space
        self.step_interval = 2
        self.theta_match_thresh = 30.0  # Degrees
        self.kernel_size = 3
        self.past_actions = deque(maxlen=3)
        self.goal_thresh = 5.0
        self.show_animation = show_animation
        self._failed_flag = False
        self._final_planning_visualization = None
        self._rng = random.Random(123)

    def act(self, occupancy_map, goal_loc, collision_prev_step):
        self._failed_flag = False
        # ============== Process occupancy map before planning ================
        processed_occupancy = self.proc_occupancy(occupancy_map)
        # Zero pad occupancy map to deal with out-of-bound targets.
        pad_width = 5
        processed_occupancy = np.pad(
            processed_occupancy, pad_width, mode="constant", constant_values=0
        )
        goal_loc = (goal_loc[0] + pad_width, goal_loc[1] + pad_width)
        curr_x = processed_occupancy.shape[1] // 2
        curr_y = processed_occupancy.shape[0] // 2
        # If the previous step resulted in a collision, update the occupancy
        # map to reflect this.
        if collision_prev_step == 1:
            processed_occupancy[
                (curr_y - 15) : curr_y, (curr_x - 3) : (curr_x + 4)
            ] = 1.0
        # Set a small region around goal location as free-space. This is done
        # to deal with cases where it is viewed as occupied due to processing.
        processed_occupancy[
            (goal_loc[1] - 2) : (goal_loc[1] + 3), (goal_loc[0] - 2) : (goal_loc[0] + 3)
        ] = 0
        # ======================== Plan shortest path =========================
        planner = FastAStarPlanner(processed_occupancy, self.show_animation)
        path_x, path_y = planner.planning(*goal_loc)
        self._final_planning_visualization = planner.planning_visualization
        # If a valid path was not found, sample a random action instead.
        if path_x is None:
            # Keep track of path planning failures.
            self._failed_flag = True
            return self._rng.choice(
                (
                    self.action_space["forward"],
                    self.action_space["left"],
                    self.action_space["right"],
                )
            )
        # ========================= Action selection ==========================
        # Determine termination condition
        d2goal = np.linalg.norm(np.array(goal_loc) - np.array([curr_x, curr_y]))
        if len(path_x) < self.step_interval or d2goal < self.goal_thresh:
            # If the target is within a certain distance, execute stop action.
            action = self.action_space["stop"]
        else:
            # Pick the subgoal to visit that is self.step_interval steps away.
            # Since the map resolution is very fine, picking this carefully
            # can result in faster navigation.
            next_x = path_x[-self.step_interval]
            next_y = path_y[-self.step_interval]
            action = self.get_next_action(curr_x, curr_y, next_x, next_y)
        self.past_actions.append(action)
        # Handles a weird edge case where the agent oscillates from left to
        # right without moving forward.
        if self._caught_in_rotation():
            action = self.action_space["forward"]
        return action

    def get_next_action(self, curr_x, curr_y, next_x, next_y):
        theta = 90 - math.degrees(math.atan2(curr_y - next_y, next_x - curr_x))
        if abs(theta) < self.theta_match_thresh:
            action = self.action_space["forward"]
        elif theta < 0:
            action = self.action_space["left"]
        else:
            action = self.action_space["right"]
        return action

    def proc_occupancy(self, occ):
        mask = np.all(occ == (0, 0, 255), axis=-1).astype(np.uint8)
        kernel_size = (self.kernel_size, self.kernel_size)
        kernel = np.ones(kernel_size, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = mask.astype(np.float32)
        return mask

    def _caught_in_rotation(self):
        if len(self.past_actions) > 1:
            l = self.action_space["left"]
            r = self.action_space["right"]
            if (self.past_actions[-2] == l and self.past_actions[-1] == r) or (
                self.past_actions[-2] == r and self.past_actions[-1] == l
            ):
                return True

        return False

    def reset(self):
        self._final_planning_visualization = None
        self._rng = random.Random(123)
        self.past_actions.clear()

    @property
    def planning_visualization(self):
        if self._final_planning_visualization is None:
            return None
        else:
            return np.copy(self._final_planning_visualization)

    @property
    def planning_failure_flag(self):
        return self._failed_flag


class AStarActorHabitat:
    """The A-star based planning policy for Habitat plans in a hierarchical
    fashion. This is a base class for that policy. This class is similar to
    AStarActorAVD, but has a slightly different set of heuristics to account
    for Habitat specific quirks.
    """

    def __init__(
        self,
        action_space,
        step_interval=4,
        goal_thresh=4.0,
        occupancy_downsample_size=400,
        show_animation=False,
        max_planning_iters=math.inf,
        dilate_occupancy=False,
    ):
        self.action_space = action_space
        self.step_interval = step_interval
        self.theta_match_thresh = 10.0  # Degrees
        self.kernel_size = 3
        self.past_actions = deque(maxlen=3)
        self.show_animation = show_animation
        self.goal_thresh = goal_thresh
        self.occupancy_downsample_size = occupancy_downsample_size
        self._failed_flag = False
        self._final_planning_visualization = None
        self._rng = random.Random(123)
        self._prev_path_length = None
        self._planned_path = None
        self.max_planning_iters = max_planning_iters
        self.dilate_occupancy = dilate_occupancy

    def act(self, occupancy_map, goal_loc, collision_prev_step):
        self._failed_flag = False
        # ============== Process occupancy map before planning ================
        # Scale down the map before planning.
        y_scaling = 1.0 * self.occupancy_downsample_size / occupancy_map.shape[0]
        x_scaling = 1.0 * self.occupancy_downsample_size / occupancy_map.shape[1]
        goal_loc = (int(goal_loc[0] * y_scaling), int(goal_loc[1] * x_scaling))
        processed_occupancy = self.proc_occupancy(occupancy_map)
        # Zero pad occupancy map to deal with out-of-bound targets.
        pad_width = 5
        processed_occupancy = np.pad(
            processed_occupancy, pad_width, mode="constant", constant_values=0
        )
        # Set a small region around goal location as free-space. This is done
        # to deal with cases where it is viewed as occupied due to processing.
        goal_loc = (goal_loc[0] + pad_width, goal_loc[1] + pad_width)
        processed_occupancy[
            (goal_loc[1] - 2) : (goal_loc[1] + 3), (goal_loc[0] - 2) : (goal_loc[0] + 3)
        ] = 0
        # Set a small region around current location as free-space. This is
        # done to deal with cases where it is viewed as occupied due to
        # processing.
        curr_x = processed_occupancy.shape[1] // 2
        curr_y = processed_occupancy.shape[0] // 2
        if processed_occupancy[curr_y, curr_x] == 1.0:
            processed_occupancy[
                (curr_y - 2) : (curr_y + 3), (curr_x - 2) : (curr_x + 3)
            ] = 0
        # ======================== Plan shortest path =========================
        planner = FastAStarPlanner(
            processed_occupancy,
            self.show_animation,
            max_planning_iters=self.max_planning_iters,
        )
        path_x, path_y = planner.planning(*goal_loc)
        self._final_planning_visualization = planner.planning_visualization
        # If a valid path was not found, keep turning left instead.
        if path_x is None:
            # Keep track of path planning failures.
            self._failed_flag = True
            self._planned_path = None
            return self.action_space["left"]
        else:
            # Store the planned path in case it is needed later.
            # Convert back to the original occupancy_map coordinates.
            x_ratio = occupancy_map.shape[0] / self.occupancy_downsample_size
            y_ratio = occupancy_map.shape[1] / self.occupancy_downsample_size
            self._planned_path = [
                [int(1.0 * (x - pad_width) * x_ratio) for x in path_x],
                [int(1.0 * (y - pad_width) * y_ratio) for y in path_y],
            ]
            self._prev_path_length = len(path_x)
        # ========================= Action selection ==========================
        # Determine termination condition
        d2goal = np.linalg.norm(np.array(goal_loc) - np.array([curr_x, curr_y]))
        self._dist2goal = d2goal
        if d2goal < self.goal_thresh:
            # If the target is within a certain distance, execute stop action.
            action = self.action_space["stop"]
        else:
            # Pick the subgoal to visit that is self.step_interval steps away.
            # Since the map resolution is very fine, picking this carefully
            # can result in faster navigation.
            next_idx = max(len(path_x) - self.step_interval, 0)
            next_x = path_x[next_idx]
            next_y = path_y[next_idx]
            action = self.get_next_action(curr_x, curr_y, next_x, next_y)
        self.past_actions.append(action)
        # Handles a weird edge case where the agent oscillates from left to
        # right without moving forward.
        if self._caught_in_rotation():
            action = self.action_space["forward"]
        return action

    def get_next_action(self, curr_x, curr_y, next_x, next_y):
        theta = math.pi / 2 - math.atan2(curr_y - next_y, next_x - curr_x)
        theta = math.degrees(math.atan2(math.sin(theta), math.cos(theta)))
        if abs(theta) < self.theta_match_thresh:
            action = self.action_space["forward"]
        elif theta < 0:
            action = self.action_space["left"]
        else:
            action = self.action_space["right"]
        return action

    def proc_occupancy(self, occ):
        mask = np.all(occ == (0, 0, 255), axis=-1).astype(np.uint8)
        if self.dilate_occupancy:
            kernel_size = (self.kernel_size, self.kernel_size)
            kernel = np.ones(kernel_size, np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
        mask = mask.astype(np.float32)
        wby2 = mask.shape[1] // 2
        hby2 = mask.shape[0] // 2
        mask[(hby2 - 1) : (hby2 + 2), (wby2 - 1) : (wby2 + 2)] = 0
        size = self.occupancy_downsample_size
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_LINEAR)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        return mask

    def _caught_in_rotation(self):
        if len(self.past_actions) > 1:
            l = self.action_space["left"]
            r = self.action_space["right"]
            if (self.past_actions[-2] == l and self.past_actions[-1] == r) or (
                self.past_actions[-2] == r and self.past_actions[-1] == l
            ):
                return True

        return False

    def reset(self):
        self._final_planning_visualization = None
        self.past_actions.clear()
        self._rng = random.Random(123)
        self._prev_path_length = None

    @property
    def planning_visualization(self):
        if self._final_planning_visualization is None:
            return None
        else:
            return np.copy(self._final_planning_visualization)

    @property
    def planned_path(self):
        return self._planned_path

    @property
    def planning_failure_flag(self):
        return self._failed_flag


class HierarchicalAStarActorHabitat:
    """The A-star based planning policy for Habitat plans in a hierarchical
    fashion. A high-level plan is created on a coarse occupancy map. A
    subgoal is sampled from this high-level plan and a new plan is created on
    a high-resolution map to reach this subgoal. This significantly
    accelerates the planning routine on large Habitat maps.
    """

    def __init__(self, action_space, map2world_scale, show_animation=False):
        self.high_level_actor = AStarActorHabitat(
            action_space,
            step_interval=2,
            goal_thresh=2.0,
            occupancy_downsample_size=200,
            show_animation=show_animation,
        )
        self.low_level_actor = AStarActorHabitat(
            action_space,
            step_interval=4,
            goal_thresh=2.0,
            occupancy_downsample_size=400,
            show_animation=False,
            max_planning_iters=2000,
        )
        self.action_space = action_space
        self._rng = random.Random(123)
        self.high_level_target = None
        self.low_level_target = None
        self.map_size = None
        self.planning_interval = 15
        self.planned_path = None
        self.intermediate_goal_idx = None
        self.goal_thresh = 0.15
        self.map2world_scale = map2world_scale
        self.past_actions = deque(maxlen=3)
        self.show_animation = show_animation
        self.rotation_stuck_timer = 0
        self.rotation_stuck_status = False
        self.rotation_stuck_thresh = 10

    def act(self, occupancy_map, goal_loc, prev_delta, collision_prev_step):
        if self.map_size is None:
            self.map_size = occupancy_map.shape[0]
        # If the target has already been reached, return STOP.
        if self.has_reached_target(goal_loc):
            return self.action_space["stop"]
        self.high_level_target = goal_loc
        # Perform high-level planning only if a low-level target has not yet
        # been set, or if it was already reached.
        if self.low_level_target is None:
            high_level_action, low_level_target = self.call_high_level_actor(
                occupancy_map, self.high_level_target, collision_prev_step
            )
            # This could be true either because:
            # (1) The high_level_actor returned stop.
            # (2) The high_level_actor failed in planning and returned a
            # random action.
            if low_level_target is None:
                return high_level_action
            self.low_level_target = low_level_target
        else:
            # Update the low_level_target using prev_delta.
            self.update_target(prev_delta)
            # Also update the planned path.
            updated_path = [[], []]
            for x, y in zip(*self.planned_path):
                tx, ty = self.ego_transform((x, y), prev_delta)
                updated_path[0].append(tx)
                updated_path[1].append(ty)
            self.planned_path = updated_path

        # Sample a low-level action to reach the intermediate goal set by the
        # high-level planner.
        low_level_action = self.low_level_actor.act(
            occupancy_map, self.low_level_target, collision_prev_step
        )
        intermediate_path = [
            self.planned_path[0][self.intermediate_goal_idx :],
            self.planned_path[1][self.intermediate_goal_idx :],
        ]
        path_feasible = self.is_feasible_path(occupancy_map, intermediate_path)
        if (
            low_level_action == self.action_space["stop"]
            or self.low_level_actor.planning_failure_flag
            or (not path_feasible)
        ):
            high_level_action, low_level_target = self.call_high_level_actor(
                occupancy_map, self.high_level_target, collision_prev_step
            )
            self.low_level_target = low_level_target
            if low_level_target is None:
                return high_level_action
            self.low_level_actor.reset()
            low_level_action = self.low_level_actor.act(
                occupancy_map, self.low_level_target, collision_prev_step
            )
            action = low_level_action
        else:
            action = low_level_action
        # ================= Prepare a map for visualization ===================
        vis_occ_map = np.copy(occupancy_map)
        # Draw final goal in red
        vis_occ_map = cv2.circle(vis_occ_map, goal_loc, 7, (255, 0, 0), -1)
        # Draw planned path to goal in gray
        for x, y in zip(*self.planned_path):
            vis_occ_map = cv2.circle(vis_occ_map, (x, y), 3, (128, 128, 128), -1)
        # Draw intermediate path in black
        for x, y in zip(*intermediate_path):
            vis_occ_map = cv2.circle(vis_occ_map, (x, y), 3, (0, 0, 0), -1)
        # Draw intermediate goal in pink
        vis_occ_map = cv2.circle(
            vis_occ_map, self.low_level_target, 7, (255, 192, 203), -1
        )
        self._final_planning_visualization = vis_occ_map
        if self.show_animation:
            cv2.imshow("Hierarchical planning status", np.flip(vis_occ_map, axis=2))
            cv2.waitKey(5)
        # ================= Prepare a map for visualization ===================
        self.past_actions.append(action)
        # If caught in rotation, try to break out of it.
        if self.rotation_stuck_status or self._caught_in_rotation():
            self.rotation_stuck_status = True
            self.rotation_stuck_timer += 1
            if self.rotation_stuck_timer > self.rotation_stuck_thresh:
                self.rotation_stuck_status = False
                self.rotation_stuck_timer = 0
            elif self.rotation_stuck_timer % 2 == 0:
                action = self.action_space["forward"]
            else:
                action = self.action_space["left"]
        return action

    def has_reached_target(self, goal_loc):
        Mby2 = self.map_size // 2
        cell_dist2goal = math.sqrt(
            (goal_loc[0] - Mby2) ** 2 + (goal_loc[1] - Mby2) ** 2
        )
        world_dist2goal = cell_dist2goal * self.map2world_scale
        return True if world_dist2goal < self.goal_thresh else False

    def call_high_level_actor(self, occupancy_map, goal_loc, collision_prev_step):
        """Executes the high-level actor to set an intermediate goal for the
        low-level actor.
        """
        high_level_action = self.high_level_actor.act(
            occupancy_map, goal_loc, collision_prev_step
        )
        # If the high level actor has determined that the target is reached,
        # return stop.
        if high_level_action == self.action_space["stop"]:
            return high_level_action, None
        # If the high level actor could not find a path, return random actions.
        self.planned_path = self.high_level_actor.planned_path
        if self.high_level_actor.planned_path is None:
            return (
                self._rng.choice(
                    [
                        self.action_space["forward"],
                        self.action_space["left"],
                        self.action_space["right"],
                    ]
                ),
                None,
            )
        else:
            path_x, path_y = self.high_level_actor.planned_path
            next_idx = max(len(path_x) - self.planning_interval, 0)
            low_level_target = (path_x[next_idx], path_y[next_idx])
            self.intermediate_goal_idx = next_idx
        return high_level_action, low_level_target

    def is_feasible_path(self, occupancy_map, path):
        """
        occupancy map - (H, W, 3) RGB input
        path - [[x0, x1, ...], [y0, y1, ...]]
        """
        path_pixels = occupancy_map[path[1], path[0]]
        path_is_blocked = np.any(np.all(path_pixels == np.array([0, 0, 255]), axis=-1))
        return not path_is_blocked

    def update_target(self, prev_delta):
        self.low_level_target = self.ego_transform(self.low_level_target, prev_delta)

    def ego_transform(self, target, prev_delta):
        """
        Update the target to the new egocentric coordinate system.
        Inputs:
            prev_delta - (dx, dy, dtheta) motion in the previous position's
                         frame of reference
        """
        # Note: X - forward, Y - rightward in egocentric frame of references
        # Perform update in egocentric coordinate
        x, y = self._convert_to_egocentric(target)
        dx, dy, dt = prev_delta
        # Translate
        xp, yp = x - dx, y - dy
        # Rotate by -dt
        xp, yp = (
            math.cos(-dt) * xp - math.sin(-dt) * yp,
            math.sin(-dt) * xp + math.cos(-dt) * yp,
        )
        # Convert back to image coordinate
        xi, yi = self._convert_to_image((xp, yp))
        xi = np.clip(xi, 1, self.map_size - 2)
        yi = np.clip(yi, 1, self.map_size - 2)
        return (int(xi), int(yi))

    def _convert_to_egocentric(self, coords):
        return (-coords[1] + self.map_size / 2, coords[0] - self.map_size / 2)

    def _convert_to_image(self, coords):
        # Forward - positive X, rightward - positive Y
        return (coords[1] + self.map_size / 2, -coords[0] + self.map_size / 2)

    def reset(self):
        self.high_level_actor.reset()
        self.low_level_actor.reset()
        self._rng = random.Random(123)
        self.high_level_target = None
        self.low_level_target = None
        self.map_size = None
        self.planned_path = None
        self.intermediate_goal_idx = None

    def _caught_in_rotation(self):
        if len(self.past_actions) > 1:
            l = self.action_space["left"]
            r = self.action_space["right"]
            if (self.past_actions[-2] == l and self.past_actions[-1] == r) or (
                self.past_actions[-2] == r and self.past_actions[-1] == l
            ):
                return True
        return False

    @property
    def planning_failure_flag(self):
        return self.high_level_actor.planning_failure_flag

    @property
    def planning_visualization(self):
        return np.copy(self._final_planning_visualization)


class HierarchicalAStarActorHabitatV2:
    """The A-star based planning policy for Habitat plans in a hierarchical
    fashion. A high-level plan is created on a coarse occupancy map. A
    subgoal is sampled from this high-level plan and a new plan is created on
    a high-resolution map to reach this subgoal. This significantly
    accelerates the planning routine on large Habitat maps.

    Note: The heuristics chosen here are slightly modified from
    HierarchicalAStarActorHabitat to work robustly for frontier exploration.
    """

    def __init__(self, action_space, map2world_scale, show_animation=False):
        self.high_level_actor = AStarActorHabitat(
            action_space,
            step_interval=2,
            goal_thresh=2.0,
            occupancy_downsample_size=200,
            show_animation=show_animation,
        )
        self.low_level_actor = AStarActorHabitat(
            action_space,
            step_interval=4,
            goal_thresh=2.0,
            occupancy_downsample_size=400,
            show_animation=False,
            max_planning_iters=2000,
        )
        self.action_space = action_space
        self._rng = random.Random(123)
        self.high_level_target = None
        self.low_level_target = None
        self.map_size = None
        self.planning_interval = 15
        self.planned_path = None
        self.intermediate_goal_idx = None
        self.goal_thresh = 0.15
        self.map2world_scale = map2world_scale
        self.past_actions = deque(maxlen=3)
        self.show_animation = show_animation
        self.rotation_stuck_timer = 0
        self.rotation_stuck_status = False
        self.rotation_stuck_thresh = 10
        self._enable_planning_visualization = True

    def act(self, occupancy_map, goal_loc, prev_delta, collision_prev_step):
        if self.map_size is None:
            self.map_size = occupancy_map.shape[0]
        # If the target has already been reached, return STOP.
        if self.has_reached_target(goal_loc):
            return self.action_space["stop"]
        self.high_level_target = goal_loc
        path_feasible = True
        # Is the current path from the intermediate goal to the target feasible?
        if self.low_level_target is not None:
            # Update the low_level_target using prev_delta
            self.update_target(prev_delta)
            # Also update the planned path
            updated_path = [[], []]
            for x, y in zip(*self.planned_path):
                tx, ty = self.ego_transform((x, y), prev_delta)
                updated_path[0].append(tx)
                updated_path[1].append(ty)
            self.planned_path = updated_path
            intermediate_path = [
                self.planned_path[0][self.intermediate_goal_idx :],
                self.planned_path[1][self.intermediate_goal_idx :],
            ]
            path_feasible = self.is_feasible_path(occupancy_map, intermediate_path)

        # If a low_level_target has not been sampled, sample one.
        if self.low_level_target is None or not path_feasible:
            reached_target = self.sample_low_level_target(
                occupancy_map, self.high_level_target, collision_prev_step
            )
            if reached_target:
                return self.action_space["stop"]
            elif self.low_level_target is None:  # Failed to sample low_level_target
                # print('======> HighLevelPlanner failed to sample any valid targets!')
                return self._rng.choice(
                    [
                        self.action_space["left"],
                        self.action_space["right"],
                        self.action_space["forward"],
                    ]
                )

        # Sample an action that takes you to the low_level_target
        low_level_action = self.low_level_actor.act(
            occupancy_map, self.low_level_target, collision_prev_step
        )

        # If the low_level_actor reached the low_level_target, return a random action and
        # set low_level_target to None
        if low_level_action == self.action_space["stop"]:
            low_level_action = self._rng.choice(
                [
                    self.action_space["left"],
                    self.action_space["right"],
                    self.action_space["forward"],
                ]
            )
            self.low_level_target = None
            self.low_level_actor.reset()
        # If the low_level_actor failed in planning, return a random action and
        # set low_level_target to None
        elif self.low_level_actor.planning_failure_flag:
            low_level_action = self._rng.choice(
                [
                    self.action_space["left"],
                    self.action_space["right"],
                    self.action_space["forward"],
                ]
            )
            self.low_level_target = None
            self.low_level_actor.reset()

        if self._enable_planning_visualization:
            vis_occ_map = np.copy(occupancy_map)
            # Draw final goal in red
            vis_occ_map = cv2.circle(vis_occ_map, goal_loc, 7, (255, 0, 0), -1)
            # Draw planned path to goal in gray
            for x, y in zip(*self.planned_path):
                vis_occ_map = cv2.circle(vis_occ_map, (x, y), 3, (128, 128, 128), -1)
            # Draw intermediate path in black
            intermediate_path = [
                self.planned_path[0][self.intermediate_goal_idx :],
                self.planned_path[1][self.intermediate_goal_idx :],
            ]
            for x, y in zip(*intermediate_path):
                vis_occ_map = cv2.circle(vis_occ_map, (x, y), 3, (0, 0, 0), -1)
            # Draw intermediate goal in pink
            if self.low_level_target is not None:
                vis_occ_map = cv2.circle(
                    vis_occ_map, self.low_level_target, 7, (255, 192, 203), -1
                )
            self._final_planning_visualization = vis_occ_map
        else:
            self._final_planning_visualization = None

        if self.show_animation:
            cv2.imshow("Hierarchical planning status", np.flip(vis_occ_map, axis=2))
            cv2.waitKey(5)

        self.past_actions.append(low_level_action)
        # If caught in rotation, try to break out of it.
        # if self.rotation_stuck_status or self._caught_in_rotation():
        #    self.rotation_stuck_status = True
        #    self.rotation_stuck_timer += 1
        #    if self.rotation_stuck_timer > self.rotation_stuck_thresh:
        #        self.rotation_stuck_status = False
        #        self.rotation_stuck_timer = 0
        #    elif self.rotation_stuck_timer % 2 == 0:
        #        action = self.action_space['forward']
        #    else:
        #        action = self.action_space['left']

        return low_level_action

    def has_reached_target(self, goal_loc):
        Mby2 = self.map_size // 2
        cell_dist2goal = math.sqrt(
            (goal_loc[0] - Mby2) ** 2 + (goal_loc[1] - Mby2) ** 2
        )
        world_dist2goal = cell_dist2goal * self.map2world_scale
        # print(f'World distance to goal: {world_dist2goal:.2f}m')
        return True if world_dist2goal < self.goal_thresh else False

    def sample_low_level_target(self, occupancy_map, goal_loc, collision_prev_step):
        reached_target = False
        high_level_action = self.high_level_actor.act(
            occupancy_map, goal_loc, collision_prev_step
        )
        self.planned_path = self.high_level_actor.planned_path
        # If the high level actor has determined that the target is reached, return stop
        if high_level_action == self.action_space["stop"]:
            self.low_level_target = None
            reached_target = True
        # If the high level actor could not find a path, return random actions
        elif self.high_level_actor.planned_path is None:
            self.low_level_target = None
            rached_target = False
        # If the high level actor found a path, sample an intermediate target.
        else:
            path_x, path_y = self.high_level_actor.planned_path
            next_idx = max(len(path_x) - self.planning_interval, 0)
            self.low_level_target = (path_x[next_idx], path_y[next_idx])
            self.intermediate_goal_idx = next_idx
            reached_target = False

        return reached_target

    def is_feasible_path(self, occupancy_map, path):
        """
        occupancy map - (H, W, 3) RGB input
        path - [[x0, x1, ...], [y0, y1, ...]]
        """
        path_pixels = occupancy_map[path[1], path[0]]
        path_is_blocked = np.any(np.all(path_pixels == np.array([0, 0, 255]), axis=-1))
        return not path_is_blocked

    def update_target(self, prev_delta):
        self.low_level_target = self.ego_transform(self.low_level_target, prev_delta)

    def ego_transform(self, target, prev_delta):
        """
        Update the target to the new egocentric coordinate system.
        Inputs:
            prev_delta - (dx, dy, dtheta) motion in the previous position's
                         frame of reference
        """
        # Note: X - forward, Y - rightward in egocentric frame of references

        # Perform update in egocentric coordinate
        x, y = self._convert_to_egocentric(target)

        dx, dy, dt = prev_delta
        # Translate
        xp, yp = x - dx, y - dy
        # Rotate by -dt
        xp, yp = (
            math.cos(-dt) * xp - math.sin(-dt) * yp,
            math.sin(-dt) * xp + math.cos(-dt) * yp,
        )

        # Convert back to image coordinate
        xi, yi = self._convert_to_image((xp, yp))
        xi = np.clip(xi, 1, self.map_size - 2)
        yi = np.clip(yi, 1, self.map_size - 2)

        return (int(xi), int(yi))

    def _convert_to_egocentric(self, coords):
        return (-coords[1] + self.map_size / 2, coords[0] - self.map_size / 2)

    def _convert_to_image(self, coords):
        # Forward - positive X, rightward - positive Y
        return (coords[1] + self.map_size / 2, -coords[0] + self.map_size / 2)

    def reset(self):
        self.high_level_actor.reset()
        self.low_level_actor.reset()
        self._rng = random.Random(123)
        self.high_level_target = None
        self.low_level_target = None
        self.map_size = None
        self.planned_path = None
        self.intermediate_goal_idx = None

    def _caught_in_rotation(self):
        if len(self.past_actions) > 1:
            l = self.action_space["left"]
            r = self.action_space["right"]
            if (self.past_actions[-2] == l and self.past_actions[-1] == r) or (
                self.past_actions[-2] == r and self.past_actions[-1] == l
            ):
                return True
        return False

    @property
    def planning_failure_flag(self):
        return self.high_level_actor.planning_failure_flag

    @property
    def planning_visualization(self):
        if self._final_planning_visualization is not None:
            return np.copy(self._final_planning_visualization)
        else:
            return None