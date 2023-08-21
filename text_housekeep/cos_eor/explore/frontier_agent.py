#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import logging
import math
import random
import numpy as np
from text_housekeep.cos_eor.explore.navigation import (
    AStarActorAVD,
    HierarchicalAStarActorHabitatV2,
)


class FrontierAgent:
    def __init__(
        self,
        action_space,
        env_name,
        occ_map_scale,
        show_animation=False,
        seed=123,
        use_contour_sampling=True,
        dilate_occupancy=True,
        max_time_per_target=-1,
    ):
        self.map_size = None
        self.action_space = action_space
        self.show_animation = show_animation
        self.frontier_target = None
        self.occ_buffer = None
        self.seed = seed
        self._rng = random.Random(seed)
        self._time_elapsed_for_target = 0
        self._failure_count = 0
        self.use_contour_sampling = use_contour_sampling
        self.env_name = env_name

        if "avd" in env_name:
            self.actor = AStarActorAVD(action_space, show_animation=show_animation)
            self.max_time_per_target = (
                20 if max_time_per_target == -1 else max_time_per_target
            )
        else:
            self.actor = HierarchicalAStarActorHabitatV2(
                action_space, occ_map_scale, show_animation=show_animation
            )
            # Manually set dilate_occupancy flag
            if dilate_occupancy:
                self.actor.high_level_actor.dilate_occupancy = True
                self.actor.low_level_actor.dilate_occupancy = True
            self.max_time_per_target = (
                200 if max_time_per_target == -1 else max_time_per_target
            )

        logging.info("========> FrontierAgent settings")
        logging.info(f"===> max_time_per_target  : {self.max_time_per_target}")
        logging.info(f"===> dilate_occupancy     : {dilate_occupancy}")

    def act(self, occ_map, prev_delta, collision_prev_step):
        if self.occ_buffer is None:
            self.map_size = occ_map.shape[0]
            self.occ_buffer = np.zeros((self.map_size, self.map_size), dtype=np.uint8)

        action = self.action_space["stop"]
        action_count = 0
        while action == self.action_space["stop"]:
            # If no target is selected or too much time was spent on a single target, pick a new target
            if (
                self.frontier_target is None
                or self._time_elapsed_for_target >= self.max_time_per_target
            ):
                self.sample_frontier_target(occ_map)
                self.actor.reset()
            # If the hierarchical planner failed twice to generate a plan to the target, then sample a new target
            elif self._failure_count == 2:
                self.sample_frontier_target(occ_map)
                self.actor.reset()
            # If a valid target is available, then update the target coordinate based on the past motion.
            else:
                self.update_target(prev_delta)
                # If the agent has reached the target or the target is occupied, then sample a new target
                if self.has_reached_target() or np.all(
                    occ_map[self.frontier_target[1], self.frontier_target[0]]
                    == (0, 0, 255)
                ):
                    self.sample_frontier_target(occ_map)
                    self.actor.reset()
                # When the hierarchical actor has reached the target, resample the target
                elif action_count > 0 and action == self.action_space["stop"]:
                    self.sample_frontier_target(occ_map)
                    self.actor.reset()

            if self.show_animation:
                cv2.imshow("Occupancy", np.flip(occ_map, axis=2))
                cv2.waitKey(20)

            action_count += 1

            # Prevents infinite loop when all frontier targets sampled return action=3
            if action_count > 3:
                logging.info("=====> Stuck in occupied region! ")
                return random.choice(
                    [
                        self.action_space["left"],
                        self.action_space["right"],
                        self.action_space["forward"],
                    ]
                )

            if "avd" in self.env_name:
                action = self.actor.act(
                    occ_map, self.frontier_target, collision_prev_step
                )
            else:
                # This does not process the occupancy map. Process it.
                action = self.actor.act(
                    occ_map, self.frontier_target, prev_delta, collision_prev_step
                )
            if self.actor.planning_failure_flag:
                self._failure_count += 1
                if self._failure_count == 2:
                    action = self.action_space["stop"]

        self._time_elapsed_for_target += 1

        return action

    def sample_frontier_target(self, occ_map):
        """
        Inputs:
            occ_map - occupancy map with the following color coding:
                      (0, 0, 255) is occupied region
                      (255, 255, 255) is unknown region
                      (0, 255, 0) is free region
        """
        self.occ_buffer.fill(0)
        self._time_elapsed_for_target = 0
        self._failure_count = 0

        unknown_mask = np.all(occ_map == (255, 255, 255), axis=-1).astype(np.uint8)
        free_mask = np.all(occ_map == (0, 255, 0), axis=-1).astype(np.uint8)

        unknown_mask_shiftup = np.pad(
            unknown_mask, ((0, 1), (0, 0)), mode="constant", constant_values=0
        )[1:, :]
        unknown_mask_shiftdown = np.pad(
            unknown_mask, ((1, 0), (0, 0)), mode="constant", constant_values=0
        )[:-1, :]
        unknown_mask_shiftleft = np.pad(
            unknown_mask, ((0, 0), (0, 1)), mode="constant", constant_values=0
        )[:, 1:]
        unknown_mask_shiftright = np.pad(
            unknown_mask, ((0, 0), (1, 0)), mode="constant", constant_values=0
        )[:, :-1]

        frontier_mask = (
            (free_mask == unknown_mask_shiftup)
            | (free_mask == unknown_mask_shiftdown)
            | (free_mask == unknown_mask_shiftleft)
            | (free_mask == unknown_mask_shiftright)
        ) & (free_mask == 1)

        frontier_idxes = list(zip(*np.where(frontier_mask)))
        if len(frontier_idxes) > 0:
            if self.use_contour_sampling:
                frontier_img = frontier_mask.astype(np.uint8) * 255
                # Reduce size for efficiency
                scaling_factor = frontier_mask.shape[0] / 200.0
                frontier_img = cv2.resize(
                    frontier_img,
                    None,
                    fx=1.0 / scaling_factor,
                    fy=1.0 / scaling_factor,
                    interpolation=cv2.INTER_NEAREST,
                )
                # Add a single channel
                frontier_img = frontier_img[:, :, np.newaxis]
                contours, _ = cv2.findContours(
                    frontier_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if len(contours) == 0:
                    tgt = self._rng.choice(frontier_idxes)  # (y, x)
                else:
                    contours_length = [len(contour) for contour in contours]
                    contours = list(zip(contours, contours_length))
                    sorted_contours = sorted(contours, key=lambda x: x[1], reverse=True)

                    contours = sorted_contours[:3]
                    # Randomly pick one of the longest contours
                    # To introduce some stochasticity in case the agent is stuck
                    max_contour = self._rng.choice(contours)[0]
                    # Pick a random sample from the longest contour
                    tgt = self._rng.choice(max_contour)[
                        0
                    ]  # Each point is [[x, y]] for some reason
                    # Scale it back to original image size
                    # Convert it to (y, x) convention as this will be reversed next
                    tgt = (int(tgt[1] * scaling_factor), int(tgt[0] * scaling_factor))
            else:
                tgt = self._rng.choice(frontier_idxes)  # (y, x)

            self.frontier_target = (
                np.clip(tgt[1], 1, self.map_size - 2).item(),
                np.clip(tgt[0], 1, self.map_size - 2).item(),
            )  # (x, y)
        else:
            self.frontier_target = (self.map_size // 2 + 4, self.map_size // 2 + 4)

        if self.show_animation:
            occ_map_copy = np.copy(occ_map)
            occ_map_copy = cv2.circle(
                occ_map_copy, self.frontier_target, 3, (255, 0, 0), -1
            )
            cv2.imshow("Occupancy map with target", np.flip(occ_map_copy, axis=2))
            cv2.imshow("Frontier mask", frontier_mask.astype(np.uint8) * 255)
            cv2.waitKey(10)

    def has_reached_target(self):
        fx, fy = self.frontier_target
        cx, cy = self.map_size / 2, self.map_size / 2
        if math.sqrt((fx - cx) ** 2 + (fy - cy) ** 2) < 3.0:
            return True
        else:
            return False

    def update_target(self, prev_delta):
        """
        Update the target to the new egocentric coordinate system.
        Inputs:
            prev_delta - (dx, dy, dtheta) motion in the previous position's
                         frame of reference
        """
        # Note: X - forward, Y - rightward in egocentric frame of references

        # Perform update in egocentric coordinate
        x, y = self._convert_to_egocentric(self.frontier_target)

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

        self.frontier_target = (int(xi), int(yi))

    def _convert_to_egocentric(self, coords):
        return (-coords[1] + self.map_size / 2, coords[0] - self.map_size / 2)

    def _convert_to_image(self, coords):
        # Forward - positive X, rightward - positive Y
        return (coords[1] + self.map_size / 2, -coords[0] + self.map_size / 2)

    def reset(self):
        self.frontier_target = None
        self._failure_count = 0
        self._time_elapsed_for_target = 0