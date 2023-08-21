from typing import Tuple

import cv2
import numpy as np

from text_housekeep.habitat_lab.habitat.core.simulator import Simulator
from text_housekeep.habitat_lab.habitat.utils.visualizations import maps

def to_grid_v2(
    realworld_x: float,
    realworld_y: float,
    x_min: float,
    y_min: float,
    map_scale: float,
) -> Tuple[int, int]:
    r"""Return gridworld index of realworld coordinates assuming top-left
    corner is the origin. This differs from to_grid() based on the map extent
    specification. x_min, y_min are the minimum x and y coordinates from the
    environment. map_scale is the real-world length that corresponds to one
    grid-cell in the map.
    """
    grid_x = int((realworld_x - x_min) / map_scale)
    grid_y = int((realworld_y - y_min) / map_scale)
    return grid_x, grid_y


def from_grid_v2(
    grid_x: int, grid_y: int, x_min: float, y_min: float, map_scale: float,
) -> Tuple[float, float]:
    r"""Inverse of to_grid_v2 function. Return real world coordinate from
    gridworld assuming top-left corner is the origin. This differs from
    from_grid() based on the map extent specification.
    x_min, y_min are the minimum x and y coordinates from the environment.
    map_scale is the real-world length that corresponds to one grid-cell
    in the map.
    """
    realworld_x = x_min + grid_x * map_scale
    realworld_y = y_min + grid_y * map_scale
    return realworld_x, realworld_y


def get_topdown_map_v2(
    sim: Simulator,
    map_extents: Tuple[int, int, int, int],
    map_scale: float,
    num_samples: int = 20000,
) -> np.ndarray:
    r"""Return a top-down occupancy map for a sim. Note, this only returns
    valid values for whatever floor the agent is currently on. This differs
    from get_topdown_map() based on the map size specification.

    Args:
        sim: The simulator.
        map_resolution: The resolution of map which will be computed and
            returned.
        num_samples: The number of random navigable points which will be
            initially
            sampled. For large environments it may need to be increased.
        draw_border: Whether to outline the border of the occupied spaces.

    Returns:
        Image containing 0 if occupied, 1 if unoccupied, and 2 if border (if
        the flag is set).
    """

    x_min, x_max, y_min, y_max = map_extents
    map_resolution = (
        int((x_max - x_min) / map_scale),
        int((y_max - y_min) / map_scale),
    )
    top_down_map = np.zeros(map_resolution, dtype=np.uint8)

    start_height = sim.get_agent_state().position[1]

    # Use sampling to find the extrema points that might be navigable.
    range_x = (map_resolution[0], 0)
    range_y = (map_resolution[1], 0)
    for _ in range(num_samples):
        point = sim.sample_navigable_point()
        # Check if on same level as original
        if np.abs(start_height - point[1]) > 0.5:
            continue
        g_x, g_y = to_grid_v2(point[0], point[2], x_min, y_min, map_scale)
        range_x = (min(range_x[0], g_x), max(range_x[1], g_x))
        range_y = (min(range_y[0], g_y), max(range_y[1], g_y))

    range_x = (max(0, range_x[0]), min(map_resolution[0], range_x[1]))
    range_y = (max(0, range_y[0]), min(map_resolution[1], range_y[1]))

    # Search over grid for valid points.
    for ii in range(range_x[0], range_x[1]):
        for jj in range(range_y[0], range_y[1]):
            realworld_x, realworld_y = from_grid_v2(ii, jj, x_min, y_min, map_scale)
            valid_point = sim.is_navigable([realworld_x, start_height, realworld_y])
            top_down_map[ii, jj] = maps.MAP_VALID_POINT if valid_point else maps.MAP_INVALID_POINT

    return top_down_map


def topdown_to_image(topdown_info: np.ndarray) -> np.ndarray:
    r"""Generate image of the topdown map.
    """
    top_down_map = topdown_info["map"]
    fog_of_war_mask = topdown_info["fog_of_war_mask"]
    top_down_map = maps.colorize_topdown_map(top_down_map, fog_of_war_mask)
    map_agent_pos = topdown_info["agent_map_coord"]

    # Add zero padding
    min_map_size = 200
    if top_down_map.shape[0] != top_down_map.shape[1]:
        H = top_down_map.shape[0]
        W = top_down_map.shape[1]
        if H > W:
            pad_value = (H - W) // 2
            padding = ((0, 0), (pad_value, pad_value), (0, 0))
            map_agent_pos = (map_agent_pos[0], map_agent_pos[1] + pad_value)
        else:
            pad_value = (W - H) // 2
            padding = ((pad_value, pad_value), (0, 0), (0, 0))
            map_agent_pos = (map_agent_pos[0] + pad_value, map_agent_pos[1])
        top_down_map = np.pad(
            top_down_map, padding, mode="constant", constant_values=255
        )

    if top_down_map.shape[0] < min_map_size:
        H, W = top_down_map.shape[:2]
        top_down_map = cv2.resize(top_down_map, (min_map_size, min_map_size))
        map_agent_pos = (
            int(map_agent_pos[0] * min_map_size // H),
            int(map_agent_pos[1] * min_map_size // W),
        )
    top_down_map = maps.draw_agent(
        image=top_down_map,
        agent_center_coord=map_agent_pos,
        agent_rotation=topdown_info["agent_angle"],
        agent_radius_px=top_down_map.shape[0] // 16,
    )
    # if top_down_map.shape[0] < min_map_size:
    #    pad_value = (min_map_size - top_down_map.shape[0]) // 2
    #    padding = ((pad_value, pad_value), (pad_value, pad_value), (0, 0))
    #    top_down_map = np.pad(top_down_map, padding, mode='constant', constant_values=255)

    return top_down_map
