import math

import numpy as np

def process_odometer(pose):
    """Converts odometer readings in polar coordinates to xyt coordinates.
    Inputs:
        pose - (bs, 4) Tensor with (r, theta, phi_head, phi_elev)
             - where angles are in radians
    Outputs:
        pose_processed - (bs, 4) Tensor with (y, x, phi_head, phi_elev)
    """
    pose_processed = np.array([
        pose[0] * math.sin(pose[1]),
        pose[0] * math.cos(pose[1]),
        pose[2],
        pose[3]
    ])
    return pose_processed


def compute_egocentric_coors(delta, prev_pos, scale):
    """
    delta --- (y, x, phi_head, phi_elev)
    prev_pos --- (y, x, phi_head, phi_elev)
    """
    dy, dx, dt = delta[0], delta[1], delta[2]
    x, y, t = prev_pos[0], prev_pos[1], prev_pos[2]
    dr = math.sqrt(dx ** 2 + dy ** 2)
    dp = math.atan2(dy, dx) - t
    dx_ego = dr * math.cos(dp) / scale
    dy_ego = dr * math.sin(dp) / scale
    dt_ego = dt

    return np.array([dx_ego, dy_ego, dt_ego])