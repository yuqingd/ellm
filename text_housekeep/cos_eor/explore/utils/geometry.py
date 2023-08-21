import scipy.stats as stats
import torch


def truncated_normal_noise(eta, width):
    """Generates truncated normal noise scalar.

    Args:
        eta - standard deviation of gaussian.
        width - maximum absolute width on either sides of the mean.
    Returns:
        Sampled noise scalar from truncated gaussian with mean=0, sigma=eta,
        and width=width.
    """
    mu = 0
    sigma = eta
    lower = mu - width
    upper = mu + width
    X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    return X.rvs()


def process_odometer(poses):
    """Converts odometer readings in polar coordinates to xyt coordinates.

    Inputs:
        pose - (bs, 4) Tensor with (r, theta, phi_head, phi_elev)
             - where angles are in radians
    Outputs:
        pose_processed - (bs, 4) Tensor with (y, x, phi_head, phi_elev)
    """
    pose_processed = torch.stack(
        [
            poses[:, 0] * torch.sin(poses[:, 1]),
            poses[:, 0] * torch.cos(poses[:, 1]),
            poses[:, 2],
            poses[:, 3],
        ],
        dim=1,
    )
    return pose_processed


def compute_egocentric_coors(delta, prev_pos, scale):
    """
    delta - (N, 4) --- (y, x, phi_head, phi_elev)
    prev_pos - (N, 4) --- (y, x, phi_head, phi_elev)
    """
    dy, dx, dt = delta[:, 0], delta[:, 1], delta[:, 2]
    x, y, t = prev_pos[:, 0], prev_pos[:, 1], prev_pos[:, 2]
    dr = torch.sqrt(dx ** 2 + dy ** 2)
    dp = torch.atan2(dy, dx) - t
    dx_ego = dr * torch.cos(dp) / scale
    dy_ego = dr * torch.sin(dp) / scale
    dt_ego = dt

    return torch.stack([dx_ego, dy_ego, dt_ego], dim=1)
