import math
import random
from copy import deepcopy

import numpy as np
from tqdm import tqdm

import habitat_sim
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_rotate_vector,
)
from typing import List

from habitat.core.utils import try_cv2_import
import magnum as mn

from habitat_sim.utils.common import quat_from_coeffs
from text_housekeep.cos_eor.utils.samplers import PolySurface
from habitat.core.registry import registry
from habitat_sim._ext.habitat_sim_bindings import Ray

from text_housekeep.cos_eor.utils.shelf_bin_packer import Rect

cv2 = try_cv2_import()


def get_polar_angle(rotation):
    heading_vector = quaternion_rotate_vector(
        rotation.inverse(), np.array([0, 0, -1])
    )

    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    z_neg_z_flip = np.pi
    return np.array(phi) + z_neg_z_flip


def geodesic_distance(pathfinder, position_a, position_b):
    """
        start = np.array([0, 0., 0])
        end = np.array([-1.8083103, 0., 0.51558936])
        ends = np.array([[-1.8083103, 0., 0.51558936]])
        path = habitat_sim.MultiGoalShortestPath()
        path.requested_start = start
        path.requested_ends = ends
        pathfinder.find_path(path)
        geo_dist = path.geodesic_distance
        l2_dist = np.linalg.norm(start - ends[0])
        print(f"l2: {l2_dist} and geo: {geo_dist}")

    """
    path = habitat_sim.MultiGoalShortestPath()
    path.requested_start = np.array(position_a, dtype=np.float32)
    if isinstance(position_b[0], List) or isinstance(
            position_b[0], np.ndarray
    ):
        path.requested_ends = np.array(position_b, dtype=np.float32)
    else:
        path.requested_ends = np.array(
            [np.array(position_b, dtype=np.float32)]
        )

    pathfinder.find_path(path)

    return path.geodesic_distance


def quaternion_xyzw_to_wxyz(v: np.array):
    return np.quaternion(v[3], *v[0:3])


def quaternion_wxyz_to_xyzw(v: np.array):
    return np.quaternion(*v[1:4], v[0])


def quaternion_to_coeff(quat: np.quaternion) -> np.array:
    r"""Converts a quaternions to coeffs in [x, y, z, w] format
    """
    coeffs = np.zeros((4,))
    coeffs[3] = quat.real
    coeffs[0:3] = quat.imag
    return coeffs


def euclidean_distance(position_a, position_b):
    return np.linalg.norm(
        np.array(position_b) - np.array(position_a), ord=2
    )


def compute_heading_from_quaternion(r):
    """
    r - rotation quaternion

    Computes clockwise rotation about Y.
    """
    # quaternion - np.quaternion unit quaternion
    # Real world rotation
    direction_vector = np.array([0, 0, -1])  # Forward vector
    heading_vector = quaternion_rotate_vector(r.inverse(), direction_vector)

    phi = -np.arctan2(heading_vector[0], -heading_vector[2]).item()
    return phi


def compute_quaternion_from_heading(theta):
    """
    Setup: -Z axis is forward, X axis is rightward, Y axis is upward.
    theta - heading angle in radians --- measured clockwise from -Z to X.

    Compute quaternion that represents the corresponding clockwise rotation about Y axis.
    """
    # Real part
    q0 = math.cos(-theta / 2)
    # Imaginary part
    q = (0, math.sin(-theta / 2), 0)

    return np.quaternion(q0, *q)


def compute_egocentric_delta(p1, r1, p2, r2):
    """
    p1, p2 - (x, y, z) position
    r1, r2 - np.quaternions

    Compute egocentric change from (p1, r1) to (p2, r2) in
    the coordinates of (p1, r1)

    Setup: -Z axis is forward, X axis is rightward, Y axis is upward.
    """
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    theta_1 = compute_heading_from_quaternion(r1)
    theta_2 = compute_heading_from_quaternion(r2)

    D_rho = math.sqrt((x1 - x2) ** 2 + (z1 - z2) ** 2)
    D_phi = (
            math.atan2(x2 - x1, -z2 + z1) - theta_1
    )  # counter-clockwise rotation about Y from -Z to X
    D_theta = theta_2 - theta_1

    return (D_rho, D_phi, D_theta)


def compute_updated_pose(p, r, delta_xz, delta_y):
    """
    Setup: -Z axis is forward, X axis is rightward, Y axis is upward.

    p - (x, y, z) position
    r - np.quaternion
    delta_xz - (D_rho, D_phi, D_theta) in egocentric coordinates
    delta_y - scalar change in height

    Compute new position after a motion of delta from (p, r)
    """
    x, y, z = p
    theta = compute_heading_from_quaternion(
        r
    )  # counter-clockwise rotation about Y from -Z to X
    D_rho, D_phi, D_theta = delta_xz

    xp = x + D_rho * math.sin(theta + D_phi)
    yp = y + delta_y
    zp = z - D_rho * math.cos(theta + D_phi)
    pp = np.array([xp, yp, zp])

    thetap = theta + D_theta
    rp = compute_quaternion_from_heading(thetap)

    return pp, rp


def grow_projected_map(proj_map, local_map, iterations=2):
    """
    proj_map - (H, W, 2) map
    local_map - (H, W, 2) map

    channel 0 - 1 if occupied, 0 otherwise
    channel 1 - 1 if explored, 0 otherwise
    """
    proj_map = np.copy(proj_map)
    HEIGHT, WIDTH = proj_map.shape[:2]

    explored_local_mask = local_map[..., 1] == 1
    free_local_mask = (local_map[..., 0] == 0) & explored_local_mask
    occ_local_mask = (local_map[..., 0] == 1) & explored_local_mask

    # Iteratively expand multiple times
    for i in range(iterations):
        # Generate regions which are predictable

        # ================ Processing free space ===========================
        # Pick only free areas that are visible
        explored_proj_map = (proj_map[..., 1] == 1).astype(np.uint8) * 255
        free_proj_map = ((proj_map[..., 0] == 0) & explored_proj_map).astype(
            np.uint8
        ) * 255
        occ_proj_map = ((proj_map[..., 0] == 1) & explored_proj_map).astype(
            np.uint8
        ) * 255

        # For the first iteration, aggressively close holes
        if i == 0:
            close_kernel = np.ones((3, 3), np.uint8)
            for itr in range(5):
                free_proj_map = cv2.morphologyEx(
                    free_proj_map, cv2.MORPH_CLOSE, close_kernel
                )
            free_proj_map = (free_proj_map > 0).astype(np.uint8) * 255

        dilate_kernel = np.ones((7, 7), np.uint8)

        # Expand only GT free area
        for itr in range(2):
            free_proj_map_edges = cv2.Canny(free_proj_map, 50, 100)
            free_proj_map_edges_dilated = cv2.dilate(
                free_proj_map_edges, dilate_kernel, iterations=3
            )
            free_mask = (
                                (free_proj_map_edges_dilated > 0) | (free_proj_map > 0)
                        ) & free_local_mask
            free_proj_map = free_mask.astype(np.uint8) * 255

        # Dilate to include some occupied area
        free_proj_map = cv2.dilate(free_proj_map, dilate_kernel, iterations=1)
        free_proj_map = (free_proj_map > 0).astype(np.uint8)

        # Close holes
        close_kernel = np.ones((3, 3), np.uint8)
        free_proj_map = cv2.morphologyEx(free_proj_map, cv2.MORPH_CLOSE, close_kernel)

        # ================ Processing occupied space ===========================
        # For the first iteration, aggressively close holes
        if i == 0:
            close_kernel = np.ones((3, 3), np.uint8)
            for itr in range(3):
                occ_proj_map = cv2.morphologyEx(
                    occ_proj_map, cv2.MORPH_CLOSE, close_kernel
                )
            occ_proj_map = (occ_proj_map > 0).astype(np.uint8) * 255

        dilate_kernel = np.ones((3, 3), np.uint8)

        # Expand only GT occupied area
        for itr in range(1):
            occ_proj_map_edges = cv2.Canny(occ_proj_map, 50, 100)
            occ_proj_map_edges_dilated = cv2.dilate(
                occ_proj_map_edges, dilate_kernel, iterations=3
            )
            occ_mask = (
                               (occ_proj_map_edges_dilated > 0) | (occ_proj_map > 0)
                       ) & occ_local_mask
            occ_proj_map = occ_mask.astype(np.uint8) * 255

        dilate_kernel = np.ones((9, 9), np.uint8)
        # Expand the free space around the GT occupied area
        for itr in range(2):
            occ_proj_map_dilated = cv2.dilate(occ_proj_map, dilate_kernel, iterations=3)
            free_mask_around_occ = (occ_proj_map_dilated > 0) & free_local_mask
            occ_proj_map = ((occ_proj_map > 0) | free_mask_around_occ).astype(
                np.uint8
            ) * 255

        # Close holes
        close_kernel = np.ones((3, 3), np.uint8)
        occ_proj_map = cv2.morphologyEx(occ_proj_map, cv2.MORPH_CLOSE, close_kernel)

        # Include originally present areas in proj_map
        predictable_regions_mask = (
                (explored_proj_map > 0) | (free_proj_map > 0) | (occ_proj_map > 0)
        )

        # Create new proj_map
        proj_map = np.zeros((HEIGHT, WIDTH, 2), np.float32)
        proj_map[predictable_regions_mask & occ_local_mask, 0] = 1
        proj_map[predictable_regions_mask, 1] = 1

    gt_map = proj_map

    return gt_map


def get_random_point_igib(pathfinder, max_h=1.0):
    tries = 100
    while True:
        point = pathfinder.get_random_navigable_point()
        if point[1] < max_h:
            return point
        else:
            tries -= 1
        if tries < 0:
            return None


def check_percent_nav(sim):
    greedy_follower = ShortestPathFollower(sim, goal_radius=1.0, return_one_hot=False)
    greedy_follower.mode = "geodesic_path"

    scene_objects = sim.get_both_existing_object_ids_with_positions()
    navigable = []

    # check how many objects are navigable from this place
    for op in scene_objects["art_pos"] + scene_objects["non_art_pos"]:
        nop = sim.pathfinder.snap_point(op)
        act = greedy_follower.get_next_action(nop)
        is_nav = act != 0
        navigable.append(is_nav)
    percent_nav = np.array(navigable, dtype=np.float).sum() / len(navigable)
    return percent_nav


def set_agent_on_floor_igib(sim, max_h=1.0, min_nav=0.33):
    nav_points = get_all_nav_points(sim, max_h)
    random.shuffle(nav_points)

    for agent_pos in nav_points:
        if agent_pos[1] < max_h:
            agent_orientation_y = np.random.randint(0, 360)
            agent_rot_mn = mn.Quaternion.rotation(mn.Deg(agent_orientation_y), mn.Vector3(0, 1.0, 0))
            agent_rot = list(agent_rot_mn.vector) + [agent_rot_mn.scalar]
            agent_rot = quat_from_coeffs(agent_rot)
            sim.set_agent_state(agent_pos, agent_rot)
            perc_nav = check_percent_nav(sim)
            if perc_nav > min_nav:
                return agent_pos
    return None


def get_intersect_ratio(obj_id, rec_id, sim):
    obj_bb = get_bb(sim, obj_id)
    rec_bb = get_bb(sim, rec_id)
    obj_vol = get_vol(obj_bb)
    intersect_vol = get_vol(mn.math.intersect(obj_bb, rec_bb))
    intersect_factor = intersect_vol / obj_vol
    return intersect_factor


def add_object_on_receptacle(obj_id, rec_id, sim, recs_packers):
    org_trans = sim.get_translation(obj_id)
    point, should_rotate = get_surface_point_for_placement(sim, recs_packers, obj_id, rec_id, "pack")
    # can't fit the object on receptacle
    if not point:
        sim.set_translation(org_trans, obj_id)
        return False

    sim.set_translation(point, obj_id)
    if should_rotate:
        obj_node = sim.get_object_scene_node(obj_id)
        obj_node.rotate_y(mn.Rad(math.pi/2))

    # num_tries = 20
    # translation = sim.get_translation(obj_id)
    # prev_translation = deepcopy(translation)
    # while not sim.contact_test(obj_id) and num_tries > 0:
    #     translation[1] -= 0.05
    #     num_tries -= 1
    #     prev_translation = deepcopy(translation)
    #     sim.set_translation(translation, obj_id)
    #
    # if sim.contact_test(obj_id):
    #     print(f"Dropped from {point} to {prev_translation}")
    #     sim.set_translation(prev_translation, obj_id)
    # else:
    #     sim.set_translation(point, obj_id)

    # test to check collision
    # timeout_tries = 100
    # while (get_intersect_ratio(obj_id, rec_id, sim) > 0.1 and (timeout_tries > 0)):
    #     point[1] += 0.01
    #     sim.set_translation(point, obj_id)
    #     timeout_tries -= 1

    # success = timeout_tries > 0
    # if not success:
    #     sim.set_translation(org_trans, obj_id)

    # return success
    return True


def get_vol(obj_bb):
    bb_size = list(obj_bb.size())
    return np.prod(bb_size)


def get_surface_point_for_placement(sim, recs_packers, obj_id, rec_id, type="center"):
    rec_node = sim.get_object_scene_node(rec_id)
    rec_bb = rec_node.cumulative_bb
    should_rotate = False

    if type == "pack":
        rec_bb = get_bb(sim, rec_id)
        obj_bb = get_bb(sim, obj_id)
        y_max = rec_bb.y().max
        obj_bb_base = get_bb_base(obj_bb)
        obj_base_rect = Rect(obj_id, obj_bb_base)
        place_match = recs_packers[rec_id].insert(obj_base_rect)
        if place_match is None:
            return None, False
        new_bb_base = place_match.rect.range
        # vertical_ray = Ray()
        # vertical_ray.direction = mn.Vector3(0, -1, 0)
        # rec_center = recs_packers[rec_id].dims.center()
        # vertical_ray.origin = mn.Vector3([rec_center[0], y_max + 0.1, rec_center[1]])
        # raycast = sim.cast_ray(vertical_ray, max_distance=2.0)
        # hits = [hit.point[1] for hit in raycast.hits if hit.object_id == rec_id]
        # if len(hits) > 0:
        #     print(f"Changing y-max from: {y_max} to {min(hits)}")
        #     y_max = min(hits)

        point = [
            (new_bb_base.x().min + new_bb_base.x().max) / 2,
            y_max + obj_bb.size_y() / 2,
            (new_bb_base.y().min + new_bb_base.y().max) / 2
        ]
        should_rotate = place_match.rect.is_rotated

    # elif type == "random":
    #     # todo: think if we need to send bboxes of already present object
    #     corners = get_corners(rec_bb)
    #     corners = [rec_node.transformation.transform_point(cor) for cor in corners]
    #     corners = [[round(cor.x, 6), round(cor.y, 6), round(cor.z, 6)] for cor in corners]
    #     y_max = max([x[1] for x in corners])
    #     surface_corners = [[x[0], x[2]] for x in corners if abs(x[1] - y_max) < 1e-4]  # equate doesn't work
    #     assert len(surface_corners) == 4
    #     height_noise = 0
    #     position_generator = PolySurface(y_max, surface_corners, height_noise)
    #     old_pos = [0, 0, 0]  # not used anywhere
    #     point = get_sampled_obj(sim, episode, task, position_generator, old_pos, obj_id, rec_id)
    #
    elif type == "center":
        corners = get_corners(rec_bb)
        corners = [rec_node.transformation.transform_point(cor) for cor in corners]
        corners = [[cor.x, cor.y, cor.z] for cor in corners]
        point = get_surface_center_from_corners(corners)

    else:
        raise AssertionError

    # point is None denotes cannot place given object
    return point, should_rotate


def get_bb_base(bb):
    bb_xz_min = [bb.min[0], bb.min[2]]
    bb_xz_max = [bb.max[0], bb.max[2]]
    bb_base = mn.Range2D(bb_xz_min, bb_xz_max)
    return bb_base


def get_bb(sim, obj_id):
    obj_node = sim.get_object_scene_node(obj_id)
    obj_bb = obj_node.cumulative_bb
    corners = get_corners(obj_bb, obj_node)
    tranformed_bb = get_bbs_from_corners(corners)
    return tranformed_bb


def get_semantic_centroids(semantic_obs):
    iids = list(np.unique(semantic_obs))
    if 0 in iids:
        iids.remove(0)
    iid_centroids = []
    for iid in iids:
        one_hot = (semantic_obs == iid)
        xis, yis = np.nonzero(one_hot)
        iid_centroids.append([xis.mean(), yis.mean()])

    return iids, iid_centroids


def get_corners(obj_bb, obj_node=None):
    corners = ["back_top_right", "back_top_left", "back_bottom_right", "back_bottom_left",
               "front_top_right", "front_top_left", "front_bottom_right", "front_bottom_left"]
    surface_corners = [getattr(obj_bb, cor) for cor in corners]

    if obj_node is not None:
        surface_corners = [obj_node.transformation.transform_point(cor) for cor in surface_corners]

    return surface_corners


def get_bbs_from_corners(cors):
    min_x, min_y, min_z = [1e3] * 3
    max_x, max_y, max_z = [-1e3] * 3

    for cor in cors:
        max_x = cor.x if cor.x > max_x else max_x
        max_y = cor.y if cor.y > max_y else max_y
        max_z = cor.z if cor.z > max_z else max_z

        min_x = cor.x if cor.x < min_x else min_x
        min_y = cor.y if cor.y < min_y else min_y
        min_z = cor.z if cor.z < min_z else min_z

    min_coords = mn.Vector3(min_x, min_y, min_z)
    size = mn.Vector3(max_x - min_x, max_y - min_y, max_z - min_z)
    bb = mn.Range3D.from_size(min_coords, size)
    return bb


def get_surface_center_from_corners(corners):
    min_cor = [100000000] * 3
    max_cor = [-100000000] * 3

    for cor in corners:
        for i, v in enumerate(cor):
            if v > max_cor[i]:
                max_cor[i] = v

            if v < min_cor[i]:
                min_cor[i] = v

    surface_center = [(min_cor[0] + max_cor[0]) / 2, max_cor[1], (min_cor[2] + max_cor[2]) / 2]
    return surface_center


def get_sampled_obj(sim, episode, task, pos_gen, pos, obj_id, rec_id, min_base_overlap=0.6, restrict_bbs=None):
    """obj_id for the object to be placed on receptacle"""
    timeout_tries = 100
    found = False

    if restrict_bbs is None:
        # obj_ids = sim.get_both_existing_object_ids()
        # obj_ids = obj_ids["art"] + obj_ids["non_art"]
        # obj_ids = list(set(obj_ids) - {obj_id, rec_id})
        obj_ids = episode.get_objects_on_rec(task, rec_id)
        # print("new num obj ids", len(obj_ids))
        restrict_bbs = [get_bb(sim, oid) for oid in obj_ids]

    rec_bb = get_bb(sim, rec_id)
    rec_bb_base = get_bb_base(rec_bb)
    # rec_base_area = np.prod(rec_bb_base.size())

    # Keep trying until we get a non-overlapping placement.
    # for i in tqdm(range(timeout_tries), total=timeout_tries, desc="sampling"):
    for i in range(timeout_tries):
        new_pos = pos_gen.sample(pos, obj_id)
        sim.set_translation(new_pos, obj_id)
        bb = get_bb(sim, obj_id)
        bb_base = get_bb_base(bb)
        bb_base_area = np.prod(bb_base.size())
        # print("rec", rec_bb_base, "obj", bb_base, "point", new_pos)

        # check if the base polygon has a sufficient overlap with receptacle surface poly
        overlap_area = np.prod(mn.math.intersect(rec_bb_base, bb_base).size())
        if overlap_area / bb_base_area < min_base_overlap:
            # print(f"overlap base ratio: {overlap_area / bb_base_area}")
            continue

        # check for collisions
        if inter_any_bb(bb, restrict_bbs):
            # print(f"overlap base ratio: {overlap_area / bb_base_area} // collided")
            continue

        # print(f"overlap base ratio: {overlap_area / bb_base_area} // not collided")
        found = True
        break

    if not found:
        # print('E: Could not get a valid position for %i with %s' % (obj_id, str(pos_gen)))
        return None

    # also provide proper vertical offset
    if pos_gen.should_add_offset():
        # new_pos[1] += (bb.size()[1] / 2.0) + 0.01
        new_pos[1] += rec_bb.max[1] - bb.min[1] + 0.01
    return new_pos


def inter_any_bb(bb0, bbs):
    for bb in bbs:
        if mn.math.intersects(bb0, bb):
            return True
    return False


# def get_nav_point(sim, point, max_dist, max_tries=100):
#     greedy_follower = ShortestPathFollower(sim, goal_radius=max_dist, return_one_hot=False)
#     greedy_follower.mode = "geodesic_path"
#
#     ap = sim.get_agent_state().position
#     pathfinder = sim.pathfinder
#     nop = pathfinder.snap_point(point)
#     act = greedy_follower.get_next_action(nop)
#     dist = sim.get_dist_pos(point, ap, 'geo')
#
#     if act == 0 and dist > max_dist:
#         num_tries = max_tries
#         while num_tries > 0:
#             random_point = pathfinder.get_random_navigable_point()
#             if sim.get_dist_pos(ap, random_point, "geo") <= max_dist:
#                 act = greedy_follower.get_next_action(random_point)
#                 if act != 0:
#                     nop = random_point
#                     break
#             num_tries = num_tries - 1
#
#     return nop, act != 0


def get_all_nav_points(sim, height_thresh=1.0):
    from collections import defaultdict, Counter
    nav_vs = np.array(sim.pathfinder.build_navmesh_vertices())
    nav_vs = nav_vs[nav_vs[:, 1] < height_thresh]
    nav_vs_r = np.array([sim.pathfinder.island_radius(nav_v) for nav_v in nav_vs])
    n_count_common = Counter(nav_vs_r).most_common()
    # largest_island_inds = nav_vs_r == np.max(nav_vs_r)
    # if np.max(nav_vs_r) != n_count_common[0][0]:
    #     import pdb
    #     pdb.set_trace()
    largest_island_inds = nav_vs_r == n_count_common[0][0]
    # nav_vs_h = np.array([v[1] for v in nav_vs])
    # largest_island_heights = nav_vs_h[largest_island_inds]
    use_vs = nav_vs[largest_island_inds]
    return use_vs


def get_random_nav_point(sim, height_thresh=1.0):
    nav_points = get_all_nav_points(sim, height_thresh)
    sel_i = np.random.randint(len(nav_points))
    point = nav_points[sel_i]
    return point


def get_closest_nav_point(sim, point, ignore_y=True, return_all=False):
    nav_points_org = get_all_nav_points(sim)
    nav_points = deepcopy(nav_points_org)
    if ignore_y:
        nav_points[:, 1] = point[1]
    closest_ind = closest_point(point, nav_points, return_all)
    return nav_points_org[closest_ind]

def closest_point(point, points, return_all=False):
    """Returns index of closest point given a list of points and point"""
    if type(points) == list:
        points = np.array(points)
    if type(point) == list:
        point = np.array(point)
    dist_2 = np.sum((points - point) ** 2, axis=1)
    if return_all:
        return np.argsort(dist_2, axis=0)
    else:
        return np.argmin(dist_2)


def extract_sensors(obs):
    refactor_sensors = [
        "gps_compass",
        "visible_obj_iids",
        "visible_obj_sids",
        "visible_rec_iids",
        "visible_rec_sids",
        "semantic_class",
        "gripped_object_id",
        "gripped_iid",
        "gripped_sid",
        "num_visible_objs",
        "num_visible_recs",
    ]
    skip_delete = [
        "gripped_object_id"
    ]
    for sensor in refactor_sensors:
        if sensor not in obs["cos_eor"]:
            print(f"Skipping: {sensor}, didn't find")
        obs[sensor] = obs["cos_eor"][sensor]
        if sensor not in skip_delete:
            del obs["cos_eor"][sensor]
    return obs
