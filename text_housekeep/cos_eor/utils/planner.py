import itertools

import numpy as np
import skfmm

from numpy import ma
from functools import partial
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from text_housekeep.cos_eor.utils.geometry import geodesic_distance
from text_housekeep.cos_eor.utils.visualization import (
    get_top_down_map
)
from text_housekeep.habitat_lab.habitat.utils.visualizations import maps, fog_of_war


def compute_traversable_map(top_down_map, fog_of_war_mask):
    """
    Compute traversable map based on GT top down map and fog of war.
    Needs a single channel top down map, 
    with navigable area = 255 and obstacles = 0
    """
    
    # Invert the fog of war so that regions outside can be considered
    # free space.
    fog_of_war_mask = (fog_of_war_mask * 255/np.max(fog_of_war_mask)).astype(np.int8)
    invert_fow = np.invert(fog_of_war_mask)

    # Combine fog of war with the top down map
    tdmap = np.logical_or(top_down_map, invert_fow)
    return tdmap


def compute_distance_using_fmm(traversible_map, anchor_point):
    """
        Compute distance from every point in the map to the anchor point. 
    """
    # all 0 values represents occlusion. 
    traversible_ma = ma.masked_values(traversible_map * 1, 0)

    traversible_ma[anchor_point[0], anchor_point[1]] = 0
    dmap = skfmm.distance(traversible_ma, dx=1)
    
    # set unreachable points to inf
    dmap_mask = np.invert(np.isnan(ma.filled(dmap, np.nan)))
    dmap = ma.filled(dmap, np.inf)
    
    return dmap


def find_dist_from_map(dmap, point):
    """
        Returns the distance of the anchor point from a given point. 
    """
    return dmap[point[0], point[1]]


def compute_distance_mat_using_fmm(tdmap, agent_pos, object_positions, goal_positions):
    """
        Compute shortest path distances between all pairs using fmm planner on top-down-map
    """
    shortest_path_lengths = []
    for i, p1 in enumerate([agent_pos] + object_positions + goal_positions):
        dmap = compute_distance_using_fmm(tdmap, p1)
        path_lengths = []
        for j, p2 in enumerate([agent_pos] + object_positions + goal_positions):
            if j == 0:
                path_lengths.append(0)
            else:
                path_lengths.append(find_dist_from_map(dmap, p2))
        shortest_path_lengths.append(path_lengths)

    return np.array(shortest_path_lengths)


def compute_distance_mat_using_navmesh(pathfinder, agent_pos, object_position, goal_position):
    """
        Computes shortest path distances between all pairs using GT geodiesic distance from navmesh
    """

    data = {}
    dist_mat = np.zeros((   
            1 + len(object_position) + len(goal_position), 
            1 + len(object_position) + len(goal_position)
    ))
    
    for i, object_pos in enumerate([agent_pos] + object_position + goal_position):
        for j, goal_pos in enumerate([agent_pos] + object_position + goal_position):
            if j == 0:  # distance from object / goal position -> "depot" is zero
                dist = 0
            else:
                dist = geodesic_distance(
                    pathfinder, object_pos, [goal_pos]
                )
            dist_mat[i][j] = dist

    return dist_mat


def create_data_model(dist_mat):
    data = {}        
    data['distance_matrix'] = dist_mat.tolist()
    number_of_objects = int((dist_mat.shape[0] - 1)/2)
    data['pickups_deliveries'] = [
        [i+1, number_of_objects+i+1] for i in range(number_of_objects)
    ]
    data['demands'] = [0] + [1]*number_of_objects + [-1]*number_of_objects
    data['vehicle_capacities'] = [1]
    data['num_vehicles'] = 1
    data['depot'] = 0

    return data


def print_solution(data, manager, routing, assignment):
    """Prints assignment on console."""
    # Display dropped nodes.
    dropped_nodes = 'Dropped nodes:'
    for node in range(routing.Size()):
        if routing.IsStart(node) or routing.IsEnd(node):
            continue
        if assignment.Value(routing.NextVar(node)) == node:
            dropped_nodes += ' {}'.format(manager.IndexToNode(node))
    # print(dropped_nodes)
    
    # Display routes
    total_distance = 0
    total_load = 0
    route_indexes = []

    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
            route_indexes.append(node_index)
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        
        plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index),
                                                 route_load)
        plan_output += 'Distance of the route: {}\n'.format(route_distance)
        plan_output += 'Load of the route: {}\n'.format(route_load)
        # print(plan_output)
        total_distance += route_distance
        total_load += route_load
    
    # print('Total Distance of all routes: {}m'.format(total_distance))
    # print('Total Load of all routes: {}'.format(total_load))
    return route_indexes


def distance_callback(manager, data, from_index, to_index):
    """Returns the distance between the two nodes."""
    # Convert from routing variable Index to distance matrix NodeIndex.
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return data['distance_matrix'][from_node][to_node]


def demand_callback(manager, data, from_index):
    """Returns the demand of the node."""
    # Convert from routing variable Index to demands NodeIndex.
    from_node = manager.IndexToNode(from_index)
    return data['demands'][from_node]
        
        
def find_shortest_path_for_multiple_objects(dist_mat):
    data = create_data_model(dist_mat)
    dist_mat = np.array(data['distance_matrix'])
    dist_max = dist_mat.max()
    dist_sum = dist_mat.sum()

    if dist_max == np.inf:
        return None, None 
        
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)


    distance_callback_fn = partial(distance_callback, manager, data)
    demand_callback_fn = partial(demand_callback, manager, data)

    # Add pickup / delivery constraints
    transit_callback_index = routing.RegisterTransitCallback(distance_callback_fn)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        3000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    for request in data['pickups_deliveries']:
            pickup_index = manager.NodeToIndex(request[0])
            delivery_index = manager.NodeToIndex(request[1])
            routing.AddPickupAndDelivery(pickup_index, delivery_index)
            routing.solver().Add(
                routing.VehicleVar(pickup_index) == routing.VehicleVar(
                    delivery_index))
            routing.solver().Add(
                distance_dimension.CumulVar(pickup_index) <=
                distance_dimension.CumulVar(delivery_index))
    # -------

    # Add constraint so that at max one object can be picked up. 
    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback_fn
    )

    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')
    # -------

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

        # Print solution on console.
    if solution:
        route_indexes = print_solution(data, manager, routing, solution)
        # skip the agent and the receptacles
        pickup_order = route_indexes[1:][::2]
        return route_indexes, pickup_order

    return None, None


def compute_oracle_pickup_order(env):
    # obs = env.reset()
    episode = env.current_episode

    agent_pos = env._sim.get_agent(0).get_state().position
    object_positions = [obj.position for obj in episode.objects]
    goal_positions = [obj.position for obj in episode.goals]
    
    dist_mat = compute_distance_mat_using_navmesh(env._task._simple_pathfinder, agent_pos, object_positions, goal_positions)
    route_indexes_navmesh, pickup_order_navmesh = find_shortest_path_for_multiple_objects(dist_mat)
    
    return {
        'episode_id': episode.episode_id, 
        'scene_id': episode.scene_id,
        'pickup_order': pickup_order_navmesh,
        'dist_mat': dist_mat
    }


def compute_path_dist(path, dist_mat):
    dist = 0
    for j in range(len(path) - 1):
        dist += dist_mat[path[j]][path[j + 1]]
    return dist

def get_permutation(episode, dist_mat):
    "returns permutations with all the different source and sinks possible"
    def idx_of(k, rs):
        for i in range(len(rs)):
            if rs[i]["key"] == k:
                return i

    ao_len = 1 + len(episode["objects"])
    keys_list = [r["paired_recs_keys"] for r in episode["objects"]]
    rec_perms = [p for p in itertools.product(*keys_list)]
    rec_indxs = []
    for perm in rec_perms:
        idx_perm = []
        for k in perm:
            # we offset all receptacle indices by (ao_len)
            idx_perm.append(idx_of(k, episode["receptacles"]) + ao_len)
        rec_indxs.append(idx_perm)

    dim = 1 + len(episode["objects"]) + len(episode["objects"])
    dist_matrices = np.empty([len(rec_perms), dim, dim])

    for i, inds in enumerate(rec_indxs):
        indices = list(range(ao_len)) + inds
        for j in range(dist_matrices.shape[1]):
            for k in range(dist_matrices.shape[2]):
                dist_matrices[i, j, k] = dist_mat[indices[j], indices[k]]

    return rec_perms, rec_indxs, dist_matrices


def compute_l2_dist_mat(agent_pos, object_positions, goal_positions):
    dist_mat = np.zeros((
        1 + len(object_positions) + len(goal_positions),
        1 + len(object_positions) + len(goal_positions)
    ))

    for i, object_pos in enumerate([agent_pos] + object_positions + goal_positions):
        for j, goal_pos in enumerate([agent_pos] + object_positions + goal_positions):
            if j == 0:  # distance from object / goal position -> "depot" is zero
                dist = 0
            else:
                dist = np.linalg.norm(
                    np.array(object_pos) - np.array(goal_pos), ord=2
                )
            dist_mat[i][j] = dist

    return dist_mat


def compute_closest_dist_mat(agent_pos, object_positions, goal_positions):
    dist_mat = np.zeros((
        1 + len(object_positions) + len(goal_positions),
        1 + len(object_positions) + len(goal_positions)
    ))

    for i, object_pos in enumerate([agent_pos] + object_positions + goal_positions):
        for j, goal_pos in enumerate([agent_pos] + object_positions + goal_positions):
            if j == 0:  # distance from object / goal position -> "depot" is zero
                dist = 0
            else:
                dist = np.linalg.norm(
                    np.array(object_pos) - np.array(goal_pos), ord=2
                )
            dist_mat[i][j] = dist

    return dist_mat


def compute_pickup_orders(pathfinder, agent_pos, object_positions, goal_positions, episode, types=["gd", "cl", "l2"]):
    """todo: verify again."""
    results = {}

    if "gd" in types:
        # geodesic
        gd_dist_mat = compute_distance_mat_using_navmesh(pathfinder, agent_pos, object_positions, goal_positions)
        gd_rec_perms, gd_rec_indxs, gd_dist_matrices = get_permutation(episode, gd_dist_mat)
        gd_pickup_orders = []
        gd_dists = []
        for i in range(gd_dist_matrices.shape[0]):
            route_indexes_navmesh, pickup_order_navmesh = find_shortest_path_for_multiple_objects(gd_dist_matrices[i])
            dist = compute_path_dist(route_indexes_navmesh, gd_dist_matrices[i])
            gd_dists.append(dist)
            gd_pickup_orders.append(pickup_order_navmesh)
        min_idx = gd_dists.index(min(gd_dists))

        results.update({
            'gd_pickup_order': gd_pickup_orders[min_idx],
            # 'gd_dist_matrix': gd_dist_matrices[min_idx],
            'gd_recs_key': gd_rec_perms[min_idx],
            "gd_recs_ind": gd_rec_indxs[min_idx],
            'gd_dist': min(gd_dists)
        })

    if "l2" in types:
        # l2
        l2_dist_mat = compute_l2_dist_mat(agent_pos, object_positions, goal_positions)
        l2_rec_perms, l2_rec_indxs, l2_dist_matrices = get_permutation(episode, l2_dist_mat)
        l2_pickup_orders = []
        l2_dists = []
        for i in range(l2_dist_matrices.shape[0]):
            route_indexes_navmesh, pickup_order_navmesh = find_shortest_path_for_multiple_objects(l2_dist_matrices[i])
            dist = compute_path_dist(route_indexes_navmesh, l2_dist_matrices[i])
            l2_dists.append(dist)
            l2_pickup_orders.append(pickup_order_navmesh)
        min_idx = l2_dists.index(min(l2_dists))
        results.update({
            'l2_pickup_order': l2_pickup_orders[min_idx],
            # 'l2_dist_matrix': l2_dist_matrices[min_idx],
            'l2_recs_key': l2_rec_perms[min_idx],
            'l2_recs_ind': l2_rec_indxs[min_idx],
            'l2_dist': min(l2_dists)
        })

    if "cl" in types:
        # closest
        cl_dist_mat = compute_closest_dist_mat(agent_pos, object_positions, goal_positions)
        cl_rec_perms, cl_rec_indxs, cl_dist_matrices = get_permutation(episode, cl_dist_mat)
        cl_pickup_orders = []
        cl_dists = []
        for j in range(cl_dist_matrices.shape[0]):
            prev_idx = 0
            cl_pickup_order = []
            route_indexes_navmesh = [0]

            while len(cl_pickup_order) != len(object_positions):
                min_dist = 10000
                min_idx = -1
                for i in range(1, len(object_positions) + 1):
                    l2dist = cl_dist_matrices[j, prev_idx, i]
                    if l2dist < min_dist and (i not in cl_pickup_order):
                        min_dist = l2dist
                        min_idx = i
                cl_pickup_order.append(min_idx)
                prev_idx = min_idx + len(object_positions)
                route_indexes_navmesh.extend([min_idx, prev_idx])
            dist = compute_path_dist(route_indexes_navmesh, cl_dist_matrices[j])
            cl_dists.append(dist)
            cl_pickup_orders.append(cl_pickup_order)

        min_idx = cl_dists.index(min(cl_dists))
        results.update({
            'cl_pickup_order': cl_pickup_orders[min_idx],
            # 'cl_dist_matrix': cl_dist_matrices[min_idx],
            'cl_recs_key': cl_rec_perms[min_idx],
            "cl_recs_ind": cl_rec_indxs[min_idx],
            'cl_dist': min(cl_dists)
        })

    return results


def compute_oracle_pickup_order_using_fmm(env):
    # obs = env.reset()
    metrics = env.get_metrics()
    episode = env.current_episode
    metrics = env.get_metrics()

    agent_pos = env._sim.get_agent(0).get_state().position
    object_positions = [obj.position for obj in episode.objects]
    goal_positions = [obj.position for obj in episode.goals]
    
    top_down_map, fog_of_war_mask = get_top_down_map(
        env, env._task._simple_pathfinder, ignore_objects=True, fog_of_war_mask=None, draw_fow=False, 
        draw_agent=False, draw_object_start_pos=False, draw_object_final_pos=False, draw_object_curr_pos=False
    )
    
    
    a_y, a_x = maps.to_grid(
        agent_pos[2],
        agent_pos[0],
        top_down_map.shape[0:2],
        sim=env._sim,
    )
    grid_object_positions = []
    grid_goal_positions = []

    for i, obj_pos in enumerate(object_positions):
        tdm_pos = maps.to_grid(
            obj_pos[2],
            obj_pos[0],
            top_down_map.shape[0:2],
            sim=env._sim,
        )
        grid_object_positions.append(tdm_pos)

    for i, goal_pos in enumerate(goal_positions):
        tdm_pos = maps.to_grid(
            goal_pos[2],
            goal_pos[0],
            top_down_map.shape[0:2],
            sim=env._sim,
        )

        grid_goal_positions.append(tdm_pos)

    tdmap = np.copy(top_down_map[:, :, 0])
    tdmap = tdmap / np.max(tdmap)
    dist_mat_map = compute_distance_mat_using_fmm(tdmap, [a_y, a_x], grid_object_positions, grid_goal_positions)
    
    
    route_indexes_map, pickup_order_map = find_shortest_path_for_multiple_objects(dist_mat_map/3)
    
    return {
        'episode_id': episode.episode_id, 
        'scene_id': episode.scene_id,
        'pickup_order_fmm': pickup_order_map,
        'dist_mat': dist_mat_map
        # 'pickup_order_fmm': pickup_order_map 
    }


def compute_l2dist_pickup_order_sim(sim, agent_pos, object_positions, goal_positions):
    dist_mat = np.zeros((   
            1 + len(object_positions) + len(goal_positions), 
            1 + len(object_positions) + len(goal_positions)
    ))
    
    for i, object_pos in enumerate([agent_pos] + object_positions + goal_positions):
        for j, goal_pos in enumerate([agent_pos] + object_positions + goal_positions):
            if j == 0:  # distance from object / goal position -> "depot" is zero
                dist = 0
            else:
                dist = np.linalg.norm(
                    np.array(object_pos) - np.array(goal_pos), ord=2
                )
            dist_mat[i][j] = dist
    
    route_indexes_map, pickup_order_map = find_shortest_path_for_multiple_objects(dist_mat)
    
    return {
#         'pickup_order': pickup_order_navmesh,
        'l2_pickup_order': pickup_order_map,
        'dist_mat': dist_mat
    }


def compute_closest_pickup_order_sim(sim, agent_pos, object_positions, goal_positions):
    dist_mat = np.zeros((   
            1 + len(object_positions) + len(goal_positions), 
            1 + len(object_positions) + len(goal_positions)
    ))
    for i, object_pos in enumerate([agent_pos] + object_positions + goal_positions):
        for j, goal_pos in enumerate([agent_pos] + object_positions + goal_positions):
            if j == 0:  # distance from object / goal position -> "depot" is zero
                dist = 0
            else:
                dist = np.linalg.norm(
                    np.array(object_pos) - np.array(goal_pos), ord=2
                )
            dist_mat[i][j] = dist
    
    prev_idx = 0
    pickup_order = []
    while len(pickup_order) != len(object_positions):
        min_dist = 10000
        min_idx = -1
        for i in range(1, len(object_positions) + 1):
            l2dist = dist_mat[prev_idx, i]
            if l2dist < min_dist and (i not in pickup_order):
                min_dist = l2dist
                min_idx = i
        pickup_order.append(min_idx)
        prev_idx = min_idx + len(object_positions)
    return {
        'cl_pickup_order': pickup_order,
        'dist_mat': dist_mat
    }


# def compute_l2dist_pickup_order(env):
#     # obs = env.reset()
#     metrics = env.get_metrics()
#     episode = env.current_episode
#     metrics = env.get_metrics()
#
#     agent_pos = env._sim.get_agent(0).get_state().position
#     object_positions = [obj.position for obj in episode.objects]
#     goal_positions = [obj.position for obj in episode.goals]
#
#     dist_mat = np.zeros((
#             1 + len(object_positions) + len(goal_positions),
#             1 + len(object_positions) + len(goal_positions)
#     ))
#
#     for i, object_pos in enumerate([agent_pos] + object_positions + goal_positions):
#         for j, goal_pos in enumerate([agent_pos] + object_positions + goal_positions):
#             if j == 0:  # distance from object / goal position -> "depot" is zero
#                 dist = 0
#             else:
#                 dist = np.linalg.norm(
#                     np.array(object_pos) - np.array(goal_pos), ord=2
#                 )
#             dist_mat[i][j] = dist
#
#     route_indexes_map, pickup_order_map = find_shortest_path_for_multiple_objects(dist_mat)
#
#     return {
#         'episode_id': episode.episode_id,
#         'scene_id': episode.scene_id,
# #         'pickup_order': pickup_order_navmesh,
#         'l2_pickup_order': pickup_order_map
#     }


def start_env_episode_distance_sim(sim, pathfinder, agent_pos, object_positions, goal_positions, pickup_order):

    prev_obj_end_pos = agent_pos
    shortest_dist = 0

    for i in range(len(pickup_order)):
        curr_idx = pickup_order[i] - 1
        curr_obj_start_pos = object_positions[curr_idx]
        curr_obj_end_pos = goal_positions[curr_idx]
        shortest_dist += geodesic_distance(
                pathfinder, prev_obj_end_pos, [curr_obj_start_pos]
            )

        shortest_dist += geodesic_distance(
                    pathfinder, curr_obj_start_pos, [curr_obj_end_pos]
                )
        prev_obj_end_pos = curr_obj_end_pos

    return shortest_dist