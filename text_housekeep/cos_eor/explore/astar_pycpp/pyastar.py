import ctypes
import numpy as np

import inspect
from os.path import abspath, dirname, join

fname = abspath(inspect.getfile(inspect.currentframe()))
lib = ctypes.cdll.LoadLibrary(join(dirname(fname), "astar.so"))

astar = lib.astar
ndmat_f_type = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
ndmat_i_type = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
astar.restype = ctypes.c_bool
astar.argtypes = [
    ndmat_f_type,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_bool,
    ndmat_i_type,
]


def astar_path(obmap, start, goal, allow_diagonal=False):
    # Ensure start is within bounds.
    if (
        start[0] < 0
        or start[0] >= obmap.shape[0]
        or start[1] < 0
        or start[1] >= obmap.shape[1]
    ):
        raise ValueError("Start of (%d, %d) lies outside grid." % (start))
    # Ensure goal is within bounds.
    if (
        goal[0] < 0
        or goal[0] >= obmap.shape[0]
        or goal[1] < 0
        or goal[1] >= obmap.shape[1]
    ):
        raise ValueError("Goal of (%d, %d) lies outside grid." % (goal))

    height, width = obmap.shape
    start_idx = np.ravel_multi_index(start, (height, width))
    goal_idx = np.ravel_multi_index(goal, (height, width))

    # The C++ code writes the solution to the paths array
    paths = np.full(height * width, -1, dtype=np.int32)
    success = astar(
        obmap.flatten(),
        height,
        width,
        start_idx,
        goal_idx,
        allow_diagonal,
        paths,  # output parameter
    )
    if not success:
        return np.array([])

    coordinates = []
    path_idx = goal_idx
    while path_idx != start_idx:
        pi, pj = np.unravel_index(path_idx, (height, width))
        coordinates.append((pi, pj))

        path_idx = paths[path_idx]

    if coordinates:
        coordinates.append(np.unravel_index(start_idx, (height, width)))
        return np.vstack(coordinates[::-1])
    else:
        return np.array([])


def astar_planner(obmap, start, goal, allow_diagonal=False):
    """
    start - (x, y) coordinates
    goal - (x, y) coordinates

    Returns:
        path_x, path_y - a list of x, y coordinates 
                         starting from GOAL to the START
    """
    # astar_path requires (y, x) as input
    path_start = (start[1], start[0])
    path_goal = (goal[1], goal[0])

    path = astar_path(obmap, path_start, path_goal, allow_diagonal=allow_diagonal)
    if path.shape[0] > 0:
        path_y = path[:, 0].tolist()[::-1]
        path_x = path[:, 1].tolist()[::-1]
    else:
        path_y = None
        path_x = None

    return path_x, path_y
