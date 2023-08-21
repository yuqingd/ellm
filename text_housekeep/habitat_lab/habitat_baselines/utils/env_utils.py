#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
import time
from collections import defaultdict
from typing import List, Type, Union

import text_housekeep.habitat_lab.habitat
from text_housekeep.habitat_lab.habitat import Config, Env, RLEnv, VectorEnv, make_dataset
from text_housekeep.habitat_lab.habitat.core.registry import registry
from tqdm import tqdm


def make_env_fn(
    config: Config, env_class: Union[Type[Env], Type[RLEnv]]
) -> Union[Env, RLEnv]:
    r"""Creates an env of type env_class with specified config and rank.
    This is to be passed in as an argument when creating VectorEnv.

    Args:
        config: root exp config that has core env config node as well as
            env-specific config node.
        env_class: class type of the env to be created.

    Returns:
        env object created according to specification.
    """
    dataset = make_dataset(
        config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET
    )
    env = env_class(config=config, dataset=dataset)
    env.seed(config.TASK_CONFIG.SEED)
    return env


def construct_envs(
    config: Config,
    env_class: Type[Union[Env, RLEnv]],
    workers_ignore_signals: bool = False,
) -> VectorEnv:
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.

    :param config: configs that contain num_processes as well as information
    :param necessary to create individual environments.
    :param env_class: class type of the envs to be created.
    :param workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor

    :return: VectorEnv object created according to specification.
    """

    # YK: There is lot of ugly code here, the dataset init is being called
    #  multiple times w/o particular reason

    num_processes = config.NUM_PROCESSES
    configs = []
    env_classes = [env_class for _ in range(num_processes)]
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    if "*" in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)

    if num_processes > 1:
        if len(scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multiple process logic relies on being able to split scenes uniquely between processes"
            )

        if len(scenes) < num_processes:
            logging.log(level=logging.WARNING, msg=
                f"you might want to reduce the number of processes as there "
                f"aren't enough number of scenes "
                f"// processes: {num_processes} and scenes: {len(scenes)}"
            )

        random.shuffle(scenes)

    # hack to run overfit faster
    if len(scenes) < num_processes:
        repeat_scenes = num_processes // len(scenes)
        scenes = scenes * repeat_scenes
        assert len(scenes) >= num_processes

    scene_splits = [[] for _ in range(num_processes)]
    for idx, scene in enumerate(scenes):
        scene_splits[idx % len(scene_splits)].append(scene)

    assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_processes):
        proc_config = config.clone()
        proc_config.defrost()

        task_config = proc_config.TASK_CONFIG
        task_config.SEED = task_config.SEED + i
        if len(scenes) > 0:
            task_config.DATASET.CONTENT_SCENES = scene_splits[i]

        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
            config.SIMULATOR_GPU_ID[i % len(config.SIMULATOR_GPU_ID)]
        )

        task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS

        proc_config.freeze()
        configs.append(proc_config)
    # task_config.DATASET.CONTENT_SCENES now contain all scenes splits

    if registry.mapping["debug"]:
        env = make_env_fn(configs[0], env_classes[0])
        env.reset()
        # t = task_env._env._task; t.rec_packers[2].shelves =[]
        from cos_eor.utils.debug import debug_viewer
        debug_viewer(env)
        # x = env.step(action={"action": 3, "action_args": {"iid": -1}})
        import pdb
        pdb.set_trace()
        del env

    if registry.mapping["time"]:
        env = make_env_fn(configs[0], env_classes[0])
        env.reset()
        time_env(env, config)
        import pdb
        pdb.set_trace()
        del env

    """
    Action: TURN_LEFT || Tries: 185 ||Avg Time / Num Processes: 0.0036 secs || Num Processes: 1
    Action: LOOK_UP || Tries: 205 ||Avg Time / Num Processes: 0.0035 secs || Num Processes: 1
    Action: TURN_RIGHT || Tries: 187 ||Avg Time / Num Processes: 0.0035 secs || Num Processes: 1
    Action: LOOK_DOWN || Tries: 210 ||Avg Time / Num Processes: 0.0036 secs || Num Processes: 1
    Action: MOVE_FORWARD || Tries: 213 ||Avg Time / Num Processes: 0.0035 secs || Num Processes: 1
    """

    # observations = []
    # for i in range(60):
    #     debug_action = env.action_space.sample()
    #     obs = env.step(action=debug_action)[0]
    #     observations.append(obs)
    #     print(f"Debug Action: {debug_action}")
    #
    # from cos_eor.play_utils import make_video_cv2
    # make_video_cv2(observations, prefix="debug-rgb-", sensor="rgb")
    # make_video_cv2(observations, prefix="debug-rgb-third-", sensor="rgb_3rd_person")

    """
    Action: LOOK_DOWN || Tries: 189 ||Avg Time / Num Processes: 0.0058 secs || Num Processes: 8
    Action: TURN_LEFT || Tries: 219 ||Avg Time / Num Processes: 0.0059 secs || Num Processes: 8
    Action: TURN_RIGHT || Tries: 200 ||Avg Time / Num Processes: 0.0058 secs || Num Processes: 8
    Action: LOOK_UP || Tries: 194 ||Avg Time / Num Processes: 0.0059 secs || Num Processes: 8
    Action: MOVE_FORWARD || Tries: 198 ||Avg Time / Num Processes: 0.006 secs || Num Processes: 8
    """

    # envs = habitat.VectorEnv(
    #     make_env_fn=make_env_fn,
    #     env_fn_args=tuple(zip(configs, env_classes)),
    #     workers_ignore_signals=workers_ignore_signals,
    # )

    envs = habitat.ThreadedVectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(zip(configs, env_classes)),
        workers_ignore_signals=workers_ignore_signals,
    )

    # pbar = tqdm(desc="Reset")
    # while True:
    #     pbar.update()
    #     envs.reset()

    """
    Without L2 calculations and PP action
    Action: LOOK_UP || Tries: 213 ||Avg Time / Num Processes: 0.0014 secs || Num Processes: 8
    Action: LOOK_DOWN || Tries: 212 ||Avg Time / Num Processes: 0.0014 secs || Num Processes: 8
    Action: MOVE_FORWARD || Tries: 197 ||Avg Time / Num Processes: 0.0014 secs || Num Processes: 8
    Action: TURN_RIGHT || Tries: 206 ||Avg Time / Num Processes: 0.0014 secs || Num Processes: 8
    Action: TURN_LEFT || Tries: 172 ||Avg Time / Num Processes: 0.0019 secs || Num Processes: 8
    """

    """
    With L2 calculations, but no PP action
    Action: LOOK_UP || Tries: 213 ||Avg Time / Num Processes: 0.0024 secs || Num Processes: 8
    Action: LOOK_DOWN || Tries: 212 ||Avg Time / Num Processes: 0.0025 secs || Num Processes: 8
    Action: MOVE_FORWARD || Tries: 197 ||Avg Time / Num Processes: 0.0024 secs || Num Processes: 8
    Action: TURN_RIGHT || Tries: 206 ||Avg Time / Num Processes: 0.0024 secs || Num Processes: 8
    Action: TURN_LEFT || Tries: 172 ||Avg Time / Num Processes: 0.0029 secs || Num Processes: 8
    """

    """
    With L2 calculations, and PP action
    """

    if registry.mapping["time"]:
        time_envs(envs, config)
        import pdb
        pdb.set_trace()

    return envs


def time_envs(envs, config, num_steps=800):
    # timing code
    num_envs = envs.num_envs
    envs_obs = envs.reset()
    # try 400 random actions and report average time for each category
    actual_actions = config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS
    # possible_actions = [idx for idx, a in enumerate(actual_actions) if a.lower() not in ["stop"]]
    possible_actions = [idx for idx, a in enumerate(actual_actions) if a.lower() in ["grab_release"]]
    sample_actions = [random.choice(possible_actions) for _ in range(num_steps)]
    envs_time = defaultdict(list)

    # obj_iids = [t.sim_obj_id_to_iid[oid] for oid in list(t.sim_obj_id_to_iid.keys()) if t.sim_obj_id_to_type[oid] == "obj"]
    # rec_iids = [t.sim_obj_id_to_iid[oid] for oid in list(t.sim_obj_id_to_iid.keys()) if t.sim_obj_id_to_type[oid] == "rec"]

    for action in tqdm(sample_actions, desc="Debug action timings"):
        start = time.time()
        # sample iid and select random action
        try:
            iids = [eo["visible_obj_iids"][:eo["num_visible_objs"]]
                    if eo["gripped_object_id"] == -1
                    else eo["visible_rec_iids"][:eo["num_visible_recs"]] for eo
                    in
                    envs_obs]
        except:
            import pdb
            pdb.set_trace()
        actions_args = [random.choice(ii) if len(ii) else -1 for ii in iids]
        envs_actions = [{"action": {"action": action, "action_args": {"iid": iid}}} for iid in actions_args]
        # envs_actions = [{"action": {"action": action, "action_args": {"iid": -1}}} for _ in range(num_envs)]

        envs_obs = [out[0] for out in envs.step(envs_actions)]
        end = time.time()
        envs_time[action].append(end - start)

    for action, times in envs_time.items():
        print(f"Action: {actual_actions[action]} || "
              f"Tries: {len(times)} ||"
              f"Avg Time / Num Processes: {round(sum(times) / (len(times) * num_envs), 5)} secs || "
              f"Num Processes: {num_envs}")


def time_env(env, config, num_steps=1000):
    actual_actions = config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS
    # possible_actions = [idx for idx, a in enumerate(actual_actions) if a.lower() not in ["stop"]]
    possible_actions = [idx for idx, a in enumerate(actual_actions) if a.lower() in ["grab_release"]]
    actions = [random.choice(possible_actions) for _ in range(num_steps)]
    envs_time = defaultdict(list)
    num_envs = 1
    obs = env.reset()

    t = env._env._task
    obj_iids = [t.sim_obj_id_to_iid[oid] for oid in list(t.sim_obj_id_to_iid.keys()) if t.sim_obj_id_to_type[oid] == "obj"]
    rec_iids = [t.sim_obj_id_to_iid[oid] for oid in list(t.sim_obj_id_to_iid.keys()) if t.sim_obj_id_to_type[oid] == "rec"]
    assert len(obj_iids) > 0 and len(rec_iids) > 0

    for a in tqdm(actions, desc="Debug action timings"):
        start = time.time()
        iids = obj_iids if obs["gripped_object_id"] == -1 else  rec_iids
        iid = random.choice(iids)
        obs, _, _, _ = env.step(action={"action": a, "action_args": {"iid": -1}})
        end = time.time()
        envs_time[a].append(end - start)

    for action, times in envs_time.items():
        print(f"Action: {actual_actions[action]} || "
              f"Tries: {len(times)} ||"
              f"Avg Time / Num Processes: {round(sum(times) / (len(times) * num_envs), 5)} secs || "
              f"Num Processes: {num_envs}")
