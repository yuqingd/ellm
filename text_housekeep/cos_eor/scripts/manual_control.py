#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
from collections import defaultdict

import cv2
import random

import numpy as np
from tqdm import tqdm
import time

cwd = os.getcwd()
sys.path.insert(1, os.path.dirname(cwd))
sys.path.insert(1, cwd)
os.chdir("/home/kyash/Documents/cos-hab2")
# os.chdir("/srv/share/ykant3/p-viz-plan")
from habitat.core.registry import registry

import habitat_sim
import habitat
import gzip
import os
import json
from cos_eor.task.play_measures import *
from cos_eor.task.play_sensors import *
from cos_eor.task.sensors import *
from cos_eor.task.measures import *
import pandas as pd
from habitat.utils.visualizations.utils import images_to_video

# %matplotlib inline
from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw
from habitat_sim.utils.viz_utils import is_notebook, display_video, get_fast_video_writer
from orp.samplers import MultiSceneSampler, PolySurface
from orp.task_settings import get_sampled_obj
from orp.utils import get_aabb, make_render_only
from cos_eor.utils.debug import get_corners, set_object_in_front_of_agent, set_art_object_in_front_of_agent
from cos_eor.utils.geometry import get_bb

# we only load the base-level config here
config = habitat.get_config("cos_eor/configs/local/igib_v2.yaml")
config = habitat.get_config_new([config.BASE_TASK_CONFIG])
config.defrost()
config.TASK.TOP_DOWN_MAP.FOG_OF_WAR.DRAW = False
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)

#########
# eplist = ['27_data/scene_datasets/gibson_train_val/Sagerton.glb',
# '8_data/scene_datasets/gibson_train_val/Cokeville.glb',
# '5_data/scene_datasets/gibson_train_val/Sands.glb',
# '32_data/scene_datasets/gibson_train_val/Sands.glb',
# '75_data/scene_datasets/gibson_train_val/Silerton.glb',
# '13_data/scene_datasets/gibson_train_val/Southfield.glb',
# '52_data/scene_datasets/gibson_train_val/Southfield.glb',
# '71_data/scene_datasets/gibson_train_val/Sands.glb',
# '6_data/scene_datasets/gibson_train_val/Silerton.glb',
# '93_data/scene_datasets/gibson_train_val/Silerton.glb',
# '61_data/scene_datasets/gibson_train_val/Wells.glb']
# i = 0
# uid = eplist[i]
# eid = eplist[i].split('_')[0]
# scene_id = eplist[i][len(eid) + 1:]
# print(eid, scene_id)
# df = pd.read_pickle('data/new_checkpoints/b9KSrZfBpB5WC5gcCUtmMX/replays/l2dist_object/7q3NhKfhy4gW67jNeGdsfA.pickle')
# data = df[df['uid'] == uid].iloc[0].to_dict()
# config.DATASET.SPLIT = "test"
# config.DATASET.DATA_PATH = "data/datasets/rearrangement/gibson/v1/{split}/{split}.json.gz"
# config.DATASET.CONTENT_SCENES = [f"rearrangement_hard_v8_{config.DATASET.SPLIT}_n=100_o=5_t=0.9_{scene_id.split('/')[-1].split('.')[0]}"]
# config.TASK.TOP_DOWN_MAP.FOG_OF_WAR.DRAW = True

#########


# config.TASK.MEASUREMENTS = ['OBJECT_TO_GOAL_DISTANCE', 'AGENT_TO_OBJECT_DISTANCE', 'EOR_TOP_DOWN_MAP']
config.TASK.MEASUREMENTS = ['COLLISIONS']
# config.TASK.MEASUREMENTS = ['OBJECT_TO_GOAL_DISTANCE', 'AGENT_TO_OBJECT_DISTANCE']
# config.TASK.SENSORS = ['GRIPPED_OBJECT_SENSOR', 'ALL_OBJECT_POSITIONS', 'ALL_OBJECT_GOALS', 'ORACLE_NEXT_OBJECT_SENSOR',]
# config.defrost()
config.SIMULATOR.DEPTH_SENSOR.HEIGHT = 1024
config.SIMULATOR.DEPTH_SENSOR.WIDTH = 1024
config.SIMULATOR.RGB_SENSOR.HEIGHT = 1024
config.SIMULATOR.RGB_SENSOR.WIDTH = 1024
config.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = 1024
config.SIMULATOR.SEMANTIC_SENSOR.WIDTH = 1024
config.SIMULATOR.CROSSHAIR_POS = [512, 512]
config.SIMULATOR.TURN_ANGLE = 30
config.SIMULATOR.RGB_SENSOR_3RD_PERSON.HEIGHT = 1024
config.SIMULATOR.RGB_SENSOR_3RD_PERSON.WIDTH = 1024
config.TASK.EOR_TOP_DOWN_MAP.MAX_RESOLUTION = 1024
config.TASK.EOR_TOP_DOWN_MAP.MAX_RESOLUTION = 1024
config.TASK.COLLISIONS = CN()
config.TASK.COLLISIONS.TYPE = "Collisions"
config.freeze()

try:
    env.close()
except:
    pass

env = habitat.Env(config)

# asserts that physics is enabled
assert "bullet" in str(env.sim.get_physics_simulation_library()).lower()


# In[3]:


# go to the desired episode
try:
    for ep in env.episodes:
        if ep.episode_id == eid:
            break
    env._current_episode = ep
except:
    env._current_episode = env.episodes[0]
    env.sim.navmesh_visualization = True

env._reset_stats()

if env._current_episode is not None:
    env._current_episode._shortest_path_cache = None

# stolen from env.reset()
env.reconfigure(env._config)
obs = env.task.reset(episode=env.current_episode)
env._task.measurements.reset_measures(
    episode=env.current_episode, task=env.task
)
metrics = env.get_metrics()

import pdb
pdb.set_trace()

# plt.imshow(obs['rgb_3rd_person'])
# plt.show()


# In[4]:

def _time(env):
    # timing code
    env.reset()
    # try 400 random actions and report average time for each category
    possible_actions = config.TASK.POSSIBLE_ACTIONS
    num_actions = len(possible_actions)
    envs_actions = [random.choice(range(num_actions)) for _ in range(80)]
    envs_time = defaultdict(list)

    # for action in tqdm(envs_actions, desc="Debug action timings w/ Env"):
    #     start = time.time()
    #     env.step(action)
    #     end = time.time()
    #     envs_time[action].append(end-start)
    #
    # for action, times in envs_time.items():
    #     print(f"Action: {possible_actions[action]} || "
    #           f"Avg. Time over {len(times)} "
    #           f"tries: {round(sum(times)/len(times), 4)} secs || "
    #           f"Num Processes: 1")

    envs_actions = [random.choice([1,2,3,4,5]) for _ in range(80)]
    envs_time = defaultdict(list)
    for action in tqdm(envs_actions, desc="Debug action timings w/ Sim"):
        start = time.time()
        env.sim.step(action)
        # super(type(env.sim), env.sim).step(action)
        end = time.time()
        envs_time[action].append(end-start)

    for action, times in envs_time.items():
        print(f"Action: {action} || "
              f"Avg. Time over {len(times)} "
              f"tries: {round(sum(times)/len(times), 4)} secs || "
              f"Num Processes: 1")

# _time(env)


def get_map(env):
    top_down_map = maps.get_topdown_map(
        env._task._simple_pathfinder,
        env._sim.get_agent(0).state.position[1],
        1024
    )

    top_down_map = maps.colorize_topdown_map(top_down_map)
    agent_position = env._sim.get_agent_state().position
    agent_rotation = env._sim.get_agent_state().rotation
    
    a_x, a_y = maps.to_grid(
        agent_position[2],
        agent_position[0],
        top_down_map.shape[0:2],
        sim=env._sim,
    )

    episode = env.current_episode
    object_positions = [obj.position for obj in episode.objects]
    goal_positions = [obj.position for obj in episode.goals]

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

    # draw the objectgoal positions.
    for i, goal_pos in enumerate(goal_positions):
        tdm_pos = maps.to_grid(
            goal_pos[2],
            goal_pos[0],
            top_down_map.shape[0:2],
            sim=env._sim,
        )

        grid_goal_positions.append(tdm_pos)

    grid_current_positions = [None] * len(object_positions)
    for sim_obj_id in env._sim.get_existing_object_ids():
        if sim_obj_id != env._task.agent_object_id:
            obj_id = env._task.sim_obj_id_to_ep_obj_id[sim_obj_id]
            position = env._sim.get_translation(sim_obj_id)
            curr_pos = maps.to_grid(position[2], position[0], top_down_map.shape[0:2], sim=env._sim)
            grid_current_positions[obj_id] = curr_pos

    polar_rotation = get_polar_angle(agent_rotation)
    
    top_down_map = maps.draw_agent(
        image=top_down_map,
        agent_center_coord=[a_x, a_y],
        agent_rotation=polar_rotation,
        agent_radius_px=min(top_down_map.shape[0:2]) / 32,
    )

    top_down_map = maps.draw_object_info(top_down_map, grid_goal_positions, suffix='g')
    top_down_map = maps.draw_object_info(top_down_map, grid_current_positions, suffix='c')
    # print(top_down_map.shape)
    plt.figure(figsize = (11,10))
    plt.imshow(top_down_map, interpolation=None)
    return top_down_map
    
# top_down_map = get_map(env)


# In[5]:


from tqdm import tqdm


def get_semantic_centroids(semantic_obs):
    sids = list(np.unique(semantic_obs))
    if 0 in sids:
        sids.remove(0)
    sid_centroids = []
    for sid in sids:
        one_hot = (semantic_obs == sid)
        xis, yis = np.nonzero(one_hot)
        sid_centroids.append([xis.mean(), yis.mean()])

    return sids, sid_centroids


def render_frame(obs, info, cross_hair, ss_th=1e3):
    # obs['rgb'][
    #     cross_hair[1] - 10 : cross_hair[1] + 10,
    #     cross_hair[0] - 10 : cross_hair[0] + 10,
    #     ] = [255, 0, 0]
    #
    # img_frame = Image.fromarray(obs['rgb_3rd_person'])
    # rgb = Image.fromarray(obs['rgb'])
    #
    # from habitat_sim.utils.common import d3_40_colors_rgb
    # semantic = obs['semantic']
    #
    # # filter sids by distance th.
    # sids = np.unique(semantic).tolist()
    # if 0 in sids:
    #     sids.remove(0)
    #
    # oids = [env._task.get_oid_from_sid(sid) for sid in sids]
    # oids = [env.task.sim_obj_id_to_ep_obj_id[x] for x in oids]
    # oids_ed = [info["agent_to_object_distance"][f"{oid}_ed"] for oid in oids]
    #
    # # sids_fil = []
    # # for sid, oid_ed in zip(sids, oids_ed):
    # #     if oid_ed < ss_th:
    # #         sids_fil.append(sid)
    # #     else:
    # #         inds = semantic == sid
    # #         semantic[inds] = 0
    #
    # sids, sid_centroids = get_semantic_centroids(semantic)
    # org_h, org_w = semantic.shape
    # semantic_img = Image.new("P", (semantic.shape[1], semantic.shape[0]))
    # semantic_img.putpalette(d3_40_colors_rgb.flatten())
    # semantic_img.putdata((semantic.flatten() % 40).astype(np.uint8))
    # semantic_img = semantic_img.convert("RGBA")
    #
    # if "depth" in obs:
    #     d_im = depth_to_rgb(obs['depth'], clip_max=1.0)[:, :, 0]
    #     depth_map = np.stack([d_im for _ in range(3)], axis=2)
    #     depth = Image.fromarray(depth_map)
    #
    # for s, sc in zip(sids, sid_centroids):
    #     obj_id = env._task.iid_to_sim_obj_id[s]
    #     ann_key = env._task.sim_obj_id_to_obj_key[obj_id]
    #     obj_ann = env._task.object_annotations[ann_key]
    #     draw_string = "P" if obj_ann["pickable"] else "NP"
    #     if obj_ann["type"] == "object_receptacle":
    #         or_string = "OR"
    #     elif obj_ann["type"] == "object":
    #         or_string = "O"
    #     elif obj_ann["type"] == "receptacle":
    #         or_string = "R"
    #     else:
    #         raise AssertionError
    #     draw_string = f"{s}"
    #     # draw_string = f"{s} | {ann_key} | {draw_string} | {or_string}"
    #     font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 100)
    #     draw = ImageDraw.Draw(semantic_img)
    #     draw.text((sc[1], sc[0]), draw_string, (0, 0, 0), font=font)
    #
    # overlay_rgb_img = rgb.resize((256, 256))
    # overlay_depth_img = depth.resize((256, 256))
    # overlay_semantic_img = semantic_img.resize((256, 256))
    #
    # img_frame.paste(overlay_rgb_img, box=(32, 32))
    # img_frame.paste(overlay_depth_img, box=(32, 320))
    # img_frame.paste(overlay_semantic_img, box=(32, 640))
    img_frame = np.array(obs["rgb"])
    if "top_down_map" in info and show_map:
        top_down_map = info["top_down_map"]
        if top_down_map.shape[0] > top_down_map.shape[1]:
            top_down_map = np.rot90(top_down_map, 1)

        # scale top down map to align with rgb view
        old_h, old_w, _ = top_down_map.shape
        top_down_height = img_frame.shape[0]
        top_down_width = int(float(top_down_height) / old_h * old_w)
        # cv2 resize (dsize is width first)
        top_down_map = cv2.resize(top_down_map, (top_down_width, top_down_height), interpolation=cv2.INTER_CUBIC,)
        img_frame = np.concatenate((top_down_map, img_frame), axis=1)
    else:
        top_down_map = None

    img_frame = Image.fromarray(img_frame).resize(window)
    return img_frame, top_down_map


def depth_to_rgb(depth_image: np.ndarray, clip_max: float = 10.0) -> np.ndarray:
    """Normalize depth image into [0, 1] and convert to grayscale rgb
    :param depth_image: Raw depth observation image from sensor output.
    :param clip_max: Max depth distance for clipping and normalization.
    :return: Clipped grayscale depth image data.
    """
    d_im = np.clip(depth_image, 0, clip_max)
    d_im /= clip_max
    # d_im = np.stack([d_im for _ in range(3)], axis=2)
    rgb_d_im = (d_im * 255).astype(np.uint8)
    return rgb_d_im


def add_semantic_ids(sim):
    global semantic_id_count
    for obj_id in sim.get_existing_object_ids():
        sim.set_object_semantic_id(semantic_id_count, obj_id)
        semantic_id_count += 10

    for obj_id in sim.get_existing_articulated_object_ids():
        sim.set_object_semantic_id_art(semantic_id_count, obj_id)
        semantic_id_count += 10

import magnum as mn
from habitat_sim.physics import MotionType
from orp.obj_loaders import load_articulated_objs, load_objs, place_viz_objs, init_art_objs
import random



import pygame
pygame.init()

show_map = True
# show_map = False
window = (1440, 720) if show_map else (720, 720)
record_path = "./debug-data/manual_control/"
assert os.path.exists(record_path)
screen = pygame.display.set_mode(window)
cross_hair = env._config.SIMULATOR.CROSSHAIR_POS
prim_ids, obj_ids = [], []

done = False
is_blue = True


def place_robot_from_agent(
    sim,
    robot_id,
    angle_correction=-1.56,
    local_base_pos=None,
):
    if local_base_pos is None:
        local_base_pos = np.array([0.0, -0.1, -2.0])
    # place the robot root state relative to the agent
    agent_transform = sim.agents[0].scene_node.transformation_matrix()
    base_transform = mn.Matrix4.rotation(
        mn.Rad(angle_correction), mn.Vector3(1.0, 0, 0)
    )
    base_transform.translation = agent_transform.transform_point(local_base_pos)
    sim.set_articulated_object_root_state(robot_id, base_transform)


def simulate(sim, dt=1.0, get_frames=True):
    # simulate dt seconds at 60Hz to the nearest fixed timestep
    print("Simulating " + str(dt) + " world seconds.")
    observations = []
    start_time = sim.get_world_time()
    while sim.get_world_time() < start_time + dt:
        sim.step_physics(1.0 / 60.0)
        if get_frames:
            observations.append(sim.get_sensor_observations())

    return observations


def add_text(screen, text, position=(0,0)):
    myfont = pygame.font.SysFont('Comic Sans MS', 30)
    textsurface = myfont.render(text, False, (0, 0, 0))
    screen.blit(textsurface, position)


def insert_non_art_obj(env):
    sim = env.sim
    obj_attr_mgr = sim.get_object_template_manager()
    obj_template_ids = obj_attr_mgr.load_configs("./data/objects")
    obj_template_id = np.random.choice(obj_template_ids)
    attr = obj_attr_mgr.get_template_by_ID(obj_template_id)
    # obj_template_id = "./data/objects/sofa.object_config.json"
    # attr = obj_attr_mgr.get_template_by_handle(obj_template_id)
    obj_attr_mgr.register_template(attr)
    object_id = sim.add_object_by_handle(attr.handle)
    set_object_in_front_of_agent(sim, object_id)
    sim.set_object_is_collidable(True, object_id)

    # add wireframe cube around it
    prim_attr_mgr = sim.get_asset_template_manager()
    solid_handle = prim_attr_mgr.get_template_handles("cubeSolid")[0]
    wire_handle = prim_attr_mgr.get_template_handles("cubeWire")[0]
    print("Wireframe Object being made using handle :{}".format(wire_handle))
    wire_template = obj_attr_mgr.create_template(wire_handle)
    obj_attr_mgr.register_template(wire_template)
    wire_id = sim.add_object_by_handle(wire_handle)
    # sim.set_translation(sim.get_translation(object_id), wire_id)

    # find scale
    obj_bb = get_bb(sim, object_id)
    prim_bb = get_bb(sim, wire_id)
    scale_x = (obj_bb.max[0] - obj_bb.min[0]) / (prim_bb.max[0] - prim_bb.min[0])
    scale_y = (obj_bb.max[1] - obj_bb.min[1]) / (prim_bb.max[1] - prim_bb.min[1])
    scale_z = (obj_bb.max[2] - obj_bb.min[2]) / (prim_bb.max[2] - prim_bb.min[2])

    # remove old object
    sim.remove_objects([wire_id])
    scaled_wire_template = obj_attr_mgr.create_template(wire_handle)
    scaled_wire_template.handle = f"{attr.handle}_{scaled_wire_template.handle}"
    scaled_wire_template.scale = np.array([scale_x, scale_y, scale_z]) * scaled_wire_template.scale
    obj_attr_mgr.register_template(scaled_wire_template)
    scaled_wire_id = sim.add_object_by_handle(scaled_wire_template.handle)
    sim.set_translation(sim.get_translation(object_id), scaled_wire_id)
    prim_ids.append(scaled_wire_id)
    obj_ids.append(object_id)

    # navmesh-reconfigure
    # sim.set_object_motion_type(MotionType.STATIC, scaled_wire_id)
    # sim.set_object_motion_type(MotionType.STATIC, object_id)
    # sim.recompute_navmesh(sim.pathfinder, sim.navmesh_settings, True)








def place_non_art_obj(env, obj_id, obj_template_id, receptacle_key=None):
    # Todo: get from episode!

    existing_obj_bbs = [get_aabb(i, sim, transformed=True) for i in existing_obj_ids]
    # insert object w/ collision check on a given receptacle
    if receptacle_key is None:
        receptacle_key = random.choice(list(mdat["bb"].keys()))
    position_generator = mdat["bb"][receptacle_key]

    # lots of redundant args in this one
    pos = [0, 0, 0]
    pos = get_sampled_obj(sim, position_generator, pos, obj_id, None, None, existing_obj_bbs, None)
    sim.set_object_motion_type(MotionType.KINEMATIC, obj_id)
    if pos is None:
        import pdb
        pdb.set_trace()

    sim.set_translation(mn.Vector3(*pos), obj_id)
    sim.set_linear_velocity(mn.Vector3(0, 0, 0), obj_id)
    sim.set_angular_velocity(mn.Vector3(0, 0, 0), obj_id)
    sim.set_object_bb_draw(True, obj_id)
    existing_obj_ids.append(obj_id)
    simulate(sim, dt=1.5, get_frames=False)
    print(f"Inserted object: {obj_template_id} on receptacle: {receptacle_key}")
    return obj_id, position_generator, pos


def visualize_point(pos, r=0.1):
    obj_mgr = sim.get_object_template_manager()
    template = obj_mgr.get_template_by_handle(obj_mgr.get_template_handles("sphere")[0])
    template.scale = mn.Vector3(r, r, r)
    new_template_handle = obj_mgr.register_template(template, "ball_new_viz")
    viz_id = sim.add_object(new_template_handle)
    make_render_only(viz_id, sim)
    sim.set_translation(mn.Vector3(*pos), viz_id)
    registry.mapping["ignore_object_ids"].append(viz_id)


def visualize_bbox(corners):
    for cor in corners:
        visualize_point(cor)


def visualize_surfaces():
    obj_ids = sim.get_both_existing_object_ids()
    for obj_id in obj_ids["art"] + obj_ids["non_art"]:
        obj_node = sim.get_object_scene_node(obj_id)
        obj_bb = obj_node.cumulative_bb
        corners = get_corners(obj_bb)
        corners = [obj_node.transformation.transform_point(cor) for cor in corners]
        corners = [[cor.x, cor.y, cor.z] for cor in corners]
        surface_center = get_surface_center_from_corners(corners)
        visualize_bbox([surface_center])


# existing_obj_ids = []
# sampler = MultiSceneSampler('data/scene_datasets', 'v3_sc%i_staging', 0.8)
# sampler.set_i("data/scene_datasets/v3_sc3_staging_13.glb")
# cur_scene_name, mdat = sampler.sample()
sim = env.sim
collision_count = 0
global semantic_id_count
semantic_id_count = 20
switch = False
frames = []

task_config = env._config.TASK
while not done:
    for event in pygame.event.get():
        # print(f"Event: {event}")
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_UP]:
            action = task_config["POSSIBLE_ACTIONS"].index("MOVE_FORWARD")
        elif pressed[pygame.K_RIGHT]:
            action = task_config["POSSIBLE_ACTIONS"].index("TURN_RIGHT")
        elif pressed[pygame.K_LEFT]:
            action = task_config["POSSIBLE_ACTIONS"].index("TURN_LEFT")
        elif pressed[pygame.K_PAGEUP]:
            action = task_config["POSSIBLE_ACTIONS"].index("LOOK_UP")
        elif pressed[pygame.K_PAGEDOWN]:
            action = task_config["POSSIBLE_ACTIONS"].index("LOOK_DOWN")
        elif pressed[pygame.K_SPACE]:
            action = task_config["POSSIBLE_ACTIONS"].index("GRAB_RELEASE")
            avail_sids = list(np.unique(obs["semantic"]))
            if 0 in avail_sids:
                avail_sids.remove(0)
            # add prompt
            add_text(screen, f"Pick/Place from Semantic Ids: {avail_sids}")
            pygame.display.update()
            chosen_sid = int(input(f"Visible sids for pick/place -- {avail_sids}: "))
            # color = (0, 0, 255)
        elif pressed[pygame.K_n]:
            # insert non-articulate object
            insert_non_art_obj(env)
        elif pressed[pygame.K_g]:
            _, obj_ids, _, _, prim_ids = sim.init_metadata_objects()
            sim.set_objects_motion(prim_ids, MotionType.STATIC)
            sim.set_objects_motion(obj_ids, MotionType.DYNAMIC)
            sim.recompute_navmesh(sim.pathfinder, sim.navmesh_settings, True)

        elif pressed[pygame.K_r]:
            # start/stop recording and dump
            if not switch:
                print(f"Started Recording...")
                switch = True
            else:
                switch = False
                try:
                    time_stamp = time.time()
                    curr_episode = env._current_episode
                    scene_name = os.path.split(curr_episode.scene_id.split(".")[0])[-1]
                    identifier = input("video identifier: ")
                    video_path = os.path.join(record_path, f"{scene_name}_{str(time_stamp)}_{identifier}")
                    images_to_video(frames, "./", video_path, fps=2)
                except:
                    print(f"Some error in saving video")
                    video_path = None
                    import pdb
                    pdb.set_trace()
                print(f"Saved: {video_path}")
                frames = []

        elif pressed[pygame.K_v]:
            # insert non-articulate object
            visualize_surfaces()
        elif pressed[pygame.K_i]:
            # add semantic-ids
            env._task.add_semantic_ids()
        elif pressed[pygame.K_d]:
            # debug
            import pdb
            pdb.set_trace()
            action = -1
        elif pressed[pygame.K_s]:
            simulate(sim)
            action = -1
        elif pressed[pygame.K_q]:
            done = True
            pygame.quit()
            break
        else:
            action = -1

        if action != -1:
            obs = env.step(action)
            metrics = env.get_metrics()
            metrics["top_down_map"] = sim.get_topdown_map()
            img_frame, top_down_map = render_frame(obs, metrics, cross_hair)
            if switch:
                frames.append(np.array(img_frame))
            img_frame.save("debug.jpeg")
            image = pygame.image.load("debug.jpeg")
            screen.blit(image,(0,0))
            # add_text(screen, f"Action ID: {action}")
            pygame.display.update()
            # print(f"Collision count: {metrics['collisions']}")
            # del metrics["top_down_map"]
            # print(metrics)


pygame.display.quit()


# In[ ]:





# In[11]:


# Debugging code to fix agent orientation in the map

def get_agent_pos_rot(env):
    state = env._sim.get_agent(0).state
    pos, rot = state.position, state.rotation
    return pos, rot


print(f"Action Space: {env.action_space}")
print(f"Position and Rotation: {get_agent_pos_rot(env)}")
# plt.imshow(obs['rgb_3rd_person'])
for i in range(18):
    obs = env.step(1)
print(f"Position and Rotation: {get_agent_pos_rot(env)}")
plt.imshow(obs['rgb'])
_ = get_map(env)
obs.keys()

