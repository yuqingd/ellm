import glob
import json
import os
import random
import sys
import time
from collections import defaultdict, Counter
from copy import deepcopy

import yaml
from habitat_sim.utils.common import quat_to_magnum, quat_from_coeffs, quat_from_magnum, quat_from_angle_axis

from text_housekeep.orp.utils import get_aabb
from text_housekeep.cos_eor.scripts.orm.utils import preprocess

cwd = os.getcwd()
sys.path.insert(1, os.path.dirname(cwd))
sys.path.insert(1, cwd)

import cv2
import numpy as np
import habitat_sim
from text_housekeep.cos_eor.task.measures import *
import pygame
from PIL import Image, ImageFont, ImageDraw
from text_housekeep.orp.samplers import PolySurface
# from orp.task_settings import get_sampled_obj
from habitat.utils.visualizations.utils import images_to_video
import magnum as mn
from tqdm import tqdm
from text_housekeep.cos_eor.utils.geometry import  get_random_point_igib, get_semantic_centroids, get_bb, \
    get_corners, get_bbs_from_corners, get_surface_point_for_placement, get_all_nav_points
from habitat_sim.physics import MotionType


def set_object_in_front_of_agent(sim, obj_id, z_offset=-1.5, y_offset=0):
    r"""
    Adds an object in front of the agent at some distance.
    """
    agent_transform = sim.agents[0].scene_node.transformation_matrix()
    obj_translation = agent_transform.transform_point(
        np.array([0, 0, z_offset])
    )

    sim.set_translation(obj_translation, obj_id)

    obj_node = sim.get_object_scene_node(obj_id)
    xform_bb = habitat_sim.geo.get_transformed_bb(
        obj_node.cumulative_bb, obj_node.transformation
    )

    # also account for collision margin of the scene
    scene_collision_margin = 0.04
    y_translation = mn.Vector3(
        0, xform_bb.size_y() / 2.0 + scene_collision_margin + y_offset, 0
    )
    sim.set_translation(y_translation + sim.get_translation(obj_id), obj_id)
    # simulate(sim, 1.0)


def set_art_object_in_front_of_agent(sim, obj_id, z_offset=-1.5):
    r"""
    Adds an object in front of the agent at some distance.
    """

    current_state = sim.get_articulated_object_root_state(obj_id)
    agent_transform = sim.agents[0].scene_node.transformation_matrix()
    obj_translation = agent_transform.transform_point(
        np.array([0, 0, z_offset])
    )

    obj_translation = list(obj_translation)
    translated_state = getattr(mn.Matrix4, 'from_')(current_state.rotation(), mn.Vector3(obj_translation))
    sim.set_articulated_object_root_state(obj_id, translated_state)


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


def add_text(screen, text, position):
    myfont = pygame.font.SysFont('Comic Sans MS', 25)
    textsurface = myfont.render(text, False, (255, 0, 255))
    screen.blit(textsurface, position)


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


def add_text_strip(img_frame, text_rows = 2, row_height = 45):
    text_strip = np.zeros((row_height * text_rows, img_frame.shape[1], img_frame.shape[2])) + 255
    img_frame = np.concatenate([text_strip, img_frame], axis=0)
    return img_frame


def render_frame(obs, info, cross_hair, episode, sim, task_env, grab_type, window):

    if grab_type == "crosshair":
        assert cross_hair[1] in range(0, obs['rgb'].shape[0])
        assert cross_hair[0] in range(0, obs['rgb'].shape[1])

        cross_hair_size = obs["rgb"].shape[0] // 64

        obs['rgb'][
        cross_hair[1] - cross_hair_size: cross_hair[1] + cross_hair_size,
        cross_hair[0] - cross_hair_size: cross_hair[0] + cross_hair_size,
        ] = [255, 0, 0]

    img_frame = Image.fromarray(obs['rgb_3rd_person'])
    rgb = Image.fromarray(obs['rgb'])
    from habitat_sim.utils.common import d3_40_colors_rgb

    if "semantic" in obs:
        semantic = obs['semantic'].squeeze()
        semantic_img = Image.new("P", (semantic.shape[1], semantic.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        iids, iid_centroids = get_semantic_centroids(semantic)
        for i, sc in zip(iids, iid_centroids):
            sim_obj_id = task_env._env._task.iid_to_sim_obj_id[i]
            obj_type = task_env._env._task.sim_obj_id_to_type[sim_obj_id]

            # when holding an object show only receptacles
            if obj_type == "obj" and obs["gripped_object_id"] != -1:
                continue

            # when holding nothing show only objects
            if obj_type == "rec" and obs["gripped_object_id"] == -1:
                continue
            # ann_key = task_env._env._task.sim_obj_id_to_obj_key[sim_obj_id]
            # obj_ann = task_env._env._task.object_annotations[ann_key]
            # obj_id = task_env._env._task.sim_obj_id_to_ep_obj_id[sim_obj_id]
            #
            # draw_string = "P" if obj_ann["pickable"] else "NP"
            # if obj_ann["type"] == "object_receptacle":
            #     or_string = "OR"
            # elif obj_ann["type"] == "object":
            #     or_string = "O"
            # elif obj_ann["type"] == "receptacle":
            #     or_string = "R"
            # else:
            #     raise AssertionError
            draw_string = f"{i}"
            # draw_string = f"{s} | {ann_key} | {draw_string} | {or_string}"
            font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 30)
            draw = ImageDraw.Draw(semantic_img)
            draw.text((sc[1], sc[0]), draw_string, (0, 0, 0), font=font)
            pass
    else:
        semantic_img = None

    if "depth" in obs:
        d_im = depth_to_rgb(obs['depth'], clip_max=1.0)[:, :, 0]
        depth_map = np.stack([d_im for _ in range(3)], axis=2)
        depth = Image.fromarray(depth_map)

    # scale up the third-person
    img_frame = img_frame.resize(window)
    overlay_rgb_img = rgb.resize((256, 256))
    img_frame.paste(overlay_rgb_img, box=(32, 32))

    overlay_depth_img = depth.resize((256, 256))
    img_frame.paste(overlay_depth_img, box=(32, 320))

    if semantic_img:
        overlay_semantic_img = semantic_img.resize((256, 256))
        img_frame.paste(overlay_semantic_img, box=(32, 640))

    img_frame = np.array(img_frame)

    if "collisions" in info and info["collisions"]["is_collision"]:
        border = 20
        # add a red border for collisions
        img_frame[:border, :] = [255, 0, 0]
        img_frame[:, :border] = [255, 0, 0]
        img_frame[-border:, :] = [255, 0, 0]
        img_frame[:, -border:] = [255, 0, 0]

    if "top_down_map" in info:
        top_down_map = maps.colorize_draw_agent_and_fit_to_height(
            info["top_down_map"], img_frame.shape[0]
        )

        if top_down_map.shape[0] > top_down_map.shape[1]:
            top_down_map = np.rot90(top_down_map, 1)

        # scale top down map to align with rgb view
        old_h, old_w, _ = top_down_map.shape
        top_down_height = img_frame.shape[0]
        top_down_width = int(float(top_down_height) / old_h * old_w)
        # cv2 resize (dsize is width first)
        top_down_map = cv2.resize(
            top_down_map,
            (top_down_width, top_down_height),
            interpolation=cv2.INTER_CUBIC,
        )

        # add object and goals to the map
        object_positions = [obj.position for obj in episode.objects]
        goal_positions = [obj.position for obj in episode.goals]

        grid_object_positions = []
        grid_goal_positions = []

        for i, obj_pos in enumerate(object_positions):
            tdm_pos = maps.to_grid(
                obj_pos[2],
                obj_pos[0],
                top_down_map.shape[0:2],
                sim=sim,
            )
            grid_object_positions.append(tdm_pos)

        # draw the objectgoal positions.
        for i, goal_pos in enumerate(goal_positions):
            tdm_pos = maps.to_grid(
                goal_pos[2],
                goal_pos[0],
                top_down_map.shape[0:2],
                sim=sim,
            )
            grid_goal_positions.append(tdm_pos)

        # get current object positions
        grid_current_positions = [None] * len(object_positions)
        for sim_obj_id in sim.get_existing_object_ids():
            if sim_obj_id != task_env._env._task.agent_object_id:
                sim_obj_id = task_env._env._task.sim_obj_id_to_ep_obj_id[sim_obj_id]
                position = sim.get_translation(sim_obj_id)
                curr_pos = maps.to_grid(position[2], position[0], top_down_map.shape[0:2], sim=sim)
                grid_current_positions[sim_obj_id] = curr_pos

        # add objects current and goal positions
        top_down_map = maps.draw_object_info(top_down_map, grid_goal_positions, suffix='g')
        top_down_map = maps.draw_object_info(top_down_map, grid_current_positions, suffix='c')

        if top_down_map.shape[0] > top_down_map.shape[1]:
            top_down_map = np.rot90(top_down_map, 1)

        # scale top down map to align with rgb view
        old_h, old_w, _ = top_down_map.shape
        top_down_height = img_frame.shape[0]
        top_down_width = int(float(top_down_height) / old_h * old_w)
        # cv2 resize (dsize is width first)
        top_down_map = cv2.resize(
            top_down_map,
            (top_down_width, top_down_height),
            interpolation=cv2.INTER_CUBIC,
        )

        # attach frames
        img_frame = np.concatenate((img_frame, top_down_map), axis=1)
    else:
        top_down_map = None

    # img_frame = add_text_strip(img_frame, text_rows=3)
    img_frame = Image.fromarray(img_frame, "RGB")
    img_frame = img_frame.resize(window )
    return img_frame, top_down_map


def render_sim_frame(obs, sim):
    img_frame = Image.fromarray(obs['rgb_3rd_person'])
    rgb = Image.fromarray(obs['rgb'])

    from habitat_sim.utils.common import d3_40_colors_rgb

    if "semantic" in obs:
        semantic = obs['semantic'].squeeze()
        semantic_img = Image.new("P", (semantic.shape[1], semantic.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
    else:
        semantic_img = None

    if "depth" in obs:
        d_im = depth_to_rgb(obs['depth'], clip_max=1.0)[:, :, 0]
        depth_map = np.stack([d_im for _ in range(3)], axis=2)
        depth = Image.fromarray(depth_map)

    # scale up the third-person
    img_frame = img_frame.resize((1024, 1024))
    overlay_rgb_img = rgb.resize((256, 256))
    img_frame.paste(overlay_rgb_img, box=(32, 32))

    overlay_depth_img = depth.resize((256, 256))
    img_frame.paste(overlay_depth_img, box=(32, 320))

    if semantic_img:
        overlay_semantic_img = semantic_img.resize((256, 256))
        img_frame.paste(overlay_semantic_img, box=(32, 640))

    img_frame = np.array(img_frame)
    img_frame = Image.fromarray(img_frame, "RGB")
    img_frame = img_frame.resize((720, 720))
    return img_frame


def debug_viewer(task_env, do_reset=False):
    """debug screen for the task"""
    pygame.init()
    window = (1024, 1024)
    screen = pygame.display.set_mode(window)
    env = task_env._env
    sim = env.sim
    semantic_id_count = 20
    switch = False
    frames = []
    grab_type = task_env.full_config.TASK_CONFIG.TASK.ACTIONS.GRAB_RELEASE.GRAB_TYPE
    cross_hair = task_env.full_config.TASK_CONFIG.TASK.ACTIONS.GRAB_RELEASE.CROSSHAIR_POS
    chosen_iid = -1

    assert "BULLET" in str(sim.get_physics_simulation_library())

    if do_reset:
        task_env.reset()

    done = False
    while not done:
        for event in pygame.event.get():
            # print(f"Event: {event}")
            if event.type == pygame.QUIT:
                done = True
                pygame.quit()

            pressed = pygame.key.get_pressed()
            task_config = env._config.TASK

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
                avail_iids = list(np.unique(obs["semantic"]))
                if 0 in avail_iids:
                    avail_iids.remove(0)
                chosen_iid = input(f"Visible iids for pick/place -- {avail_iids}: ").strip()
                # fails occasionally if you accidentally press other keys before the prompt
                try:
                    chosen_iid = int(chosen_iid)
                except:
                    chosen_iid = int(input(f"Visible iids for pick/place -- {avail_iids}: "))
            elif pressed[pygame.K_r]:
                # start/stop recording and dump
                if not switch:
                    print(f"Started Recording...")
                    switch = True
                else:
                    switch = False
                    try:
                        record_path = "./debug-data/manual_control/"
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
            # debug
            elif pressed[pygame.K_d]:
                eps = env._current_episode
                import pdb
                pdb.set_trace()
                action = -1
            elif pressed[pygame.K_s]:
                # simulate(sim)
                action = -1
            elif pressed[pygame.K_r]:
                task_env.reset()
                action = -1
            elif pressed[pygame.K_q]:
                done = True
                pygame.quit()
                break
            elif pressed[pygame.K_i]:
                for obj_id in sim.get_existing_object_ids():
                    sim.set_object_semantic_id(semantic_id_count, obj_id)
                    semantic_id_count += 10
            else:
                action = -1

            if action != -1:
                actions_list = task_config["POSSIBLE_ACTIONS"]
                action_name = actions_list[action]

                if action_name == "GRAB_RELEASE":
                    obs, rewards, done, info = task_env.step(action={"action_args": {"iid": chosen_iid}, "action": action})
                else:
                    obs, rewards, done, info = task_env.step(action={"action": action})

                metrics = env.get_metrics()
                img_frame, top_down_map = render_frame(obs, metrics, cross_hair,
                                                       task_env.current_episode,
                                                       sim, task_env, grab_type, window)

                if switch:
                    frames.append(np.array(img_frame))

                # print(img_frame.shape)
                # frames.append(img_frame)
                # top_down_map_frames.append(top_down_map)
                # obs_list.append(obs)
                # metric_list.append(metrics)
                # print(f"Action {action} | color: {color}")
                img_frame.save("debug.jpeg")
                image = pygame.image.load("debug.jpeg")
                screen.blit(image, (0, 0))
                text_position = np.array([0, 0])

                # merge everything in metrics
                metrics.update(obs['cos_eor'])
                metrics.update(info)

                # add obj-key to iid map
                metrics["obj_key_to_iid"] = {}
                metrics["iid_to_obj_key"] = {}
                for ok, oid in metrics["obj_key_to_sim_obj_id"].items():
                    metrics["obj_key_to_iid"][ok] = metrics["sim_obj_id_to_iid"][oid]
                    metrics["iid_to_obj_key"][metrics["sim_obj_id_to_iid"][oid]] = ok

                # skip tracking correctly placed objects
                skip_keys = []
                for ok in list(metrics["object_to_goal_distance"].keys()):
                    ogd = metrics["object_to_goal_distance"][ok]
                    if ogd["current_rec"] in ogd["recs_keys"]:
                        skip_keys.append(ok)

                # get agent-object and object-goal distances
                # obj_goal = {metrics["obj_key_to_iid"][k]: round(sum(v["l2_dists"]) / len(v["l2_dists"]), 2) for k, v in metrics['object_to_goal_distance'].items() if k not in skip_keys}
                obj_goal = {metrics["obj_key_to_iid"][k]: round(min(v["l2_dists"]), 2) for k, v in metrics['object_to_goal_distance'].items() if k not in skip_keys}
                agent_goal = {metrics["obj_key_to_iid"][k]: round(v["l2_dist"], 2) for k, v in metrics['agent_obj_dists'].items() if k not in skip_keys}
                obj_rec_map = {}
                for o, rs in metrics['correct_mapping'].items():
                    if o not in skip_keys:
                        obj_rec_map[metrics['obj_key_to_iid'][o]] = [metrics['obj_key_to_iid'][r] for r in rs]


                add_text(screen, f"Distances -- Object-Goal Geo: {obj_goal} "
                                 f"|| Agent-Object L2: {agent_goal} "
                         , text_position)
                text_position += [0, 30]
                add_text(screen, f"|| Obj-Rec map: {obj_rec_map} ", text_position)
                text_position += [0, 30]
                add_text(screen, f"Rewards -- Object-Goal: {round(rewards['object_to_goal_dist_reward'], 2)} "
                                 f"|| Agent-Object: {round(rewards['agent_to_object_dist_reward'], 2)} "
                                 f"|| Grip: {round(rewards['gripped_success_reward'], 2)} "
                                 f"|| Drop: {round(rewards['drop_success_reward'], 2)} "
                                 f"|| Fail: {round(rewards['gripped_dropped_fail_reward'], 2)} "
                                 f"|| Episode: {round(rewards['episode_success_reward'], 2)} ",
                         text_position)
                text_position += [0, 30]
                episode = task_env.current_episode
                # if holding an object show target receptacles
                if obs["gripped_object_id"] != -1:
                    obj_id = obs['gripped_object_id']
                    obj_key = task_env._env._task.sim_obj_id_to_obj_key[obj_id]
                    recs_keys = episode.get_correct_recs(obj_key)
                    rec_iids = [task_env._env._task.get_iid_from_key(rk) for rk in recs_keys]
                    # print("Receptacle", recs_keys)
                else:
                    rec_iids = []

                objs_success = task_env._episode_obj_success()
                for i, s in enumerate(objs_success):
                    if not s:
                        obj_key = task_env.current_episode.objs_keys[i]
                        obj_iid = task_env._env._task.get_iid_from_key(obj_key)
                        break
                else:
                    obj_iid = -1

                gripped_iid = -1 if obs["gripped_object_id"] == -1 \
                    else task_env._env._task.sim_obj_id_to_iid[obs["gripped_object_id"]]

                add_text(screen, f"Gripped Object: {gripped_iid}"
                                # [[obj_id], agent_obj_pointgoal, agent_goal_pointgoal],
                                 f"|| GD Next Object: {obj_iid}"
                                 f"|| Episode-id: {os.path.split(task_env.current_episode.scene_id)[-1]}"
                                 f"|| Target Receptacles: {str(rec_iids)}"
                                 f"|| Failed Placement: {obs.get('too_far', False)}",
                         text_position)

                pygame.display.update()

                # del metrics["top_down_map"]
                print(f"Collision count: {metrics['collisions']}")
                # del metrics["top_down_map"]
                # print(metrics)


def adjust_igib_objects(sim, meta_keys, object_ids, metadata, metadata_dir, global_mapping):
    relabel_obj_types(sim, meta_keys, object_ids, metadata)
    scene = os.path.split(metadata_dir)[-1]

    new_metadata = {
        "default_mapping": []
    }
    for oidx, (omk, obj_id) in tqdm(enumerate(zip(meta_keys, object_ids)), desc="adjusting", total=len(meta_keys)):
        # filters floor-lamps, wall-pictures
        if "floor" in omk or "picture" in omk:
            metadata[omk]["type"] = "receptacle"

        if "object" not in metadata[omk]["type"]:
            # add receptacle
            if (omk, scene) not in global_mapping["receptacles"]:
                global_mapping["receptacles"].append((metadata[omk]["category"], scene, "igib"))
            continue

        # add object
        if (omk, scene) not in global_mapping["objects"]:
            global_mapping["objects"].append((metadata[omk]["category"], scene, "igib"))


        found_rec = False
        drop_step = 0.001
        dist_moved = 0
        initial_translation = sim.get_translation(obj_id)

        while not found_rec:
            obj_translation = sim.get_translation(obj_id)
            prev_translation = deepcopy(obj_translation)
            obj_translation[1] -= drop_step
            sim.set_translation(obj_translation, obj_id)
            obj_bb = get_bb(sim, obj_id)
            dist_moved += drop_step

            if dist_moved >= 0.5:
                print(f"couldn't place receptacle for {omk} resetting to initial position")
                # this resetting helps the contact-test to work better!
                sim.set_translation(initial_translation, obj_id)
                break

            for ridx, (rmk, rec_id) in enumerate(zip(meta_keys, object_ids)):
                if "receptacle" not in metadata[rmk]["type"]:
                    continue
                # no overlap
                assert ridx != oidx

                try:
                    if mn.math.intersects(obj_bb, get_bb(sim, rec_id)):
                        print(f"{omk} is placed on top of {rmk} // dropped {omk} by {dist_moved}")
                        found_rec = True
                        # set and ensure prev-translation did not intersect
                        sim.set_translation(prev_translation, obj_id)
                        if dist_moved >= 2 * drop_step:
                            assert not mn.math.intersects(get_bb(sim, obj_id), get_bb(sim, rec_id))

                        # save prev non-intersecting position in metadata
                        metadata[omk]["pos"] = prev_translation.tolist()
                        # save mappings
                        if metadata[rmk]["category"] in global_mapping["mapping_igib"]:
                            global_mapping["mapping_igib"][metadata[rmk]["category"]].append((omk, scene, "igib"))
                        else:
                            global_mapping["mapping_igib"][metadata[rmk]["category"]] = [(omk, scene, "igib")]
                        new_metadata["default_mapping"].append((omk, rmk))
                except:
                    import pdb
                    pdb.set_trace()

    relabel_path = os.path.join(metadata_dir, "metadata_v2_readjusted.yaml")
    new_metadata["urdfs"] = metadata
    with open(relabel_path, 'w') as f:
        yaml.dump(new_metadata, f)
    print(f"Dumped: {relabel_path}")
    global_mapping["scenes_parsed"].append(scene)

    for obj_id, mk in zip(object_ids, meta_keys):
        if not sim.contact_test(obj_id):
            num_tries = 5e3
            translation = sim.get_translation(obj_id)
            prev_translation = deepcopy(translation)
            while not sim.contact_test(obj_id) and num_tries > 0:
                translation[1] -= 0.001
                num_tries -= 1
                prev_translation = deepcopy(translation)
                sim.set_translation(translation, obj_id)
            if num_tries >0:
                metadata[mk]["pos"] = list(prev_translation)
                print(f"readjusted: {mk}")
            else:
                print(f"contact test failed for: {mk}")


def relabel_obj_types(sim, meta_keys, object_ids, metadata):
    for oid, mk in tqdm(zip(object_ids, meta_keys), desc="relabeling", total=len(object_ids)):
        if "object" in metadata[mk]["type"] and "receptacle" in metadata[mk]["type"]:
            raise AssertionError # done manually
            typ = input(f"{mk} is o/r:  ")
            metadata[mk]["type"] = "object" if typ == "o" else "receptacle"
            print(f"{mk} is {metadata[mk]['type']} \n")


def manual_adjust_igib_objects(sim, meta_keys, object_ids):
    print(f"Avaialble keys: {meta_keys}")
    stop = False
    while not stop:
        drop_key = input(f"Enter object key to drop:")
        if drop_key.lower() == "stop":
            stop = True
        else:
            for idx, k in enumerate(meta_keys):
                if drop_key in k:
                    drop_value = input(f"Dropping: {k} value")
                    obj_id = object_ids[idx]
                    translation = sim.get_translation(obj_id)
                    translation[1] -= float(drop_value)
                    sim.set_translation(translation, obj_id)
                    print(f"dropped {k} by {drop_value} m")


def get_interesects(sim, meta_keys, object_ids, transformed=False):
    object_nodes = [sim.get_object_scene_node(obj_id) for obj_id in object_ids]
    object_bbs = [node.cumulative_bb for node in object_nodes]
    corners = [get_corners(bb, node) for bb, node in zip(object_bbs, object_nodes)]
    transformed_bbs = [get_bbs_from_corners(cors) for cors in corners]

    for i in range(len(transformed_bbs)):
        for j in range(len(transformed_bbs)):
            if mn.math.intersects(transformed_bbs[i], transformed_bbs[j]):
                if i != j:
                    print(f"{meta_keys[i]} intersects w/ {meta_keys[j]}")
    # if transformed:
    #     obj_bb = habitat_sim.geo.get_transformed_bb(obj_node.cumulative_bb, obj_node.transformation)
    # return obj_bb


def set_agent_on_floor(sim):
    # set agent on the floor
    agent_pos = get_random_point_igib(sim.pathfinder)
    prev_agent_pos = sim.get_agent_state().position
    agent_rot = sim.get_agent_state().rotation
    sim.set_agent_state(agent_pos, agent_rot)
    print(f"Agent position: {agent_pos}")


# def rescale_non_art_objects(sim, global_mapping):
#     import pdb
#     pdb.set_trace()
#
#     # harsh's code
#     with open('data/ycb_object_templates_renamed.json') as f:
#         old_templates = json.load(f)
#
#     obj_attr_mgr = sim.get_object_template_manager()
#     templates = obj_attr_mgr.load_configs("./data/objects")
#     random.shuffle(templates)
#
#     for obj_template in templates:
#         default_scale = [1.0] * 3 if obj_template not in templates else templates[obj_template]["scale"]
#         print(f"Default scale: {default_scale}")
#         attr = obj_attr_mgr.get_template_by_ID(obj_template)
#         obj_attr_mgr.register_template(attr)
#         object_id = sim.add_object_by_handle(attr.handle)
#         point = get_surface_point_for_placement(sim, 120)
#         obj_bb = get_aabb(object_id, sim)
#         point[1] += (obj_bb.size()[1] / 2.0) + 0.01
#         sim.set_translation(point, object_id)


def set_pos_rot(obj_attr_mgr, obj_temp, scale, rot, sim, curr_obj_id, rec_id=120):
    sim.remove_objects([curr_obj_id])
    if type(obj_temp) == int:
        attr = obj_attr_mgr.get_template_by_ID(obj_temp)
    else:
        attr = obj_temp
    attr.scale = scale
    obj_attr_mgr.register_template(attr)
    object_id = sim.add_object_by_handle(attr.handle)
    if rec_id != -1:
        point = get_surface_point_for_placement(sim, None, object_id, rec_id)
        obj_bb = get_aabb(object_id, sim)
        point[1] += (obj_bb.size()[1] / 2.0) + 0.01
        sim.set_translation(point, object_id)
    rot = quat_from_coeffs(rot)
    rot = quat_to_magnum(rot)
    sim.set_rotation(rot, object_id)
    rot_list = [*rot.vector, rot.scalar]
    return object_id, scale, rot_list


def spin_and_dump_vid(sim, obj_id, dump_path, y_rot=None, rotate=True):
    if y_rot is None:
        y_rot = quat_from_angle_axis(np.pi / 8, np.array([0, 0, 1]))

    frames = []
    for i in range(16):
        if rotate:
            start_rot = quat_from_magnum(sim.get_rotation(obj_id))
            new_rot = start_rot * y_rot
            sim.set_rotation(quat_to_magnum(new_rot), obj_id)
        # run failed pick-up action for stepping
        obs = sim.step(action=6)
        frames.append(obs["rgb_3rd_person"])
    images_to_video(frames, os.path.dirname(dump_path), os.path.split(dump_path)[-1], fps=2)
    print(f"Dumped: {dump_path}")


def build_viz(
        sim,
        templates,
        manual_rescale_idx,
        obj_attr_mgr,
        ycb_visual_dump_dir,
        dump_visuals,
        ycb_scale_rots,
        default_scale_ab,
        default_rot,
        ab_metadata,
        ab_models_dir,
        ab_visual_dump_dir,
        ab_scale_rots_path,
        object_id
    ):
    # build visualizations of ycb objects
    """
     ['coffee_table_5_0.urdf: 40', 'table_18_0.urdf: 121',
     'table_9_0.urdf: 122', 'table_lamp_22_0.urdf: 123']
    """

    # arrange viewing table when running for first time
    if 40 in sim.get_both_existing_object_ids()["art"]:
        coffee_table_pos_rot = sim.get_translation(40), sim.get_rotation(40)
        table_obj_id = 122
        table_pos_rot = sim.get_translation(table_obj_id), sim.get_rotation(table_obj_id)
        obj_ids = sim.get_both_existing_object_ids()
        obj_ids = obj_ids["art"] + obj_ids["non_art"]
        obj_ids.remove(table_obj_id)
        sim.remove_objects(obj_ids)
        # set at coffee table with original height
        sim.set_translation([coffee_table_pos_rot[0][0], table_pos_rot[0][1], coffee_table_pos_rot[0][2]], table_obj_id)
        sim.set_rotation(coffee_table_pos_rot[-1], table_obj_id)

    agent_state = {
        "position": np.array([-0.04932895, 0.0314267, 1.052413], dtype=np.float32),
        "rotation": quat_from_coeffs([0.0, -0.819166600704193, 0.0, -0.573555707931519])
    }
    sim.set_agent_state(**agent_state)
    if dump_visuals == "ycb":
        print(f"Dumping {len(templates)} objects in total")
        count = 10
        while manual_rescale_idx < len(templates):
            handle = obj_attr_mgr.get_template_by_ID(templates[manual_rescale_idx]).handle
            file_path = f"{ycb_visual_dump_dir}/{os.path.split(handle)[-1]}" + ".mp4"

            if handle not in ycb_scale_rots["accepted"] or os.path.exists(file_path):
                manual_rescale_idx += 1
                print(f"skipping: {file_path}")
                continue
            scale, rot = ycb_scale_rots["accepted"][handle]["scale"], ycb_scale_rots["accepted"][handle]["rotation"]
            object_id, scale, rot = \
                set_pos_rot(obj_attr_mgr, templates[manual_rescale_idx], scale, rot, sim, object_id, table_obj_id)
            spin_and_dump_vid(sim, object_id, f"{ycb_visual_dump_dir}/{os.path.split(handle)[-1]}")
            manual_rescale_idx += 1
            count = count - 1
            if count == 0:
                break

    elif dump_visuals == "ab":
        ab_metadata_scale_path = "cos_eor/scripts/dump/ab_manual_scale_fil.npy"
        # load all handles
        ab_model_paths = [os.path.join(ab_models_dir, item["path"]) for item in ab_metadata["data"]]
        ab_config_paths = [path.split(".")[0] + ".object_config.json" for path in ab_model_paths]
        for path in tqdm(ab_config_paths, desc="Loading AB objects"):
            obj_attr_mgr.load_configs(path)

        cat_metadata = defaultdict(list)
        cat_counter = Counter()
        for item in ab_metadata["data"]:
            item_type = item['product_type'][0]['value']
            cat_metadata[item_type].append(item)
            cat_counter[item_type] += 1
        print(f"Category distribution: \n{cat_counter.most_common()}")
        print(f"Total categories: \n{len(cat_counter.keys())} and Total count: {sum(cat_counter.values())}")

        rot_dict = {
            0: [0, 0, 0, 1.0],
            1: [-0.5, 0.5, 0.5, 0.5]
        }
        axis_dict = {
            0: [0, 1, 0],
            1: [0, 0, 1]
        }

        # store default rots and scales
        cat_rot_scale = np.load(ab_scale_rots_path, allow_pickle=True).item()
        fil_ab_metadata = []

        max_downscale = 0.8
        max_side_possible = 0.5
        ascend_cats = cat_counter.most_common()[::-1]
        for cat, cat_cnt in ascend_cats:
            print(f"Processing {cat} which has {len(cat_metadata[cat])} objects")
            cat_dump_dir = os.path.join(ab_visual_dump_dir, cat)
            os.makedirs(cat_dump_dir, exist_ok=True)
            # scale, rot, axis = default_scale_ab, rot_dict[0], axis_dict[0]
            rotate = True
            for item in tqdm(cat_metadata[cat], desc=f"processing {cat}"):
                scale, rot, axis = default_scale_ab, cat_rot_scale[cat]["rot"], cat_rot_scale[cat]["rot_axis"]
                path = item["path"]
                handle = path.split(".")[0] + ".object_config.json"
                handle = os.path.join(ab_models_dir, handle)
                item["path"] = handle
                it_template = obj_attr_mgr.get_template_by_handle(handle)
                y_rot = quat_from_angle_axis(np.pi / 8, np.array(axis))
                bb_max_side = max([item['3d_extent_x'], item['3d_extent_y'], item['3d_extent_z']])
                if bb_max_side > max_side_possible:
                    item_name = item["item_name"][0]["value"]

                    if bb_max_side * max_downscale <= max_side_possible:
                        scale = [max_side_possible / bb_max_side] * 3
                        print(f"rescaled: {item_name}")
                    else:
                        print(f"skipping too large: {item_name}")
                        continue
                item["scale"] = scale

                # try:
                #     # first annotate rotations manually
                #     while True:
                #         y_rot = quat_from_angle_axis(np.pi / 8, np.array(axis))
                #         object_id, scale, rot = set_pos_rot(obj_attr_mgr, it_template, scale, rot, sim, object_id,
                #                                             table_obj_id)
                #         spin_and_dump_vid(sim, object_id, f"debug-data/{cat}-debug", y_rot, rotate)
                #
                #         inp = input("c: change rot | r: manual | s: save and exit | e: exit: ").lower()
                #         if inp == "c":
                #             if rot == rot_dict[0]:
                #                 rot, axis = rot_dict[1], axis_dict[1]
                #             else:
                #                 rot, axis = rot_dict[0], axis_dict[0]
                #         elif inp == "s":
                #             cat_rot_scale[cat] = {
                #                 "rot": rot,
                #                 "rot_axis": axis
                #             }
                #             break
                #         elif inp == "e":
                #             break
                #         elif inp == "d":
                #             import pdb
                #             pdb.set_trace()
                # except:
                #     import pdb
                #     pdb.set_trace()

                # dump videos
                # object_id, scale, rot = set_pos_rot(obj_attr_mgr, it_template, scale, rot, sim, object_id, table_obj_id)
                # spin_and_dump_vid(sim, object_id, f"{cat_dump_dir}/{os.path.split(handle)[-1]}", y_rot, rotate)
                fil_ab_metadata.append(item)

        import pdb
        pdb.set_trace()

        # store
        ab_metadata["data"] = fil_ab_metadata
        np.save(ab_metadata_scale_path, ab_metadata)

    elif dump_visuals == "rcad":
        rcad_dir = "data/replica-cad/configs/objects/"
        rcad_scale_rots = "cos_eor/scripts/dump/rcad_scale_fil.npy"
        rcad_visual_dump_dir = "cos_eor/scripts/dump/rcad_obj_viz"

        rcad_configs = [cfg for cfg in os.listdir(rcad_dir) if cfg.endswith(".object_config.json")]
        rcad_configs = [os.path.join(rcad_dir, cfg) for cfg in rcad_configs]
        rcad_metadata = defaultdict(list)

        rot_dict = {
            0: [0, 0, 0, 1.0],
            1: [-0.5, 0.5, 0.5, 0.5]
        }
        axis_dict = {
            0: [0, 1, 0],
            1: [0, 0, 1]
        }
        skip_cats = [
            "bike",
            "table",
            "cabinet",
            "stool",
            "tv_screen",
            "refrigerator",
            "tvstand",
            "monitor",
            "rack",
            "sofa",
            "beanbag",
            "mat",
            "monitor_stand",
            "indoor_plant",
            "rug",
            "bin",
            "chair",
            "picture",
            "tv screen",
            "tv object",
            "indoor plant",
            "monitor stand",
        ]

        cat_counter = Counter()
        for path in tqdm(rcad_configs, desc="Loading Replica-CAD objects"):
            obj_attr_mgr.load_configs(path)
            cat = preprocess(path)
            cat = cat.replace("frl apartment", "").replace("wall", "").strip()
            if cat not in skip_cats:
                rcad_metadata[cat].append({
                    "path": path,
                    "cat": cat
                })
                cat_counter[cat] += 1
            else:
                print(f"Skipping {cat} due to skip list!")

        max_downscale = 0.7
        max_side_possible = 0.5
        ascend_cats = cat_counter.most_common()[::-1]
        fil_rcad_metadata = []
        for cat, cat_cnt in ascend_cats:
            # print(f"Processing {cat} which has {cat_counter[cat]} objects")
            cat_dump_dir = os.path.join(rcad_visual_dump_dir, cat)
            os.makedirs(cat_dump_dir, exist_ok=True)
            scale, rot, axis = [1.0] * 3, rot_dict[0], axis_dict[0]
            rotate = True

            for item in tqdm(rcad_metadata[cat], desc=f"processing {cat}"):
                path = item["path"]
                it_template = obj_attr_mgr.get_template_by_handle(path)
                y_rot = quat_from_angle_axis(np.pi / 8, np.array(axis))
                object_id, scale, rot = set_pos_rot(obj_attr_mgr, it_template, scale, rot, sim, object_id, -1)
                # spin_and_dump_vid(sim, object_id, f"{cat_dump_dir}/{os.path.split(path)[-1]}", y_rot, rotate)
                bb = get_bb(sim, object_id)
                bb_max_side = max(bb.size())
                if bb_max_side > max_side_possible:
                    if bb_max_side * max_downscale <= max_side_possible:
                        scale = [max_side_possible / bb_max_side] * 3
                        print(f"rescaled: {path}")
                    elif cat in ["cloth", "umbrella"]:
                        pass
                    else:
                        import pdb
                        pdb.set_trace()
                        print(f"skipping too large: {path}")
                        continue
                item["scale"] = scale
                item["rot"] = rot
                fil_rcad_metadata.append(item)

        import pdb
        pdb.set_trace()

        data = {
            "data": fil_rcad_metadata,
            "skip_cats": skip_cats
        }
        np.save(rcad_scale_rots, data)
        cat_counter = Counter([item["cat"] for item in fil_rcad_metadata])
        print(f"Category distribution: \n{cat_counter.most_common()}")
        print(f"Total categories: \n{len(cat_counter.keys())} and Total count: {sum(cat_counter.values())}")
        import pdb
        pdb.set_trace()

    elif dump_visuals == "gso":
        gso_dir = "data/google_object_dataset"
        gso_meta_path = "cos_eor/scripts/dump/gso_dump.npy"
        gso_scale_rots = "cos_eor/scripts/dump/gso_scale_fil.npy"
        gso_visual_dump_dir = "cos_eor/scripts/dump/gso_obj_viz"
        glob_dump = "data/gso_glob.npy"
        paths = list(np.load(glob_dump, allow_pickle=True))
        gso_configs = [cfg for cfg in paths if cfg.endswith(".object_config.json")]
        gso_metadata = np.load(gso_meta_path, allow_pickle=True).item()

        rot_dict = {
            0: [0, 0, 0, 1.0],
            1: [-0.5, 0.5, 0.5, 0.5]
        }
        axis_dict = {
            0: [0, 1, 0],
            1: [0, 0, 1]
        }
        skip_cats = [
            "Car Seat"
        ]

        # read metadata and categorize
        cat_counter = Counter()
        na_name_counter = Counter()
        for k,v in gso_metadata.items():
            v['name'] = v['name'].split("_")
            v['name'] = ' '.join(v['name'])
            if "categories" in v and len(v["categories"]) > 0:
                cats = list(v["categories"].values())
                if len(cats) > 1:
                    # never executed
                    print(f"Cats: {cats}")
                cat = cats[0]
            else:
                cat = "n/a"
            if cat in skip_cats:
                continue
            if cat == "n/a":
                na_name_counter[v['name']] += 1
            cat_counter[cat] += 1

        # filter and load configs
        cat_gso_metadata = defaultdict(list)
        assert len(gso_configs) == len(gso_metadata)
        for path in tqdm(gso_configs, desc="Loading GSO objects"):
            obj_attr_mgr.load_configs(path)
            gso_key = os.path.split(path)[-1].split('.object_config.json')[0]
            if "categories" in gso_metadata[gso_key]:
                cats = list(gso_metadata[gso_key]["categories"].values())
            else:
                cats = []

            cat = "n/a" if len(cats) == 0 else cats[0]
            if cat not in skip_cats:
                cat_gso_metadata[cat].append({
                    "path": path,
                    "cat": cat,
                    "gso_metadata": gso_metadata[gso_key]
                })
            else:
                print(f"Skipping {cat} due to skip list!")

        max_downscale = 0.8
        max_side_possible = 0.5
        ascend_cats = cat_counter.most_common()[::-1]
        fil_gso_metadata = []
        for cat, cat_cnt in ascend_cats:
            print(f"Processing {cat} which has {cat_counter[cat]} objects")
            cat_dump_dir = os.path.join(gso_visual_dump_dir, cat)
            os.makedirs(cat_dump_dir, exist_ok=True)
            scale, rot, axis = [1.0] * 3, rot_dict[1], axis_dict[1]
            rotate = True

            for item in tqdm(cat_gso_metadata[cat], desc=f"processing {cat}"):
                path = item["path"]
                it_template = obj_attr_mgr.get_template_by_handle(path)
                y_rot = quat_from_angle_axis(np.pi / 8, np.array(axis))
                object_id, scale, rot = set_pos_rot(obj_attr_mgr, it_template, scale, rot, sim, object_id, table_obj_id)
                spin_and_dump_vid(sim, object_id, f"{cat_dump_dir}/{os.path.split(path)[-1]}", y_rot, rotate)
                bb = get_bb(sim, object_id)
                bb_max_side = max(bb.size())
                if bb_max_side > max_side_possible:
                    if bb_max_side * max_downscale <= max_side_possible:
                        scale = [max_side_possible / bb_max_side] * 3
                        print(f"rescaled: {path}")
                    else:
                        print(f"skipping too large: {path}")
                        continue
                item["scale"] = scale
                item["rot"] = rot
                fil_gso_metadata.append(item)

        import pdb
        pdb.set_trace()

        data = {
            "data": fil_gso_metadata,
            "skip_cats": skip_cats
        }
        np.save(gso_scale_rots, data)
        cat_counter = Counter([item["cat"] for item in fil_gso_metadata])
        print(f"Category distribution: \n{cat_counter.most_common()}")
        print(f"Total categories: \n{len(cat_counter.keys())} and Total count: {sum(cat_counter.values())}")
        import pdb
        pdb.set_trace()


def debug_sim_viewer(sim, do_reset=False, meta_keys=[], object_ids=[],
                     metadata={}, metadata_dir=None, global_mapping=None, adjust_igib=False, dump_visuals=""):
    """debugging / annotation viewer using only simulator."""
    window = (720, 720)
    semantic_id_count = 20
    switch = False
    frames = []

    assert "BULLET" in str(sim.get_physics_simulation_library())
    if do_reset:
        obs = sim.reset()
    obj_attr_mgr = sim.get_object_template_manager()

    # ab viz
    ab_metadata_path = "cos_eor/scripts/dump/ab_manual_fil.npy"
    ab_metadata_scale_path = "cos_eor/scripts/dump/ab_manual_scale_fil.npy"
    ab_scale_rots_path = 'cos_eor/scripts/dump/ab_scale_rotation.npy'
    ab_models_dir = f"data/amazon-berkeley/3dmodels/original/"
    ab_metadata = np.load(ab_metadata_path, allow_pickle=True).item()
    ab_visual_dump_dir = "cos_eor/scripts/dump/ab_obj_viz"

    # ycb viz
    ycb_visual_dump_dir = "cos_eor/scripts/dump/ycb_obj_viz"
    ycb_scale_rots_path = 'cos_eor/scripts/dump/ycb_scale_rotation.yaml'
    if os.path.exists(ycb_scale_rots_path):
        with open(ycb_scale_rots_path, "r") as f:
            ycb_scale_rots = yaml.load(f)

    # "accepted" / "rejected" keys are used to denote if we have filtered the object during
    # manual annotation
    ycb_scale_rots_accepted = ycb_scale_rots["accepted"]

    templates = obj_attr_mgr.load_configs("./data/objects")
    manual_rescale_idx = 0
    rescale_done = True
    object_id = -1
    # default scale and rots
    default_scale = [1.2] * 3
    default_scale_ab = [1.0] * 3
    default_rot = [-0.5, 0.5, 0.5, 0.5]

    if dump_visuals != "":
        build_viz(
            sim,
            templates,
            manual_rescale_idx,
            obj_attr_mgr,
            ycb_visual_dump_dir,
            dump_visuals,
            ycb_scale_rots,
            default_scale_ab,
            default_rot,
            ab_metadata,
            ab_models_dir,
            ab_visual_dump_dir,
            ab_scale_rots_path,
            object_id
        )

    if adjust_igib:
        adjust_igib_objects(sim, meta_keys, object_ids, metadata, metadata_dir, global_mapping)
        return

    pygame.init()
    screen = pygame.display.set_mode(window)
    done = False
    while not done:
        for event in pygame.event.get():
            # print(f"Event: {event}")
            if event.type == pygame.QUIT:
                done = True
                pygame.quit()

            pressed = pygame.key.get_pressed()

            if pressed[pygame.K_UP]:
                action = 1
            elif pressed[pygame.K_RIGHT]:
                action = 3
            elif pressed[pygame.K_LEFT]:
                action = 2
            # elif pressed[pygame.K_SPACE]:
            #     avail_iids = list(np.unique(obs["semantic"]))
            #     if 0 in avail_iids:
            #         avail_iids.remove(0)
            #     # add prompt
            #     add_text(screen, f"Pick/Place from Semantic Ids: {avail_iids}", (0, 0))
            #     pygame.display.update()
            #     chosen_iid = int(input(f"Visible iids for pick/place -- {avail_iids}: "))
            #     action = 3
            elif pressed[pygame.K_r]:
                # start/stop recording and dump
                if not switch:
                    print(f"Started Recording...")
                    switch = True
                else:
                    switch = False
                    try:
                        record_path = "./debug-data/manual_control/"
                        time_stamp = time.time()
                        # curr_episode = env._current_episode
                        # scene_name = os.path.split(curr_episode.scene_id.split(".")[0])[-1]
                        identifier = input("video identifier: ")
                        video_path = os.path.join(record_path, f"sim_{str(time_stamp)}_{identifier}")
                        images_to_video(frames, "./", video_path, fps=2)
                    except:
                        print(f"Some error in saving video")
                        video_path = None
                        import pdb
                        pdb.set_trace()
                    print(f"Saved: {video_path}")
                    frames = []
            elif pressed[pygame.K_PAGEUP]:
                action = 4
            elif pressed[pygame.K_PAGEDOWN]:
                action = 5
            # debug
            elif pressed[pygame.K_d]:
                import pdb
                pdb.set_trace()
                action = -1
            # auto-adjust heights using contact test.
            elif pressed[pygame.K_a]:
                adjust_igib_objects(sim, meta_keys, object_ids, metadata, metadata_dir,global_mapping)
            elif pressed[pygame.K_m]:
                manual_adjust_igib_objects(sim, meta_keys, object_ids)
            elif pressed[pygame.K_c]:
                get_interesects(sim, meta_keys, object_ids)
            elif pressed[pygame.K_s]:
                # simulate(sim)
                action = -1
            elif pressed[pygame.K_f]:
                set_agent_on_floor(sim)
                action = -1
            elif pressed[pygame.K_r]:
                sim.reset()
                action = -1
            elif pressed[pygame.K_y]:
                sim.set_random_semantic_ids()
                handle = obj_attr_mgr.get_template_by_ID(templates[manual_rescale_idx]).handle
                if rescale_done:
                    scale, rot = default_scale, default_rot

                    # skip already scaled / rejected objects
                    while (manual_rescale_idx < len(templates) and handle in ycb_scale_rots_accepted) \
                            or handle in ycb_scale_rots["rejected"]:
                        manual_rescale_idx += 1
                        handle = obj_attr_mgr.get_template_by_ID(templates[manual_rescale_idx]).handle
                        print(f"skipping {handle}: {manual_rescale_idx+1}/{len(templates)}")

                    # add current object in scene
                    if manual_rescale_idx < len(templates):
                        print(f"adjusting {handle}: {manual_rescale_idx+1}/{len(templates)} "
                              f"w/ default scale: {scale} and rotation:{rot}")
                        object_id, scale, rot = \
                            set_pos_rot(obj_attr_mgr, templates[manual_rescale_idx], scale, rot, sim, object_id)
                        rescale_done = False
                    else:
                        print(f"Finished labeling all!")
                else:
                    scale_or_rot_finish = input(f"curr scale: {scale} and rot: {rot} for {handle}: ")
                    if scale_or_rot_finish == "f":
                        obj_name = os.path.split(handle)[-1].split(".")[0]
                        ycb_scale_rots_accepted[handle] = {"rotation": rot,
                                                "scale": scale,
                                                "global_object": (obj_name, "none", "ycb_or_ab_adjusted")}
                        manual_rescale_idx += 1
                        rescale_done = True
                        ycb_scale_rots["accepted"] = ycb_scale_rots_accepted
                        # save current progress
                        with open(ycb_scale_rots_path, "w") as f:
                            yaml.dump(ycb_scale_rots, f)
                        print(f"Saved data: {scale} for {obj_name}")

                    elif scale_or_rot_finish == "r":
                        rot = eval(input("rot: "))
                        # run loop for manual adjustment
                        object_id, scale, rot = \
                            set_pos_rot(obj_attr_mgr, templates[manual_rescale_idx], scale, rot, sim, object_id)

                    elif scale_or_rot_finish == "s":
                        scale = eval(input("scale: "))
                        if type(scale) == int or type(scale) == float:
                            scale = [float(scale)] * 3
                        else:
                            assert type(scale) == list
                        # run loop for manual adjustment
                        object_id, scale, rot = \
                            set_pos_rot(obj_attr_mgr, templates[manual_rescale_idx], scale, rot, sim, object_id)

                    # reject the item
                    elif scale_or_rot_finish == "n":
                        ycb_scale_rots["rejected"].append(handle)
                        # save current progress
                        with open(ycb_scale_rots_path, "w") as f:
                            yaml.dump(ycb_scale_rots, f)
                        manual_rescale_idx += 1
                        rescale_done = True
                        print(f"Skipped and saved data")

                action = -1


            elif pressed[pygame.K_q]:
                done = True
                pygame.quit()
                break
            elif pressed[pygame.K_i]:
                for obj_id in sim.get_existing_object_ids():
                    sim.set_object_semantic_id(semantic_id_count, obj_id)
                    semantic_id_count += 10
            else:
                action = -1

            if action != -1:
                obs = sim.step(action)
                img_frame = render_sim_frame(obs, sim)
                if switch:
                    frames.append(np.array(img_frame))

                # very ugly hack, couldn't figure out a better way to refresh screen :/
                img_frame.save("debug_sim.jpeg")
                image = pygame.image.load("debug_sim.jpeg")
                screen.blit(image, (0, 0))
                pygame.display.update()
