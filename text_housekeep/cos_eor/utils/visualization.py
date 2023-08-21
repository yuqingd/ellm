import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw

from text_housekeep.habitat_lab.habitat.utils.visualizations import maps, fog_of_war
from text_housekeep.cos_eor.utils.geometry import get_polar_angle, get_semantic_centroids

def get_top_down_map_sim(sim, pathfinder, object_positions, goal_positions, navmesh_settings, fog_of_war_mask=None, ignore_objects=True, 
    draw_fow=True, draw_agent=False, draw_object_start_pos=False, draw_object_final_pos=False
):
    sim.recompute_navmesh(pathfinder, navmesh_settings, (not ignore_objects))

    agent_position = sim.get_agent(0).get_state().position
    top_down_map = maps.get_topdown_map(
        pathfinder,
        agent_position[1],
        256
    )

    a_y, a_x = maps.to_grid(
        agent_position[2],
        agent_position[0],
        top_down_map.shape[0:2],
        sim=sim,
    )

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
    
    agent_state = sim.get_agent(0).get_state()
    # quaternion is in x, y, z, w format
    ref_rotation = agent_state.rotation
    agent_rotation = get_polar_angle(ref_rotation)
    # print(agent_rotation)
    
    if draw_fow:
        if fog_of_war_mask is None:
            fog_of_war_mask = np.zeros_like(top_down_map)
        
        fog_of_war_mask = fog_of_war.reveal_fog_of_war(
            np.ones(top_down_map.shape),
            fog_of_war_mask,
            np.array([a_y, a_x]),
            agent_rotation,
            fov = 180,
            max_line_len = 500
        )
    top_down_map = maps.colorize_topdown_map(top_down_map)
    
    if draw_agent:
        top_down_map = maps.draw_agent(
            image=top_down_map,
            agent_center_coord=[a_y, a_x],
            agent_rotation=agent_rotation,
            agent_radius_px=min(top_down_map.shape[0:2]) / 32,
        )
    
    if draw_object_start_pos:
        top_down_map = maps.draw_object_info(top_down_map, grid_object_positions, suffix="")

    if draw_object_final_pos:
        top_down_map = maps.draw_object_info(top_down_map, grid_goal_positions, suffix="g")
    
    return top_down_map, fog_of_war_mask


def get_top_down_map(env, pathfinder, fog_of_war_mask=None, ignore_objects=True, draw_fow=True, draw_agent=False, draw_object_start_pos=False, draw_object_final_pos=False, draw_object_curr_pos=False):
    
    episode = env.current_episode
    object_positions = [obj.position for obj in episode.objects]
    goal_positions = [obj.position for obj in episode.goals]

    top_down_map, fog_of_war_mask = get_top_down_map_sim(
        env._sim, 
        pathfinder,
        object_positions,
        goal_positions,
        env._sim.navmesh_settings,
        fog_of_war_mask, 
        ignore_objects, 
        draw_fow, 
        draw_agent, 
        draw_object_start_pos, 
        draw_object_final_pos
    )


    object_positions = [obj.position for obj in episode.objects]
    grid_current_positions = [None] * len(object_positions)
    
    for sim_obj_id in env._sim.get_existing_object_ids():
        if sim_obj_id != env._task.agent_object_id:
            obj_id = env._task.sim_obj_id_to_ep_obj_id[sim_obj_id]
            position = env._sim.get_translation(sim_obj_id)
            curr_pos = maps.to_grid(position[2], position[0], top_down_map.shape[0:2], sim=env._sim)
            grid_current_positions[obj_id] = curr_pos

    if draw_object_curr_pos:
        top_down_map = maps.draw_object_info(top_down_map, filter(None, grid_current_positions), suffix="c")

    return top_down_map, fog_of_war_mask


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


def render_frame_explore_sim(observations, window=(1024, 1024)):
    rgb = np.array(observations["rgb"].squeeze().cpu(), dtype=np.uint8)
    semantic = np.array(observations["semantic"].squeeze().cpu(), dtype=np.uint8)
    depth = np.array(observations["depth"].squeeze().unsqueeze(dim=-1).cpu())
    occ_map = np.array(observations["coarse_occupancy"].squeeze().cpu(), dtype=np.uint8)
    cos_eor = observations["cos_eor"][0]
    gripped_object_id = observations["gripped_object_id"][0]
    collision = observations["collision"][0]

    img_frame = Image.fromarray(rgb)

    from habitat_sim.utils.common import d3_40_colors_rgb
    semantic_img = Image.new("P", (semantic.shape[1], semantic.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGBA")
    semantic_img = semantic_img.resize((256, 256))
    iids, iid_centroids = get_semantic_centroids(semantic)
    for i, sc in zip(iids, iid_centroids):
        sim_obj_id = cos_eor["iid_to_sim_obj_id"][i]
        obj_type = cos_eor["sim_obj_id_to_type"][sim_obj_id]

        # when holding an object show only receptacles
        if obj_type == "obj" and gripped_object_id != -1:
            continue

        # when holding nothing show only objects
        if obj_type == "rec" and gripped_object_id == -1:
            continue
        ann_key = cos_eor["sim_obj_id_to_obj_key"][sim_obj_id]
        # obj_ann = cos_eor["object_annotations"][ann_key]
        # obj_id = cos_eor["sim_obj_id_to_ep_obj_id"][sim_obj_id]
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
        # draw_string = f"{i}"
        draw_string = f"{obj_type}:{ann_key}"
        # draw_string = f"{s} | {ann_key} | {draw_string} | {or_string}"
        # font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 10)
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 20)
        draw = ImageDraw.Draw(semantic_img)
        draw.text((sc[1], sc[0]), draw_string, (0, 0, 0), font=font)

    d_im = depth_to_rgb(depth, clip_max=1.0)[:, :, 0]
    depth_map = np.stack([d_im for _ in range(3)], axis=2)
    depth = Image.fromarray(depth_map)

    # scale up the third-person
    img_frame = img_frame.resize(window)

    overlay_depth_img = depth.resize((256, 256))
    img_frame.paste(overlay_depth_img, box=(32, 32))

    # overlay_semantic_img = semantic_img.resize((256, 256))
    img_frame.paste(semantic_img, box=(32, 32+256+32))

    occ_map_im = Image.fromarray(occ_map)
    overlay_occ_map = occ_map_im.resize((256, 256))
    img_frame.paste(overlay_occ_map, box=(32, 32+(256+32)*2))

    img_frame = np.array(img_frame)

    if collision == 1:
        border = 20
        # add a red border for collisions
        img_frame[:border, :] = [255, 0, 0]
        img_frame[:, :border] = [255, 0, 0]
        img_frame[-border:, :] = [255, 0, 0]
        img_frame[:, -border:] = [255, 0, 0]

    return img_frame


def add_text(frame, text, pos):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,)*3)
    return frame
