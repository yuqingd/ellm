import numpy as np
import pybullet as p
from PIL import Image
import attr
from habitat_sim.nav import NavMeshSettings
from habitat_sim.physics import MotionType
import habitat_sim
import magnum as mn
import quaternion

def make_border_red(img):
    border_color = [255,0,0]
    border_width = 10
    img[:, :border_width] = border_color
    img[:border_width, :] = border_color
    img[-border_width:, :] = border_color
    img[:, -border_width:] = border_color
    return img

def get_angle(x,y):
    """
    Gets the angle between two vectors in radians.
    """
    x_norm = x / np.linalg.norm(x)
    y_norm = y / np.linalg.norm(y)
    return np.arccos(np.clip(np.dot(x_norm, y_norm), -1, 1))


def make_render_only(obj_idx, sim):
    if hasattr(MotionType, 'RENDER_ONLY'):
        sim.set_object_motion_type(MotionType.RENDER_ONLY, obj_idx)
    else:
        sim.set_object_motion_type(MotionType.KINEMATIC, obj_idx)
        sim.set_object_is_collidable(False, obj_idx)

def has_collision(name, colls):
    for coll0, coll1 in colls:
        if coll0['name'] == name or coll1['name'] == name:
            return True
    return False


def get_collision_matches(link, colls, search_key='link'):
    matches = []
    for coll0, coll1 in colls:
        if coll0[search_key] == link or coll1[search_key] == link:
            matches.append((coll0, coll1))
    return matches

def get_other_matches(link, colls):
    matches = get_collision_matches(link, colls)
    other_surfaces = [
            b if a['link'] == link else a
            for a,b in matches
            ]
    return other_surfaces

def coll_name(coll, name):
    return coll_prop(coll, name, 'name')

def coll_prop(coll, val, prop):
    return coll[0][prop] == val or coll[1][prop] == val

def coll_link(coll, link):
    return coll_prop(coll, link, 'link')

def swap_axes(x):
    x[1], x[2] = x[2], x[1]
    return x

class PbRenderer:
    def __init__(self, cam_start, render_dim, use_sim, args):
        self.cam_pos = np.array(cam_start)
        self.rot = np.zeros((2,))
        self.render_dim = render_dim
        self.pb_sim = use_sim
        self.client_id = use_sim.pc_id
        self.targ_ee_pos = np.array([0.5, 0.0, 1.0])
        self.render_i = 0
        self.sphere_id = None
        if args.mp_margin is not None:
            self.sphere_id = self.pb_sim.add_sphere(args.mp_margin)

    def move_cam(self, delta_pos, delta_rot):
        self.cam_pos += delta_pos
        self.rot += delta_rot

    def render(self):
        #cam_pos = swap_axes(self.cam_pos)
        #look_at = swap_axes(self.look_at)
        cam_pos = self.cam_pos
        if self.pb_sim.is_setup:
            look_at = np.array(self.pb_sim.get_robot_transform().translation)
        else:
            look_at = [0,0,0]

        view_mat = p.computeViewMatrix(cam_pos, look_at, [0.0,1.0,0.0])
        #view_mat = p.computeViewMatrixFromYawPitchRoll(cam_pos,
        #        1.0, self.rot[0], self.rot[1], 0, 2)
        proj_mat = p.computeProjectionMatrixFOV(
                fov=90, aspect=1,
                nearVal=0.1, farVal=100.0)
        img = p.getCameraImage(self.render_dim, self.render_dim, viewMatrix=view_mat,
                projectionMatrix=proj_mat, physicsClientId=self.client_id)[2]
        p.stepSimulation(physicsClientId=self.client_id)

        if self.sphere_id is not None:
            self.pb_sim.set_position(self.pb_sim.get_ee_pos(), self.sphere_id)
        colls = self.pb_sim.get_collisions()
        if len(colls) != 0:
            print('Update', self.render_i, colls)

        self.render_i += 1

        return img[:,:,:3]

    def move_ee(self, delta_ee_pos):
        #if delta_ee_pos[0] > 0:
        #    self.pb_sim.arm_start += 1
        #    print('Increasing arm start to ', self.pb_sim.arm_start)
        #delta_ee_pos = [0,0,0]
        if not self.pb_sim.is_setup:
            return
        self.targ_ee_pos = self.targ_ee_pos + delta_ee_pos
        js = self.pb_sim._ik.calc_ik(self.targ_ee_pos)
        self.pb_sim.set_arm_pos(js)


class IkHelper:
    def __init__(self, arm_start):
        self._arm_start = arm_start
        self._arm_len = 7

    def setup_sim(self):
        self.pc_id = p.connect(p.DIRECT)

        self.robo_id = p.loadURDF(
                "./orp/robots/opt_fetch/robots/fetch_onlyarm.urdf",
                basePosition=[0, 0, 0],
                useFixedBase=True,
                flags=p.URDF_USE_INERTIA_FROM_FILE,
                physicsClientId=self.pc_id
                )

        p.setGravity(0, 0, -9.81, physicsClientId=self.pc_id)
        JOINT_DAMPING = 0.5
        self.pb_link_idx = 7

        for link_idx in range(15):
            p.changeDynamics(
                    self.robo_id,
                    link_idx,
                    linearDamping=0.0,
                    angularDamping=0.0,
                    jointDamping=JOINT_DAMPING,
                    physicsClientId=self.pc_id
                    )
            p.changeDynamics(self.robo_id, link_idx, maxJointVelocity=200,
                    physicsClientId=self.pc_id)

    def set_arm_state(self, joint_pos, joint_vel=None):
        if joint_vel is None:
            joint_vel = np.zeros((len(joint_pos),))
        for i in range(7):
            p.resetJointState(self.robo_id, i, joint_pos[i],
                    joint_vel[i], physicsClientId=self.pc_id)

    def get_joint_limits(self):
        lower = []
        upper = []
        for joint_i in range(self._arm_len):
            ret = p.getJointInfo(self.robo_id, joint_i, physicsClientId=self.pc_id)
            lower.append(ret[8])
            if ret[9] == -1:
                upper.append(np.pi)
            else:
                upper.append(ret[9])
        return np.array(lower), np.array(upper)

    def calc_ik(self, targ_ee):
        js = p.calculateInverseKinematics(self.robo_id, self.pb_link_idx,
                targ_ee, physicsClientId=self.pc_id)
        return js[:self._arm_len]


@attr.s(auto_attribs=True, kw_only=True)
class CollDetails:
    obj_scene_colls: int = 0
    robo_obj_colls: int = 0
    robo_scene_colls: int = 0




def rearrang_collision(colls, snapped_obj_id, count_obj_colls, verbose=False,
        ignore_names=[], ignore_base=True):
    """
    Defines what counts as a collision for the Rearrange environment execution
    """
    # Filter out any collisions from the base
    if ignore_base:
        colls = [x for x in colls
                if not ('base' in x[0]['link'] or 'base' in x[1]['link'])]

    def should_ignore(x):
        for ignore_name in ignore_names:
            if coll_name(x, ignore_name):
                return True
        return False

    # Filter out any collisions with the ignore objects
    colls = [x for x in colls if not should_ignore(x)]

    # Check for robot collision
    robo_obj_colls = 0
    robo_scene_colls = 0
    robo_scene_matches = get_collision_matches('fetch', colls, 'name')
    for match in robo_scene_matches:
        urdf_on_urdf = match[0]['type'] == 'URDF' and match[1]['type'] == 'URDF'
        with_stage = coll_prop(match, 'Stage', 'type')
        #fetch_on_fetch = match[0]['name'] == 'fetch' and match[1]['name'] == 'fetch'
        #if fetch_on_fetch:
        #    continue
        if urdf_on_urdf or with_stage:
            robo_scene_colls += 1
        else:
            robo_obj_colls += 1

    # Checking for holding object collision
    obj_scene_colls = 0
    if count_obj_colls:
        if snapped_obj_id is not None:
            matches = get_collision_matches("group 16", colls, 'link')
            for match in matches:
                if coll_name(match, 'fetch'):
                    continue
                obj_scene_colls += 1

    total_colls = robo_obj_colls + robo_scene_colls + obj_scene_colls
    return total_colls > 0, CollDetails(
            obj_scene_colls=min(obj_scene_colls,1),
            robo_obj_colls=min(robo_obj_colls,1),
            robo_scene_colls=min(robo_scene_colls,1))


def get_nav_mesh_settings(agent_config):
    navmesh_settings = NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_radius = 0.2
    navmesh_settings.agent_height = agent_config.HEIGHT
    return navmesh_settings



def convert_legacy_cfg(obj_list):
    if len(obj_list) == 0:
        return obj_list

    def convert_fn(obj_dat):
        if len(obj_dat) == 2 and len(obj_dat[1]) == 4 and np.array(obj_dat[1]).shape == (4,4):
            # Specifies the full transformation, no object type
            return (obj_dat[0], (obj_dat[1], int(MotionType.DYNAMIC)))
        elif len(obj_dat) == 2 and len(obj_dat[1]) == 3:
            # Specifies XYZ, no object type
            trans = mn.Matrix4.translation(mn.Vector3(obj_dat[1]))
            return (obj_dat[0], (trans, int(MotionType.DYNAMIC)))
        else:
            # Specifies the full transformation and the object type
            return (obj_dat[0], obj_dat[1])

    return list(map(convert_fn, obj_list))

def get_aabb(obj_id, sim, transformed=False):
    obj_node = sim.get_object_scene_node(obj_id)
    obj_bb = obj_node.cumulative_bb
    if transformed:
        obj_bb = habitat_sim.geo.get_transformed_bb(obj_node.cumulative_bb, obj_node.transformation)
    return obj_bb

def inter_any_bb(bb0, bbs):
    for bb in bbs:
        if mn.math.intersects(bb0, bb):
            return True
    return False

def euler_to_quat(rpy):
    rot = quaternion.from_euler_angles(rpy)
    rot = mn.Quaternion(mn.Vector3(rot.vec), rot.w)
    return rot

def allowed_region_to_bb(allowed_region):
    if len(allowed_region) == 0:
        return allowed_region
    return mn.Range2D(allowed_region[0], allowed_region[1])
