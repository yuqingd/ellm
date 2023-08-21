from collections import defaultdict, OrderedDict
import math

import magnum as mn
import numpy as np
import torch

from text_housekeep.cos_eor.utils.objects_to_byte_tensor import dec_bytes2obj
from text_housekeep.habitat_lab.habitat_baselines.common.baseline_registry import baseline_registry
from text_housekeep.habitat_lab.habitat_baselines.rl.ppo import Policy
from text_housekeep.cos_eor.policy.explore import ExploreModule
from text_housekeep.cos_eor.policy.nav import NavModule, NavTargetInfo
from text_housekeep.cos_eor.utils.visualization import add_text, render_frame_explore_sim
from text_housekeep.habitat_lab.habitat.utils.visualizations.utils import images_to_video

# CODES
SUC = "succeeded"
INP = "in-progress"
FAIL = "failed"

OBJ = "object"
REC = "receptacle"

NAV = "navigating"
PP = "pick-place"
LOOK = "look-at"
EXP = "exploring"


class HiePolicy:
    """
    Contains:
     1. The nav-agent that is to be trained for reaching goals
     given semantic-id
    2. The rule-based / learned exploration strategy
    3. The internal-state matrix and scoring/ranking function for ORM.

    """
    def __init__(self, envs, nav_module, rank_module, explore_module, policy_params, task_params, debug_params=None):
        self.envs = envs
        self.nav_module = nav_module
        self.rank_module = rank_module
        self.explore_module = explore_module
        self.policy_params = policy_params
        self.task_params = task_params
        if policy_params.oracle:
            self.fail_thresholds = {
                f"{NAV}-{OBJ}-{INP}": 500,  # max-steps for nav
                f"{NAV}-{REC}-{INP}": 500,  # max-steps for nav
                f"{PP}-{REC}-{INP}": 0,  # max-tries for pp
                f"{PP}-{OBJ}-{INP}": 0,  # max-tries for pp
                f"{LOOK}-{OBJ}-{INP}": 100,  # max-steps for look
                f"{LOOK}-{REC}-{INP}": 100,  # max-steps for look
                f"{EXP}-{None}-{INP}": 100,  # max-tries for exp
            }
        else:
            self.fail_thresholds = {
                f"{NAV}-{OBJ}-{INP}": 100,  # max-steps for nav
                f"{NAV}-{REC}-{INP}": 100,  # max-steps for nav
                f"{PP}-{REC}-{INP}": 0,  # max-tries for pp
                f"{PP}-{OBJ}-{INP}": 0,  # max-tries for pp
                f"{LOOK}-{OBJ}-{INP}": 20,  # max-steps for look
                f"{LOOK}-{REC}-{INP}": 20,  # max-steps for look
                f"{EXP}-{None}-{INP}": self.policy_params.explore.max_steps,  # max-tries for exp
            }
        self.reset()

        self.debug_video = False
        self.measures = {}
        self.turn_measures = {}

    def reset(self):
        # we want them to be ordered so that indices can map 1-1 to matrices
        self.rec_rooms = OrderedDict()
        self.objs = OrderedDict()
        self.bad_recs = set()

        self.pending_rearrangements = []  # fifo queue
        self.past_rearrangements = OrderedDict()
        self.curr_rearrangment = None  # should be (obj_idx, curr_rec_idx, curr_best_rec_idx)
        self.tracker = defaultdict(int)
        self.measures = {
            "success_look_at": 0,
            "fail_look_at": 0,
        }

        # debug
        self.obs = []
        self.raw_obs = []
        self.debug_count = 0
        self.step_count = 0

        # reset all internal components
        self.reset_state()
        self.rank_module.reset()
        self.explore_module.reset()
        self.nav_module.reset()

    def reset_state(self):
        self.curr_state = {
            "act": EXP,  # one from ["nav", "pick-place", "look-at", "explore"]
            "target": None,  # one from ["obj", "rec"]
            "status": INP  # one from ["in-progress", "success", "fail"]
        }

    def get_current_state(self):
        return [self.curr_state]

    def _rearrange_sort_key(self, rearrangement, scores):
        oi, cmi, (ssi, ss) = rearrangement
        order = self.policy_params.rearrange_order
        if order == "discovery":
            return oi
        elif order == "score_gain":
            cur_score = scores[cmi, oi]
            return cur_score - ss[0]
        elif order == "agent_distance":
            return self._get_agent_obj_dist(ssi[0], REC, "l2")
        elif order == "obj_distance":
            return self._get_obj_obj_dist(cmi, REC, ssi[0], REC, "l2")
        else:
            raise ValueError

    def skip_initial_bad_recs(self, rearrangement):
        oi, cmi, (ssi, ss) = rearrangement
        start_idx = 0
        for ri in ssi:
            past_rearrangement_status = self.past_rearrangements.get((oi, cmi, ri))
            if past_rearrangement_status is not None:
                past_rearrangement_status = past_rearrangement_status[1]
            if past_rearrangement_status != FAIL and ri not in self.bad_recs:
                break
            start_idx += 1
        return start_idx

    def get_rearrangements(self):
        if len(self.objs) == 0 or len(self.rec_rooms) == 0:
            return
        # sorted receptacle indices from high-low similarity scores
        combined_scores = self.rank_module.scores + self.rank_module.room_scores
        combined_scores[self.rank_module.scores < self.policy_params.score_threshold] = 0
        sort_scores_inds = (combined_scores * -1).argsort(axis=0).T
        sort_scores = np.array([self.rank_module.scores[ssi, idx] for idx, ssi in enumerate(sort_scores_inds)])

        curr_match_keys = [obj["rec"] for obj in self.objs.values()]
        curr_recs_keys = self.get_seen("obj_key", types=["rec"])
        curr_recs_key_to_idx = {key: idx for idx, key in enumerate(curr_recs_keys)}
        curr_match_inds = [curr_recs_key_to_idx[rk] if rk in curr_recs_keys else -1 for rk in curr_match_keys]

        rearrangements = []
        # match and return
        for obj_idx, (cmi, ssi, ss) in enumerate(zip(
                curr_match_inds,
                sort_scores_inds,
                sort_scores,
        )):
            if cmi == -1:
                continue
            # skip if we are already committed
            if self.curr_rearrangment is not None and obj_idx == self.curr_rearrangment[0]:
                continue
            start_idx = self.skip_initial_bad_recs((obj_idx, cmi, (ssi, ss)))
            ssi, ss = ssi[start_idx:], ss[start_idx:]

            curr_score = self.rank_module.scores[cmi, obj_idx]
            if (
                ss[0] > self.policy_params.score_threshold and
                curr_score <= self.policy_params.score_threshold and
                ssi.size > 0 and
                cmi != ssi[0]
            ):
                rearrangements.append((obj_idx, cmi, (ssi, ss)))

        # sort rearrangements by scheme in config
        rearrangements.sort(key=lambda r: self._rearrange_sort_key(r, combined_scores))

        # merge with current-list
        pending_rearrangements_map = {pr[0]: idx for idx, pr in enumerate(self.pending_rearrangements)}
        for nr in rearrangements:
            oi, _, _ = nr
            # remove stale matches if better ones exist
            if oi in pending_rearrangements_map:
                pr_idx = pending_rearrangements_map[oi]
                self.pending_rearrangements[pr_idx] = nr
            else:
                self.pending_rearrangements.append(nr)

    def print_pending(self, log=True):
        self.debug_count += 1
        for idx, pr in enumerate(self.pending_rearrangements):
            oi, cmi, cbi, cbs = pr
            ok = self.get_value(oi, OBJ, "obj_key")
            crk = self.get_value(cmi, REC, "obj_key")
            brk = self.get_value(cbi, REC, "obj_key")
            if self.debug_count % 10 == 0 or log:
                self.log(f"R-{idx}: move obj {ok} from {crk} to {brk}")

    def assert_consistency(self):
        assert self.curr_state["status"] in [INP, FAIL, SUC, None]
        assert self.curr_state["target"] in [OBJ, REC, None]
        if self.policy_params.explore.type == "oracle":
            assert self.curr_state["act"] in [LOOK, PP, NAV, EXP, None]
        else:
            assert self.curr_state["act"] in [LOOK, PP, NAV, EXP]
        # ensure every obj-id in pending rearrangments in unique
        # ensure curr_state and curr_rearrangment consistency

    def track(self, only_return=False, global_state=False):
        status_key = f"{self.curr_state['act']}-{self.curr_state['target']}-{self.curr_state['status']}"
        if global_state:
            track_key = status_key
        else:
            track_key = f"{status_key}-{str(self.curr_rearrangment)}"
        if not only_return:
            self.tracker[track_key] += 1
        else:
            return self.tracker[track_key]

    def assert_threshold(self, global_state=False):
        status_key = f"{self.curr_state['act']}-{self.curr_state['target']}-{self.curr_state['status']}"
        return self.fail_thresholds[status_key] >= self.track(only_return=True, global_state=global_state)

    def reset_tracker_current_state(self):
        tracker_key = f"{self.curr_state['act']}-{self.curr_state['target']}-{self.curr_state['status']}"
        self.tracker[tracker_key] = 0

    def load_next_state(self):
        if self.curr_rearrangment is not None:
            return

        explore_type = self.policy_params.explore.type
        if explore_type == "phasic":
            if self.curr_state["act"] == EXP and self.curr_state["status"] == INP:
                return
            elif len(self.pending_rearrangements) > 0:
                self.log("rearranging now!")
                self.curr_rearrangment = self.pending_rearrangements.pop(0)
                self.curr_state["act"] = NAV
                self.curr_state["target"] = OBJ
                self.curr_state["status"] = INP
            elif len(self.pending_rearrangements) == 0:
                if self.curr_state["act"] != EXP:
                    self.log("exploring now!")
                    self.curr_state["act"] = EXP
                    self.curr_state["target"] = None
                    self.curr_state["status"] = INP
                    self.explore_module.reset_steps_since_new_area()
                    self.reset_tracker_current_state()
        elif explore_type == "oracle":
            if len(self.pending_rearrangements) > 0:
                self.log("rearranging now!")
                self.curr_rearrangment = self.pending_rearrangements.pop(0)
                self.curr_state["act"] = NAV
                self.curr_state["target"] = OBJ
                self.curr_state["status"] = INP
            else:
                self.curr_state["act"] = None
        else:
            raise ValueError

    def dump_vid(self, video_path, video_name, raw_numpy=False):
        frames = []
        for raw_obs in self.raw_obs:
            observations, text_logs = raw_obs["obs"], raw_obs["text"]
            frame = self.create_frame(observations)
            self.add_text_logs_to_frame(frame, text_logs)
            frames.append(frame)
        video_name += f"-frames_{len(frames)}"
        if raw_numpy:
            np.save(video_path/video_name, frames)
            self.log(f"Dumped: {video_name}.npy frames")
        else:
            images_to_video(frames, video_path, video_name, fps=4)
            self.log(f"Dumped: {video_name}.mp4 video")


    def wrap_action(self, action):
        self.step_count += 1
        if self.step_count % 100 == 0:
            print(f"Num steps: {self.step_count}")
        return [{"action": action}]

    def loop_action(self, observations):
        action = None
        while action is None:
            self.load_next_state()
            if self.curr_state["act"] is None:
                action = {"action": 0}
            elif self.curr_state["act"] == NAV:
                action = self.nav(observations)
            elif self.curr_state["act"] == LOOK:
                action = self.look_at(observations)
            elif self.curr_state["act"] == PP:
                action = self.pick_place(observations)
            else:
                assert self.curr_state["act"] == EXP
                action = self.explore(observations)
        else:
            return self.wrap_action(action)

    def create_text_logs(self, action, cos_eor_sensor):
        if "action" not in action[0] or "action" not in action[0]["action"]:
            import pdb
            pdb.set_trace()
        action = action[0]["action"]["action"]
        possible_actions = self.task_params.POSSIBLE_ACTIONS
        action_text = possible_actions[action]

        gripped_object_id = cos_eor_sensor["gripped_object_id"]
        gripped_object_key = cos_eor_sensor["sim_obj_id_to_obj_key"].get(gripped_object_id)

        obj_key = None
        pos = (0, 0, 0)
        if self.curr_rearrangment is not None:
            oi, cmi, (ssi, ss) = self.curr_rearrangment
            if self.curr_state["target"] == OBJ:
                obj_key = self.get_value(oi, OBJ, "obj_key")
                ep_idx = cos_eor_sensor["objs_keys"].index(obj_key)
                pos = cos_eor_sensor["objs_pos"][ep_idx]
            else:
                obj_key = self.get_value(ssi[0], REC, "obj_key")
                ep_idx = cos_eor_sensor["recs_keys"].index(obj_key)
                pos = cos_eor_sensor["recs_pos"][ep_idx]

        lines = [
            f"Gripped Key: {gripped_object_key}",
            f"Target Pos: {pos[0]:.3f}, {pos[2]:.3f}",
            f"Target Key: {obj_key}",
            f"Action: {action_text}",
            f"State: {self.curr_state['act']}",
        ]

        return lines

    def add_text_logs_to_frame(self, frame, lines):
        cur_line_y = 5
        line_space = 20
        for line in reversed(lines):
            add_text(frame, line, (0, frame.shape[0] - cur_line_y))
            cur_line_y += line_space

    def create_frame(self, observations):
        frame = render_frame_explore_sim(observations)
        return frame

    def cache_raw_obs(self, observations, action):
        text_logs = self.create_text_logs(action, observations["cos_eor"][0])
        self.raw_obs.append({"obs": observations, "text": text_logs})

    def act(
        self,
        observations,
    ):
        self.turn_measures = {
            "seen_area": observations["seen_area"][0][0].item()
        }

        # decode task-sensor info
        observations["cos_eor"] = [dec_bytes2obj(obs) for obs in observations["cos_eor"]]
        self.update(observations)
        self.rank_module.rerank(observations["cos_eor"][0], self.rec_rooms, self.objs, True)
        self.explore_module.update(observations)

        # build new rearrangements
        self.get_rearrangements()
        self.assert_consistency()

        # next-state
        self.load_next_state()

        # ongoing rearrangement
        action = None
        if self.curr_rearrangment is not None:
            if self.curr_state["act"] == NAV:
                action = self.nav(observations)
                if action is not None:
                    action = self.wrap_action(action)
                else:
                    action = self.loop_action(observations)
            elif self.curr_state["act"] == LOOK:
                action = self.look_at(observations)
                if action is not None:
                    action = self.wrap_action(action)
                else:
                    action = self.loop_action(observations)
            elif self.curr_state["act"] == PP:
                action = self.pick_place(observations)
                if action is not None:
                    action = self.wrap_action(action)
                else:
                    action = self.loop_action(observations)
            else:
                raise ValueError
        else:
            # explore if there are no pending rearrangements
            if self.curr_state["act"] == EXP:
                action = self.explore(observations)
                if action is not None:
                    action = self.wrap_action(action)
                else:
                    action = self.loop_action(observations)
            else:
                assert (
                    self.policy_params.explore.type == "oracle" and
                    self.curr_state["act"] is None
                )
                action = self.wrap_action({"action": 0})
        self.cache_raw_obs(observations, action)
        return action, [self.turn_measures]

    def explore(self, observations):
        did_steps_exceed_threshold = (
            not self.assert_threshold(global_state=True) or
            self.explore_module.steps_since_new_area >= self.policy_params.explore.max_steps_since_new_area
        )
        if did_steps_exceed_threshold and len(self.pending_rearrangements) > 0:
            self.curr_state["status"] = SUC
            return None
        action = self.explore_module.act(observations)
        self.track(global_state=True)
        return action

    def _signed_angle(self, a: mn.Vector2, b: mn.Vector2) -> float:
        return math.atan2(a.x*b.y - a.y*b.x, mn.math.dot(a, b))

    def _look_at_pos(self, cos_eor_sensor, obj_pos):
        center_ray = cos_eor_sensor["camera_center_ray"]
        center_ray_origin = mn.Vector3(center_ray["origin"])
        center_ray_dir = mn.Vector3(center_ray["direction"])

        obj_pos = (mn.Vector3(obj_pos) - center_ray_origin).normalized()

        # calculate difference between current camera gaze direction
        # and gaze direction needed to look at object
        y_rot = self._signed_angle(center_ray_dir.xz, obj_pos.xz)

        # difference in angles of elevation between agent and object
        up_vec = mn.Vector3(0, 1, 0)
        up_vec_yz = up_vec.yz.normalized()
        elevation_diff = float(
            mn.math.angle(center_ray_dir.yz.normalized(), up_vec_yz) - \
            mn.math.angle(obj_pos.yz.normalized(), up_vec_yz)
        )

        # return action that reduces the larger of the angle differences
        if abs(y_rot) > abs(elevation_diff):
            if y_rot < 0:
                return 2 # left
            else:
                return 3 # right
        else:
            if elevation_diff > 0:
                return 4 # up
            else:
                return 5 # down

    def look_at(self, observations):
        oi, cmi, (ssi, ss) = self.curr_rearrangment
        cbi = ssi[0]  # current-best index

        cos_eor_sensor = observations["cos_eor"][0]
        obj_key = self.get_value(oi, OBJ, "obj_key")

        if self.curr_state["target"] == OBJ:
            rec_key = self.get_value(cmi, REC, "obj_key")
            iid = self.get_value(cmi, REC, "iid")
        elif self.curr_state["target"] == REC:
            rec_key = self.get_value(cbi, REC, "obj_key")
            iid = self.get_value(cbi, REC, "iid")
        else:
            raise ValueError

        ep_idx = cos_eor_sensor["recs_keys"].index(rec_key)
        pos = cos_eor_sensor["recs_pos"][ep_idx]

        # check if obj/rec is within sensor frame
        visible_ids = torch.unique(observations["semantic"][0]).tolist()
        if iid in visible_ids:
            self.curr_state["act"] = PP
            self.measures["success_look_at"] += 1
            return None

        # if num steps for looking is exceeded, fail this rearrangement
        if not self.assert_threshold():
            self.measures["fail_look_at"] += 1
            if self.curr_state["target"] == OBJ:
                self.past_rearrangements[(oi, cmi, ssi[0])] = (self.curr_rearrangment, FAIL)
                self.curr_rearrangment = None
                self.log(f"couldn't look at {obj_key} on {rec_key}!")
                return None
            elif self.curr_state["target"] == REC:
                self.past_rearrangements[(oi, cmi, ssi[0])] = (self.curr_rearrangment, FAIL)
                self.reset_to_next_best_receptacle()
                self.log(f"trying next best receptacle for {obj_key}, failed to look at {rec_key}!")
                return None

        self.track()
        action = self._look_at_pos(cos_eor_sensor, pos)
        return {"action": action}

    def pick_place(self, observations):
        gripped_object_id = observations["cos_eor"][0]["gripped_object_id"]
        oi, cmi, (ssi, ss) = self.curr_rearrangment
        cbi = ssi[0]  # current-best index

        obj_id = self.get_value(oi, OBJ, "obj_id")
        ok = self.get_value(oi, OBJ, "obj_key")
        rk = self.get_value(cbi, REC, "obj_key")

        if self.curr_state["target"] == OBJ:
            iid = self.get_value(oi, OBJ, "iid")
            # picked correct object, navigate to rec
            if obj_id == gripped_object_id:
                self.curr_state["status"] = SUC
                self.track()
                self.curr_state["act"] = NAV
                self.curr_state["status"] = INP
                self.curr_state["target"] = REC
                self.log(f"picked {ok}!")
                self.turn_measures["pick_place_objs"] = f"pick,{ok}"
                return None
            # move-on if failed to pick and tries > threshold
            if not self.assert_threshold():
                self.past_rearrangements[(oi, cmi, ssi[0])] = (self.curr_rearrangment, FAIL)
                self.curr_rearrangment = None
                self.log(f"couldn't pick {ok}!")
                return None
            # failed pick attempt, increase counter
            elif gripped_object_id == -1:
                self.track()
                return {"action": 6, "action_args": {"iid": iid}}
            # oracle agent can't pick wrong object
            else:
                raise ValueError

        elif self.curr_state["target"] == REC:
            iid = self.get_value(cbi, REC, "iid")
            # placed on correct receptacle, move-on
            if gripped_object_id == -1:
                self.curr_state["status"] = SUC
                self.track()
                self.past_rearrangements[(oi, cmi, ssi[0])] = (self.curr_rearrangment, SUC)
                self.curr_rearrangment = None
                self.objs[oi]["rec"] = rk
                self.log(f"placed {ok} on {rk}!")
                self.turn_measures["pick_place_objs"] = f"place,{ok},{rk}"
                return None
            # move-on to next best receptacle if failed to place and tries > threshold
            elif not self.assert_threshold():
                self.past_rearrangements[(oi, cmi, ssi[0])] = (self.curr_rearrangment, FAIL)
                self.reset_to_next_best_receptacle()
                self.log(f"trying next best receptacle for {ok}, failed to place on {rk}!")
                return None
            # try placing

            elif gripped_object_id == obj_id:
                self.track()
                return {"action": 6, "action_args": {"iid": iid}}
            else:
                raise ValueError

    def reset_to_next_best_receptacle(self):
        oi, cmi, (ssi, ss) = self.curr_rearrangment

        # remove current best receptacle
        ssi, ss = ssi[1:], ss[1:]

        self.curr_rearrangment = oi, cmi, (ssi, ss)
        # reset state to navigate
        self.curr_state["status"] = INP
        self.curr_state["target"] = REC
        self.curr_state["act"] = NAV

    def log(self, text):
        file = "oracle-log.txt"
        with open(file, 'a') as f:
            print(text, file=f)
        print(text)

    def in_view(self, obs, oi):
        iid = self.get_value(oi, OBJ, "obj_key")
        avail_sids = obs["sid"].squeeze().unique()

    def fail_nav(self):
        oi, cmi, (ssi, ss) = self.curr_rearrangment
        self.curr_state["status"] = FAIL
        self.track()
        self.past_rearrangements[(oi, cmi, ssi[0])] = (self.curr_rearrangment, FAIL)
        self.curr_rearrangment = None

    def _get_agent_obj_dist(self, obj_idx, obj_type, dist_type):
        obj_id = self.get_value(obj_idx, obj_type, "obj_id")
        return self.envs.call_at(0, "get_agent_object_distance", {"obj_id": obj_id, "dist_type": dist_type})

    def _get_obj_obj_dist(self, obj1_idx, obj1_type, obj2_idx, obj2_type, dist_type):
        obj1_id = self.get_value(obj1_idx, obj1_type, "obj_id")
        obj2_id = self.get_value(obj2_idx, obj2_type, "obj_id")
        return self.envs.call_at(
            0,
            "get_object_object_distance",
            {"obj1_id": obj1_id, "obj2_id": obj2_id, "dist_type": dist_type}
        )

    def nav(self, observations):
        gripped_object_id = observations["cos_eor"][0]["gripped_object_id"]
        oi, cmi, (ssi, ss) = self.curr_rearrangment
        cbi = ssi[0]
        grab_dist = self.task_params.ACTIONS.GRAB_RELEASE.GRAB_DISTANCE
        if self.curr_state["target"] == OBJ:
            agent_rec_dist = self._get_agent_obj_dist(cmi, REC, "l2")
            ok = self.get_value(oi, OBJ, "obj_key")
            rk = self.get_value(cmi, REC, "obj_key")
            # move-on if failed tothreshol navigate
            if not self.assert_threshold():
                self.fail_nav()
                self.log(f"can't reach {ok} on {rk}! exceeded step threshold")
                return None
            else:
                # take a step towards goal
                num_nav_steps = self.track(only_return=True)
                targets = {
                    "nav": NavTargetInfo(REC, cmi),
                    "look": NavTargetInfo(REC, cmi)
                }
                _, action, _, _, module, _ = self.nav_module.act(observations, targets, self, num_nav_steps)
                if num_nav_steps == 0:
                    print(f"snap to {ok} on {rk}")
                self.track()
                # navigation completed
                if action == 0:
                    # check for success, and move to picking
                    if agent_rec_dist < grab_dist:
                        self.curr_state["status"] = SUC
                        self.track()
                        self.curr_state["act"] = LOOK
                        self.curr_state["status"] = INP
                        self.log(f"reached {ok} on {rk}!")
                        return None
                    # failed to navigate
                    else:
                        self.fail_nav()
                        self.log(
                            f"can't reach {ok} on {rk}! "
                            f"pathplanner failed with dist {agent_rec_dist} after {num_nav_steps} steps"
                        )
                        return None
                return {"action": action}

        elif self.curr_state["target"] == REC:
            assert gripped_object_id != -1
            agent_rec_dist = self._get_agent_obj_dist(cbi, REC, "l2")
            rk = self.get_value(cbi, REC, "obj_key")
            # try next best receptacle if failed
            if not self.assert_threshold():
                self.past_rearrangements[(oi, cmi, ssi[0])] = (self.curr_rearrangment, FAIL)
                self.reset_to_next_best_receptacle()
                self.log(f"trying next best receptacle, failed to reach {rk}! exceeded step threshold")
                return None
            # take a step towards goal
            else:
                num_nav_steps = self.track(only_return=True)
                targets = {
                    "nav": NavTargetInfo(REC, cbi),
                    "look": NavTargetInfo(REC, cbi)
                }
                _, action, _, _, module, _ = self.nav_module.act(observations, targets, self, num_nav_steps)
                self.track()
                if action == 0:
                    # check success, and move to placing
                    if agent_rec_dist < grab_dist:
                        self.curr_state["status"] = SUC
                        self.track()
                        self.curr_state["act"] = LOOK
                        self.curr_state["status"] = INP
                        self.log(f"reached {rk}!")
                        return None
                    else:
                        self.past_rearrangements[(oi, cmi, ssi[0])] = (self.curr_rearrangment, FAIL)
                        self.reset_to_next_best_receptacle()
                        self.log(
                            f"trying next best receptacle, failed to reach {rk}! "
                            f"pathplanner failed with dist {agent_rec_dist} after {num_nav_steps} steps"
                        )
                        return None
                return {"action": action}
        else:
            raise AssertionError

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        pass

    def get_seen(self, attr, types=["rec", "obj"]):
        """get specified attributes of already seen objects/receptacles"""
        attr_list = []
        if "rec" in types:
            attr_list.extend([ed[attr] for ed in self.rec_rooms.values()])
        if "obj" in types:
            attr_list.extend([ed[attr] for ed in self.objs.values()])
        return attr_list

    def get_value(self, idx, type, key):
        if type == REC:
            # floor
            if idx == -1:
                if key == "obj_key":
                   return "floor"
                else:
                    raise ValueError
            return self.rec_rooms[idx][key]
        elif type == OBJ:
            return self.objs[idx][key]
        else:
            raise ValueError

    def debug(self, task_data, visible_iids):
        obj_keys = [
            "laptop_7_0.urdf",
            "table_9_0.urdf",
            "coffee_table_5_0.urdf",
            "013_apple_1",
            "026_sponge_2",
            "counter_26_0.urdf",
            "sink_35_0.urdf",
            "sink_42_0.urdf"
        ]
        visible_iids += [task_data["sim_obj_id_to_iid"][task_data['obj_key_to_sim_obj_id'][ok]] for ok in obj_keys]
        visible_iids = list(set(visible_iids))
        return visible_iids

    def update(self, obs):
        task_data = obs["cos_eor"][0]
        if self.policy_params.explore.type == "oracle":
            visible_iids = task_data["iid_to_sim_obj_id"].keys()
        else:
            # currently visible iids
            visible_iids = obs["semantic"][0].unique().tolist()
        if 0 in visible_iids:
            visible_iids.remove(0)
        visible_obj_ids = [task_data["iid_to_sim_obj_id"][iid] for iid in visible_iids]
        visible_obj_keys = [task_data["sim_obj_id_to_obj_key"][oid] for oid in visible_obj_ids]
        obj_iid_to_idx = {obj["iid"]: idx for idx, obj in self.objs.items()}
        rec_iid_to_idx = {rr["iid"]: idx for idx, rr in self.rec_rooms.items()}
        novel_objects = [
            vk
            for vk, vi in zip(visible_obj_keys, visible_iids)
            if (vi not in rec_iid_to_idx) and (vi not in obj_iid_to_idx) and ("door" not in vk)
        ]
        self.turn_measures["novel_objects"] = ",".join(novel_objects)
        if len(novel_objects):
            self.log(f"discovered: {novel_objects}")
        for iid, obj_id, obj_key in zip(visible_iids,  visible_obj_ids, visible_obj_keys):
            # floor etc
            if iid == 0:
                continue
            sid = task_data["iid_to_sid"][iid]
            obj_type = task_data["sim_obj_id_to_type"][obj_id]
            entity_dict = {
                "sid": sid,
                "iid": iid,
                "obj_id": obj_id,
                "sem_class": task_data["sid_class_map"][sid],
                "room": task_data["obj_id_to_room"][obj_id],
                "obj_type": obj_type,
                "obj_key": obj_key,
            }

            if obj_type == "rec" and "door" not in obj_key:
                ep_idx = task_data["recs_keys"].index(obj_key)
                entity_dict["pos"] = task_data["recs_pos"][ep_idx]
                entity_dict["objs"] = [o for o,r in task_data["current_mapping"].items() if r == obj_key and r in visible_obj_keys]
                rec_idx = rec_iid_to_idx.get(iid, len(self.rec_rooms))
                if iid in rec_iid_to_idx:
                    # if we have previously seen this receptacle, accumulate objs
                    entity_dict["objs"] = list(set(entity_dict["objs"]) or set(self.rec_rooms[rec_idx]["objs"]))
                self.rec_rooms[rec_idx] = entity_dict
            elif obj_type == "obj":
                ep_idx = task_data["objs_keys"].index(obj_key)
                entity_dict["pos"] = task_data["objs_pos"][ep_idx]
                # map receptacle if it is visible
                entity_dict["rec"] = task_data["current_mapping"][obj_key]
                obj_idx = obj_iid_to_idx.get(iid, len(self.objs))
                self.objs[obj_idx] = entity_dict
            elif "door" not in obj_key:
                raise ValueError

        # TODO: support non-interactable object type
        bad_rec_substrs = ["-picture_", "-window_"]
        self.bad_recs = set()
        for idx, rec in self.rec_rooms.items():
            key = rec["obj_key"]
            if any(bad_sub in key for bad_sub in bad_rec_substrs):
                self.bad_recs.add(idx)

        self.assert_consistency()

    def update_pick_place(self, type, oid, rid):
        # update the internal mapping when pick/place occurs
        self.assert_consistency()
        pass

    def get_info(self):
        return self.measures
