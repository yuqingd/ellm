import collections
import copy
import string
import os
import pathlib
import pickle as pkl
import wandb
import time
import fcntl

import numpy as np  

from text_housekeep.cos_eor.env.env import CosRearrangementRLEnv
from text_housekeep.habitat_lab.habitat_baselines.common.baseline_registry import baseline_registry
from text_housekeep.lm import GPTHouseKeep, OracleHouseKeep, HousekeepSinglePrompt, BaselineHousekeep

try:
  import gym
  DiscreteSpace = gym.spaces.Discrete
  BoxSpace = gym.spaces.Box
  DictSpace = gym.spaces.Dict
  BaseClass = gym.Env
  MultiDiscreteSpace = gym.spaces.MultiDiscrete
except ImportError:
  DiscreteSpace = collections.namedtuple('DiscreteSpace', 'n')
  BoxSpace = collections.namedtuple('BoxSpace', 'low, high, shape, dtype')
  DictSpace = collections.namedtuple('DictSpace', 'spaces')
  BaseClass = object

@baseline_registry.register_env(name="BaseTextRearrangementRLEnv")
class BaseTextRearrangementEnv(CosRearrangementRLEnv):
    def __init__(self, task_config, env_type='easier', use_sbert=True, use_language_state=False, full_sentence=True, max_seq_len=100, resuggest=False, lm_spec=None, **kwargs):
        # Base text environment for running baselines, where we can get text observations 
        super().__init__(task_config)

        self.use_sbert = use_sbert
        self.use_sbert_sim = False
        self.use_language_state = use_language_state
        
        self.sbert_time = 0
        self.cache_time = 0
        self.cache_load_time = 0
        self.cache_hits = 0
        self.cache_misses = 0

        self.cache_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__))) / 'embedding_cache.pkl'
        self.tokenizer = None
        if use_sbert:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-MiniLM-L3-v2', use_fast=True)

        self.max_seq_len = max_seq_len
        self.goal_str = "" # Text describing the goal, e.g. "chop tree"
        self.oracle_goal_str = ""
        self.prev_goal = ""
        self.single_task = None

        self.seen_receptacles = set()
        self.seen_objs = set()
        self.action_space = DiscreteSpace(len(self.action_list))# + len(self.obj_iids) - 1)

        # Set up LM
        prompt_format = HousekeepSinglePrompt(prob_threshold=lm_spec['prob_threshold'])
        self.novelty_bonus = lm_spec['novelty_bonus']

        self.use_lm = lm_spec['lm_class'] != 'NoLM'
        if self.use_lm:
          if lm_spec['lm_class'] == 'SimpleOracle':
            self.lm = OracleHouseKeep()
          elif lm_spec['lm_class'] == 'BaselineLM':
            self.lm = BaselineHousekeep()
          else: 
            self.lm = GPTHouseKeep(prompt_format=prompt_format, **lm_spec)
          self.oracle_lm = OracleHouseKeep()
          self.suggested_actions = []
          self.all_suggested_actions = []
          self.oracle_suggested_actions = []

        # Set up logging
        self.last_hundred_suggestions = []
        self.all_goals_suggested = {}
        self.all_goals_achieved = {}
        self.all_oracle_goals_suggested = {}

    def save_caches(self):
        start_time = time.time()
        if self.use_sbert_sim:
            with open(self.cache_path, 'wb') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                pkl.dump(self.cache, f)
                fcntl.flock(f, fcntl.LOCK_UN)
        self.cache_load_time += time.time() - start_time

    def load_and_save_caches(self):
        if self.use_sbert_sim:  
            new_cache = self.load_cache()
            # Combine existing and new cache
            self.cache = {**new_cache, **self.cache}
        if self.use_sbert_sim or self.use_sbert:
            self.save_caches()
    
    def set_resuggest(self, resuggest):
        self.resuggest = resuggest
    
    def set_env_reward(self, reward):
        self.env_reward = reward

    @property
    def observation_space(self):
      if not self.use_language_state:
        return BoxSpace(0, 255, self._size, np.uint8)
      observation_dict = {
        'obs' : BoxSpace(0, 255, self._size, np.uint8),
      }

      if self.use_language_state:
        observation_dict.update({
          'text_obs' : BoxSpace(-np.inf, np.inf, (self.max_seq_len,)),
          'goal' : BoxSpace(-np.inf, np.inf, (self.max_seq_len,)),
        })

      return DictSpace(
        observation_dict
      )

    @property
    def action_names(self):
        return self.action_list

    def make_obs_dict(self, obs):
        return {'obs': obs, 
            'text_obs': np.zeros(self.max_seq_len, dtype=int), 
            'inv_status_obs': np.zeros(self.max_seq_len, dtype=int), 
            'text_goal': np.zeros(self.max_seq_len, dtype=int),
            'old_goals': ""}

    def set_single_task(self, task):
        self.single_task = task

    def get_action_name(self, action):
        if action >= len(self.action_list):
          action = len(self.action_list) - 1
        return self.action_list[action]
  
    def get_embedding(self, string : string):
      return np.where(np.asarray(self.action_names) == string)[0][0]

    def text_obs(self, obs):
        text_obs = ""
        # get objects
        visible_objects = obs['visible_obj_sids']
        visible_objects_classes = []
        visible_obj_dict = {}
        for obj_sid in visible_objects:
            if obj_sid > 0:
                visible_objects_classes.append(' '.join(obs['cos_eor']['sid_class_map'][obj_sid].split('_')))
                # Receptacle the object is on
                for obj_str, receptacle_str in obs['cos_eor']['current_mapping'].items():
                    obj_name = obs['cos_eor']['sid_class_map'][obj_sid]
                    if obj_name in obj_str:
                        obj_name = ' '.join(obj_name.split('_'))
                        receptacle_obj_key = receptacle_str
                        if receptacle_obj_key == 'agent':
                          receptacle_class = 'agent'
                        else:
                          receptacle_sim_obj_id = obs['cos_eor']['obj_key_to_sim_obj_id'][receptacle_obj_key]
                          receptacle_iid = obs['cos_eor']['sim_obj_id_to_iid'][receptacle_sim_obj_id]
                          receptacle_sid = obs['cos_eor']['iid_to_sid'][receptacle_iid]
                          receptacle_class = obs['cos_eor']['sid_class_map'][receptacle_sid]
                          receptacle_class = ' '.join(receptacle_class.split('_'))

                          receptacle_room_name = obs['cos_eor']['obj_id_to_room'][receptacle_sim_obj_id] 
                          receptacle_room_name = ' '.join(receptacle_room_name.split('_')[:-1]) # room names end with an integer.
                
                          receptacle_class = receptacle_room_name + ' ' + receptacle_class

                          visible_obj_dict[obj_name] = receptacle_class
                        break
        
        self.seen_objs.update(set(visible_objects_classes))
        if len(self.seen_objs) > 0:
            text_obs += 'Objects: ' + ', '.join(self.seen_objs) + '.'
            
        # get receptacles and add to seen receptacles
        all_recs = obs['visible_rec_iids']
        all_rec_classes = []
        for rec_iid in all_recs:
            if rec_iid > 0:
                rec_obj_id = obs['cos_eor']['iid_to_sim_obj_id'][rec_iid] 
                receptacle_room_name = obs['cos_eor']['obj_id_to_room'][rec_obj_id] 
                receptacle_room_name = ' '.join(receptacle_room_name.split('_')[:-1]) # room names end with an integer.

                rec_sid = obs['cos_eor']['iid_to_sid'][rec_iid]
                rec_name = ' '.join(obs['cos_eor']['sid_class_map'][rec_sid].split('_'))
                all_rec_classes.append(receptacle_room_name + ' ' + rec_name)
        self.seen_receptacles.update(set(all_rec_classes))
        
        if len(self.seen_receptacles) > 0:
            text_obs += 'Receptacles: ' + ', '.join(self.seen_receptacles) + '.'
            self.oracle_goal_str = ', '.join(self.seen_receptacles) 
        
        # get gripped objects
        if obs['gripped_sid'] == -1:
            gripped_obj_class = None
            self.gripped_obj = ''
        else:
            gripped_obj_class = ' '.join(obs['cos_eor']['sid_class_map'][obs['gripped_sid']].split('_'))
            text_obs += 'You are holding: ' + gripped_obj_class + '.'
            self.gripped_obj = gripped_obj_class
                
        text_obs_dict = {
          'holding': gripped_obj_class,
          'all_receptacles': self.seen_receptacles,
          'visible_objects': visible_obj_dict,          
        }
        return text_obs.lower(), text_obs_dict

    def pad_sbert(self, input):
        arr = np.zeros(self.max_seq_len, dtype=int)
        if len(input) > self.max_seq_len:
            input = input[:self.max_seq_len]
        arr[:len(input)] = input
        return arr 

    def tokenize_obs(self, obs_dict):
      """
      Takes in obs dict (same as output to get_full_obs) and returns tokenized version
      """
      new_obs = {}
      for k, v in obs_dict.items():
        if type(v) in [int, float, bool, np.ndarray] or self.tokenizer is None:
          new_obs[k] = v
          continue
        elif type(v) is dict:
          v = " ".join(v.values())
        if type(v) is str:
          arr = self.tokenizer(v)['input_ids']
          new_obs[k] = arr
        else:
          raise TypeError("unrecognized obs type", type(v))
      if self.use_sbert:
        new_obs['text_obs'] = self.pad_sbert(new_obs['text_obs']) 
        new_obs['goal'] = self.pad_sbert(new_obs['goal']) 
        new_obs['old_goals'] = self.pad_sbert(new_obs['old_goals']) 

      if not self.use_language_state:
        new_obs = new_obs['obs']

      return new_obs


    def setup_logging(self):
      self.episode_actions_suggested, self.episode_actions_achieved = set(), set()
      self.episode_goals_suggested, self.episode_goals_achieved = set(), set()

      self.good_action_good_time_count = 0
      self.good_action_good_time_rew_count = 0

    def make_predictions(self, obs):
      text_obs, inv_status = self.text_obs(obs)
      self.suggested_actions = self.lm.predict_options({'obs': text_obs, **inv_status}, self)
      self.oracle_suggested_actions = self.oracle_lm.predict_options({'obs': text_obs, **inv_status}, self)

      # set goal 
      self.goal_str = '. '.join(' in '.join(list(t)) if t[1] != '' else 'pick ' + t[0] for t in self.suggested_actions) + '.'
     # Logging

      # Store the last 100 suggestions
      self.last_hundred_suggestions += [' in '.join(list(t)) if t[1] != '' else 'pick ' + t[0] for t in self.suggested_actions]
      self.last_hundred_suggestions = self.last_hundred_suggestions[-100:]

      # Check whether any of the valid actions have been suggested, and if so, log them
      for suggestion in self.suggested_actions:
        suggestion_str = ' in '.join(list(suggestion)) if suggestion[1] != '' else 'pick ' + suggestion[0]
        self.episode_goals_suggested.add(suggestion_str)
        if suggestion_str in self.all_goals_suggested:
            self.all_goals_suggested[suggestion_str] += 1
        else:
            self.all_goals_suggested[suggestion_str] = 1

        if suggestion in self.oracle_suggested_actions:
          self.good_action_good_time_count += 1
          if suggestion_str in self.all_oracle_goals_suggested:
            self.all_oracle_goals_suggested[suggestion_str] += 1
          else:
            self.all_oracle_goals_suggested[suggestion_str] = 1


      return text_obs

    def reset(self):
      # Parametrize goal as enum.
      obs = super().reset()
      self.goal_str = ""
      self.oracle_goal_str = ""
      self.gripped_obj = ""
      self.seen_receptacles = set()
      self.seen_objs = set()

      self.setup_logging()
      self.raw_obs = obs
      
      # Language goals and LM call
      if self.use_lm:
        self.goal_str = ""
        self.oracle_goal_str = ""
        self.lm.reset(self)
        self.oracle_lm.reset(self)
        self.make_predictions(obs)
        self.old_all_suggested_actions = copy.deepcopy(self.suggested_actions)

      text_obs, text_dict = self.text_obs(obs)
      obs = {
        'obs': np.asarray(obs['rgb'], dtype=np.float64),
        'text_obs' :  text_obs, 
        'other' : 0,
        'goal': self.goal_str,
        'old_goals' : self.prev_goal
      }
      obs['success'] = False
      obs['goal_success'] = np.array(0)
      return self.tokenize_obs(obs), {}

    def step(self, action):
      obs, reward, done, info = super().step(action)

      self.raw_obs = obs
      text_obs, text_dict = self.text_obs(obs)
      if self.use_lm:
        self.make_predictions(obs)
      self.prev_obs = obs
      obs = {
        'obs': np.asarray(obs['rgb'], dtype=np.float64),
        'text_obs' :  text_obs, 
        'other' : self._first_correct,
        'goal': self.goal_str,
        'old_goals' : self.prev_goal
      }
      
      info['reward'] = reward
      info['env_reward'] = reward
      info['rearrange_success'] =  np.sum(self._episode_obj_success(obs))/len(self._episode_obj_success(obs))
      info['initial_success'] =  np.sum(list(self.initial_success.values()))/len(self.initial_success)
      info['rearrange_misplaced_success'] = float(np.sum(self._episode_obj_success(obs)))

      info['pick_success'] = self._pick_success        
      info['place_success'] = self._place_success 

      if self.use_lm:
        # log lm
        info['oracle_lm_rearrange_success'] = np.sum(list(self.oracle_lm.predict_success(text_dict['visible_objects']).values()))/len(self._episode_obj_success(obs))
        info['lm_rearrange_success'] = np.sum(list(self.lm.predict_success(text_dict['visible_objects']).values()))/len(self._episode_obj_success(obs))

      obs['success'] = 0
      obs['goal_success'] = 0
      
      return self.tokenize_obs(obs), reward, done, info   

    def render(self, size, mode="rgb") -> np.ndarray:
        img = self._sim.render(mode)
        return img

    def get_reward(self, observations, metrics):
        if not self.use_lm:
          return super().get_reward(observations, metrics)
        
        reward = self._rl_config.SLACK_REWARD
        gripped_success_reward = 0.0
        episode_success_reward = 0.0
        drop_success_reward = 0.0
        gripped_dropped_fail_reward = 0.0
        agent_to_object_dist_reward = 0.0
        object_to_goal_dist_reward = 0.0
        collision_reward = 0.0

        action_name = self.action_list[self._previous_action['action']]
        objs_success = np.array(self._episode_obj_success(observations))
        observations["gripped_object_id"] = observations["cos_eor"].get("gripped_object_id", -1)
        observations["fail_action"] = observations["cos_eor"].get("fail_action", False)
        objs_keys = self._env._current_episode.objs_keys
        objs_ids = [self._env.task.obj_key_to_sim_obj_id[k] for k in objs_keys]

        # misplaced objects discovered
        misplaced_objs_found = []
        placed_objs_found = []
        for viid in observations["visible_obj_iids"]:
            if viid == 0:
                break
            voi = self._env.task.iid_to_sim_obj_id[viid]
            vkey = self._env.task.sim_obj_id_to_obj_key[voi]
            if vkey not in self._prev_measure["misplaced_objs_keys"] and vkey in self._misplaced_objs_start:
                misplaced_objs_found.append(vkey)
            if vkey not in self._prev_measure["placed_objs_keys"] and vkey not in self._misplaced_objs_start:
                placed_objs_found.append(vkey)

        objs_dict = {}
        for ok, oid, os in zip(objs_keys, objs_ids, objs_success):
            objs_dict[oid] = {"success": os, "obj_key": ok}

        # update for the first-time in the start
        if len(self._prev_measure["objs_dict"]) == 0:
            self._prev_measure["objs_dict"] = objs_dict

        # grip

        gripped_success_reward = 0
        drop_success_reward = 0
        if action_name == "GRAB_RELEASE" and self._gripped_success(observations): # Agent has picked up an object
          assert not observations["fail_action"]
          obj_id = observations["gripped_object_id"]
          self._prev_measure['gripped_object_count'][obj_id] += 1
          first_time_pickup = self._prev_measure['gripped_object_count'][obj_id] == 1 or not self.novelty_bonus # Don't care about first pickup if we're not using novelty bonus
                
          suggested_objs = [obj if receptacle == '' else '' for obj, receptacle in self.suggested_actions]  # Objects the LM says to pick up
          gripped_obj_id = observations['gripped_object_id']
          gripped_iid = self._env.task.sim_obj_id_to_iid[gripped_obj_id]
          gripped_sid = self._env.task.iid_to_sid[gripped_iid]
          obj_picked_up = ' '.join(observations['cos_eor']['sid_class_map'][gripped_sid].split('_'))
            
          gripped_success_reward = self._rl_config.GRIPPED_SUCCESS_REWARD if obj_picked_up in suggested_objs else -.1 * self._rl_config.GRIPPED_SUCCESS_REWARD  # Reward is 1 if the agent picked up an object the LM said to pick up, 0 otherwise
          

          if obj_picked_up in suggested_objs and not first_time_pickup:
            gripped_success_reward = 0

          if self._prev_measure['gripped_object_count'][obj_id] == 1 and not self._prev_measure["objs_dict"][obj_id]["success"]:
            self._pick_success += 1

          print(gripped_success_reward,'pick ' + obj_picked_up)
          if obj_picked_up in suggested_objs:
              completed_suggestion = 'pick ' + obj_picked_up
              if completed_suggestion in self.all_goals_achieved:
                self.all_goals_achieved[completed_suggestion] += 1
              else:
                self.all_goals_achieved[completed_suggestion] = 1
          pass
        # Drop
        elif action_name == "GRAB_RELEASE" and self._prev_measure['gripped_object_id'] != -1  and observations['gripped_object_id'] == -1:
          assert not observations["fail_action"]
          # first time placing
          gripped_obj_id = self._prev_measure['gripped_object_id']
          gripped_obj_key = self._env.task.sim_obj_id_to_obj_key[gripped_obj_id]
          gripped_iid = self._env.task.sim_obj_id_to_iid[gripped_obj_id]
          gripped_sid = self._env.task.iid_to_sid[gripped_iid]
          obj_name  = ' '.join(observations['cos_eor']['sid_class_map'][gripped_sid].split('_'))

          receptacle_obj_key = observations['cos_eor']['current_mapping'][gripped_obj_key]
          receptacle_sim_obj_id = observations['cos_eor']['obj_key_to_sim_obj_id'][receptacle_obj_key]
          receptacle_iid = observations['cos_eor']['sim_obj_id_to_iid'][receptacle_sim_obj_id]
          receptacle_sid = observations['cos_eor']['iid_to_sid'][receptacle_iid]
          receptacle_name = observations['cos_eor']['sid_class_map'][receptacle_sid]
          receptacle_name = ' '.join(receptacle_name.split('_'))

          receptacle_room_name = observations['cos_eor']['obj_id_to_room'][receptacle_sim_obj_id] 
          receptacle_room_name = ' '.join(receptacle_room_name.split('_')[:-1]) # room names end with an integer.

          receptacle_name = receptacle_room_name + ' ' + receptacle_name
          
          # Loop through suggested actions to see if the agent placed the object in the correct receptacle
          correct_placement = False
          for obj, receptacle in self.suggested_actions:
            if receptacle == receptacle_name and obj == obj_name:
              correct_placement = True
              completed_suggestion = ' in '.join([obj, receptacle])
              break
          if correct_placement:  # Only reward the first time the agent places an object in the correct receptacle
            drop_success_reward = self._rl_config.DROP_SUCCESS_REWARD * 1
            self.lm.take_action(obj_name)  

            self.episode_goals_achieved.add(completed_suggestion) # (#10 log)
            self.good_action_good_time_rew_count += 1
            if completed_suggestion in self.all_goals_achieved:
              self.all_goals_achieved[completed_suggestion] += 1
            else:
              self.all_goals_achieved[completed_suggestion] = 1
          else:
            drop_success_reward = self._rl_config.DROP_SUCCESS_REWARD * -.1

          print(drop_success_reward,' in '.join([obj_name, receptacle_name]))

          new_success = (
                objs_success.astype(int) - self._prev_measure['objs_success'].astype(int)
          ).sum()
          if new_success == 1:
            self._place_success += 1

        # failed pick/drop
        if action_name == "GRAB_RELEASE" and observations["fail_action"]:
            gripped_dropped_fail_reward = self._rl_config.GRIPPED_DROPPED_FAIL_REWARD

        self._prev_measure['objs_success'] = objs_success
        self._prev_measure['gripped_object_id'] = observations['gripped_object_id']
        self._prev_measure["objs_dict"] = objs_dict
        self._prev_measure["misplaced_objs_keys"].extend(misplaced_objs_found)
        self._prev_measure["placed_objs_keys"].extend(placed_objs_found)

        assert not(
            len(set(self._prev_measure["misplaced_objs_keys"])) != len(self._prev_measure["misplaced_objs_keys"]) or
            len(set(self._prev_measure["placed_objs_keys"])) != len(self._prev_measure["placed_objs_keys"])
        )

        reward = (
            gripped_success_reward + drop_success_reward + gripped_dropped_fail_reward
        )

        action = self.action_list[self._previous_action['action']]
        self.counter += 1
        debug_string = (f""
                  f"reward: {reward} | \n"
                  f"a: {action}")

        assert not(action_name != "GRAB_RELEASE" and (gripped_success_reward + gripped_dropped_fail_reward) != 0)

        self.save_frames(observations, action_name, debug_string)
        return reward

    def log_lm(self, step):
      if not isinstance(step, int):
        step = step.state.step
      # self.lm.log(step)

      all_goals_suggested = [[label, val] for (label, val) in self.all_goals_suggested.items()] # (#1 save)
      all_oracle_goals_suggested = [[label, val] for (label, val) in self.all_oracle_goals_suggested.items()] # (#1 save)
      all_goals_achieved = [[label, val] for (label, val) in self.all_goals_achieved.items()] # (#1 save)

      all_goals_suggested = wandb.Table(data=all_goals_suggested, columns = ["Suggested Goal", "Count"])
      all_oracle_goals_suggested = wandb.Table(data=all_oracle_goals_suggested, columns = ["Oracle Goal", "Count"])
      all_goals_achieved = wandb.Table(data=all_goals_achieved, columns = ["Achieved Goal", "Count"])

      # Ratio of lm suggested / oracle suggested
      suggest_ratio = [[label, val/self.all_oracle_goals_suggested[label]] for (label, val) in self.all_goals_suggested.items() if label in self.all_oracle_goals_suggested] # (#6 save)
      # Ratio of achieved / suggested 
      achieved_ratio = [[label, val/self.all_goals_suggested[label]] for (label, val) in self.all_goals_achieved.items() if label in self.all_goals_suggested] # (#6 save)
      # Count lm suggested not in oracle
      good_suggestions = [[label, val] for (label, val) in self.all_goals_suggested.items() if label in self.all_oracle_goals_suggested] # (#6 save)
      bad_suggestions = [[label, val] for (label, val) in self.all_goals_suggested.items() if label not in self.all_oracle_goals_suggested] # (#6 save)

      suggest_ratio = wandb.Table(data=suggest_ratio, columns = ["Suggested/Oracle Goal", "Fraction"])
      achieved_ratio = wandb.Table(data=achieved_ratio, columns = ["Achieved/Suggested Goal", "Fraction"])
      good_suggestions = wandb.Table(data=good_suggestions, columns = ["Good Suggestions", "Count"])
      bad_suggestions = wandb.Table(data=bad_suggestions, columns = ["Bad Suggestions", "Count"])

      episode_goals_suggested_table = wandb.Table(data=[['Goals', ', '.join(self.episode_goals_suggested)]], columns = ["Goal", "Goals"])
      episode_goals_achieved_table = wandb.Table(data=[['Goals', ', '.join(self.episode_goals_achieved)]], columns = ["Goal", "Goal Text"])
      last_hundred = wandb.Table(data=[['Goals', ', '.join(self.last_hundred_suggestions)]], columns = ["Goal", "Last 100"])

      try:
        wandb_dict = {
            'lm_acc/all_goals_suggested': wandb.plot.bar(all_goals_suggested, "Suggested Goal", "Count",
                                title="Suggestion Counts for Goals"),
            'lm_acc/all_oracle_goals_suggested': wandb.plot.bar(all_oracle_goals_suggested, "Oracle Goal", "Count",
                                title="Suggestion Counts for Oracle Goals"),
            'lm_acc/all_goals_achieved': wandb.plot.bar(all_goals_achieved, "Achieved Goal", "Count",
                                title="Achievement Counts for Goals"),
            'lm_acc/suggest_ratio': wandb.plot.bar(suggest_ratio, "Suggested/Oracle Goal", "Fraction",
                                title="P(suggested | oracle suggested)"),
            'lm_acc/achieved_ratio': wandb.plot.bar(achieved_ratio, "Achieved/Suggested Goal", "Fraction",
                                title="P(achieved | suggested)"),
            'lm_acc/good_suggestions': wandb.plot.bar(good_suggestions, "Good Suggestions", "Count",
                                title="Good Suggestion Counts"),
            'lm_acc/bad_suggestions': wandb.plot.bar(bad_suggestions, "Bad Suggestions", "Count",
                                title="Bad Suggestion Counts"),
            'lm_acc/episode_goals_suggested': episode_goals_suggested_table,
            'lm_acc/episode_goals_achieved': episode_goals_achieved_table,
            'lm_acc/good_action_good_time_count': self.good_action_good_time_count,
            'lm_acc/good_action_good_time_rew_count': self.good_action_good_time_rew_count,
        }
        wandb_dict['lm_acc/last_hundred'] = last_hundred


        if not isinstance(step, int):
          step = step.state.step
        wandb.log(wandb_dict, step=step)
      except Exception as e:
        import pdb; pdb.set_trace()
        print(e)
    