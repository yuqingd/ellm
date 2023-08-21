""" Env wrapper which adds goals and rewards """

import pickle as pkl
import time
import copy
import os
import pathlib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util as st_utils
from text_crafter import lm
import fasteners
from captioner import get_captioner

class CrafterGoalWrapper:
    """ Goal wrapper for baselines. Used for baselines and single-goal eval. """
    def __init__(self, env, env_reward, single_task=None, single_goal_hierarchical=False):
        self.env = env
        self._single_task = self.set_single_task(single_task)
        self._single_goal_hierarchical = single_goal_hierarchical
        self._use_env_reward = env_reward
        self.prev_goal = ""
        self.use_sbert_sim = False
        if self.env.action_space_type == 'harder':
            self.goals_so_far = dict.fromkeys(self.env.good_action_names) # for Eval purposes only
        else:
            self.goals_so_far = dict.fromkeys(self.env.action_names) # for Eval purposes only
        self._cur_subtask = 0
        self.custom_goals = ['plant row', 'make workshop', 'chop grass with wood pickaxe', 'survival', 'vegetarianism', 'deforestation',
                             'work and sleep', 'gardening', 'wilderness survival']

    # If the wrapper doesn't have a method, it will call the method of the wrapped environment
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self.env, name)

    def set_env_reward(self, use_env_reward):
        """ If this is true, we use the env reward, not the text reward."""
        self._use_env_reward = use_env_reward
        
    def on_final_subtask(self):
        return self._cur_subtask == len(self.goal_compositions()[self._single_task]) - 1
    
    def get_subtask(self):
        if not self._single_task:
            return None
        if self._single_goal_hierarchical:
            try:
                return self.goal_compositions()[self._single_task][self._cur_subtask]
            except:
                print(f'Error finding subtask {self._cur_subtask} for task {self._single_task}')
                import pdb; pdb.set_trace()
        else:
            return self._single_task

    def goal_compositions(self):
        """ Returns a dictionary with each goal, and the prereqs needed to achieve it. """
        goal_comps = {
            'eat plant' : ['chop grass', 'place plant', 'eat plant'],
            'attack zombie' : ['attack zombie'],
            'attack skeleton' : ['attack skeleton'],
            'attack cow' : ['attack cow'],
            'chop tree' : ['chop tree'],
            'mine stone': ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'mine stone'],
            'mine coal' : ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'mine coal'],
            'mine iron': ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'mine stone', 'make stone pickaxe', 'mine iron'],
            'mine diamond' : ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'mine stone', 'make stone pickaxe', 'mine stone', 'place furnace', 'mine coal', 'mine iron', 'make iron pickaxe', 'mine diamond'],
            'drink water' : ['drink water'],
            'chop grass' : ['chop grass'],
            'sleep' : ['sleep'],
            'place stone' : ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'mine stone', 'place stone'],
            'place crafting table' : ['chop tree', 'place crafting table'],
            'place furnace' : ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'mine stone', 'mine stone', 'mine stone', 'mine stone', 'place furnace'],
            'place plant': ['chop grass', 'place plant'],
            'make wood pickaxe' : ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe'],
            'make stone pickaxe' : ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'mine stone', 'make stone pickaxe'],
            'make iron pickaxe' : ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'mine stone', 'make stone pickaxe', 'mine stone', 'place furnace', 'mine coal', 'mine iron', 'make iron pickaxe'],
            'make wood sword' : ['chop tree', 'chop tree', 'place crafting table', 'make wood sword'],
            'make stone sword' : ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'mine stone', 'make stone sword'],
            'make iron sword' : ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'mine stone', 'make stone pickaxe', 'mine stone', 'place furnace', 'mine coal', 'mine iron', 'make iron sword'],
            'plant row': ['chop grass', 'place plant', 'chop grass', 'plant grass'],
            'chop grass with wood pickaxe': ['chop tree', 'chop tree', 'place crafting table', 'make wood pickaxe', 'chop grass with wood pickaxe'],
            'vegetarianism': ['drink water', 'chop grass'],
            'make workshop': ['chop tree', 'place crafting table', 'chop tree', 'place crafting table'],
            'survival': ['survival'],
            'deforestation': ['chop tree', 'chop tree', 'chop tree', 'chop tree', 'chop tree'],
            'work and sleep': ['chop tree', 'sleep', 'place crafting table'],
            'gardening': ['chop grass', 'chop tree', 'place plant'],
            'wilderness survival': ['sleep', 'chop grass', 'attack zombie'],
            
        }
        if self.env.action_space_type == 'harder':
            return self.filter_hard_goals(goal_comps)
        else:
            return goal_comps

    def check_multistep(self, action):
        """Check if a given action has prereqs"""
        if isinstance(action, str):
            action_name = action
        else:
            action_name = self.action_names[action]
        return action_name not in ['attack zombie', 'attack skeleton', 'attack cow', 'chop tree',  'drink water', 'chop grass', 'sleep']

    def _tokenize_goals(self, new_obs):
        if self.use_sbert:  # Use SBERT tokenizer
            new_obs['goal'] = self.env.pad_sbert(new_obs['goal'])
            new_obs['old_goals'] = self.env.pad_sbert(new_obs['old_goals'])
        return new_obs

    def reset(self):
        """Reset the environment, adding in goals."""
        # Parametrize goal as enum.
        obs, info = self.env.reset()
        self.goal_str = ""
        self.oracle_goal_str = ""
        obs['goal'] = self.goal_str
        obs['old_goals'] = self.prev_goal
        obs['goal_success'] = np.array(0) # 0 b/c we can't succeed on the first step
        obs = self.tokenize_obs(obs)
        self._cur_subtask = 0
        self._reset_custom_task()
        return self._tokenize_goals(obs), info
    
    def _reset_custom_task(self):
        self.drank_water = False
        self.trees_chopped = 0

    def set_single_task(self, task):
        """ When single_task is set, we only give the agent this goal."""
        self._single_task = task
        return task
    
    def set_end_on_success(self, end_on_success):
        """ When end_on_success is set, we end the episode when the agent succeeds."""
        self._end_on_success = end_on_success
        return end_on_success
    
    def set_single_goal_hierarchical(self, single_goal_hierarchical):
        """ 
        There are 3 options for single_goal_hierarchical:
        - False/None: don't use hierarchical goals
        - 'reward_last': use hierarchical goals, but don't reward the agent for each subtask
        - True: use hierarchical goals, and reward the agent for each subtask
        """
        self._single_goal_hierarchical = single_goal_hierarchical
        return single_goal_hierarchical

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Replace env reward with text reward
        goal_reward = 0
        if not self._use_env_reward:
            if info['action_success']:  # Only reward on an action success
                action_name = self.env.get_action_name(action)
                if self._single_task is not None: # If there's a single task, check if we've achieved it
                    # If reward_last is true, don't reward for intermediate tasks
                    task = self.get_subtask()
                    achieved_task = action_name == task
                    goal_reward = int(achieved_task)
                    if self._single_goal_hierarchical == 'reward_last' and (not self.on_final_subtask()):
                        goal_reward = 0
                    if achieved_task and self._single_goal_hierarchical:
                        self._cur_subtask = min(self._cur_subtask + 1, len(self.goal_compositions()[self._single_task]) - 1)
                else:
                    goal_reward = 1 # reward for any env success if no task is specified
                    reward = info['health_reward'] + goal_reward
            else:
                reward = 0 # Don't compute reward if action failed.

        obs['goal'] = self.goal_str
        obs['old_goals'] = self.prev_goal
        obs['goal_success'] = info['eval_success'] and goal_reward > 0
        obs = self.tokenize_obs(obs)
        return self._tokenize_goals(obs), reward, done, info

    def filter_hard_goals(self, inputs):
        good_actions = self.env.good_action_names + self.custom_goals
        if isinstance(inputs, dict):
            return {k : v for k, v in inputs.items() if k in good_actions}
        elif isinstance(inputs, list):
            return [v for v in inputs if v in good_actions]
        else:
            raise NotImplementedError

class CrafterLMGoalWrapper(CrafterGoalWrapper):

    def __init__(self, env, lm_spec, env_reward, device=None, threshold=.5, debug=True, single_task=None, single_goal_hierarchical=False,
                 use_state_captioner=False, use_transition_captioner=False, check_ac_success=True): 
        super().__init__(env, env_reward, single_task, single_goal_hierarchical)
        self.env = env
        self.debug = debug
        self.goal_str = "" # Text describing the goal, e.g. "chop tree"
        self.oracle_goal_str = ""
        self.prev_goal = ""
        self.goals_so_far = {}
        self.sbert_time = 0
        self.cache_time = 0
        self.cache_load_time = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_successes = 0
        self.cache_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__))) / 'embedding_cache.pkl'
        self.rw_lock = fasteners.InterProcessReaderWriterLock(self.cache_path)

        # Language model setup.
        prompt_format = getattr(lm, lm_spec['prompt'])()
        lm_class = getattr(lm, lm_spec['lm_class'])
        self.check_ac_success = check_ac_success
        if 'Baseline' in lm_spec['lm_class']:
            lm_spec['all_goals'] = self.action_names.copy()
            self.check_ac_success = False
        self.lm = lm_class(prompt_format=prompt_format, **lm_spec)
        self.oracle_lm = lm.SimpleOracle(prompt_format=prompt_format, **lm_spec) 
        self.use_sbert_sim = True
        self.device = device
        assert self.device is not None
        self.threshold = threshold
        self.embed_lm = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
        self.device = torch.device(device)
        self.cache = {}
        self.suggested_actions = []
        self.all_suggested_actions = []
        self.oracle_suggested_actions = []

        self.all_frac_valid, self.all_frac_covered, self.all_frac_correct = [], [], []
        self.unit_cache_time, self.unit_query_time = [], []
        self._end_on_success = False
        # Get the captioner model
        self._use_state_captioner = use_state_captioner
        self._use_transition_captioner = use_transition_captioner
        if use_state_captioner or use_transition_captioner:
            self.transition_captioner, self.state_captioner, self.captioner_logging = get_captioner()
        self.transition_caption = self.state_caption = None
        self.prev_info = None

    def _save_caches(self):
        if self.debug: pass
        start_time = time.time()
        # The cache will be used by multiple processes, so we need to lock it.
        # We will use the file lock to ensure that only one process can write to the cache at a time.
        
        self.rw_lock.acquire_write_lock()
        with open(self.cache_path, 'wb') as f:
            pkl.dump(self.cache, f)
        self.rw_lock.release_write_lock()
        self.cache_load_time += time.time() - start_time

    def load_and_save_caches(self):
        new_cache = self._load_cache()
        # Combine existing and new cache
        self.cache = {**new_cache, **self.cache}
        self._save_caches()

    def _load_cache(self):
        if self.debug:
            self.cache = {}
            return {}
        start_time = time.time()
        if not self.cache_path.exists():
            cache = {}
            with open(self.cache_path, 'wb') as f:
                pkl.dump({}, f)
        else:
            try:
                self.rw_lock.acquire_read_lock()
                with open(self.cache_path, 'rb') as f:
                    cache = pkl.load(f)
                self.rw_lock.release_read_lock()
            except FileNotFoundError:
                cache = {}
        self.cache_load_time += time.time() - start_time
        return cache

    def text_reward(self, action_embedding, rewarding_actions, update_suggestions=True):
        """
            Return a sparse reward based on how close the task is to the list of actions
            the  LM proposed.
        """
        text_rew = 0
        best_suggestion = None

        # If there are no suggestions, there is no reward
        if len(rewarding_actions) == 0:
            return 0, None

        if self.device is None:
            raise ValueError("Must specify device for real LM")

        # Cosine similarity reward
        suggestion_embeddings, suggestion_strs, updated_cache = self._get_model_embeddings(rewarding_actions)

        # action_name = self.get_action_name(action_embedding)
        action_name = action_embedding
        action_embedding, updated_cache_action = self._get_model_embedding(action_name)

        # Compute the cosine similarity between the action embedding and the suggestion embeddings
        cos_scores = st_utils.pytorch_cos_sim(action_embedding, suggestion_embeddings)[0].detach().cpu().numpy()

        # Compute reward for every suggestion over the threshold
        for suggestion, cos_score in zip(suggestion_strs, cos_scores):
            # print("High score:", cos_score, "for action", action_name, "suggestion", suggestion)
            if cos_score > self.threshold:
                print("High score:", cos_score, "for action", action_name, "suggestion", suggestion)
                if suggestion in self.all_suggested_actions and update_suggestions:
                    self.all_suggested_actions.remove(suggestion)
                text_rew = max(cos_score, text_rew)
        if text_rew > 0:
            best_suggestion = suggestion_strs[np.argmax(cos_scores)]
            print(text_rew, best_suggestion)
        return text_rew, best_suggestion

    def _get_model_embeddings(self, str_list):
        assert isinstance(str_list, list)
        # Split strings into those in cache and those not in cache
        strs_in_cache = []
        strs_not_in_cache = []
        for str in str_list:
            if str in self.cache:
                strs_in_cache.append(str)
            else:
                strs_not_in_cache.append(str)
        all_suggestions = strs_in_cache + strs_not_in_cache

        # Record how many strings are in/not in the cache
        self.cache_hits += len(strs_in_cache)
        self.cache_misses += len(strs_not_in_cache)

        # Encode the strings which are not in cache
        if len(strs_not_in_cache) > 0:
            start_time = time.time()
            embeddings_not_in_cache = self.embed_lm.encode(strs_not_in_cache, convert_to_tensor=True, device=self.device)
            self.sbert_time += time.time() - start_time
            self.unit_query_time.append(time.time() - start_time)
            self.unit_query_time = self.unit_query_time[-100:]
            assert embeddings_not_in_cache.shape == (len(strs_not_in_cache), 384) # size of sbert embeddings
            # Add each (action, embedding) pair to the cache
            for suggestion, embedding in zip(strs_not_in_cache, embeddings_not_in_cache):
                self.cache[suggestion] = embedding
            updated_cache = True
        else:
            embeddings_not_in_cache = torch.FloatTensor([]).to(self.device)
            updated_cache = False

        # Look up the embeddings of the strings which are in cache
        if len(strs_in_cache) > 0:
            start_time = time.time()
            embeddings_in_cache = torch.stack([self.cache[suggestion] for suggestion in strs_in_cache]).to(self.device)
            self.cache_time += time.time() - start_time
            self.unit_cache_time.append(time.time() - start_time)
            self.unit_cache_time = self.unit_cache_time[-100:]
            assert embeddings_in_cache.shape == (len(strs_in_cache), 384) # size of sbert embeddings
        else:
            embeddings_in_cache = torch.FloatTensor([]).to(self.device)

        # Concatenate the embeddings of the suggestions in the cache and the suggestions not in the cache
        suggestion_embeddings = torch.cat((embeddings_in_cache, embeddings_not_in_cache), dim=0)
        return suggestion_embeddings, all_suggestions, updated_cache

    def _get_model_embedding(self, action_name):
        " return the embedding for the action name, and a boolean indicating if the cache was updated"
        if action_name in self.cache:
            start_time = time.time()
            embedding = self.cache[action_name]
            self.cache_time += time.time() - start_time
            self.unit_cache_time.append(time.time() - start_time)
            self.unit_cache_time = self.unit_cache_time[-100:]
            return embedding, False
        else:
            start_time = time.time()
            embedding = self.embed_lm.encode(action_name, convert_to_tensor=True, device=self.device)
            self.sbert_time += time.time() - start_time
            self.unit_query_time.append(time.time() - start_time)
            self.unit_query_time = self.unit_query_time[-100:]
            self.cache[action_name] = embedding
            return embedding, True

    def _get_full_obs(self, obs):
        all_goal_str = f" {self.split_str} ".join([s.lower().strip() for s in self.old_all_suggested_actions])

        goal_str = " ".join([s.lower().strip() for s in self.suggested_actions])
        self.goal_str = goal_str
        self.oracle_goal_str = " ".join([s.lower().strip() for s in self.oracle_suggested_actions])
        if self.use_sbert:
            goal_str = 'Your goal is: ' + goal_str + '.'
        obs['goal'] = goal_str
        obs['old_goals'] = all_goal_str
        return obs

    def reset(self):
        obs, info = self.env.reset()
        self._cur_subtask = 0
        self.goal_str = None
        self.oracle_goal_str = None
        self.lm.reset()
        self.oracle_lm.reset()
        self.prev_info = info
        self._make_predictions()
        self.old_all_suggested_actions = copy.deepcopy(self.suggested_actions)
        self.old_oracle_suggested_actions = []
        obs = self._get_full_obs(obs)
        obs['success'] = False
        obs['goal_success'] = np.array(0)
        obs = self.env.tokenize_obs(obs)
        self._reset_custom_task()
        return self._tokenize_goals(obs), info

    def _make_predictions(self):
        if self._single_task is not None:
            task = self.get_subtask()
            self.suggested_actions = [task]
            self.all_suggested_actions = [task]
            self.goals_so_far[task] = None
            return

        text_obs, inv_status = self.env.text_obs()
        if self._use_state_captioner:
            state_caption = self.state_captioner(self.prev_info)
            self.state_caption = state_caption
            caption = state_caption
        else:
            caption = text_obs
        
        self.suggested_actions = self.lm.predict_options({'obs': caption, **inv_status}, self)
        self.oracle_suggested_actions = self.oracle_lm.predict_options({'obs': text_obs, **inv_status}, self)

        # Filter out bad suggestions
        if self.threshold == 1:  # Exact match
            self.suggested_actions = [s for s in self.suggested_actions if s in self.action_names]
        
        self.all_suggested_actions = copy.deepcopy(self.suggested_actions)

        for suggestion in self.suggested_actions:
            self.goals_so_far[suggestion] = None

    def custom_reward(self, action_name):
        # First compute the normal text reward (not custom)
        task = self.get_subtask()
        text_reward = int(self.check_actions_same(action_name, task))
        closest_suggestion = action_name
        if action_name == 'make crafting table' and task == 'place crafting table':
            text_reward = 1
            closest_suggestion = 'place crafting table'
        achieved_goal = text_reward == 1
        if self._single_task is None:
            return False
        if self._single_task == 'plant row':
            # The only change from the normal text reward is that we only reward for planting if the new plant is a row (2+ plants in adjacent squares)
            if (not self._single_goal_hierarchical) or (self._cur_subtask == 3):  # final subtask
                achieved_goal = False
                text_reward = 0
                if action_name == 'place plant':
                    # confirm the space ahead of the player now contains a plant
                    player = self.player
                    target = (player.pos[0] + player.facing[0], player.pos[1] + player.facing[1])
                    material, obj = player.world[target]
                    if (not obj) and obj.name.lower() == 'plant':
                        print('huh?????')
                    # Loop through all the spaces around the plant and make sure there is at least one plant
                    found_row = False
                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            if i == j:# or i == -j:  # skip the center and diagonals
                                continue
                            material, obj = player.world[(target[0] + i, target[1] + j)]
                            if obj and obj.name.lower() == 'plant':
                                found_row = True
                    if found_row:
                        text_reward = 1
                        achieved_goal = True
                    else:
                        print('not a row :(')
                    closest_suggestion = 'plant row'
        elif self._single_task == 'deforestation':
            # If we are using single_goal_hierarchical, then this task already works
            # If we're not, then we will compute the reward manually
            if not self._single_goal_hierarchical:
                if action_name == 'chop tree':
                    self.trees_chopped += 1
                    achieved_goal = True if self._single_goal_hierarchical else False
                    closest_suggestion = 'chop tree'
                if self.trees_chopped >= len(self.goal_compositions()['deforestation']):
                    achieved_goal = True
                    text_reward = 1
                    closest_suggestion = 'deforestation'       
        elif self._single_task == 'make workshop':
            # The only change from the normal text reward is that we only reward for placing the second table if it puts it in a row.
            if (not self._single_goal_hierarchical) or self._cur_subtask >= 2:  # final subtask (this lets us go tree - table - tree - table or  tree - tree - table -table)
                achieved_goal = False
                text_reward = 0
                if action_name == 'place crafting table' or action_name == 'make crafting table':
                    # confirm the space ahead of the player now contains a table
                    player = self.player
                    target = (player.pos[0] + player.facing[0], player.pos[1] + player.facing[1])
                    material, obj = player.world[target]
                    if not material == 'table':
                        print('huh?????')
                    # Loop through all the spaces around the table and make sure there is at least one table
                    found_row = False
                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            if i == j:# or i == -j:  # skip the center and diagonals
                                continue
                            material, obj = player.world[(target[0] + i, target[1] + j)]
                            if material == 'table':
                                found_row = True
                    if found_row:
                        text_reward = 1
                        achieved_goal = True
                    closest_suggestion = 'make workshop'  
        elif self._single_task == 'vegetarianism':
            achieved_goal = False
            text_reward = 0
            # This is only called when the action is successful, so we jsut need to check the action names
            if not self.drank_water:  # First we need to drink water
                if action_name == 'drink water':
                    self.drank_water = True
                    text_reward = 1 if self._single_goal_hierarchical == True else 0
                    achieved_goal = True if self._single_goal_hierarchical else 0
                    closest_suggestion = 'drink water'
            else:  # Then we need to eat a plant
                if action_name == 'chop grass':
                    text_reward = 1
                    achieved_goal = True
                    closest_suggestion = 'chop grass'
        elif self._single_task == 'chop grass with wood pickaxe':
            # The only change from the normal text reward is that we only reward for chopping grass if we have a wood pickaxe
            if task == 'chop grass with wood pickaxe':
                achieved_goal = False
                text_reward = 0
                if self.env.player.inventory['wood_pickaxe'] > 0 and action_name == 'chop grass':
                    text_reward = 1
                    achieved_goal = True
                    closest_suggestion = 'chop grass with wood pickaxe'
        return achieved_goal, text_reward, closest_suggestion
    
    def use_custom_reward(self):
        return self._single_task and self._single_task in self.custom_goals
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.action_space_type == 'harder':
            verb, noun = self.unflatten_ac(action)
            action = tuple([verb, noun])

        self.old_all_suggested_actions = copy.deepcopy(self.all_suggested_actions)
        self.old_suggested_actions =  copy.deepcopy(self.suggested_actions)
        self.old_oracle_suggested_actions = copy.deepcopy(self.oracle_suggested_actions)
        # Make a copy of the LM and oracle achievement sets
        self.old_lm_achievements = copy.deepcopy(self.lm.achievements)
        self.old_oracle_achievements = copy.deepcopy(self.oracle_lm.achievements)
        
        info['env_reward'] = reward

        health_reward = info['health_reward']
        # replace with text reward
        text_reward = 0
        closest_suggestion = None
        info['goal_achieved'] = None

        if (info['action_success'] and self.check_ac_success) or not self.check_ac_success:
            # If single_task is true, we only reward for exact matches
            action_name = self.get_action_name(action)
            if self._single_task:
                if not info['eval_success']:
                    text_reward = 0
                    closest_suggestion = None
                else:
                    task = self.get_subtask()
                    if self.use_custom_reward():
                        achieved_goal, text_reward, closest_suggestion = self.custom_reward(action_name)
                    else:
                        text_reward = int(self.check_actions_same(action_name, task))
                        closest_suggestion = action_name
                        # Only one case where the action is not a suggested action
                        if action_name == 'eat cow' and task == 'attack cow':
                            text_reward = 1
                            closest_suggestion = 'attack cow'
                        if action_name == 'make crafting table' and task == 'place crafting table':
                            text_reward = 1
                            closest_suggestion = 'place crafting table'
                        achieved_goal = text_reward == 1
                    # if we're only rewarding for the final task, then we don't want to reward for intermediate tasks
                    if self._single_goal_hierarchical == 'reward_last' and not self.on_final_subtask():
                        text_reward = 0
                
                    # If end_on_success is true, end on success
                    achieved_final_subtask = (self.on_final_subtask() or not self._single_goal_hierarchical) and achieved_goal
                    if achieved_final_subtask and self._end_on_success:
                        done = True
                    if achieved_final_subtask:
                        self.total_successes += 1
                    if achieved_goal and self._single_goal_hierarchical:
                        self._cur_subtask = min(self._cur_subtask + 1, len(self.goal_compositions()[self._single_task]) - 1)
            else:
                if len(self.old_all_suggested_actions) > 0:  # Since the captioner is slow, only compute this when there's a suggestion it could match to
                    if self._use_transition_captioner:
                        caption = self.transition_captioner(self.prev_info, info)
                        self.transition_caption = caption
                        caption_meaningful = caption.strip() != 'you did nothing.' and caption.strip() != 'nothing happened.'
                        # action id can be replaced by a string from the captioner model
                        if caption_meaningful:
                            text_reward, closest_suggestion = self.text_reward(caption, self.old_suggested_actions, update_suggestions=True)
                    else:
                        text_reward, closest_suggestion = self.text_reward(action_name, self.old_suggested_actions, update_suggestions=True)

            if not self._use_env_reward:
                reward = health_reward + text_reward
            self.lm.take_action(closest_suggestion)
            self.oracle_lm.take_action(action_name)

            info['goal_achieved'] = closest_suggestion

        else:
            if not self._use_env_reward:
                reward = health_reward # Don't compute lm reward if action failed
                
        if self._single_task == 'survival' and not self._use_env_reward:
            reward = text_reward = .05
            info['action_success'] = False
            info['eval_success'] = False

        info['text_reward'] = text_reward
        self.prev_info = info
        self._make_predictions()
        obs = self._get_full_obs(obs)
        obs['success'] = info['action_success']
        obs['goal_success']  = int(info['eval_success'] and text_reward > 0)
        if text_reward > 0:
            print(f"Goal success {obs['goal_success']}, {info['action_success']}, {text_reward}")
        obs = self.env.tokenize_obs(obs)
        return self._tokenize_goals(obs), reward, done, info

    # If the wrapper doesn't have a method, it will call the method of the wrapped environment
    def __getattr__(self, name):
        return getattr(self.env, name)
