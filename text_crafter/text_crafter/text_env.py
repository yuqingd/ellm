"""Crafter environment with text observations."""

import string
import numpy as np
import gym
from transformers import AutoTokenizer

from text_crafter.text_crafter import constants
from text_crafter.text_crafter import engine
from text_crafter.text_crafter import objects
from text_crafter.text_crafter.env import Env

STATUS_ITEMS = ['health', 'food', 'drink', 'energy']
VERB_ONLY = ['do nothing', 'move left', 'move right', 'move up', 'move down', 'sleep']
STATUS_THRESHOLD = 9

class BaseTextEnv(Env):
    """Base text environment for running baselines, where we can get text observations"""
    def __init__(self, action_space_type='easier', use_sbert=False, max_seq_len=100,  **kwargs):
        super().__init__(**kwargs)
        self.action_space_type = action_space_type  # Easier or harder

        # Tokenizer to encode all strings
        self.use_sbert = use_sbert
        if use_sbert:
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-MiniLM-L3-v2', use_fast=True)

        # Obs configuration
        view = self._view
        item_rows = int(np.ceil(len(constants.items) / view[0]))
        self._text_view = engine.EmbeddingView(self.world, [
            objects.Player, objects.Cow, objects.Zombie,
            objects.Skeleton, objects.Arrow, objects.Plant], [view[0], view[1] - item_rows])
        self._max_seq_len = max_seq_len
        self._vocab = self._get_action_vocab()

    @property
    def action_space(self):
        """Define the discrete action space.
        With the harder env, each discrete action represents a unique (verb, noun) pair"""
        if self.action_space_type == 'harder':
            verb_only_count = len(VERB_ONLY)  # intices for actions like 'sleep' that only have a verb
            other_verb_count = len(constants.decomposed_actions['verbs']) - len(VERB_ONLY)  # indices for verbs like 'chop' that have a noun
            noun_count = len(constants.decomposed_actions['nouns']) # indices for nouns like 'tree' that pair with verbs
            return gym.spaces.Discrete(verb_only_count + other_verb_count * noun_count)
        else:
            return super().action_space

    def reset(self):
        """ Reset the env, return a dictionary of tokenized string observations."""
        obs, info = super().reset()
        text_obs, inv_status = self.text_obs()
        obs = {
            'obs': obs,
            'text_obs' :  text_obs,
            'inv_status': inv_status,
            'success': False
        }
        return self.tokenize_obs(obs), info

    def step(self, action):
        """ Step the env. Action is passed in as an int."""
        if self.action_space_type == 'harder':
            verb, noun = self.unflatten_ac(action)
            action = tuple([verb, noun])
        obs, reward, done, info = super().step(action)
        info['env_reward'] = reward
        text_obs, inv_status = self.text_obs()
        obs = {
            'obs': obs,
            'text_obs' :  text_obs,
            'inv_status': inv_status,
            'success': info['action_success']
        }
        return self.tokenize_obs(obs), reward, done, info

    @property
    def action_names(self):
        """List of strings, one for each action (including invalid actions)."""
        if self.action_space_type == 'harder':
            # action names is a flattened list of verbs x nouns.
            # Some verbs have no corresponding nouns.
            action_combinations = VERB_ONLY.copy()
            for verb in constants.decomposed_actions['verbs'][len(VERB_ONLY):]:
                verb_row = [verb + ' ' + noun for  noun in constants.decomposed_actions['nouns']]
                action_combinations.extend(verb_row)
            return action_combinations
        else:
            return super().action_names

    @property
    def good_action_names(self):
        """Return good actions - i.e. actions which are valid in the environment."""
        return constants.good_actions

    def check_action_interesting(self, action):
        """True unless the action is noop or movement."""
        if isinstance(action, int):
            action_name = self.get_action_name(action)
        else:
            action_name = action
        return action_name not in ['do nothing', 'move left', 'move right', 'move up', 'move down']

    def get_action_name(self, action):
        """Get action name. Input is either an int (the action index) or a tuple (verb index, noun index)"""
        if isinstance(action, tuple):
            verb, noun = action
            if noun is None:
                ac = constants.decomposed_actions['verbs'][verb]
            else:
                ac = constants.decomposed_actions['verbs'][verb] + ' ' + constants.decomposed_actions['nouns'][noun]
            return ac
        else:
            return self.action_names[action]

    def unflatten_ac(self, flattened_action : int):
        """Takes in an action id, and returns a tuple of (verb_id, noun_id)"""
        noun = None
        if flattened_action < len(VERB_ONLY):
            verb = flattened_action
        else:
            noun = (flattened_action - len(VERB_ONLY))  % len(constants.decomposed_actions['nouns'])
            verb = (flattened_action - len(VERB_ONLY)) // len(constants.decomposed_actions['nouns'])
            verb += len(VERB_ONLY)
        return verb, noun

    def check_actions_same(self, action_str_1, action_str_2):
        """Check if two actions are the same. Input is a string"""
        if action_str_1 == 'eat cow' and action_str_2 == 'attack cow' or action_str_1 == 'attack cow' and action_str_2 == 'eat cow':
           return True
        if action_str_1 == 'make crafting table' and action_str_2 == 'place crafting table' or action_str_1 == 'place crafting table' and action_str_2 == 'make crafting table':
            return True
        return action_str_1 == action_str_2
    
    def text_obs(self):
        """Return a dictionary of text observations"""
        inv, status = self._inventory_to_text()
        obs = self._text_view.local_sentence_view(self.player)
        return obs.lower(), {'inv' : inv.lower(), 'status': status.lower()}

    def _inventory_to_text(self):
        """
        Returns a list of strings for the inventory, and list of strings for player status.
        else returns a sentence formed from the inventory lists: "You have axe, wood..", and status lists "You feel hungry, sleepy..."
        """
        inv = []
        status = []

        # Text description only mentions low status items and inventory items the player currently has
        for k, v in self.player.inventory.items():
            if k in STATUS_ITEMS and v < STATUS_THRESHOLD: # First four elements are status items. Only include status item if it is low.
                status.append(k)
            elif k not in STATUS_ITEMS and v > 0: # Only add to inv if we have 1 or more
                inv.append(k)

        # Concatenate the status list and inventory list into strings
        inv_str, status_str = "", ""
        if len(inv) > 0:
            inv_str = "you have in your inventory "
            for item in inv:
                if item == 'sapling':
                    item = 'plant'
                inv_str += item + ", "
            inv_str = inv_str[:-2] + "."

        if len(status) > 0:
            status_str = "you feel "
            for status in status:
                if status == 'health':
                    status_str += "hurt, "
                elif status == 'food':
                    status_str += "hungry, "
                elif status == 'drink':
                    status_str += "thirsty, "
                elif status == 'energy':
                    status_str += "sleepy, "
            status_str = status_str[:-2] + "."
        return inv_str, status_str

    def tokenize_str(self, s):
        """Tokenize a string using the vocab index"""
        if self.use_sbert:  # Use SBERT tokenizer
            return np.array(self.tokenizer(s)['input_ids'])
        # Use the vocab index
        arr = np.zeros(self._max_seq_len, dtype=int)
        if " " in s:
            word_list = [w.strip(string.punctuation + ' ').lower() for w in s.split()]
            word_list = [w for w in word_list if len(w) > 0]
        else:
            word_list = [s.lower()]
        assert len(word_list) <= self._max_seq_len, f"word list length {len(word_list)} too long; increase max seq length: {self._max_seq_len}"

        for i, word in enumerate(word_list):
            if len(word) == 0:
                continue
            assert word in self._vocab, f"Invalid vocab word: |{word}|. {s}"
            arr[i] = self._vocab.index(word)
        return arr

    def pad_sbert(self, input_arr):
        """Pad array to max seq length"""
        arr = np.zeros(self._max_seq_len, dtype=int)
        if len(input_arr) > self._max_seq_len:
            input_arr = input_arr[:self._max_seq_len]
        arr[:len(input_arr)] = input_arr
        return arr

    def tokenize_obs(self, obs_dict):
        """
        Takes in obs dict and returns a dict where all strings are tokenized.
        """
        if self.use_sbert and isinstance(obs_dict['inv_status'], dict):
            inv_status = ""
            for k, v in obs_dict['inv_status'].items():
                if v != '.' and 'null' not in v:
                    inv_status += v + " "
            obs_dict['text_obs'] = obs_dict['text_obs'] + " " + inv_status

        new_obs = {}
        for k, v in obs_dict.items():
            # If the value is a dictionary of strings, concatenate them into a single string
            if isinstance(v, dict) and isinstance(list(v.values())[0], str):
                v = " ".join(v.values())
            # If the value is a string, tokenize it
            if isinstance(v, str):
                arr = self.tokenize_str(v)
                new_obs[k] = arr
            else:
                # Value is already tokenized (int, array, etc)
                new_obs[k] = v
        if self.use_sbert:
            new_obs['text_obs'] = self.pad_sbert(new_obs['text_obs'])
        return new_obs

    def untokenize_arr(self, arr):
        """Takes in an array of tokenized words and returns a string"""
        if self.use_sbert:
            # Trim off zero padding
            arr = arr[:np.argmax(arr == 0)]
            # Trim off the [CLS] token at the beginning and the [SEP] token at the end
            arr = arr[1:-1]
            return self.tokenizer.decode(arr)
        else:
            # 0 is the padding token
            return " ".join([self._vocab[token] for token in arr.tolist() if not token == 0])

    def untokenize_obs(self, obs):
        """" Takes in either a tokenized array or an obs_dict (same as output of tokenize_obs)
        Turns input into strings (or an obs_dict of strings) """
        if isinstance(obs, np.ndarray):
            return self.untokenize_arr(obs)
        assert isinstance(obs, dict)
        new_obs = {}
        for k, v in obs.items():
            if not k == 'obs':
                v = self.untokenize_arr(v)
            new_obs[k] = v
        return new_obs

    def _get_action_vocab(self):
        """Create a list of all possible vocab words."""
        # split string is the transformers library split token
        self.split_str = ' [SEP] '
        vocab = {self.split_str}
        vocab.update("you have in your inventory".split())
        vocab.update("you feel hurt hungry thirsty sleepy".split())
        vocab.update("you see".split())
        vocab.update("you are targeting".split())
        vocab.update('arrow player and'.split())
        vocab.update(constants.materials)

        split_actions = [ac.split() for ac in constants.actions]
        split_actions = [item for sublist in split_actions for item in sublist]

        vocab.update(split_actions)
        vocab.update(constants.walkable)
        vocab.update(constants.items.keys())
        vocab.update(constants.collect.keys())
        vocab.update(constants.place.keys())
        vocab.update(constants.make.keys())
        vocab.update(constants.achievements)

        vocab_list = ['null'] + sorted(list(vocab))
        return vocab_list
