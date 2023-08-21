import collections
import os
import pickle as pkl
import time
import openai
import wandb
import pathlib
import fcntl
import time
import numpy as np

# Language models provided by OpenAI. They are ranked from best to worst (quality-wise)
LANGUAGE_MODELS = [
    'code-davinci-002', # Codex Model
    'text-davinci-002',  # Best model, slowest and priciest
    'text-curie-001',
    'text-babbage-001',
    'text-ada-001',  # Worst model, fastest and cheapest
]

class PromptFormat:
    def format_prompt(self, state_dict):
        raise NotImplementedError

    def parse_response(self, response):
        raise NotImplementedError


class BulletPrompt(PromptFormat):
    def __init__(self):
        self.prefix = """Valid actions: sleep, eat, attack, chop, drink, place, make, mine
You are a player playing a game. Suggest the best actions the player can take based on the things you see and the items in your inventory. Only use valid actions and objects.

You see plant, tree, and skeleton. You are targeting skeleton. What do you do?
- Eat plant
- Chop tree
- Attack skeleton

You see water, grass, cow, and diamond. You are targeting grass. You have in your inventory plant. What do you do?
- Drink water
- Chop grass
- Attack cow
- Place plant

"""
        self.suffix = "What do you do?\n"

    def format_prompt(self, state_dict):        
        # Capitalize first letter of each element of state_dict
        obs = state_dict['obs'][0].upper() + state_dict['obs'][1:]
        if 'targeting' in obs:
            next_sentence_idx = obs.find('.') + 2
            obs = obs[:next_sentence_idx] + obs[next_sentence_idx].upper() + obs[next_sentence_idx+1:] 
        
        if 'inv' in state_dict:
            inv = state_dict['inv']
            status = 'you feel sleepy.' if 'sleepy' in state_dict['status'] else '.'
        else:
            inv = status = state_dict['inv_status']
        
        if len(inv) > 1 or len(status) > 1 :
            obs += " "
            if len(inv) > 1:
                inv = inv[0].upper() + inv[1:]
                obs += inv
            if len(status) > 1:
                if len(inv) > 1:
                    obs += " "
                status = status[0].upper() + status[1:]
                obs += status
        # Strip punctuation and convert to lowercase
        continuation = f"{obs} {self.suffix}"
        full_prompt = self.prefix + continuation
        return full_prompt

    def parse_response(self, response):
        """
        response: string, probably contains suggestions. Each suggestion starts with a dash.
        """
        
        # account for codex generating ``` or """ because it thinks it's generating comments
        if response[-4:] == '\n"""' or response[-4:] == '\n```':
            response = response[:-4]
            
        # Check that response is made of sentences which start with a dash and are separated by a newline
        if not all([s.startswith('-') for s in response.split('\n')]):
            raise ValueError(f"Response {response} is not a valid response to a bullet prompt")

        result = response.lower().split("-")
        result = [r.strip(' .\n') for r in result if len(r.strip(' .\n')) > 0]
        return result


class LanguageModel:

    def __init__(self, **kwargs):
        super().__init__()
        self.achievements = set() 
        self.verbose = kwargs.get('verbose', False)

    def reset(self):
        self.achievements = set()

    def take_action(self, suggestion):
        """
        action: action taken, in the form used in the constants file
        """
        if suggestion is not None:
            # Don't double count same suggestion
            if suggestion == 'place crafting table':
                self.achievements.add('make crafting table')
            elif suggestion == 'make crafting table':
                self.achievements.add('place crafting table')
            elif suggestion == 'eat cow':
                self.achievements.add('attack cow')
            elif suggestion == 'attack cow':
                self.achievements.add('eat cow')
            self.achievements.add(suggestion)

    def log(self, step):
        pass
        
    def predict_options(self, _, _2):
        raise NotImplementedError

    def load_and_save_cache(self):
        pass
    
    # 
    def prereq_map(self, env='yolo'):
        prereqs = { # values are [inv_items], [world_items]
                'eat plant': ([], ['plant']),
                'attack zombie': ([], ['zombie']),
                'attack skeleton': ([], ['skeleton']),
                'attack cow': ([], ['cow']),
                'eat cow': ([], ['cow']),
                'chop tree': ([], ['tree']),
                'mine stone': (['wood_pickaxe'], ['stone']),
                'mine coal': (['wood_pickaxe'], ['coal']),
                'mine iron': (['stone_pickaxe'], ['iron']),
                'mine diamond': (['iron_pickaxe'], ['diamond']),
                'drink water': ([], ['water']),
                'chop grass': ([], ['grass']),
                'sleep': ([], []),
                'place stone': (['stone'], []),
                'place crafting table': (['wood'], []),
                'make crafting table': (['wood'], []),
                'place furnace': (['stone', 'stone', 'stone', 'stone'], []),
                'place plant': (['sapling'], []),
                'make wood pickaxe': (['wood'], ['table']),
                'make stone pickaxe': (['stone', 'wood'], ['table']),
                'make iron pickaxe': (['wood', 'coal', 'iron'], ['table', 'furnace']),
                'make wood sword': (['wood'], ['table']),
                'make stone sword': (['wood', 'stone'], ['table']),
                'make iron sword': (['wood', 'coal', 'iron'], ['table', 'furnace']),
            }
        if env.action_space_type == 'harder':
            return env.filter_hard_goals(prereqs)
        else:
            return prereqs


class BaselineModel(LanguageModel):
    """
    Only propose novel combinations of verbs + nouns.
    """
    def __init__(self, all_goals, max_num_goals, novelty_bonus, **kwargs):
        super().__init__(**kwargs)
        self.novelty_bonus = novelty_bonus
        self.all_goals = all_goals
        self.novelty_bonus = novelty_bonus
        if max_num_goals == 'all':
            max_num_goals = len(self.all_goals)
        self.max_num_goals = max_num_goals
    
    def predict_options(self, state_dict, env):
        if self.novelty_bonus:
            filtered_goals =  [x for x in self.all_goals if x not in self.achievements]   
        else:
            filtered_goals = self.all_goals
        if self.max_num_goals < len(self.all_goals):
            return np.random.choices(filtered_goals, k=self.max_num_goals)
        return filtered_goals


class GPTLanguageModel(LanguageModel):
    def __init__(self, lm:str='text-curie-001', 
                 prompt_format:PromptFormat=None, 
                 max_tokens:int=100, 
                 temperature:float=.7, 
                 budget:float=3,
                 dummy_lm:bool=False,
                 openai_key=None,
                 stop_token=['\n\n'],
                 openai_org=None,
                 novelty_bonus=True,
                 **kwargs):
        """
        lm: which language model to use
        prompt format: ID of which prompt format to use
        max_tokens: maximum number of tokens returned by the lm
        temperature: temperature in [0, 1], higher is more random
        logger: optional logger
        """
        super().__init__(**kwargs)

        assert lm in LANGUAGE_MODELS, f"invalid language model {lm}; valid options are {LANGUAGE_MODELS}"
        assert 0 <= temperature <= 1, f"invalid temperature {temperature}; must be in [0, 1]"
        self.lm = lm
        self.prompt_format = prompt_format
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.budget = budget
        self.novelty_bonus = novelty_bonus
        prices = {
            'code-davinci-002' : 0.0/ 1000,
            'text-davinci-002': 0.06 / 1000,
            'text-curie-001': 0.006 / 1000,
            'text-babbage-001': 0.0012 / 1000, 
            'text-ada-001': 0.0008 / 1000,
        }
        self.price_per_token = prices[lm]
        self.tokens_per_word = 4/3
        self.budget_used = 0
        self.cached_queries = 0
        self.api_queries = 0
        self.all_queries = 0
        self.num_parse_errors = 0
        self.time = 0
        self.query_time = 0
        self.cache_time = 0
        self.cache_load_time = 0
        self.cache_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__))) / 'lm_cache.pkl'
        self.cache = self.load_cache()
        self.prices = []
        self.dummy_lm = dummy_lm
        self.attempts = 0
        self.unit_query_time = []
        self.unit_cache_time = []
        self.api_key_list = openai_key.split('_')
        openai.api_key = self.api_key_list[0]
        self.api_key_idx = 0
        self.stop = stop_token
        openai.organization = openai_org

    def load_cache(self):
        if not self.cache_path.exists():
            cache = {}
            with open(self.cache_path, 'wb') as f:
                pkl.dump({}, f)
        else:
            try:
                with open(self.cache_path, 'rb') as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    cache = pkl.load(f)
                    fcntl.flock(f, fcntl.LOCK_UN)
            except:
                cache = {}
        return cache

    def save_cache(self):
        with open(self.cache_path, 'wb') as f:
            # Lock file while saving cache so multiple processes don't overwrite it.
            fcntl.flock(f, fcntl.LOCK_EX)
            pkl.dump(self.cache, f)
            fcntl.flock(f, fcntl.LOCK_UN)

    def log(self, step):
        wandb.log({
            'lm/budget_used': self.budget_used,
            'lm/cached_queries': self.cached_queries,
            'lm/api_queries': self.api_queries,
            'lm/all_queries': self.all_queries,
            'lm/num_parse_errors': self.num_parse_errors,
            'lm/time': self.time,
            'lm/query_time': self.query_time / self.time,
            'lm/cache_time': self.cache_time / self.time,
            'lm/cache_load_time': self.cache_load_time / self.time,
            'lm/cache_size': len(self.cache),
            'lm/attempts': self.attempts,
            'lm/unit_query_time': np.mean(self.unit_query_time),
            'lm/unit_cache_time': np.mean(self.unit_cache_time),
        }, step=step)

    def load_and_save_cache(self):
        start_time = time.time()
        new_cache = self.load_cache()
        # Combine existing and new cache
        self.cache = {**new_cache, **self.cache}
        self.save_cache()
        self.cache_load_time += time.time() - start_time
        self.time += time.time() - start_time

    def check_in_cache(self, inputs):
        return inputs in self.cache

    def retrieve_from_cache(self, inputs):
        return self.cache[inputs]
    
    def try_query(self, inputs):
        query_start_time = time.time()
        response = None
        attempts = 0
        while response is None:
            try:
                response = openai.Completion.create(**inputs)
            except Exception as e:
                if self.budget_used > self.budget:
                    import pdb; pdb.set_trace()
                    if self.budget_used > self.budget:
                        print('over budget')
                    raise e
                if 'code' not in self.lm:
                    avg_price = len(inputs['prompt']) * self.tokens_per_word * self.price_per_token
                    self.prices.append(avg_price)
                    self.budget_used += avg_price
                attempts += 1
                print('LM attempts', attempts, e)
                time.sleep(4)
        self.attempts = .99 * self.attempts + .01 * attempts
        self.query_time += time.time() - query_start_time
        self.unit_query_time.append(time.time() - query_start_time)
        self.unit_query_time = self.unit_query_time[-100:]
        avg_price = len(inputs['prompt']) * self.tokens_per_word * self.price_per_token
        self.prices.append(avg_price)
        self.budget_used += avg_price
        if self.verbose: print(f"Budget used (x 1000): {self.budget_used * 1000}, avg price: {avg_price}")
        self.api_queries += 1
        return response

    def predict_options(self, state_dict, env=None):
        """
        state_dict: a dictionary with language strings as values. {'inv' : inventory, 'status': health status, 'actions': actions, 'obs': obs}

        """
        start_time = time.time()
        prompt = self.prompt_format.format_prompt(state_dict)
        inputs = (self.lm, prompt, self.max_tokens, self.temperature, tuple(self.stop))
        if self.check_in_cache(inputs):
            if self.verbose: print("Fetching from cache", prompt[-2000:-50])
            response = self.retrieve_from_cache(inputs)
            self.cached_queries += 1
            new_api_query = False
            self.cache_time += time.time() - start_time
            self.unit_cache_time.append(time.time() - start_time)
            self.unit_cache_time = self.unit_cache_time[-100:]
        else:
            if self.budget_used > self.budget:
                raise RuntimeError("Budget exceeded")
            if self.verbose: print("Fetching new inputs and response", prompt[-200:])
            
            if self.dummy_lm:
                response = '- you'
            else:
                response = None
                max_attempts = float('inf')#1000
                attempts = 0
                while response is None:
                    try:
                        response = openai.Completion.create(engine=self.lm,
                                                            prompt=prompt,
                                                            max_tokens=self.max_tokens,
                                                            stop=self.stop,
                                                            temperature=self.temperature,
                                                            frequency_penalty=0)
                    except Exception as e:
                        self.api_key_idx = (self.api_key_idx + 1) % len(self.api_key_list)
                        openai.api_key = self.api_key_list[self.api_key_idx]
                        if attempts > max_attempts or not 'code' in self.lm:
                            if attempts > max_attempts:
                                print('max attempts exceeded')
                            raise e
                        attempts += 1
                        print('attempts', attempts)
                        time.sleep(4)
                self.attempts = .99 * self.attempts + .01 * attempts
                response = response.choices[0]['text']
            new_api_query = True
        self.all_queries += 1
        if self.verbose: print("QUERIES SO FAR", self.all_queries, "cached", self.cached_queries, "api", self.api_queries)
        try:
            response_list = self.prompt_format.parse_response(response)
            # If we have not reached an error by this point, it means the response is well-formed and can be added to the cache
            if new_api_query:
                if not self.dummy_lm:
                    self.store_in_cache(inputs, response)
        except ValueError:
            print("=" * 40)
            print("         response", response)
            print("/" * 40)
            self.num_parse_errors += 1
            with open('bad_lm_strings.txt', 'a') as f:
                f.write(f"NEW FAILED QUERY:\n INPUTS \n{inputs}\n RESPONSE \n{response}\n")
            self.time += time.time() - start_time
            return []

        response_list = list(set(response_list)) # Remove duplicates

        # Lowercase and filter out items which are already achieved
        response_list = [l.lower() for l in response_list]
        if self.novelty_bonus:
            response_list = [l for l in response_list if not l in self.achievements]
        self.time += time.time() - start_time
        return response_list

    def store_in_cache(self, inputs, response):
        self.cache[inputs] = response


class SimpleOracle(LanguageModel):
    """Agent only reacts to the items it can see. No filtering by achievements."""
    
    def __init__(self, novelty_bonus, **kwargs):
        super().__init__(**kwargs)
        self.novelty_bonus = novelty_bonus

    def predict_options(self, state_dict, env):
        obs = state_dict['obs']
        if 'inv' in state_dict:
            inv = state_dict['inv']
            status = state_dict['status']
        elif 'inv_status' in state_dict:
            status = state_dict['inv_status']
        actions_list = []
        for item_str, (inv_prereqs, world_prereqs) in self.prereq_map(env).items():
            inv_prereq_dict = collections.Counter(inv_prereqs)
            if all([item in obs for item in world_prereqs]) and all([env.player.inventory[item] >= count for item, count in inv_prereq_dict.items()]):
                actions_list.append(item_str)
        # Only include 'sleep' in the list if 'sleepy' is in the status
        if not 'sleepy' in status:
            actions_list.remove('sleep')
        # Only include 'place crafting table' if we don't have a crafting table yet
        if 'crafting table' in obs and 'place crafting table' in actions_list:
            actions_list.remove('place crafting table')
        if 'thirsty' in inv and not 'drink water' in actions_list:
            actions_list.append('drink water')
        # Filter out items in achievements
        if self.novelty_bonus:
            actions_list = [action for action in actions_list if not action in self.achievements]
        return actions_list
