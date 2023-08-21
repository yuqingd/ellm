import time
import numpy as np
from text_crafter.lm import GPTLanguageModel
from text_crafter.lm import PromptFormat

class HousekeepSinglePrompt(PromptFormat):
    def __init__(self, prob_threshold):
        self.threshold = prob_threshold
        self.prefix = 'You are a robot in a house. You have the ability to pick up objects and place them in new locations. For each example, state if the item should be stored in/on the receptacle.'
        self.prefix += f'\nShould you store a dirty spoon in/on the chair: No.'
        self.prefix += f'\nShould you store a mixing bowl in/on the dishwasher: Yes.'
        self.prefix += f'\nShould you store a clean sock in/on the drawer: Yes.'

    def format_prompt(self, state_dict):
        self.item = state_dict['item']
        self.receptacle = state_dict['receptacle']
        prompt = self.prefix
        prompt += f'\nShould you store a {state_dict["item"]} in/on the {state_dict["receptacle"]}:'
        return prompt

    def parse_response(self, response):
        """
        response: dict, contains the response from the LM
        """
        choice_idx = 0
        top = response['choices'][0]["logprobs"]["top_logprobs"][choice_idx]
        
        # collect the log probs of the two answers (or give a small value if not found)
        no_lp = top.get(' No', -100)
        yes_lp = top.get(' Yes', -100)
        no_probs = np.exp(no_lp)
        yes_probs = np.exp(yes_lp)
        prob_sum = no_probs + yes_probs
        # renormalize
        no_probs /= prob_sum
        yes_probs /= prob_sum
        return yes_probs > self.threshold

class GPTHouseKeep(GPTLanguageModel):
    def __init__(self, lm:str='text-curie-001', 
                 prompt_format:PromptFormat=None, 
                 max_tokens:int=2, 
                 temperature:float=.7, 
                 budget:float=3,
                 dummy_lm:bool=False,
                 openai_key=None,
                 stop_token=['\n'],
                 num_log_probs=5,
                 **kwargs):
        self.num_log_probs = num_log_probs
        super().__init__(lm=lm, prompt_format=prompt_format, max_tokens=max_tokens, temperature=temperature, budget=budget,
                         dummy_lm=dummy_lm, openai_key=openai_key, stop_token=stop_token, **kwargs)

    def predict_options_helper(self, state_dict, prob_threshold=0.5):
        """
        state_dict: a dictionary with language strings as values. {'inv' : inventory, 'status': health status, 'actions': actions, 'obs': obs}

        """
        start_time = time.time()
        prompt = self.prompt_format.format_prompt(state_dict)
        inputs = (self.lm, prompt, self.max_tokens, self.temperature, tuple(self.stop), self.num_log_probs)
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
                lm_inputs = dict(engine=self.lm,  prompt=prompt, max_tokens=self.max_tokens, stop=self.stop, temperature=self.temperature, frequency_penalty=0, logprobs=self.num_log_probs) # Get top k
                response = self.try_query(lm_inputs)
            new_api_query = True
        self.all_queries += 1
        if self.verbose: print("QUERIES SO FAR", self.all_queries, "cached", self.cached_queries, "api", self.api_queries)
        response_bool = self.prompt_format.parse_response(response)
        # If we have not reached an error by this point, it means the response is well-formed and can be added to the cache
        if new_api_query:
            if not self.dummy_lm:
                self.store_in_cache(inputs, response)

        self.time += time.time() - start_time
        return response_bool
    
    def predict_options(self, state_dict, env=None):
        # Split the prompt into separate prompts for each obj pair
        holding = state_dict['holding']
        results = []
        lm_api_queries = self.api_queries
        if holding is not None:
            if self.novelty_bonus and holding in self.achievements:
                pass
            else:
                for receptacle in state_dict['all_receptacles']:
                    good_placement = self.predict_options_helper({'item': holding, 'receptacle': receptacle})
                    if good_placement:
                        results.append((holding, receptacle))
        else:
            for obj, receptacle in state_dict['visible_objects'].items():
                if obj in self.achievements and self.novelty_bonus:
                    continue
                good_placement = self.predict_options_helper({'item': obj, 'receptacle': receptacle})
                if not good_placement:  # If the placement is not good, then we should pick up the object
                    results.append((obj, ''))
        # Save any new API queries to the cache
        if self.api_queries != lm_api_queries:
            self.load_and_save_cache()
        return results

    def predict_success(self, obs_dict):
        success = {}
        for obj, receptacle in obs_dict.items():
            if 'agent' in receptacle:
                success[obj] = False
            else:
                good_placement = self.predict_options_helper({'item': obj, 'receptacle': receptacle})
                success[obj] = good_placement
        
        return success
            


    
    def reset(self, env=None):
        self.achievements = set()
    
    def take_action(self, object):
        self.achievements.add(object)

class BaselineHousekeep:
    def predict_options(self, state_dict, env=None):
        holding = state_dict['holding']
        results = []
        if holding is not None and holding not in self.achievements:
            for receptacle in state_dict['all_receptacles']:
                results.append((holding, receptacle))
        else:
            for obj, receptacle in state_dict['visible_objects'].items():
                if obj in self.achievements:
                    continue
                results.append((obj, ''))
        return results

    def predict_success(self, obs_dict):
        success = {}
        for obj, receptacle in obs_dict.items():
            success[obj] = True
        
        return success
    
    def take_action(self, object):
        self.achievements.add(object)
    
    def reset(self, env=None):
        self.achievements = set()

    def log(self, step):
        return



class OracleHouseKeep:
    def predict_options(self, state_dict, env=None):
        holding = state_dict['holding']
        results = []
        if holding is not None and holding not in self.achievements:
            for receptacle in state_dict['all_receptacles']:
                good_placement =  self.combos_dict[(holding, receptacle)]
                if good_placement:
                    results.append((holding, receptacle))
        else:
            for obj, receptacle in state_dict['visible_objects'].items():
                if obj in self.achievements:
                    continue
                good_placement = self.combos_dict[(obj, receptacle)]
                if not good_placement:
                    results.append((obj, ''))
        return results

    def predict_success(self, obs_dict):
        success = {}
        for obj, receptacle in obs_dict.items():
            good_placement = self.combos_dict[(obj, receptacle)]
            success[obj] = good_placement
        
        return success
    
    def take_action(self, object):
        self.achievements.add(object)
    
    def reset(self, env=None):
        self.combos_dict = self.get_oracle_placements(env)
        self.achievements = set()

    def log(self, step):
        return

    def get_oracle_placements(self, env):
        combos_dict = {}
        obs = env.raw_obs
        receptacles_list = obs['cos_eor']['recs_keys']
        objs_list = obs['cos_eor']['objs_keys']
        correct_mapping = obs['cos_eor']['correct_mapping']
        sid_to_class = obs['cos_eor']['sid_class_map']
        key_to_sim_obj_id = obs['cos_eor']['obj_key_to_sim_obj_id']
        iid_to_sid = obs['cos_eor']['iid_to_sid']
        sim_obj_id_to_iid = obs['cos_eor']['sim_obj_id_to_iid']
        obj_id_to_room = obs['cos_eor']['obj_id_to_room']
        # This is for the *single one* option
        for obj in objs_list:
            # Objs are in the format obj_name_num. We trim the num
            obj_name = ' '.join(obj.split('_')[:-1])
            for rec in receptacles_list:
                # Recs are urdf files, which we convert to keys and then to names
                if rec == 'agent': # None:
                    continue
                
                receptacle_room_name = obj_id_to_room[key_to_sim_obj_id[rec]] 
                receptacle_room_name = ' '.join(receptacle_room_name.split('_')[:-1]) # room names end with an integer.
                
                rec_name_und = sid_to_class[iid_to_sid[sim_obj_id_to_iid[key_to_sim_obj_id[rec]]]]
                rec_name = ' '.join(rec_name_und.split('_'))

                rec_name = receptacle_room_name + ' ' + rec_name
                
                valid_recs = correct_mapping[obj]
                # True if correct mapping and false if incorrect mapping
                correct_placement = rec in valid_recs
                combos_dict[(obj_name, rec_name)] = correct_placement
        return combos_dict