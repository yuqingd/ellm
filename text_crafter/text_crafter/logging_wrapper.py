import wandb
import numpy as np

class CrafterLoggingWrapper:
    
    def __init__(self, env):
        self.env = env
        self.setup_logging()
        self.crafter_init_logging()
        self.all_good_rewards = {}
        self.all_poorly_timed_rewards = {}
        self.all_bad_rewards = {}
        self.all_nonsensical_rewards = {}
        self.all_good_acts = {}
        self.all_poorly_timed_acts = {}
        self.all_bad_acts= {}
        self.all_nonsensical_acts = {}

    # If the wrapper doesn't have a method, it will call the method of the wrapped environment
    def __getattr__(self, name):
        return getattr(self.env, name)
    
    def reset(self):
        self.setup_logging()
        output, info = self.env.reset()
        self.log_make_predictions()
        return output, info
    
    def save_counts(self, exp_name):
        mega_dict = {
            'good_rew': self.all_good_rewards,
            'poorly_timed_rew': self.all_poorly_timed_rewards,
            'bad_rew': self.all_bad_rewards,
            'nonsensical_rew': self.all_nonsensical_rewards,
            'good_acts': self.all_good_acts,
            'poorly_timed_acts': self.all_poorly_timed_acts,
            'bad_acts': self.all_bad_acts,
            'nonsensical_acts': self.all_nonsensical_acts,
        }
        import pickle as pkl
        with open(f'{exp_name}_lm_counts.pkl', 'wb') as f:
            pkl.dump(mega_dict, f)
        
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Save the captioner suggestions
        if hasattr(self.env, 'captioner_logging'):
            self.captioner_times = self.env.captioner_logging()
        
        # TODO: If we remove check_ac_success, remove this check as well
        text_reward = info['text_reward']
        if (info['action_success'] and self.check_ac_success) or not self.check_ac_success:
            if self.action_space_type == 'harder':
                verb, noun = self.unflatten_ac(action)
                action = tuple([verb, noun])
            action_name = self.get_action_name(action)
            self.all_actions_achieved[action_name] += 1 # (#4 log, #5 log)

            self.episode_actions_achieved.add(action_name)  # (#6 log, #7 log)
            closest_suggestion = info['goal_achieved']
            if closest_suggestion is not None:
                self.episode_goals_achieved.add(closest_suggestion) # (#10 log)

            if text_reward > 0:
                if sum([self.check_actions_same(action_name, act) for act in self.old_oracle_suggested_actions]) > 0:
                    self.good_action_good_time_rew_count += 1
                    if not closest_suggestion in self.all_good_rewards:
                        self.all_good_rewards[closest_suggestion] = 0
                    self.all_good_rewards[closest_suggestion] += 1
                elif closest_suggestion in self.good_action_names:
                    self.good_action_bad_time_rew_count += 1
                    if not closest_suggestion in self.all_poorly_timed_rewards:
                        self.all_poorly_timed_rewards[closest_suggestion] = 0
                    self.all_poorly_timed_rewards[closest_suggestion] += 1
                elif closest_suggestion in self.action_names:
                    self.bad_action_rew_count += 1
                    if not closest_suggestion in self.all_bad_rewards:
                        self.all_bad_rewards[closest_suggestion] = 0
                    self.all_bad_rewards[closest_suggestion] += 1
                else:
                    self.invalid_action_rew_count += 1
                    if not closest_suggestion in self.all_nonsensical_rewards:
                        self.all_nonsensical_rewards[closest_suggestion] = 0
                    self.all_nonsensical_rewards[closest_suggestion] += 1
        
        if done:
            for episode_action in self.episode_actions_suggested: # (#6 log, #7 log)
                achieved = int(episode_action in self.episode_actions_achieved)
                self.all_actions_achieved_when_suggested[episode_action] += achieved
                self.all_episode_actions_suggested[episode_action] += 1
        self.log_make_predictions()
        return obs, reward, done, info
    
    def setup_logging(self):
        self.episode_actions_suggested, self.episode_actions_achieved = set(), set()
        self.episode_goals_suggested, self.episode_actions_achieved = set(), set()
        
    def log_make_predictions(self):
        # Store the last 100 suggestions
        self.last_hundred_suggestions += self.suggested_actions
        self.last_hundred_suggestions = self.last_hundred_suggestions[-100:]

        # Check whether any of the valid actions have been suggested, and if so, log them
        for suggestion in self.suggested_actions:
            if suggestion in self.action_names:
                self.all_actions_suggested[suggestion] += 1  # (#1 log, #2 log)
                self.episode_actions_suggested.add(suggestion)  # (#6 log, #7 log)
            self.episode_goals_suggested.add(suggestion)  # (#10 log)

            # Mark each action as good or not
            if sum([self.check_actions_same(suggestion, act) for act in self.oracle_suggested_actions]) > 0:
                self.good_action_good_time_count += 1
                if not suggestion in self.all_good_acts:
                    self.all_good_acts[suggestion] = 0
                self.all_good_acts[suggestion] += 1
            elif suggestion in self.good_action_names:
                self.good_action_bad_time_count += 1
                if not suggestion in self.all_poorly_timed_acts:
                    self.all_poorly_timed_acts[suggestion] = 0
                self.all_poorly_timed_acts[suggestion] += 1
            elif suggestion in self.action_names:
                self.bad_action_count += 1
                if not suggestion in self.all_bad_acts:
                    self.all_bad_acts[suggestion] = 0
                self.all_bad_acts[suggestion] += 1
            else:
                self.invalid_action_count += 1
                if not suggestion in self.all_nonsensical_acts:
                    self.all_nonsensical_acts[suggestion] = 0
                self.all_nonsensical_acts[suggestion] += 1

            for obj in self.objs:  # (#9 log)
                if obj in suggestion:
                    if suggestion in self.good_action_names:
                        self.all_objs_good_suggestions[obj] += 1
                    elif suggestion in self.action_names:
                        self.all_objs_bad_suggestions[obj] += 1
                    else:
                        self.all_objs_broken_suggestions[obj] += 1

        for suggestion in self.oracle_suggested_actions:
            self.all_actions_oracle_suggested[suggestion] += 1  # (#3 log)
            in_suggestions = int(suggestion in self.suggested_actions)
            self.all_actions_when_oracle_suggested[suggestion] += in_suggestions  # (#8 log)

        len_before = len(self.suggested_actions)

        len_after = len(self.suggested_actions)
        self.total_suggestions += 1
        self.filtered_suggestions += len_before - len_after

        
    def crafter_init_logging(self):
        self.captioner_times = {}, {}
        
        # Count how often each action is suggested (#1 init, #2 init)
        self.all_actions_suggested = {a: 0 for a in self.action_names}
        # Count how often the oracle suggests each good action (#3 init)
        self.all_actions_oracle_suggested = {a: 0 for a in self.good_action_names}
        # Count how often each action is achieved (#4 init, #5 init)
        self.all_actions_achieved = {a: 0 for a in self.action_names}
        # Count how often when an action is suggested, it is achieved (#6 init, #7 init)
        self.episode_actions_suggested = set()
        self.episode_actions_achieved = set()
        self.all_actions_achieved_when_suggested = {a: 0 for a in self.action_names}
        self.all_episode_actions_suggested = {a: 0 for a in self.action_names}
        # Count how often when an action is suggested, it was also suggested by the oracle (#8 init)
        self.all_actions_when_oracle_suggested = {a: 0 for a in self.good_action_names}
        # All objects mentioned in prereq_dict keys (#9 init)
        self.objs = ['plant', 'zombie', 'skeleton', 'cow', 'tree', 'stone', 'coal', 'iron', 'diamond', 'water', 'grass', 'crafting table', 'furnace', 'wood pickaxe', 'stone pickaxe', 'iron pickaxe', 'wood sword', 'stone sword', 'iron sword']
        self.all_objs_good_suggestions = {o: 0 for o in self.objs}
        self.all_objs_bad_suggestions = {o: 0 for o in self.objs}
        self.all_objs_broken_suggestions = {o: 0 for o in self.objs}

        # Set of goals suggested (#10 init)
        self.episode_goals_suggested = set()
        self.episode_goals_achieved = set()

        # Count # suggestions of each type
        self.good_action_good_time_count = 0
        self.good_action_bad_time_count = 0
        self.bad_action_count = 0
        self.invalid_action_count = 0
        self.last_hundred_suggestions = []

        self.good_action_good_time_rew_count = 0
        self.good_action_bad_time_rew_count = 0
        self.bad_action_rew_count = 0
        self.invalid_action_rew_count = 0

        self.filtered_suggestions = self.total_suggestions = 0
        
    def log_lm(self, step):
        try:
            wandb_dict = {
                'lm/sbert_query_time': self.sbert_time,
                'lm/sbert_cache_time': self.cache_time,
                'lm/sbert_cache_load_time': self.cache_load_time,
                'lm/sbert_unit_cache_time': np.mean(self.unit_cache_time),
                'lm/sbert_unit_query_time': np.mean(self.unit_query_time),
                'lm/cache_hits': self.cache_hits,
                'lm/cache_misses': self.cache_misses,
            }
            for k, v in self.captioner_times[0].items():
                wandb_dict[f'captioner_transition/{k}'] = v
            for k, v in self.captioner_times[1].items():
                wandb_dict[f'captioner_state/{k}'] = v
            wandb.log(wandb_dict, step=step)
        except Exception as e:
            print(e)
        
    
    def log_real_lm(self, step):
        self.lm.log(step)

        good_actions_suggested_table = [[label, val] for (label, val) in self.all_actions_suggested.items() if label in self.good_action_names] # (#1 save)
        top_actions_suggested_table = [[label, val] for (label, val) in self.all_actions_suggested.items() if label in self.action_names] # (#2 save)
        # Take the top 20 actions suggested, sorting by value
        top_actions_suggested_table = sorted(top_actions_suggested_table, key=lambda x: x[1], reverse=True)[:20]
        oracle_actions_suggested_table = [[label, val] for (label, val) in self.all_actions_oracle_suggested.items()] # (#3 save)
        good_actions_achieved_table = [[label, val] for (label, val) in self.all_actions_achieved.items() if label in self.good_action_names] # (#4 save)
        top_actions_achieved_table = [[label, val] for (label, val) in self.all_actions_achieved.items() if label in self.action_names] # (#5 save)
        # Take the top 20 actions achieved, sorting by value
        top_actions_achieved_table = sorted(top_actions_achieved_table, key=lambda x: x[1], reverse=True)[:20]
        good_actions_achieved_when_suggested_table = [[label, val/self.all_episode_actions_suggested[label]] for (label, val) in self.all_actions_achieved_when_suggested.items() if label in self.good_action_names and self.all_episode_actions_suggested[label] > 0] # (#6 save)
        top_actions_achieved_when_suggested_table = [[label, val/self.all_episode_actions_suggested[label]] for (label, val) in self.all_actions_achieved_when_suggested.items() if label in self.action_names and self.all_episode_actions_suggested[label] > 0] # (#7 save)
        # Take the top 20 actions achieved when suggested, sorting by value
        top_actions_achieved_when_suggested_table = sorted(top_actions_achieved_when_suggested_table, key=lambda x: x[1], reverse=True)[:20]
        suggest_when_oracle_table = [[label, val/self.all_actions_oracle_suggested[label]] for (label, val) in self.all_actions_when_oracle_suggested.items() if self.all_actions_oracle_suggested[label] > 0] # (#8 save)
        nonzero_objs_good_suggestions_table = [[label, val] for (label, val) in self.all_objs_good_suggestions.items() if val > 0] # (#9 save)
        nonzero_objs_bad_suggestions_table = [[label, val] for (label, val) in self.all_objs_bad_suggestions.items() if val > 0]
        nonzero_objs_broken_suggestions_table = [[label, val] for (label, val) in self.all_objs_broken_suggestions.items() if val > 0]

        # Turn these into actual tables
        good_actions_suggested_table = wandb.Table(data=good_actions_suggested_table, columns = ["Good Action", "Count"])
        top_actions_suggested_table = wandb.Table(data=top_actions_suggested_table, columns = ["Action", "Count"])
        oracle_actions_suggested_table = wandb.Table(data=oracle_actions_suggested_table, columns = ["Oracle Action", "Count"])
        good_actions_achieved_table = wandb.Table(data=good_actions_achieved_table, columns = ["Good Action", "Count"])
        top_actions_achieved_table = wandb.Table(data=top_actions_achieved_table, columns = ["Action", "Count"])
        if not good_actions_achieved_when_suggested_table == []:
            good_actions_achieved_when_suggested_table = wandb.Table(data=good_actions_achieved_when_suggested_table, columns = ["Good Action", "Fraction"])
        if not top_actions_achieved_when_suggested_table == []:
            top_actions_achieved_when_suggested_table = wandb.Table(data=top_actions_achieved_when_suggested_table, columns = ["Action", "Fraction"])
        if not suggest_when_oracle_table == []:
            suggest_when_oracle_table = wandb.Table(data=suggest_when_oracle_table, columns = ["Action", "Fraction"])
        if not nonzero_objs_good_suggestions_table == []:
            nonzero_objs_good_suggestions_table = wandb.Table(data=nonzero_objs_good_suggestions_table, columns = ["Object", "Count"])
        if not nonzero_objs_bad_suggestions_table == []:
            nonzero_objs_bad_suggestions_table = wandb.Table(data=nonzero_objs_bad_suggestions_table, columns = ["Object", "Count"])
        if not nonzero_objs_broken_suggestions_table == []:
            nonzero_objs_broken_suggestions_table = wandb.Table(data=nonzero_objs_broken_suggestions_table, columns = ["Object", "Count"])
        episode_goals_suggested_table = wandb.Table(data=[['Goals', ', '.join(self.episode_goals_suggested)]], columns = ["Goal", "Goals"])
        episode_goals_achieved_table = wandb.Table(data=[['Goals', ', '.join(self.episode_goals_achieved)]], columns = ["Goal", "Goal Text"])
        last_hundred = wandb.Table(data=[['Goals', ', '.join(self.last_hundred_suggestions)]], columns = ["Goal", "Last 100"])
        action_breakdown = wandb.Table(data=[ ['good actions', self.good_action_good_time_count], 
                                                ['poorly timed', self.good_action_bad_time_count], 
                                                ['bad actions', self.bad_action_count],
                                                ['invalid actions', self.invalid_action_count]], columns=['Action Type', 'Count'])
        reward_breakdown = wandb.Table(data=[ ['good actions', self.good_action_good_time_rew_count], 
                                                ['poorly timed', self.good_action_bad_time_rew_count], 
                                                ['bad actions', self.bad_action_rew_count],
                                                ['invalid actions', self.invalid_action_rew_count]], columns=['Action Type', 'Count'])

        try:
            wandb_dict = {
                'lm_acc/fraction_filtered': self.filtered_suggestions / self.total_suggestions,
                'lm/sbert_query_time': self.sbert_time,
                'lm/sbert_cache_time': self.cache_time,
                'lm/sbert_cache_load_time': self.cache_load_time,
                'lm/sbert_unit_cache_time': np.mean(self.unit_cache_time),
                'lm/sbert_unit_query_time': np.mean(self.unit_query_time),
                'lm/cache_hits': self.cache_hits,
                'lm/cache_misses': self.cache_misses,
                'lm/sbert_time': self.sbert_time + self.cache_time + self.cache_load_time,
                'lm_acc/good_actions_suggested': wandb.plot.bar(good_actions_suggested_table, "Good Action", "Count",
                                    title="Suggestion Counts for Good Actions"),
                'lm_acc/top_actions_suggested': wandb.plot.bar(top_actions_suggested_table, "Action", "Count",
                                    title="Suggestion Counts for 20 Most Frequent Actions"),
                'lm_acc/oracle_actions_suggested': wandb.plot.bar(oracle_actions_suggested_table, "Oracle Action", "Count",
                                    title="Suggestion Counts for Oracle Actions"),
                'lm_acc/good_actions_achieved': wandb.plot.bar(good_actions_achieved_table, "Good Action", "Count",
                                    title="Achievement Counts for Good Actions"),
                'lm_acc/top_actions_achieved': wandb.plot.bar(top_actions_achieved_table, "Action", "Count",
                                    title="Achievement Counts for 20 Most Frequent Actions"),
                'lm_acc/episode_goals_suggested': episode_goals_suggested_table,
                'lm_acc/episode_goals_achieved': episode_goals_achieved_table,
            }
            if not good_actions_achieved_when_suggested_table == []:
                wandb_dict['lm_acc/good_achieve_rate'] = wandb.plot.bar(good_actions_achieved_when_suggested_table, "Good Action", "Fraction",
                                    title="P(episode achieved goal | episode suggested goal)")
            if not top_actions_achieved_when_suggested_table == []:
                wandb_dict['lm_acc/top_achieve_rate'] = wandb.plot.bar(top_actions_achieved_when_suggested_table, "Action", "Fraction",
                                    title="Top 20: P(episode achieved goal | episode suggested goal)")
            if not suggest_when_oracle_table == []:
                wandb_dict['lm_acc/suggest_when_oracle'] = wandb.plot.bar(suggest_when_oracle_table, "Action", "Fraction",
                                    title="P(suggested goal | oracle suggested goal)")
            if not nonzero_objs_good_suggestions_table == []:
                wandb_dict['lm_acc/objs_good_suggestions'] = wandb.plot.bar(nonzero_objs_good_suggestions_table, "Object", "Count",
                                    title="Good Actions With Object Suggested")
            if not nonzero_objs_bad_suggestions_table == []:
                wandb_dict['lm_acc/objs_bad_suggestions'] = wandb.plot.bar(nonzero_objs_bad_suggestions_table, "Object", "Count",
                                    title="Bad Actions With Object Suggested")
            if not nonzero_objs_broken_suggestions_table == []:
                wandb_dict['lm_acc/objs_broken_suggestions'] = wandb.plot.bar(nonzero_objs_broken_suggestions_table, "Object", "Count",
                                    title="Invalid Actions With Object Suggested")
            wandb_dict['lm_acc/action_breakdown'] = wandb.plot.bar(action_breakdown, 'Action Type', 'Count', 'Frequency of Action Types')
            wandb_dict['lm_acc/reward_breakdown'] = wandb.plot.bar(reward_breakdown, 'Action Type', 'Count', 'Frequency of Actions Rewarded For')
            wandb_dict['lm_acc/last_hundred'] = last_hundred

            for k, v in self.captioner_times[0].items():
                wandb_dict[f'captioner_transition/{k}'] = v
            for k, v in self.captioner_times[1].items():
                wandb_dict[f'captioner_state/{k}'] = v

            wandb.log(wandb_dict, step=step)
        except Exception as e:
            print(e)
        
