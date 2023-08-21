import time
import PIL
import skimage.io as io
import torch
import numpy as np
import torch.nn.functional as nnf
import clip  # git+https://github.com/openai/CLIP.git
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import gym
import text_crafter.text_crafter
from text_crafter.text_crafter import constants
import pathlib
import os
import time

path_to_transition_weights = pathlib.Path(os.path.dirname(os.path.realpath(__file__))) / 'transition_captioner_model.pt'
path_to_state_weights = pathlib.Path(os.path.dirname(os.path.realpath(__file__))) / 'state_captioner_model.pt'

inventory_keys_ordered = ['health', 'food', 'drink', 'energy', 'sapling', 'wood', 'stone', 'coal', 'iron', 'diamond', 'wood_pickaxe', 'stone_pickaxe',
                          'iron_pickaxe', 'wood_sword', 'stone_sword', 'iron_sword']
actions_sorted = ['do', 'make_iron_pickaxe', 'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword', 'make_wood_pickaxe', 'make_wood_sword', 'move_down', 'move_left',
              'move_right', 'move_up', 'noop', 'place_furnace', 'place_plant', 'place_stone', 'place_table', 'sleep']
objects = ['arrow', 'coal', 'cow', 'diamond', 'furnace', 'grass', 'iron', 'iron_pickaxe', 'iron_sword', 'lava', 'path', 'plant', 'sand', 'skeleton', 'stone',
           'stone_pickaxe', 'stone_sword', 'table', 'tree', 'water', 'wood_pickaxe', 'wood_sword', 'zombie']
class Predictor():
    def __init__(self, prefix_size_transition=1024+56, prefix_size_state=512+56, transition_weight_path=path_to_transition_weights, state_weight_path=path_to_state_weights):
        self.device = torch.device("cuda")
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.prefix_length = 10
        self.transition_model = ClipCaptionModel(self.prefix_length, prefix_size=prefix_size_transition)
        self.transition_model.load_state_dict(torch.load(transition_weight_path, map_location=torch.device("cpu")))
        self.transition_model = self.transition_model.eval()
        self.transition_model = self.transition_model.to(self.device)

        self.state_model = ClipCaptionModel(self.prefix_length, prefix_size=prefix_size_state)
        self.state_model.load_state_dict(torch.load(state_weight_path, map_location=torch.device("cpu")))
        self.state_model = self.state_model.eval()
        self.state_model = self.state_model.to(self.device)

        self.time_dict_transition = {
            'make_pil': 0,
            'preprocess': 0,
            'encode_image': 0,
            'clip_project': 0,
            'generate': 0,
            'overall_predict': 0,
            'overall': 0,
        }
        self.time_dict_state = {
            'make_pil': 0,
            'preprocess': 0,
            'encode_image': 0,
            'clip_project': 0,
            'generate': 0,
            'overall_predict': 0,
            'overall': 0,
        }

    def predict_state(self, image, semantic_emb_pre):
        """Run a single prediction on the model"""
        predict_start_time = time.time()
        start_time = time.time()
        pil_image = PIL.Image.fromarray(image)
        self.time_dict_state['make_pil'] += time.time() - start_time
        start_time = time.time()
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        semantic_emb_pre = torch.tensor(semantic_emb_pre).to(self.device, dtype=torch.float32).unsqueeze(0)
        self.time_dict_state['preprocess'] += time.time() - start_time
        start_time = time.time()

        with torch.no_grad():
            prefix = self.clip_model.encode_image(image).to(self.device, dtype=torch.float32)
            self.time_dict_state['encode_image'] += time.time() - start_time
            start_time = time.time()
            prefix = torch.cat([prefix, semantic_emb_pre], dim=1)
            prefix_embed = self.state_model.clip_project(prefix).reshape(1, self.prefix_length, -1)
            self.time_dict_state['clip_project'] += time.time() - start_time
        start_time = time.time()
        output = generate2(self.state_model, self.tokenizer, embed=prefix_embed)
        self.time_dict_state['generate'] += time.time() - start_time
        self.time_dict_state['overall_predict'] += time.time() - predict_start_time
        return output

    def predict_transition(self, image, next_image, semantic_emb_pre, semantic_emb_post):
        """Run a single prediction on the model"""
        predict_start_time = time.time()
        start_time = time.time()
        pil_image = PIL.Image.fromarray(image)
        pil_next_image = PIL.Image.fromarray(next_image)
        self.time_dict_transition['make_pil'] += time.time() - start_time
        start_time = time.time()
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        next_image = self.preprocess(pil_next_image).unsqueeze(0).to(self.device)
        semantic_emb_pre = torch.tensor(semantic_emb_pre).to(self.device, dtype=torch.float32)
        semantic_emb_post = torch.tensor(semantic_emb_post).to(self.device, dtype=torch.float32)
        semantic_emb_diff = (semantic_emb_post - semantic_emb_pre).unsqueeze(0)
        self.time_dict_transition['preprocess'] += time.time() - start_time
        start_time = time.time()

        with torch.no_grad():
            prefix = self.clip_model.encode_image(image).to(self.device, dtype=torch.float32)
            prefix_next = self.clip_model.encode_image(next_image).to(self.device, dtype=torch.float32)
            self.time_dict_transition['encode_image'] += time.time() - start_time
            start_time = time.time()
            prefix = torch.cat([prefix, prefix_next, semantic_emb_diff], dim=1)
            prefix_embed = self.transition_model.clip_project(prefix).reshape(1, self.prefix_length, -1)
            self.time_dict_transition['clip_project'] += time.time() - start_time
        start_time = time.time()
        output = generate2(self.transition_model, self.tokenizer, embed=prefix_embed)
        self.time_dict_transition['generate'] += time.time() - start_time
        self.time_dict_transition['overall_predict'] += time.time() - predict_start_time
        return output

class MLP(torch.nn.Module):
    def __init__(self, sizes, bias=True, act=torch.nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(torch.nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ClipCaptionModel(torch.nn.Module):

    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:  # not enough memory
            self.clip_project = torch.nn.Linear(prefix_size, self.gpt_embedding_size * prefix_length)
        else:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2, self.gpt_embedding_size * prefix_length))

    def get_dummy_token(self, batch_size, device):
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens, prefix, mask=None, labels = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out


def generate2(model, tokenizer, embed, entry_length=67, top_p=0.8, temperature=1.0, stop_token= ".",):
    model.eval()
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    tokens = None
    with torch.no_grad():
        generated = embed
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = filter_value
            next_token = torch.argmax(logits, -1).unsqueeze(0)
            next_token_embed = model.gpt.transformer.wte(next_token)
            if tokens is None: tokens = next_token
            else: tokens = torch.cat((tokens, next_token), dim=1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            if stop_token_index == next_token.item():
                break
        output_list = list(tokens.squeeze().cpu().numpy())
        output_text = tokenizer.decode(output_list)
        generated_list.append(output_text)
    return generated_list[0]

def get_captioner():
    captioners = Predictor()
    def caption_transition(info_pre, info_post):
        start_time = time.time()
        image_pre = info_pre['large_obs']
        image_post = info_post['large_obs']
        semantic_emb_pre = extract_representation(info_pre)
        semantic_emb_post = extract_representation(info_post)
        output = captioners.predict_transition(image_pre, image_post, semantic_emb_pre, semantic_emb_post)
        captioners.time_dict_transition['overall'] += time.time() - start_time
        return output

    def caption_state(info_pre):
        start_time = time.time()
        image_pre = info_pre['large_obs']
        semantic_emb_pre = extract_representation(info_pre)
        output = captioners.predict_state(image_pre, semantic_emb_pre)
        captioners.time_dict_state['overall'] += time.time() - start_time
        return output

    def get_log_dicts():
        return captioners.time_dict_transition, captioners.time_dict_state

    return caption_transition, caption_state, get_log_dicts

def extract_representation(info):
    action_embedding = np.zeros(len(actions_sorted))
    if not info['player_action'] is None:
        simple_act = convert_hard_act_to_simple(info['player_action'])
        assert simple_act in actions_sorted
        action_index = actions_sorted.index(simple_act)
        action_embedding[action_index] = 1
    inventory_embedding = np.array([info['inventory'][k] for k in inventory_keys_ordered])
    object_embedding = np.zeros(len(objects))
    obj_set = set()
    for line in info['local_token']:
        obj_set = obj_set.union(set(line))
    for o in obj_set:
        if o not in ['Player', 'Null']:
            object_embedding[objects.index(o.lower())] = 1
    representation = np.concatenate([action_embedding, inventory_embedding, object_embedding])
    return representation

def decode_representation(sem_emb):
    action_embedding = list(sem_emb[:len(actions_sorted)])
    action = actions_sorted[action_embedding.index(1)]
    text = f'Action: {action}'
    inv_emb = sem_emb[len(actions_sorted):len(actions_sorted) + len(inventory_keys_ordered)]
    text += '\nInventory: '
    for i in range(len(inventory_keys_ordered)):
        if inv_emb[i] != 0:
            text += inventory_keys_ordered[i] + ', '
    text = text[:-2] + '.\nObjects: '
    obj_emb = sem_emb[-len(objects):]
    for i in range(len(objects)):
        if obj_emb[i] == 1:
            text += objects[i] + ', '
    text = text[:-2] + '.'
    print(text)
    return text

def sample_random_hard_action(only_valid=True):
    if only_valid:
        return np.random.choice(constants.good_actions)
    else:
        verb = np.random.choice(constants.decomposed_actions['verbs'])
        if verb in ['mine', 'eat', 'attack', 'chop', 'drink', 'place', 'make']:
            verb += ' ' + np.random.choice(constants.decomposed_actions['nouns'])
        return verb

def convert_hard_act_to_simple(hard_act):

    if hard_act == 'do nothing' or hard_act not in constants.good_actions:
        return 'noop'
    elif hard_act.replace(' ', '_') in ['move_up', 'move_down', 'move_left', 'move_right', 'make_iron_pickaxe', 'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword',
                                'make_wood_pickaxe', 'make_wood_sword', 'place_furnace', 'place_plant', 'place_stone', 'place_table']:
        return hard_act.replace(' ', '_')
    elif hard_act == 'place crafting table':
        return 'place_table'
    elif hard_act == 'make crafting table':
        return 'place_table'
    elif hard_act == 'eat cow':
        return 'do'
    elif hard_act == 'sleep':
        return 'sleep'
    else:
        return 'do'


def test_action_conversion():
    for verb in constants.decomposed_actions['verbs']:
        if verb in ['mine', 'eat', 'attack', 'chop', 'drink', 'place', 'make']:
            for noun in constants.decomposed_actions['nouns']:
                hard_act = verb + ' ' + noun
                simple_act = convert_hard_act_to_simple(hard_act)
                if hard_act in constants.good_actions:
                    print(hard_act,  "==", simple_act, ' - VALID')
                assert simple_act in actions_sorted
        else:
            hard_act = verb
            simple_act = convert_hard_act_to_simple(hard_act)
            if hard_act in constants.good_actions:
                print(hard_act, "==", simple_act, ' - VALID')
            assert simple_act in actions_sorted


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    env_spec = dict(action_space_type='harder',
                    dying=True,
                    length=400,
                    simple=False,
                    threshold=0.999,
                    single_goal_hierarchical=False)

    # test_action_conversion()
    # print(actions_sorted)
    env = gym.make("CrafterTextEnv-v1", **env_spec)

    show = True  # show
    caption_transition, caption_state, get_time_dicts = get_captioner()

    obs, info = env.reset()
    state_caption = caption_state(info)
    print(f'State caption: {state_caption}')
    if show:
        plt.figure()
        figure_generated = plt.imshow(np.concatenate([info['large_obs'], info['large_obs']], axis=1))
        plt.draw()
        plt.pause(0.001)
    for i in range(len(constants.good_actions)):
        print(i, constants.good_actions[i])
    for i in range(300):
        act = sample_random_hard_action(only_valid=True)
        act = 'move right'
        act = 'move left'
        act = 'move up'
        act = 'move down'
        act = 'chop tree'
        act = 'place crafting table'
        act = 'make wood pickaxe'
        # act = 'make wood sword'
        # act = 'attack skeleton'
        act = 'eat cow'
        act = 'mine stone'
        act_idx = env.action_names.index(act)
        new_obs, _, _, new_info = env.step(act_idx)
        init = time.time()
        transition_caption = caption_transition(info, new_info)
        print(act)
        print(f'Transition caption: {transition_caption}')
        state_caption = caption_state(new_info)
        print(f'State caption: {state_caption}')
        print(time.time() - init)
        figure_generated = plt.imshow(np.concatenate([info['large_obs'], new_info['large_obs']], axis=1))
        plt.draw()
        plt.pause(0.001)
        info, obs = new_info, new_obs
        stop = 1

    stop = 1