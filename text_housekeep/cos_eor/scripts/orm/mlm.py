# %%
import torch
from transformers import pipeline
import pandas as pd
from collections import Counter, defaultdict
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from lists import mlm_templates, stop_words, spatial_prepositions, build_lists, get_mlm_templates, living_room_syns
from utils import build_map, add_row, get_list
from registry import registry
from tokenizers import AddedToken
import os
from scipy.special import softmax as softmax_func


def get_prep_freqs(receptacle, df_objects):
    template_prep_dict = {}

    for tem in rec_templates:
        template_list = []
        object_list = []
        for idx, row in df_objects.iterrows():
            # Fill -- "Put A <mask> B"
            template_list.append(tem.format(row["entity"], "<mask>", receptacle))
            object_list.append(row["entity"])

        filled_template_list = unmasker(template_list)
        filled_preps = []

        for preds, obj in zip(filled_template_list, object_list):
            for item in preds:
                if item["token_str"][1:].lower() in spatial_prepositions:
                    filled_preps.append((item["token_str"][1:].lower(), obj))
                    break
            else:
                import pdb
                pdb.set_trace()

        template_prep_dict[tem] = {
            "preps": filled_preps,
            "object_list": object_list,
            "top_prep": Counter(prep_obj[0] for prep_obj in filled_preps).most_common(1)[0][0]
        }

    return template_prep_dict


def add_spatial_prepositions(df_receptacles, df_objects):
    top_preps = []
    for idx, row in tqdm(df_receptacles.iterrows(), "Matching spatial-prepositions"):
        prep_dict = get_prep_freqs(row["entity"], df_objects)
        # aggregate top across all templates
        top_prep_templates = [v["top_prep"] for k, v in prep_dict.items()]
        top_prep = Counter(top_prep_templates).most_common(1)[0][0]
        top_preps.append(top_prep)
    df_receptacles["spatial-prep"] = top_preps


def get_room_receptacle(obj, prep, rr_list, rr_type="receptacle", obj_room=None, debug=False, rev=False, softmax=True):
    """
    The outputs w/ a tokenizer with and without spaces is different.

    > (Pdb) unmasker.tokenizer('bedroom')
    > {'input_ids': [0, 15112, 2], 'attention_mask': [1, 1, 1]}
    > (Pdb) unmasker.tokenizer(' bedroom')
    > {'input_ids': [0, 8140, 2], 'attention_mask': [1, 1, 1]}`

    Todo: Check lstrip and rstrip arguments in the tokenizer
    These arguments are set by the folks who trained the model, doesn't seem right to change them during
    inference now.

    (Pdb) unmasker.tokenizer
    > PreTrainedTokenizer(name_or_path='roberta-large', vocab_size=50265,
    model_max_len=512, is_fast=False, padding_side='right',
    special_tokens={
    'mask_token': AddedToken("<mask>", rstrip=False, lstrip=True, single_word=False, normalized=True)
    }

    Culprit is line 156 in "tokenization_roberta.py"

    """
    rec_templates, room_templates, room_rec_templates = get_mlm_templates(debug)

    # rr_type defines if we want to match objects with receptacles/rooms
    if rr_type == "receptacle":
        # if we know what room the receptacles lie in use 
        # that information and corresponding templates
        if obj_room is None:
            templates = deepcopy(rec_templates)
            templates = [tem.format(obj, prep, "<mask>") for tem in templates]
        else:
            templates = deepcopy(room_rec_templates)
            templates = [tem.format(obj_room, obj, prep, "<mask>") for tem in templates]

    else:
        templates = deepcopy(room_templates)

        if rev:
            templates = [tem.format("<mask>", obj) for tem in templates]
        else:
            templates = [tem.format(obj, "<mask>") for tem in templates]

    tokenizer = unmasker.tokenizer

    tokenized_rr = tokenizer(rr_list, add_special_tokens=False)
    dec_step_probs = [[[] for _ in rr_list] for _ in templates]

    # print a sample to check decoding and encoding are same
    # print(f"Enc-Dec: '{tokenizer.decode(tokenized_rr['input_ids'][0])}'"
    #       f" vs Original: '{rr_list[0]}' vs Tok: '{tokenized_rr['input_ids'][0]}'")

    # one-step decoding
    outputs = unmasker(templates, registry=registry)

    # fill probs for one-step
    assert len(registry["mlm_probs"]) == len(templates)
    for temp_idx in range(len(templates)):
        for rr_idx in range(len(rr_list)):
            tok_idx = tokenized_rr["input_ids"][rr_idx][0]
            dec_step_probs[temp_idx][rr_idx].append(float(registry["mlm_probs"][temp_idx][tok_idx]))

    # deal with multi-step
    # (rec_idx, steps_remain)
    multistep_rr = [(rr_idx, len(tokenized_rr["input_ids"][rr_idx]))
                    for rr_idx in range(len(rr_list)) if len(tokenized_rr["input_ids"][rr_idx]) > 1]

    # build_input
    multistep_input = [[] for _ in range(len(templates))]
    unrolled_input = []

    masked_tok_idx = unmasker.tokenizer.mask_token_id
    tokenized_temps = unmasker._parse_and_tokenize(templates)

    for temp_idx in range(len(templates)):
        for rr_idx, rr_len in multistep_rr:
            # build the input for given template and rr; replace masked tok-idx
            temp_inds = tokenized_temps["input_ids"][temp_idx].tolist()
            replace_inds = [tokenized_rr["input_ids"][rr_idx][:i + 1] for i in range(rr_len - 1)]
            mask_pos = temp_inds.index(masked_tok_idx)

            # build input with mask token over all positions for a given receptacle
            _input = []
            for rep_ind in replace_inds:
                __in = deepcopy(temp_inds)
                __in[mask_pos:mask_pos] = rep_ind
                _input.append(__in)

            # attach to multistep_input
            multistep_input[temp_idx].append(_input)
            unrolled_input.extend(_input)

    # print a sample to check decoding and encoding are same
    # print(f"Enc-Dec: '{tokenizer.decode(unrolled_input[0])}' vs Tok: '{unrolled_input[0]}'")

    # add padding and build attention mask
    max_seq_len = max([len(_) for _ in unrolled_input])
    attention_mask = []

    for idx in range(len(unrolled_input)):
        if len(unrolled_input[idx]) < max_seq_len:
            unrolled_input[idx] += [unmasker.tokenizer.pad_token_id] * (max_seq_len - len(unrolled_input[idx]))
        attention_mask.append([0 if tok == unmasker.tokenizer.pad_token_id else 1 for tok in unrolled_input[idx]])

    # pipelines API replace the input <mask> in-place w/ an incorrect token!
    tokenized_input = {"input_ids": torch.tensor(unrolled_input), "attention_mask": torch.tensor(attention_mask)}
    multistep_output = unmasker(tokenized_input, registry=registry, do_tokenize=False)

    # fill probs
    count_idx = 0
    for temp_idx in range(len(templates)):
        for rr_idx, rr_len in multistep_rr:
            num_probs = rr_len - 1
            tok_inds = tokenized_rr["input_ids"][rr_idx][1:]
            assert num_probs == len(tok_inds)
            probs = []

            for idx in range(num_probs):
                probs.append(float(registry["mlm_probs"][count_idx][tok_inds[idx]]))
                count_idx += 1

            # add to input
            dec_step_probs[temp_idx][rr_idx].extend(probs)

    # sanity check
    assert len(unrolled_input) == count_idx
    # single-step probs
    for temp_idx in range(len(templates)):
        rr_probs = dec_step_probs[temp_idx]
        rr_probs = [x[0] for x in rr_probs if type(x) == list]
        dec_step_probs[temp_idx] = rr_probs

    # num_templates x num_rr
    rr_inds = np.array(dec_step_probs).argmax(-1)
    rr_choices = [rr_list[ind] for ind in rr_inds]
    dec_step_probs = np.array(dec_step_probs)

    # return ranked-list
    if rev:
        # if softmax:
        #     dec_step_probs = dec_step_probs/dec_step_probs.sum(axis=1, keepdims=1)
        #     # dec_step_probs = softmax_func(dec_step_probs, axis=1)

        asc_sorted_inds = np.argsort(dec_step_probs, axis=-1)
        desc_sorted_inds = np.flip(asc_sorted_inds, axis=-1)
        return desc_sorted_inds, dec_step_probs

    # print(f"object: {obj}, rr_choices: {rr_choices}")
    return rr_choices, dec_step_probs


def get_room(df_objects, df_receptacles, debug=False):
    rooms = get_list(df_receptacles, "room", remove_none_dup=True, insert_spaces=True, append_list=living_room_syns)
    objects = get_list(df_objects, "entity")
    target_rooms = get_list(df_objects, "room")

    object_room_scores = []
    for obj, tar in tqdm(zip(objects, target_rooms), desc="Matching rooms given objects", total=len(objects)):
        rr_choices, one_step_probs = get_room_receptacle(obj, None, rooms, rr_type="room")
        object_room_scores.append(one_step_probs.squeeze())

    scores = np.stack(object_room_scores, axis=0)
    result = {
        "rooms": rooms,
        "objects": objects,
        "scores": scores
    }
    return result


def get_receptacle(df_objects, df_receptacles, use_rooms=False):
    objects = get_list(df_objects, "entity")
    receptacles = get_list(df_receptacles, "entity", insert_spaces=True, remove_none_dup=True)
    rooms = get_list(df_receptacles, "room", remove_none_dup=True, insert_spaces=True, append_list=living_room_syns)

    if use_rooms:
        scores = []
        for obj in tqdm(objects, desc="Matching receptacles-rooms", total=len(objects)):
            object_room_rec_probs = []
            for room in rooms:
                one_step_probs_preps = []
                for prep in spatial_prepositions:
                    rr_choices, one_step_probs = get_room_receptacle(obj, prep, receptacles, "receptacle", room)
                    one_step_probs_preps.append(one_step_probs)
                object_room_rec_probs.append(np.sum(one_step_probs_preps, axis=0).squeeze())
            object_room_rec_probs = np.stack(object_room_rec_probs, axis=0)
            scores.append(object_room_rec_probs)
        scores = np.stack(scores, axis=0)
    else:
        object_receptacle_scores = []
        for obj in tqdm(objects, desc="Matching receptacles", total=len(objects)):
            one_step_probs_preps = []
            for prep in spatial_prepositions:
                rr_choices, one_step_probs = get_room_receptacle(obj, prep, receptacles, "receptacle")
                one_step_probs_preps.append(one_step_probs)
            object_receptacle_scores.append(np.sum(one_step_probs_preps, axis=0).squeeze())
        scores = np.stack(object_receptacle_scores, axis=0)

    result = {
        "receptacles": [rec.strip() for rec in receptacles],
        "objects": objects,
        "scores": scores,
        "use_rooms": use_rooms,
        "rooms": [room.strip() for room in rooms]
    }
    return result


def quick_fix(dump_path, df_objects, df_receptacles):
    data = np.load(dump_path, allow_pickle=True).item()
    receptacles = get_list(df_receptacles, "entity", insert_spaces=True, remove_none_dup=True)
    rooms = get_list(df_receptacles, "room", remove_none_dup=True, insert_spaces=True, append_list=living_room_syns)
    # fix some labels
    data["object_receptacle_room_scores"]["receptacles"] =  [rec.strip() for rec in receptacles]
    data["object_receptacle_room_scores"]["rooms"] =  [room.strip() for room in rooms]
    data["object_receptacle"]["receptacles"] =  [rec.strip() for rec in receptacles]
    data["object_receptacle"]["rooms"] =  [room.strip() for room in rooms]

    # rename
    data["object_receptacle_room"] =  deepcopy(data["object_receptacle_room_scores"])
    del data["object_receptacle_room_scores"]

    # restore
    np.save(dump_path, data)


# we get lists of object/receptacles alongwith the rooms they're found in from AI2-THOR and ObjectNet
obj_room_list, rec_room_list = build_lists()

# we build dataframes from these lists
df_objects = build_map(obj_room_list, True)
df_receptacles = build_map(rec_room_list, True)

rank_top = 5
use_fast = False
model_name = "roberta-large"
dump_path = f"cos_eor/utils/{model_name}_scores.npy"
# rev=True will match objects given a room, rev=False will match rooms given an object
rev = False
task = "fill-mask"

# fix some issues
# quick_fix(dump_path, df_objects, df_receptacles)

import pdb
pdb.set_trace()

# initialize model
unmasker = pipeline(task, model=model_name, use_fast=use_fast)

# match object to rooms
object_room_scores = get_room(df_objects, df_receptacles)

# match object to receptacles
object_receptacle_scores = get_receptacle(df_objects, df_receptacles)

# match object to room-receptacle pairs
object_receptacle_room_scores = get_receptacle(df_objects, df_receptacles, use_rooms=True)

lm_dump = {
    "object_room": object_room_scores,
    "object_receptacle": object_receptacle_scores,
    "object_receptacle_room": object_receptacle_room_scores,
    "config": {
        "rank_top": rank_top,
        "use_fast": use_fast,
        "model_name": model_name,
        "rev": rev,
        "task": task
    }
}

np.save(dump_path, lm_dump)
print(f"Dumped: {dump_path}")