import os

import pandas as pd

stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]


def no_digits(s):
    return ''.join([i for i in s if not i.isdigit()])


# use this function to preprocess orm strings
def preprocess(word):
    word = word.strip().lower()
    word = process_ycb(word)
    return word


def has_digits(s):
    return any([i.isdigit() for i in s])


def process_ycb(s):
    s = os.path.split(s)[-1]
    if s.endswith(".object_config.json"):
        s = s.split(".object_config.json")[0]
    splits = s.split("_")
    remove_digits = []
    for split in splits:
        if not has_digits(split):
            remove_digits.append(split)
    return " ".join(remove_digits)


def build_map(entity_list, allow_multiword=False):
    cols = ["entity", "room", "multi-word-entity", "multi-word-room"]
    df = pd.DataFrame(columns=cols)
    for en_ro in entity_list:
        en, ro = en_ro.split(",")
        en, ro = en.strip().lower(), ro.strip().lower()
        en, ro = process_ycb(en), process_ycb(ro)
        df_row = [en, ro, len(en.split()), len(ro.split())]
        if len(en.split()) == 1 or allow_multiword:
            df = add_row(df, df_row)
    return df


def add_row(df, row):
    df.loc[-1] = row  # adding a row
    df.index = df.index + 1  # shifting index
    df = df.sort_index()  # sorting by index
    return df


def get_list(df, col, remove_none_dup=False, insert_spaces=False, append_list=None):
    _list = [s.lower().strip() for s in df[col].to_list()]

    if append_list:
        _list += append_list

    if remove_none_dup:
        _list = list(set(_list))
        if "none" in _list:
            _list.remove("none")


    if insert_spaces:
        _list = [f" {s}" for s in _list]

    return _list
