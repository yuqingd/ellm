import yaml

# generic code to load all receptacles and objects
global_mapping_path = "cos_eor/utils/dump_archive/global_mapping_v3.yaml"
global_mapping = yaml.load(open(global_mapping_path, "r"), Loader=yaml.BaseLoader)
igib_objs = list(set([item[0] for item in global_mapping["objects"] if item[-1] == 'igib']))
ycb_objs = list(set([item[0] for item in global_mapping["objects"] if item[-1] == 'ycb_or_ab_adjusted']))
igib_rooms = []
for rooms in global_mapping["object_room_map"].values():
    igib_rooms.extend(rooms)
igib_rooms = list(set(igib_rooms))
igib_recs = list(set([item[0] for item in global_mapping["receptacles"] if item[-1] == 'igib']))
ycb_recs = list(set([item[0] for item in global_mapping["receptacles"] if item[-1] == 'ycb_or_ab_adjusted']))
igib_rooms = sorted(igib_rooms)
igib_recs = sorted(igib_recs)
ycb_recs = sorted(ycb_recs)

objects = igib_objs + ycb_objs
objects = [f"{o}, None" for o in objects]
receptacles = []
for rec in igib_recs:
    if rec in global_mapping["object_room_map"]:
        rooms = set(global_mapping["object_room_map"][rec])
        receptacles.extend([f"{rec}, {room}"for room in rooms])

stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
              'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
              'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
              'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
              'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
              'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
              'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
              'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
              'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
              'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
              'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
              'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
              "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
              "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
              'wouldn', "wouldn't"]

spatial_prepositions = ["in", "on"]

# MLM templates
rec_templates = [
    # "Put {} {} {}",
    # "Store {} {} {}",
    # "Keep {} {} {}",
    # "Usually, we put {} {} {}",
    # "Usually, we store {} {} {}",
    # "Usually, we keep {} {} {}",
    "In a household, usually, we put {} {} {}",

]
room_rec_templates = [
    # "In {}, store {} {} {}",
    # "In {}, put {} {} {}",
    "In {}, usually you put {} {} {}"
]

room_templates = [
    # "The room where you find {} is called {}",  # 0.7142857143
    # "The room where {} is found is called {}",  # 0.5918367347 (past-tense bad?)
    # "In a house, the room where {} is found is called {}",  # 0.693877551 (In a house helps?)
    # "In a house, the room where you find {} is called {}",  # 0.7755102041
    # "In a household, the room where you find {} is called {}",  # 0.
    # "In a household, usually, the room where you find {} is called {}",  # 0.7959183673 (best)
    # "In a household, usually, you can find {} in the room called {}",  # 0.
    # "In a household, usually, you can find {} in a room called {}",  # 0.
    # "In a household, often, you can find {} in the room called {}",  # 0.
    # "In a household, likely, you can find {} in the room called {}",  # 0.
    "In a household, it is likely that you can find {} in the room called {}",  # 0.
    # "Within a household, the room where you find {} is called {}",  # 0.
    # "Within a household, often times you can find {} in the room called {}",  # 0.
    # "In a house, the room where {} is kept is called {}",   # 0.693877551
    # "In a house, the room where you keep {} is called {}",  # 0.7551020408
    # "In a house, the room where {} is stored is called {}", # 0.7142857143
    # "In a house, the room where you store {} is called {}", # 0.7551020408
    # "In a house, the room where {} is placed is called {}", # 0.7346938776
    # "In a house, the room where you place {} is called {}", # 0.7551020408


  ]

mlm_templates = {
    "rec_templates": rec_templates,
    "room_templates": room_templates,
    "room_rec_templates": room_rec_templates,
}


# Living room synonyms
living_room_syns = ["lounge", "parlor", "salon"]


def get_mlm_templates(debug=False):
    if debug:
        return mlm_templates["rec_templates"][1:], mlm_templates["room_templates"][1:], mlm_templates["room_rec_templates"][1:]

    return mlm_templates["rec_templates"], mlm_templates["room_templates"], mlm_templates["room_rec_templates"]


def build_lists():
    obj_list, rec_list = [], []

    for objs in [objects]:
        obj_list.extend(objs)

    for recs in [receptacles]:
        rec_list.extend(recs)

    return obj_list, rec_list