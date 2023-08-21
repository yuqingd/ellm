import json

import numpy as np
from cos_eor.scripts.orm.utils import preprocess

class RankModule:
    def __init__(self, params):
        self.params = params
        self.data = np.load(params["file"], allow_pickle=True).item()
        self.min_score = self.data["scores"].min()
        self.build_key_to_idx_maps()
        if params.room_select == "stats":
            self.obj_room_scores = self.load_obj_room_stats(params.room_stats_file)
        elif params.room_select == "model_scores":
            self.obj_room_scores = self.load_obj_room_scores(params.room_scores_file)
        elif params.room_select == "none":
            self.obj_room_scores = np.zeros((len(self.data["objects"]), len(self.data["rooms"])))
        else:
            raise ValueError
        self.reset()

    def _list_to_idx_dict(self, l):
        return {val: idx for idx, val in enumerate(l)}

    def build_key_to_idx_maps(self):
        self.key_to_idx = {
            "rooms": {},
            "objects": {},
            "receptacles": {},
        }
        avail_rooms = self.data["rooms"]
        avail_rooms = [preprocess(r.strip()) for r in avail_rooms]  # remove spaces
        self.key_to_idx["rooms"] = self._list_to_idx_dict(avail_rooms)
        self.key_to_idx["objects"] = self._list_to_idx_dict(self.data["objects"])
        self.key_to_idx["receptacles"] = self._list_to_idx_dict(self.data["receptacles"])

    def load_obj_room_scores(self, scores_file):
        data = np.load(scores_file, allow_pickle=True).item()
        scores = data["scores"]

        avail_rooms = [preprocess(r.strip()) for r in data["rooms"]]
        obj_key2idx = self._list_to_idx_dict(data["objects"])
        room_key2idx = self._list_to_idx_dict(avail_rooms)
        obj_reorder_idxs = [obj_key2idx[obj] for obj in self.data["objects"]]
        room_reorder_idxs = [room_key2idx[room] for room in self.key_to_idx["rooms"].keys()]
        scores = scores[obj_reorder_idxs, :]
        scores = scores[:, room_reorder_idxs]

        scores = scores.argsort(axis=1).argsort(axis=1) + 1

        return scores

    def load_obj_room_stats(self, stats_file):
        with open(stats_file) as f:
            stats = json.load(f)

        avail_objs = self.data["objects"]
        avail_rooms = self.data["rooms"]

        stats_matrix = np.zeros((len(avail_objs), len(avail_rooms)))
        for obj_key, rooms in stats.items():
            obj_idx = self.key_to_idx["objects"][obj_key]
            sorted_rooms = sorted(rooms.items(), key=lambda r: r[1])
            for rank, (room_key, score) in enumerate(sorted_rooms):
                room_idx = self.key_to_idx["rooms"][room_key]
                stats_matrix[obj_idx, room_idx] = rank + 1

        return stats_matrix
    
    def reset(self):
        self.scores = [None]
        self.room_scores = [None]

    def assert_consistency(self):
        # assert consistency after every update
        pass

    def rerank(self, task_data, rec_rooms, objs, use_room=True, score_threshold=0):
        self.scores = np.zeros(shape=(len(rec_rooms), len(objs)))
        if self.params.room_select != "none":
            self.room_scores = np.zeros(shape=(len(rec_rooms), len(objs)))
        # build scores matrices
        for ri, rec_room in enumerate(rec_rooms.values()):
            rec_dump_key = preprocess(rec_room["sem_class"])
            # hack for this receptacle
            if rec_dump_key == "bottom cabinet no top":
                rec_dump_key = "bottom cabinet"
            room = preprocess(rec_room["room"])
            try:
                room_idx = self.key_to_idx["rooms"][room]
                rec_idx = self.key_to_idx["receptacles"][rec_dump_key]
            except:
                self.scores[ri] = self.min_score

            for oi, obj in enumerate(objs.values()):
                obj_dump_key = preprocess(obj["sem_class"])
                try:
                    obj_idx = self.key_to_idx["objects"][obj_dump_key]
                except:
                    print(f"couldn't find: {obj_dump_key}")
                    self.scores[ri][oi] = self.min_score
                    continue

                if use_room and room != "null":
                    try:
                        score = self.data["scores"][obj_idx][room_idx][rec_idx]
                        if self.params.room_select != "none":
                            self.room_scores[ri, oi] = self.obj_room_scores[obj_idx][room_idx]
                    except:
                        score = self.min_score
                else:
                    try:
                        score = self.data["scores"][obj_idx, :, rec_idx].mean()
                    except:
                        score = self.min_score
                self.scores[ri, oi] = score

if __name__ == "main":
    dump_path = f"cos_eor/scripts/orm/clip_scores.npy"
    rank_module = RankModule({"file": dump_path})
