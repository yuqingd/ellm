import numpy as np

class OracleRankModule(object):
    def __init__(self, params):
        self.params = params

    def reset(self):
        self.scores = [None]

    def assert_consistency(self):
        pass

    def rerank(self, task_data, rec_rooms, objs, use_room=True):
        self.scores = np.zeros(shape=(len(rec_rooms), len(objs)))
        self.room_scores = np.zeros(shape=(len(rec_rooms), len(objs)))
        correct_mapping = task_data["correct_mapping"]
        rec_key_to_idx = {}
        for ri, rec_room in enumerate(rec_rooms.values()):
            rec_key_to_idx[rec_room["obj_key"]] = ri
        for oi, obj in enumerate(objs.values()):
            obj_key = obj["obj_key"]
            obj_rec_keys = correct_mapping[obj_key]
            for rec_key in obj_rec_keys:
                ri = rec_key_to_idx.get(rec_key, None)
                if ri is None:
                    continue
                self.scores[ri][oi] = 1
