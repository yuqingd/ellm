import numpy as np
import pdb

data = np.load('cos_eor/scripts/orm/amt_data/data.npy', allow_pickle=True).item()

# ranks: a 3d numpy array: [#room-receps, #objects, #annotations]
ranks = data["ranks"]
objects = data["objects"]
# room_receptacles: a list of room-receptacle pairs with the room name and receptacle name concatenated with '|'
room_receptacles = data["room_receptacles"]

obj_to_idx = {idx:obj for obj,idx in enumerate(objects)}
room_recep_to_idx = {idx:room_rec for room_rec, idx in enumerate(room_receptacles)}


'''
returns a np.array containing 10 numbers - 1 for each annotation
a positive rank indicates the recep was placed under 'after'
a negative rank indicates the recep was placed under 'before'
zero value for rank indicates the recep marked as 'implausible'
'''
def get_ranks_for_room_recep(obj, room, recep):
    return ranks[room_recep_to_idx[room + '|' + recep], obj_to_idx[obj], :] 

if __name__ == '__main__':
    print(get_ranks_for_room_recep('pear', 'kitchen', 'fridge'))