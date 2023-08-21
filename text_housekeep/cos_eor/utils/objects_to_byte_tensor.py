# Adopted from
# https://github.com/pytorch/fairseq/blob/master/fairseq/distributed_utils.py

import pickle

import torch

# MAX_SIZE_LIMIT = max value of uint32 - num bytes to store uint32
MAX_SIZE_LIMIT = 2**32-1-4
BYTE_SIZE = 256


def enc_obj2bytes(obj, max_size=128000):
    """
    Encode Python objects to PyTorch byte tensors
    """
    assert max_size <= MAX_SIZE_LIMIT

    obj_enc = pickle.dumps(obj)
    obj_size = len(obj_enc)
    if obj_size > max_size:
        import pdb
        pdb.set_trace()
        raise Exception(
            "objects too large: object size {}, max size {}".format(obj_size, max_size)
        )

    byte_tensor = torch.zeros(obj_size+4, dtype=torch.uint8)
    obj_size_parts = torch.ByteTensor(4)
    cur_obj_size = obj_size
    for i in range(4):
        obj_size_parts[i] = cur_obj_size % 256
        cur_obj_size //= 256
    byte_tensor[:4] = obj_size_parts
    byte_tensor[4 : 4 + obj_size] = torch.ByteTensor(list(obj_enc))
    return byte_tensor


def dec_bytes2obj(byte_tensor, max_size=128000):
    """
    Decode PyTorch byte tensors to Python objects
    """
    assert max_size <= MAX_SIZE_LIMIT

    obj_size = 0
    for i in range(4):
        obj_size += int(byte_tensor[i].item()) * (256**(4-i))
    obj_enc = bytes(byte_tensor[4 : 4 + obj_size].byte().tolist())
    obj = pickle.loads(obj_enc)
    return obj


if __name__ == "__main__":
    test_obj = [1, "2", {3: 4}, [5]]
    test_obj_bytes = enc_obj2bytes(test_obj)
    test_obj_dec = dec_bytes2obj(test_obj_bytes)
    print(test_obj_dec == test_obj)