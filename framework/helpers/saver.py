import torch
import os
import time
from typing import Optional, List
from collections import defaultdict


class SaverElement:
    def save(self):
        raise NotImplementedError()

    def load(self, saved_state):
        raise NotImplementedError()


class PyObjectSaver(SaverElement):
    def __init__(self, obj):
        self._obj = obj

    def load(self, state):
        def _load(target, state):
            if hasattr(target, "load_state_dict"):
                target.load_state_dict(state)
            elif isinstance(target, dict):
                for k, v in state.items():
                    target[k] = _load(target.get(k), v)
            elif isinstance(target, list):
                if len(target) != len(state):
                    target.clear()
                    for v in state:
                        target.append(v)
                else:
                    for i, v in enumerate(state):
                        target[i] = _load(target[i], v)
            else:
                return state
            return target

        _load(self._obj, state)
        return True

    def save(self):
        def _save(target):
            if isinstance(target, (defaultdict, dict)):
                res = target.__class__()
                res.update({k: _save(v) for k, v in target.items()})
            elif hasattr(target, "state_dict"):
                res = target.state_dict()
            elif isinstance(target, list):
                res = [_save(v) for v in target]
            else:
                res = target

            return res

        return _save(self._obj)


class Saver:
    def __init__(self, dir: str, keep_every_n_hours: Optional[int] = 4, keep_last: int = 1):
        self.savers = {}
        self.dir = dir
        assert keep_last >= 1
        self.keep_last = keep_last
        self._keep_every_n_seconds = keep_every_n_hours * 3600 if keep_every_n_hours else None

    def register(self, name: str, saver, replace: bool = False):
        if not replace:
            assert name not in self.savers, "Saver %s already registered" % name

        if isinstance(saver, SaverElement):
            self.savers[name] = saver
        else:
            self.savers[name] = PyObjectSaver(saver)

    def __setitem__(self, key: str, value):
        if value is not None:
            self.register(key, value)

    def get_data(self):
        return {name: fns.save() for name, fns in self.savers.items()}

    def save(self, fname: Optional[str] = None, dir: Optional[str] = None, iter: Optional[int]=None):

        if fname is None:
            assert iter is not None, "If fname is not given, iter should be."
            if dir is None:
                dir = self.dir
            fname = os.path.join(dir, self.model_name_from_index(iter))

        dname = os.path.dirname(fname)
        if dname:
            os.makedirs(dname, exist_ok=True)

        print("Saving %s" % fname)
        state = self.get_data()

        try:
            torch.save(state, fname)
        except:
            print("WARNING: Save failed. Maybe running out of disk space?")
            try:
                os.remove(fname)
            except:
                pass
            return None

        return fname

    def tick(self, iter: int):
        self.save(iter=iter)
        self.cleanup()

    @staticmethod
    def model_name_from_index(index: int) -> str:
        return f"model-{index}.pth"

    @staticmethod
    def get_checkpoint_index_list(dir: str) -> List[int]:
        return list(reversed(sorted(
            [int(fn.split(".")[0].split("-")[-1]) for fn in os.listdir(dir) if fn.split(".")[-1] == "pth"])))

    def get_ckpts_in_time_window(self, dir: str, index_list: Optional[List[int]]=None):
        if index_list is None:
            index_list = Saver.get_checkpoint_index_list(dir)

        names = [Saver.model_name_from_index(i) for i in index_list]
        if self._keep_every_n_seconds is None:
            return names

        now = time.time()

        res = []
        for name in names:
            mtime = os.path.getmtime(os.path.join(dir, name))
            if now - mtime > self._keep_every_n_seconds:
                break

            res.append(name)

        return res

    @staticmethod
    def do_load(fname):
        return torch.load(fname)

    def load_last_checkpoint(self) -> Optional[any]:
        if not os.path.isdir(self.dir):
            return None

        last_checkpoint = Saver.get_checkpoint_index_list(self.dir)

        if last_checkpoint:
            for index in last_checkpoint:
                fname = Saver.model_name_from_index(index)
                try:
                    data = self.do_load(os.path.join(dir, fname))
                except:
                    continue
                return data
        return None

    def cleanup(self):
        index_list = self.get_checkpoint_index_list(self.dir)
        new_files = self.get_ckpts_in_time_window(self.dir, index_list[self.keep_last:])
        new_files = new_files[:-1] if self._keep_every_n_seconds is not None else new_files

        for f in new_files:
            os.remove(os.path.join(self.dir, f))

    def load_data(self, state) -> bool:
        if not state:
            return False

        for k, s in state.items():
            if k not in self.savers:
                print("WARNING: failed to load state of %s. It doesn't exists." % k)
                continue

            print(f"Loading {k}")
            if not self.savers[k].load(s):
                print(f"Failed to load {k}")
                return False

        return True

    def load(self, fname=None) -> bool:
        if fname is None:
            state = self.load_last_checkpoint()
        else:
            state = self.do_load(fname)

        return self.load_data(state)
