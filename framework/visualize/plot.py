import torch
import os
import numpy as np
from ..utils import U
from typing import Dict, Tuple, List, Optional, Callable, Union
import threading
import atexit
from torch.multiprocessing import Process, Queue, Event
from queue import Empty as EmptyQueue
import sys
import itertools
import PIL
import time

wandb = None
plt = None
make_axes_locatable = None


def import_matplotlib():
    global plt
    global make_axes_locatable
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable


class CustomPlot:
    def to_tensorboard(self, name: str, summary_writer, global_step: int):
        pass

    def to_wandb(self):
        return None


class Histogram(CustomPlot):
    def __init__(self, data: Union[torch.Tensor, np.ndarray]):
        if torch.is_tensor(data):
            data = data.detach().cpu()

        self.data = data

    def to_tensorboard(self, name: str, summary_writer, global_step: int):
        summary_writer.add_histogram(name, self.data, global_step)

    def to_wandb(self):
        return wandb.Histogram(self.data)


class Image(CustomPlot):
    def __init__(self, data: Union[torch.Tensor, np.ndarray], caption: Optional[str] = None):
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()

        self.data = data.astype(np.float32)
        self.caption = caption

    def to_tensorboard(self, name, summary_writer, global_step):
        if self.data.shape[-1] in [1,3]:
            data = np.transpose(self.data, (2,0,1))
        else:
            data = self.data
        summary_writer.add_image(name, data, global_step)

    def to_wandb(self):
        if self.data.shape[0] in [1, 3]:
            data = np.transpose(self.data, (1,2,0))
        else:
            data = self.data

        if data.shape[-1] == 1:
            data = np.repeat(data, 3, axis=-1)

        data = PIL.Image.fromarray(np.ascontiguousarray((data*255.0).astype(np.uint8)), mode="RGB")
        return wandb.Image(data, caption = self.caption)


class Video(CustomPlot):
    def __init__(self, data: Union[torch.Tensor, np.ndarray], caption: Optional[str] = None):
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()

        # self.data = data.astype(np.float32)
        self.data = data
        self.caption = caption

    def to_tensorboard(self, name, summary_writer, global_step):
        if self.data.shape[-1] in [1, 3]:
            data = np.transpose(self.data, (0, 3, 1, 2))
            data = np.expand_dims(data, axis=0)
        else:
            data = self.data
        data = data.astype(np.float32) / 255
        summary_writer.add_video(name, data, global_step)

    def to_wandb(self):
        if self.data.shape[-1] in [1, 3]:
            data = np.transpose(self.data, (0, 3, 1, 2))
        else:
            data = self.data

        # data = np.ascontiguousarray((data * 255.0).astype(np.uint8))
        return wandb.Video(data, fps=10, format="gif")


class Scalars(CustomPlot):
    def __init__(self, scalar_dict: Dict[str, Union[torch.Tensor, np.ndarray, int, float]]):
        self.values = {k: v.item() if torch.is_tensor(v) else v for k, v in scalar_dict.items()}
        self.leged = sorted(self.values.keys())

    def to_tensorboard(self, name, summary_writer, global_step):
        v = {k: v for k, v in self.values.items() if v == v}
        summary_writer.add_scalars(name, v, global_step)

    def to_wandb(self):
        return self.values


class Scalar(CustomPlot):
    def __init__(self, val: Union[torch.Tensor, np.ndarray, int, float]):
        if torch.is_tensor(val):
            val = val.item()

        self.val = val

    def to_tensorboard(self, name, summary_writer, global_step):
        summary_writer.add_scalar(name, self.val, global_step)

    def to_wandb(self):
        return self.val


class XYChart(CustomPlot):
    def __init__(self, data: Dict[str, List[Tuple[float, float]]], markers: List[Tuple[float,float]] = [],
                 xlim = (None, None), ylim = (None, None)):
        import_matplotlib()

        self.data = data
        self.xlim = xlim
        self.ylim = ylim
        self.markers = markers

    def matplotlib_plot(self):
        f = plt.figure()
        names = list(sorted(self.data.keys()))

        for n in names:
            plt.plot([p[0] for p in self.data[n]], [p[1] for p in self.data[n]])

        if self.markers:
            plt.plot([a[0] for a in self.markers], [a[1] for a in self.markers], linestyle='', marker='o',
                 markersize=2, zorder=999999)

        plt.legend(names)
        plt.ylim(*self.xlim)
        plt.xlim(*self.ylim)

        return f

    def to_tensorboard(self, name, summary_writer, global_step):
        summary_writer.add_figure(name, self.matplotlib_plot(), global_step)

    def to_wandb(self):
        return self.matplotlib_plot()



class Heatmap(CustomPlot):
    def __init__(self, map: Union[torch.Tensor, np.ndarray], xlabel: str, ylabel: str,
                 round_decimals: Optional[int] = None, x_marks: Optional[List[str]] = None,
                 y_marks: Optional[List[str]] = None):

        if torch.is_tensor(map):
            map = map.detach().cpu().numpy()

        self.round_decimals = round_decimals
        self.map = map
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.x_marks = x_marks
        self.y_marks = y_marks

    def to_matplotlib(self):
        figure, ax = plt.subplots(figsize=(self.map.shape[0]*0.25 + 2, self.map.shape[1]*0.15+1.5))
        im = plt.imshow(self.map, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto')

        x_marks = self.x_marks if self.x_marks is not None else [str(i) for i in range(self.map.shape[1])]
        assert len(x_marks) == self.map.shape[1]

        y_marks = self.y_marks if self.y_marks is not None else [str(i) for i in range(self.map.shape[0])]
        assert len(y_marks) == self.map.shape[0]

        plt.xticks(np.arange(self.map.shape[1]), x_marks, rotation=45, fontsize=8, ha="right", rotation_mode="anchor")
        plt.yticks(np.arange(self.map.shape[0]), y_marks, fontsize=8)

        # Use white text if squares are dark; otherwise black.
        threshold = self.map.max() / 2.

        rmap = np.around(self.map, decimals=self.round_decimals) if self.round_decimals is not None else self.map
        for i, j in itertools.product(range(self.map.shape[0]), range(self.map.shape[1])):
            color = "white" if self.map[i, j] > threshold else "black"
            plt.text(j, i, rmap[i, j], ha="center", va="center", color=color, fontsize=8)

        plt.ylabel(self.ylabel)
        plt.xlabel(self.xlabel)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.25, pad=0.1)
        plt.colorbar(im, cax)

        plt.tight_layout()
        return figure

    def to_tensorboard(self, name, summary_writer, global_step):
        summary_writer.add_figure(name, self.to_matplotlib(), global_step)

    def to_wandb(self):
        return wandb.Image(self.to_matplotlib())


class ConfusionMatrix(Heatmap):
    def __init__(self, map: Union[torch.Tensor, np.ndarray], class_names: Optional[List[str]] = None,
                 x_marks: Optional[List[str]] = None, y_marks: Optional[List[str]] = None):

        if torch.is_tensor(map):
            map = map.detach().cpu().numpy()

        map = np.transpose(map, (1, 0))
        map = map.astype('float') / map.sum(axis=1).clip(1e-6, None)[:, np.newaxis]

        if class_names is not None:
            assert x_marks is None and y_marks is None
            x_marks = y_marks = class_names

        super().__init__(map, "predicted", "real", round_decimals=2, x_marks = x_marks, y_marks = y_marks)


class TextTable(CustomPlot):
    def __init__(self, header: List[str], data: List[List[str]]):
        self.header = header
        self.data = data

    def to_markdown(self):
        res = " | ".join(self.header)+"\n"
        res += " | ".join("---" for _ in self.header)+"\n"
        return res+"\n".join([" | ".join(l) for l in self.data])

    def to_tensorboard(self, name, summary_writer, global_step):
        summary_writer.add_text(name, self.to_markdown(), global_step)

    def to_wandb(self):
        return wandb.Table(data=self.data, columns=self.header)


class Logger:
    @staticmethod
    def parse_switch_string(s: str) -> Tuple[bool,bool]:
        s = s.lower()
        if s=="all":
            return True, True
        elif s=="none":
            return False, False

        use_tb, use_wandb =  False, False
        s = s.split(",")
        for p in s:
            if p=="tb":
                use_tb = True
            elif p=="wandb":
                use_wandb = True
            else:
                assert False, "Invalid visualization switch: %s" % p

        return use_tb, use_wandb

    def create_loggers(self):
        self.is_sweep = False
        self.wandb_id = {
            "run_name": "tb",
        }
        global wandb

        if self.use_wandb:
            import wandb
            # CSCS failed many times with the following error message (from `slurm-33699807`):
            # wandb.errors.UsageError: Error communicating with wandb process
            # This is a workaround from here:
            # https://github.com/wandb/client/issues/1409#issuecomment-723371808
            while True:
                try:
                    wandb.init(**self.wandb_init_args)
                    break
                except:
                    print("wandb init failed. Retrying after 10 seconds..")
                    time.sleep(10)
            self.wandb_id = {
                "sweep_id": wandb.run.sweep_id,
                "run_id": wandb.run.id,
                "run_name": wandb.run.name,
            }
            self.is_sweep = bool(wandb.run.sweep_id)
            wandb.config["is_sweep"] = self.is_sweep
            wandb.config.update(self.wandb_extra_config)

            self.save_dir = os.path.join(wandb.run.dir)

        os.makedirs(self.save_dir, exist_ok=True)
        self.tb_logdir = os.path.join(self.save_dir, "tensorboard")

        if self.use_tb:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(self.tb_logdir, exist_ok=True)
            self.summary_writer = SummaryWriter(log_dir=self.tb_logdir, flush_secs=10)
        else:
            self.summary_writer = None

    def __init__(self, 
                 save_dir: Optional[str] = None, 
                 use_tb: bool = False, 
                 use_wandb: bool = False,
                 get_global_step: Optional[Callable[[], int]] = None, 
                 get_global_epoch: Optional[Callable[[], int]] = None, 
                 wandb_init_args={}, 
                 wandb_extra_config={}):
        global plt
        global wandb

        import_matplotlib()

        self.use_wandb = use_wandb
        self.use_tb = use_tb
        self.save_dir = save_dir
        self.get_global_step = get_global_step
        self.get_global_epoch = get_global_epoch
        self.wandb_init_args = wandb_init_args
        self.wandb_extra_config = wandb_extra_config

        # self.create_loggers()

    def flatten_dict(self, dict_of_elems: Dict) -> Dict:
        res = {}
        for k, v in dict_of_elems.items():
            if isinstance(v, dict):
                v = self.flatten_dict(v)
                for k2, v2 in v.items():
                    res[k+"/"+k2] = v2
            else:
                res[k] = v
        return res

    def get_step(self, step: Optional[int] = None) -> Optional[int]:
        if step is None and self.get_global_step is not None:
            step = self.get_global_step()
        return step

    def get_epoch(self, epoch: Optional[int] = None) -> Optional[int]:
        if epoch is None and self.get_global_epoch is not None:
            epoch = self.get_global_epoch()
        return epoch

    def log(self, plotlist: Union[List, Dict], step: Optional[int] = None):
        if not isinstance(plotlist, list):
            plotlist = [plotlist]

        plotlist = [p for p in plotlist if p]
        if not plotlist:
            return

        d = {}
        for p in plotlist:
            d.update(p)

        self.log_dict(d, step)

    def log_dict(self, 
                 dict_of_elems: Dict, 
                 step: Optional[int] = None, 
                 epoch: Optional[int] = None):
        dict_of_elems = self.flatten_dict(dict_of_elems)

        if not dict_of_elems:
            return

        dict_of_elems = {k: v.item() if torch.is_tensor(v) and v.nelement()==1 else v for k, v in dict_of_elems.items()}
        dict_of_elems = {k: Scalar(v) if isinstance(v, (int, float)) else v for k, v in dict_of_elems.items()}

        step = self.get_step(step)
        epoch = self.get_epoch(epoch)

        if self.use_wandb:
            wandbdict = {}
            for k, v in dict_of_elems.items():
                if isinstance(v, CustomPlot):
                    v = v.to_wandb()
                    if v is None:
                        continue

                    if isinstance(v, dict):
                        for k2, v2 in v.items():
                            wandbdict[k+"/"+k2] = v2
                    else:
                        wandbdict[k] = v
                elif isinstance(v, plt.Figure):
                    wandbdict[k] = v
                else:
                    assert False, f"Invalid data type {type(v)} for key {k}"

            wandbdict["step"] = step
            wandbdict["epoch"] = epoch
            wandb.log(wandbdict)

        if self.summary_writer is not None:
            for k, v in dict_of_elems.items():
                if isinstance(v, CustomPlot):
                    v.to_tensorboard(k, self.summary_writer, step)
                elif isinstance(v, plt.Figure):
                    self.summary_writer.add_figure(k, v, step)
                else:
                    assert False, f"Unsupported type {type(v)} for entry {k}"

    def __call__(self, *args, **kwargs):
        self.log(*args, **kwargs)

    def flush(self):
        pass

    def finish(self):
        pass
