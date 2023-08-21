

import os
import sys
import time
import socket
import subprocess
from copy import deepcopy
from datetime import datetime

import torch
import torch.distributed

import framework
from ..data_structures import DotDict
from ..utils import seed, U
from ..visualize import plot
from .argument_parser import ArgumentParser
from .distributed import SLURMEnv
from .saver import Saver


def get_plot_config(args):
    assert args.logger.type in ["all", "tb", "wandb"]
    return args.logger.type in ["all", "tb"], args.logger.type in ["all", "wandb"]


def master(func):
    def wrapper(self, *args, **kwargs):
        if self.dist_env.is_master():
            func(self, *args, **kwargs)
    return wrapper


class TrainingHelper:
    class Dirs:
        pass

    def __init__(self, register_args, extra_dirs=[]):

        self.is_sweep = False
        self.all_dirs = ["checkpoint", "tensorboard"] + extra_dirs
        self.create_parser()
        self.dist_env = SLURMEnv()
        self.dist_env.init_env()

        if register_args is not None:
            register_args(self.arg_parser)
        self.start()

    def print_env_info(self):
        try:
            import pkg_resources
            print("---------------- Environment information: ----------------")
            installed_packages = pkg_resources.working_set
            print(list(sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])))
            print("----------------------------------------------------------")
        except:
            pass

        try:
            git = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], stderr=subprocess.DEVNULL, 
                                stdout=subprocess.PIPE)

            if git.returncode == 0:
                print(f"Git hash: {git.stdout.decode().strip()}")
        except:
            pass

    def create_parser(self):
        get_train_dir = lambda x: os.path.join("exps", x.profile_name) if x.profile_name is not None else None
        self.arg_parser = ArgumentParser(get_train_dir=get_train_dir)

    @master
    def create_dirs(self):
        self.dirs = self.Dirs()
        self.dirs.base = self.summary.save_dir

        for d in self.all_dirs:
            assert d not in self.dirs.__dict__, f"Directory {d} already exists"
            self.dirs.__dict__[d] = os.path.join(self.dirs.base, d)

        for d in self.all_dirs:
            os.makedirs(self.dirs.__dict__[d], exist_ok=True)

    @master
    def save_startup_log(self):
        self.arg_parser.save(os.path.join(self.summary.save_dir, "args.json"))
        with open(os.path.join(self.summary.save_dir, "startup_log.txt"), "a+") as f:
            f.write(f"{str(datetime.now())} {socket.gethostname()}: {' '.join(sys.argv)}\n")

    @master
    def start_tensorboard(self):
        if self.use_tensorboard:
            os.makedirs(self.dirs.tensorboard, exist_ok=True)
            framework.visualize.tensorboard.start(log_dir=self.dirs.tensorboard)

    def use_cuda(self) -> bool:
        return torch.cuda.is_available() and self.args.gpu.lower() != "none"

    def setup_environment(self):
        # TODO: Re-enable if needed. I've disabled it as it was "clashing" with the multi-gpu setup.
        # use_gpu(self.args.gpu)
        if self.args.seed is not None:
            assert not self.dist_env.is_distributed
            seed.fix(self.args.seed)

        self.device = torch.device("cuda") if self.use_cuda() else torch.device("cpu")

    def get_batch_size(self):
        batch_size = self.args.batch_size
        if self.dist_env.is_distributed:
            bs = batch_size // self.dist_env.world_size
            if self.dist_env.rank == 1:
                bs = bs + batch_size % self.dist_env.world_size
            return bs
        else:
            return batch_size

    def get_world_size(self):
        if self.dist_env.is_distributed:
            return self.dist_env.world_size
        else:
            return 1

    def start(self):
        self.args = self.arg_parser.parse_and_try_load()
        self.restore_pending = None

        if self.dist_env.is_master():
            if self.dist_env.is_distributed:
                torch.distributed.broadcast_object_list([self.arg_parser.to_dict()], src=0)
        else:
            a = [None]
            torch.distributed.broadcast_object_list(a, src=0)
            self.args = self.arg_parser.from_dict(a[0])

        self.use_tensorboard, self.use_wandb = get_plot_config(self.args)

        self.state = DotDict()
        self.state.step = 0
        self.state.epoch = 0

        self.run_invariants = {
            "args": self.arg_parser.to_dict()
        }

        self.wandb_project_name = self.args.wandb_project_name
        assert (not self.use_wandb) or (self.wandb_project_name is not None), \
            'Must specify wandb project name if logging to wandb.'

        assert self.args.profile_name is not None or self.use_wandb, "Either name must be specified or W&B should be used"

        wandb_args = {
            "project": self.wandb_project_name,
            "name" : self.args.profile_name,
            "config": self.arg_parser.to_dict(),
            "sync_tensorboard": self.args.logger.sb3.sync_tb,  # auto-upload sb3's tensorboard metrics
            "monitor_gym": self.args.logger.sb3.monitor_gym,  # auto-upload the videos of agents playing the game
            "save_code": self.args.logger.sb3.save_code,  # optional
            "entity": 'word-bots', 
            "resume" : 'must' if self.args.wandb_resume_id is not None else None,
            "id": self.args.wandb_resume_id
        }

        if self.dist_env.is_master():

            if self.args.profile_name is not None:
                timestamp = datetime.now().strftime("_%Y_%m_%d_%H_%M_%S_%f")
                save_dir = os.path.join("exps", self.args.profile_name + timestamp)
            else:
                save_dir = None
            self.summary = plot.Logger(
                save_dir=save_dir,
                use_tb=self.use_tensorboard,
                use_wandb=self.use_wandb,
                wandb_init_args=wandb_args,
                wandb_extra_config={
                    "profile_name": self.args.profile_name,
                    "n_nodes": self.dist_env.world_size or 1,
                    "job_id": self.dist_env.job_id,
                },
                get_global_step = lambda: self.state.step,
                get_global_epoch = lambda: self.state.epoch,
            )

            

            self.create_dirs()
            self.save_startup_log()
            self.start_tensorboard()

        self.saver = Saver(self.dirs.checkpoint if self.dist_env.is_master() else None, \
                            keep_every_n_hours=None if self.use_wandb else 4)
        self.saver["state"] = self.state
        self.saver["run_invariants"] = deepcopy(self.run_invariants)

        if self.dist_env.is_master():
            if self.dist_env.is_distributed:
                torch.distributed.broadcast_object_list([self.wandb_id], src=0)
        else:
            a = [None]
            torch.distributed.broadcast_object_list(a, src=0)
            self.wandb_id = a[0]

        self.setup_environment()

    def start_logging(self):
        self.summary.create_loggers()
        self.run_invariants["wandb_id"] = self.summary.wandb_id
        if self.use_wandb:
            self.print_env_info()
        
        # configs to save for later
        self.save_dir = self.summary.save_dir
        self.wandb_id = self.summary.wandb_id
        self.wandb_id = deepcopy(self.summary.wandb_id)
        self.wandb_id.update({'save_dir': self.summary.save_dir})

    @master
    def wait_for_termination(self):
        if self.args.keep_alive and self.use_tensorboard and not self.use_wandb:
            print("Done. Waiting for termination")
            while True:
                time.sleep(100)

    @master
    def save(self):
        if not self.dist_env.is_master():
            return

        self.saver.save(iter=self.state.step)
        self.saver.cleanup()

    @master
    def tick(self):
        pass
        # self.saver.tick(iter=self.state.step)

    @master
    def finish(self):
        self.summary.finish()
        if self.is_sweep:
            self.save()

        self.wait_for_termination()

    def to_device(self, data):
        return U.apply_to_tensors(data, lambda d: d.to(self.device))


    def restore(self):
        if self.dist_env.is_master():
            if self.restore_pending is not None:
                assert self.saver.load_data(self.restore_pending), "Restoring failed."
                self.restore_pending = None
                restored = True
            else:
                restored = self.saver.load()

            if restored:
                # Do not restore these things
                self.saver.register("run_invariants", deepcopy(self.run_invariants), replace=True)

            # if distributed, send the full state to all workers
            if self.dist_env.is_distributed:
                torch.distributed.broadcast_object_list([self.saver.get_data()], src=0)
        else:
            # if dsitributed and worker, restore state from master
            a = [None]
            torch.distributed.broadcast_object_list(a, src=0)
            self.saver.load_data(a[0])

    @master
    def log(self, logs, step=None):
        self.summary.log(logs, step)
