# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import csv
import datetime
from collections import defaultdict

import torch
import torchvision
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter

ONLINE_TRAIN_FORMAT = [('step', 'S', 'int'),
                       ('episode', 'E', 'int'), ('episode_length', 'L', 'int'),
                       ('episode_reward', 'R', 'float'),
                       ('buffer_size', 'BS', 'int'), ('fps', 'FPS', 'float'),
                       ('total_time', 'T', 'time')]

ONLINE_EVAL_FORMAT = [('step', 'S', 'int'),
                      ('episode', 'E', 'int'), ('episode_length', 'L', 'int'),
                      ('episode_reward', 'R', 'float'),
                      ('total_time', 'T', 'time')]

OFFLINE_TRAIN_FORMAT = [('step', 'S', 'int'), ('loss', 'L', 'float'),
                        ('lr', 'LR', 'float'), ('fps', 'FPS', 'float'),
                        ('time', 'T', 'time'), ('total_time', 'TT', 'time')]

OFFLINE_EVAL_FORMAT = [('step', 'S', 'int'),
                       ('episode_return[@20]', 'R@20', 'float'),
                       ('episode_length[@20]', 'R@20', 'int'),
                       ('episode_return[@10]', 'R@10', 'float'),
                       ('episode_length[@10]', 'R@10', 'int'),
                       ('episode_return[@5]', 'R@5', 'float'),
                       ('episode_length[@5]', 'R@5', 'int'),
                       ('episode_return[@0]', 'R@0', 'float'),
                       ('episode_length[@0]', 'R@0', 'int'),
                       ('time', 'T', 'time')]


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, csv_file_name, formating):
        self._csv_file_name = csv_file_name
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self._csv_file = None
        self._csv_writer = None

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith('train'):
                key = key[len('train') + 1:]
            else:
                key = key[len('eval') + 1:]
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _remove_old_entries(self, data):
        rows = []
        with self._csv_file_name.open('r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row['step']) >= data['step']:
                    break
                rows.append(row)
        with self._csv_file_name.open('w') as f:
            writer = csv.DictWriter(f,
                                    fieldnames=sorted(data.keys()),
                                    restval=0.0)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _dump_to_csv(self, data):
        if self._csv_writer is None:
            should_write_header = True
            if self._csv_file_name.exists():
                self._remove_old_entries(data)
                should_write_header = False

            self._csv_file = self._csv_file_name.open('a')
            self._csv_writer = csv.DictWriter(self._csv_file,
                                              fieldnames=sorted(data.keys()),
                                              restval=0.0)
            if should_write_header:
                self._csv_writer.writeheader()

        self._csv_writer.writerow(data)
        self._csv_file.flush()

    def _format(self, key, value, ty):
        if ty == 'int':
            value = int(value)
            return f'{key}: {value}'
        elif ty == 'float':
            return f'{key}: {value:.04f}'
        elif ty == 'time':
            value = str(datetime.timedelta(seconds=int(value)))
            return f'{key}: {value}'
        else:
            raise f'invalid format type: {ty}'

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        pieces = [f'| {prefix: <14}']
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print(' | '.join(pieces))

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data['step'] = step
        self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self, log_dir, use_tb, use_wandb, config, resume_id):
        self._log_dir = log_dir
        self._train_mg = MetersGroup(log_dir / 'train.csv',
                                     formating=ONLINE_TRAIN_FORMAT)
        self._eval_mg = MetersGroup(log_dir / 'eval.csv',
                                    formating=ONLINE_EVAL_FORMAT)
        if use_tb:
            self._sw = SummaryWriter(str(log_dir / 'tb'))
        else:
            self._sw = None
        if use_wandb:
            import wandb
            self._wb = wandb
            if resume_id is None:
                resume = False
            else:
                resume = 'must'
            self._wb.init(project='text-crafter_{}'.format(config.exp_name[0]), config=dict(config), name=config.exp_name, entity='word-bots', resume=resume, id=resume_id)
        else:
            self._wb = None

    def _try_log(self, key, value, step):
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)
        if self._wb is not None:
            self._wb.log({key: value}, step=step)


    def _try_log_image(self, key, image, step):
        if self._sw is not None:
            assert image.dim() == 3
            grid = torchvision.utils.make_grid(image.unsqueeze(1))
            self._sw.add_image(key, grid, step)
        if self._wb is not None:
            self._wb.log({key: self._wb.Image(image)}, step=step)

    def log(self, key, value, step):
        # assert key.startswith('train') or key.startswith('eval')
        if type(value) == torch.Tensor:
            value = value.item()
        self._try_log(key, value, step)
        if key.startswith('train'):
            self._train_mg.log(key, value)
        elif key.startswith('eval'):
            self._train_mg.log(key, value)

    def log_image(self, key, image, step, log_frequency=None):
        assert key.startswith('train') or key.startswith('eval')
        self._try_log_image(key, image, step)

    def log_metrics(self, metrics, step, ty):
        for key, value in metrics.items():
            if key.startswith('image'):
                self.log_image(f'{ty}/{key}', value, step)
            else:
                self.log(f'{ty}/{key}', value, step)

    def dump(self, step, ty=None):
        if ty is None or ty == 'eval':
            self._eval_mg.dump(step, 'eval')
        if ty is None or ty == 'train':
            self._train_mg.dump(step, 'train')

    def log_and_dump_ctx(self, step, ty):
        return LogAndDumpCtx(self, step, ty)


class LogAndDumpCtx:
    def __init__(self, logger, step, ty):
        self._logger = logger
        self._step = step
        self._ty = ty

    def __enter__(self):
        return self

    def __call__(self, key, value):
        self._logger.log(f'{self._ty}/{key}', value, self._step)

    def __exit__(self, *args):
        self._logger.dump(self._step, self._ty)
