#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# from cos_eor.task.sensors import *
# from cos_eor.task.play_sensors import *
# from cos_eor.task.measures import *
# from cos_eor.task.play_measures import *
from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry


def _try_register_cos_eor_task():
    from text_housekeep.habitat_lab.cos_eor.task.task import CosRearrangementTask  # noqa: F401
    from text_housekeep.habitat_lab.cos_eor.task.play_task import CosRearrangementTaskPlay  # noqa: F401
