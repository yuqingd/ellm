#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from text_housekeep.habitat_lab.habitat.core.embodied_task import EmbodiedTask
from text_housekeep.habitat_lab.habitat.core.registry import registry


def _try_register_nav_task():
    try:
        from text_housekeep.habitat_lab.habitat.tasks.nav.nav import NavigationTask  # noqa
    except ImportError as e:
        navtask_import_error = e

        @registry.register_task(name="Nav-v0")
        class NavigationTaskImportError(EmbodiedTask):
            def __init__(self, *args, **kwargs):
                raise navtask_import_error
