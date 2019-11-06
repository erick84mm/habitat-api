#!/usr/bin/env python3

# Copyright (c) National Institute of Advanced Industrial Science and Technology （AIST).
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry


def _try_register_r2r_task():
    try:
        from habitat.tasks.r2r.r2r import Room2RoomTask

        has_r2rtask = True
    except ImportError as e:
        has_r2rtask = False
        r2rtask_import_error = e

    if has_r2rtask:
        from habitat.tasks.r2r.r2r import Room2RoomTask
    else:

        @registry.register_task(name="R2R-v0")
        class Room2RoomTaskImportError(EmbodiedTask):
            def __init__(self, *args, **kwargs):
                raise r2rtask_import_error
