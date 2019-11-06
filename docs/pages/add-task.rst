How to add a task in Habitat API Demo
#####################################

The tasks are declared under /habitat-api/habitat/tasks/<task name>

The task contains two files:

1. __init__.py file: The definition of the _try_register_<task name>_task method
to try to register the new task, if it cannot it raise an error.

The __init__ file has the following structure

```python
#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry


def _try_register_eqa_task():
    try:
        from habitat.tasks.eqa.eqa import EQATask

        has_eqatask = True
    except ImportError as e:
        has_eqatask = False
        eqatask_import_error = e

    if has_eqatask:
        from habitat.tasks.eqa.eqa import EQATask
    else:
        @registry.register_task(name="EQA-v0")
        class EQATaskImportError(EmbodiedTask):
            def __init__(self, *args, **kwargs):
                raise eqatask_import_error


```

2. Second file is the actual task definition. <task name>.py

This file contains the definition for:
  * Task definition
  * SimulatorTaskAction definitions
  * Measure definitions
  * Sensor definitions
  * Attributes in the task (Task inputs)
  * merge_sim_episode_config method, which merges the configuration.



3. Register the task into the /habitat-api/habitat/registration.py

You will need to add the import and then the
```

from habitat.tasks.<task name> import _try_register_<task name>_task

_try_register_r2r_task()

```
