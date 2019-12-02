#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.dataset import Dataset
from habitat.core.registry import registry

def _try_register_r2r_dataset():
    try:
        from habitat.datasets.vln.r2r_dataset import (
            R2RDatasetV1,
        )

        has_r2r = True
    except ImportError as e:
        has_r2r = False
        r2r_import_error = e

    if has_r2r:
        from habitat.datasets.vln.r2r_dataset import (
            R2RDatasetV1,
        )
    else:

        @registry.register_dataset(name="VLNR2R-v1")
        class R2RDatasetImportError(Dataset):
            def __init__(self, *args, **kwargs):
                raise r2r_import_error
