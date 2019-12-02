#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os
from typing import List, Optional

from habitat.config import Config
from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.core.simulator import AgentState
from habitat.datasets.utils import VocabDict
from habitat.tasks.vln.vln import VLNEpisode, InstructionData, ViewpointData
from habitat.datasets.vln.r2r_utils import serialize_r2r

DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"


def get_default_r2r_v1_config(split: str = "val"):
    config = Config()
    config.name = "VLNR2R-v1"
    config.DATA_PATH = "data/datasets/vln/r2r/v1/{split}/R2R_{split}.json.gz"
    config.SPLIT = split
    return config


@registry.register_dataset(name="VLNR2R-v1")
class R2RDatasetV1(Dataset):
    r"""Class inherited from Dataset that loads Matterport3D
    Room to Room dataset.

    This class can then be used as follows::
        r2r_config.dataset = get_default_r2r_v1_config()
        r2r = habitat.make_task(r2r_config.task_name, config=r2r_config)
    """

    episodes: List[VLNEpisode]
    train_vocab: VocabDict
    trainval_vocab: VocabDict

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(config.DATA_PATH.format(split=config.SPLIT))

    def __init__(self, config: Config = None) -> None:
        serialize_r2r(config)
        self.episodes = []
        self.train_vocab = []
        self.trainval_vocab = []
        self.connectivity = {}

        if config is None:
            return

        with open(config.CONNECTIVITY_PATH) as f:
            self.connectivity = json.load(f)

        with gzip.open(config.DATA_PATH.format(split=config.SPLIT), "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR)


    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)

        self.train_vocab = VocabDict(
            word_list=deserialized["train_vocab"]["word_list"]
        )
        self.trainval_vocab = VocabDict(
            word_list=deserialized["trainval_vocab"]["word_list"]
        )

        for ep_index, r2r_episode in enumerate(deserialized["episodes"]):
            try:
                episode = VLNEpisode(**r2r_episode)
            except:
                print(ep_index)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]
                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)
            episode.instruction = InstructionData(
                instruction=episode.instruction
            )

            for v_index, viewpoint in enumerate(episode.goals):
                scan = episode.scan
                pos = self.connectivity[scan][viewpoint]["start_position"]
                rot = self.connectivity[scan][viewpoint]["start_rotation"]
                episode.path[v_index] = ViewpointData(
                    image_id=viewpoint,
                    view_point=AgentState(position=pos,rotation=rot)
                    )

            self.episodes.append(episode)
