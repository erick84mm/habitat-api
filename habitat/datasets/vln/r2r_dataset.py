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
from habitat.datasets.vln.r2r_utils import serialize_r2r, load_connectivity
from habitat.tasks.utils import heading_to_rotation

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
    def get_scenes_to_load(config: Config) -> List[str]:
        if not R2RDatasetV1.check_config_paths_exist(config):
            serialize_r2r(config, splits=[config.SPLIT])
        with gzip.open(config.DATA_PATH.format(split=config.SPLIT), "rt") as f:
            data = json.loads(f.read())
            scenes = data["scenes"]
        return scenes

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(config.DATA_PATH.format(split=config.SPLIT))

    def __init__(self, config: Config = None) -> None:

        if config is None:
            return

        serialize_r2r(config, splits=[config.SPLIT])  # R2R to Habitat convertion
        self.episodes: List[VLNEpisode] = []
        self.train_vocab: VocabDict = []
        self.trainval_vocab: VocabDict = []
        self.action_tokens = {}
        self.connectivity = []
        self.scenes: List[str] = []
        self.config = config
        self.mini_alignments = {}

        with gzip.open(config.DATA_PATH.format(split=config.SPLIT), "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        default_rotation = [0,0,0,1]

        self.train_vocab = VocabDict(
            word_list=deserialized["train_vocab"]["word_list"]
        )
        self.trainval_vocab = VocabDict(
            word_list=deserialized["trainval_vocab"]["word_list"]
        )

        self.action_tokens = deserialized["BERT_vocab"]["action_tokens"]
        self.mini_alignments = deserialized["mini_alignments"]

        self.scenes = deserialized["scenes"]

        self.connectivity = load_connectivity(
            self.config.CONNECTIVITY_PATH,
            self.scenes
        )

        for ep_index, r2r_episode in enumerate(deserialized["episodes"]):

            r2r_episode["curr_viewpoint"] = ViewpointData(
                image_id=r2r_episode["goals"][0],
                view_point=AgentState(
                    position=r2r_episode["start_position"] ,
                    rotation=r2r_episode["start_rotation"] )
                )
            instruction_encoding = r2r_episode["instruction_encoding"]
            mask = r2r_episode["mask"]
            del r2r_episode["instruction_encoding"]
            del r2r_episode["mask"]
            episode = VLNEpisode(**r2r_episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX):
                    ]
                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)
            episode.instruction = InstructionData(
                instruction=r2r_episode["instruction"],
                tokens=instruction_encoding,
                tokens_length=sum(mask),
                mask=mask
            )

            scan = episode.scan
            for v_index, viewpoint in enumerate(episode.goals):
                viewpoint_id = self.connectivity[scan]["idxtoid"][viewpoint]
                pos = self.connectivity[scan]["viewpoints"][viewpoint_id]
                rot = default_rotation
                episode.goals[v_index] = ViewpointData(
                    image_id=viewpoint,
                    view_point=AgentState(position=pos, rotation=rot)
                    )
            episode.distance = self.get_distance_to_target(
                scan,
                episode.goals[0].image_id,
                episode.goals[-1].image_id
            )
            self.episodes.append(episode)

    def get_distance_to_target(self, scan, start, end):
        start_vp = self.connectivity[scan]["idxtoid"][start]
        end_vp = self.connectivity[scan]["idxtoid"][end]
        return self.connectivity[scan]["distances"][start_vp][end_vp]

    def get_shortest_path_to_target(self, scan, start, end):
        start_vp = self.connectivity[scan]["idxtoid"][start]
        end_vp = self.connectivity[scan]["idxtoid"][end]
        shortest_path = self.connectivity[scan]["paths"][start_vp][end_vp]
        return [self.connectivity[scan]["idtoidx"][vp] for vp in shortest_path]

    def get_navigable_locations(self, scan, viewpoint):
        observations = []
        default_rotation = [0,0,0,1]
        scan_inf = self.connectivity[scan]
        viewpoint_id = scan_inf["idxtoid"][viewpoint]
        viewpoint_inf = scan_inf["visibility"][viewpoint_id]
        for i in range(len(viewpoint_inf["visible"])):
            if viewpoint_inf["included"] and viewpoint_inf["unobstructed"][i]:
                if i != viewpoint:
                    adjacent_viewpoint_name = scan_inf["idxtoid"][i]
                    adjacent_viewpoint_pos = \
                        scan_inf["viewpoints"][adjacent_viewpoint_name]
                    adjacent_viewpoint = \
                        scan_inf["visibility"][adjacent_viewpoint_name]
                    if adjacent_viewpoint["included"]:
                        observations.append(
                            {
                                "image_id": i,
                                "start_position":
                                    adjacent_viewpoint_pos,
                                "start_rotation":
                                    default_rotation
                            }
                        )
        # In VLN the observations are from left to right but here is backwards.
        # The observations must be sorted by absolute relative heading
        return observations[::1]

    def get_action_tokens(self):
        return self.action_tokens

    def get_mini_alignments_tokens(self):
        return self.mini_alignments
