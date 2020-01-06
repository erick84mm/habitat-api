import json
import os
import sys
import numpy as np
import random
import time
import cv2
from pprint import pprint
from PIL import Image

import torch
import torch.nn as nn
import torch.distributions as D
from torch.autograd import variable
from torch import optim
import torch.nn.functional as F

import habitat
import random

from collections import defaultdict
from typing import Dict, Optional
from habitat.core.agent import Agent
from habitat.config.default import get_config
from habitat.core.env import Env
from habitat.tasks.vln.vln import ViewpointData

from habitat.core.simulator import (
    AgentState,
)
import argparse
from habitat_baselines.vln.models.Seq2Seq import EncoderLSTM, AttnDecoderLSTM
from habitat_baselines.vln.agents import seq2seqAgent


class VLNBenchmark(habitat.Benchmark):

    def __init__(self, config_paths: Optional[str] = None) -> None:
        self.action_history: Dict = defaultdict()
        self.agg_metrics: Dict = defaultdict(float)
        config_env = get_config(config_paths)
        self._env = Env(config=config_env)

    def save_json(self, filename, data):
        if data:
            with open(filename, "w+") as outfile:
                json.dump(data, outfile)
        else:
            print("Error: There are no data to write")

    def save_action_history(self, filename):
        self.save_json(filename, self.action_history)

    def save_evaluation_report(self, filename):
        self.save_json(filename, self.agg_metrics)

    def reset_benchmark(self):
        self.action_history: Dict = defaultdict()
        self.agg_metrics: Dict = defaultdict(float)

    def save_action_summary(self, filename):
        if self.action_history:
            action_summary = {}

            for elem in self.action_history:
                summary = {
                    "gold_path" : elem["gold_path"],
                    "actions" : elem["actions"],
                    "path": elem["path"]
                }

                if not (elem["scan"] in action_summary):
                    action_summary[elem["scan"]] = {}

                action_summary[elem["scan"]].update({
                    elem["path_id"] : summary
                })

            self.save_json(filename, action_summary)


class Seq2SeqBenchmark(VLNBenchmark):
    def train(
            self,
            agent: Agent,
            num_episodes: Optional[int] = None
        ) -> Dict[str, float]:

        self.reset_benchmark()  # Removing action history and such
        print("Training for %s episodes" % str(num_episodes))
        if num_episodes is None:
            num_episodes = len(self._env.episodes)
        else:
            assert num_episodes <= len(self._env.episodes), (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(
                    num_episodes, len(self._env.episodes)
                )
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        count_episodes = 0
        agent.train()
        while count_episodes < num_episodes:
            agent.reset()
            observations = self._env.reset()
            action_history = []
            elapsed_steps = 0
            goal_idx = 1
            last_goal_idx = len(self._env._current_episode.goals) - 1

            while not self._env.episode_over:
                goal_viewpoint = self._env._current_episode.goals[goal_idx]
                action = agent.act(
                    observations,
                    self._env._current_episode,
                    goal_viewpoint
                    )

                action["action_args"].update(
                    {
                    "episode": self._env._current_episode
                    }
                )

                if action["action"] == "TELEPORT":
                    if goal_idx < last_goal_idx:
                        goal_idx += 1
                    else:
                        goal_idx = -1

                prev_state = self._env._sim.get_agent_state()
                prev_image_id = self._env._current_episode.curr_viewpoint.image_id
                prev_heading = observations["heading"]
                prev_nav_locations = observations["adjacentViewpoints"]

                observations = self._env.step(action)

                state = self._env._sim.get_agent_state()
                image_id = self._env._current_episode.curr_viewpoint.image_id
                heading = observations["heading"]
                nav_locations = observations["adjacentViewpoints"]

                action_history.append({
                    "action": action["action"],
                    "prev_image_id": prev_image_id,
                    "prev_heading": prev_heading,
                    "prev_pos": prev_state.position,
                    "prev_rot": prev_state.rotation,
                    "prev_nav_locations": prev_nav_locations,
                    "new_image_id": image_id,
                    "new_heading": heading,
                    "new_pos": state.position,
                    "new_rot": state.rotation,
                    #"nav_locations": nav_locations,
                    })

            pprint(self._env._current_episode.goals)
            pprint(action_history)
            agent.train_step()
            count_episodes += 1

        return {"losses": agent.losses}

    def evaluate(
            self, agent: Agent, num_episodes: Optional[int] = None
        ) -> Dict[str, float]:
            r"""..

            :param agent: agent to be evaluated in environment.
            :param num_episodes: count of number of episodes for which the
                evaluation should be run.
            :return: dict containing metrics tracked by environment.
            """
            self.reset_benchmark()

            if num_episodes is None:
                num_episodes = len(self._env.episodes)
            else:
                assert num_episodes <= len(self._env.episodes), (
                    "num_episodes({}) is larger than number of episodes "
                    "in environment ({})".format(
                        num_episodes, len(self._env.episodes)
                    )
                )

            assert num_episodes > 0, "num_episodes should be greater than 0"

            count_episodes = 0
            while count_episodes < num_episodes:
                agent.reset()
                observations = self._env.reset()
                action_history = []
                elapsed_steps = 0
                goal_idx = 1
                last_goal_idx = len(self._env._current_episode.goals) - 1

                while not self._env.episode_over:
                    goal_viewpoint = self._env._current_episode.goals[goal_idx]
                    action = agent.act(
                        observations,
                        self._env._current_episode,
                        goal_viewpoint
                        )

                    if action["action"] == "TELEPORT":
                        if goal_idx < last_goal_idx:
                            goal_idx += 1
                        else:
                            goal_idx = -1
                    break
                break
                    #observations = self._env.step(action)
            return



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-config", type=str, default="configs/tasks/vln_r2r.yaml"
    )
    parser.add_argument(
        "--num-episodes", type=int, default=100
    )
    parser.add_argument(
        "--agent_type", type=int, default=0
    )
    parser.add_argument(
        "--discrete", action='store_true'
    )
    args = parser.parse_args()

    encoder = EncoderLSTM(1300, 100, 128, 0, 0.1, bidirectional=True, num_layers=3).cuda()
    decoder = AttnDecoderLSTM(8, 6, 32, 256, 0.1).cuda()

    agent = seq2seqAgent(3.0, "SPL", encoder, decoder)
    benchmark = Seq2SeqBenchmark(args.task_config)

    metrics = benchmark.train(agent, num_episodes=args.num_episodes)

    for k, v in metrics.items():
        print("{}: {:.3f}".format(k, v))

if __name__ == "__main__":
    main()
