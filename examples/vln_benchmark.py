#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import os
import sys
import numpy as np
import random
import time
from pprint import pprint

import torch
import torch.nn as nn
import torch.distributions as D
from torch.autograd import variable
from torch import optim
import torch.nn.functional as F

import argparse
import habitat
import random

from collections import defaultdict
from typing import Dict, Optional
from habitat.core.agent import Agent
from habitat.tasks.vln.vln import ViewpointData
from habitat.core.simulator import (
    AgentState,
)


class VLNRandomBenchmark(habitat.Benchmark):
    def evaluate(
            self, agent: Agent, num_episodes: Optional[int] = None
        ) -> Dict[str, float]:
            r"""..

            :param agent: agent to be evaluated in environment.
            :param num_episodes: count of number of episodes for which the
                evaluation should be run.
            :return: dict containing metrics tracked by environment.
            """

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

            agg_metrics: Dict = defaultdict(float)

            count_episodes = 0
            while count_episodes < num_episodes:
                agent.reset()
                observations = self._env.reset()
                action_history = []
                print("*"*20 + "Starting new episode" + "*"*20,
                    self._env._current_episode.curr_viewpoint.image_id)
                elapsed_steps = 0
                while not self._env.episode_over:
                    action = agent.act(
                        observations,
                        elapsed_steps,
                        self._env._sim.previous_step_collided,
                        )
                    action["action_args"].update(
                        {
                        "episode": self._env._current_episode
                        }
                    )

                    if elapsed_steps == 0 or action["action"] == "TELEPORT":
                        elapsed_steps += 1

                    prev_state = self._env._sim.get_agent_state()
                    prev_image_id = self._env._current_episode.curr_viewpoint.image_id
                    prev_heading = observations["heading"]
                    prev_nav_locations = observations["adjacentViewpoints"]
                    #print("Taking action %s from %s \n" % (action["action"], self._env._current_episode.curr_viewpoint.image_id))
                    observations = self._env.step(action)
                    #print("Result of Action in position %s\n" %  self._env._current_episode.curr_viewpoint.image_id)
                    state = self._env._sim.get_agent_state()
                    image_id = self._env._current_episode.curr_viewpoint.image_id
                    heading = observations["heading"]
                    nav_locations = observations["adjacentViewpoints"]
                    #print("Current position", state.position)
                    #print("Current rotation", state.rotation)
                    #print("\n\n")

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
                        "nav_locations": nav_locations,
                        })

                print("Target path ", [str(goal) for goal in self._env._current_episode.goals])
                pprint(action_history)
                metrics = self._env.get_metrics()
                pprint(metrics)
                for m, v in metrics.items():
                    agg_metrics[m] += v
                count_episodes += 1

            avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

            return avg_metrics

class RandomAgent(habitat.Agent):
    def __init__(self, success_distance, goal_sensor_uuid):
        self.dist_threshold_to_stop = success_distance
        self.goal_sensor_uuid = goal_sensor_uuid

    def reset(self):
        pass

    def is_goal_reached(self, observations):
        dist = observations[self.goal_sensor_uuid][0]
        return dist <= self.dist_threshold_to_stop

    def act(self, observations, elapsed_steps, previous_step_collided):
        action = ""
        action_args = {}

        if elapsed_steps == 0:
            # Turn right (direction choosing)
            action = "TURN_RIGHT"
            num_steps = random.randint(0,11)
            if num_steps > 0:
                action_args= {"num_steps": num_steps}
            else:
                action = "MOVE_FORWARD"
        elif elapsed_steps >= 5:
            # Stop action after 5 tries.
            action = "STOP"
        elif previous_step_collided:
            # Turn right until we can go forward
            action = "TURN_RIGHT"
        else:
            action = "MOVE_FORWARD"
        return {"action": action, "action_args": action_args}

class RandomDiscreteAgent(habitat.Agent):
    def __init__(self, success_distance, goal_sensor_uuid):
        self.dist_threshold_to_stop = success_distance
        self.goal_sensor_uuid = goal_sensor_uuid

    def reset(self):
        pass

    def is_goal_reached(self, observations):
        dist = observations[self.goal_sensor_uuid][0]
        return dist <= self.dist_threshold_to_stop

    def act(self, observations, elapsed_steps, previous_step_collided):
        action = ""
        action_args = {}
        if elapsed_steps == 0:
            # Turn right (direction choosing)
            action = "TURN_RIGHT"
            num_steps = random.randint(0,11)
            if num_steps > 0:
                action_args = {"num_steps": num_steps}

        # After going forward 6 times stop. 0 counts in R2R.
        elif elapsed_steps >= 6:
            # Stop action after 5 tries.
            action = "STOP"
        elif len(observations["adjacentViewpoints"]) > 1:
            # Turn right until we can go forward
            action = "TELEPORT"
            pos = observations["adjacentViewpoints"][1]["start_position"]

            # Keeping the same rotation as the previous step
            rot = observations["adjacentViewpoints"][0]["start_rotation"]
            image_id = observations["adjacentViewpoints"][1]["image_id"]

            viewpoint = ViewpointData(
                image_id=image_id,
                view_point=AgentState(position=pos, rotation=rot)
            )
            action_args.update({"target": viewpoint})

        else:
            action = "TURN_RIGHT"
        return {"action": action, "action_args": action_args}

class ShortestPathAgent(habitat.Agent):
    def __init__(self, success_distance, goal_sensor_uuid):
        self.dist_threshold_to_stop = success_distance
        self.goal_sensor_uuid = goal_sensor_uuid

    def reset(self):
        pass

    def is_goal_reached(self, observations):
        dist = observations[self.goal_sensor_uuid][0]
        return dist <= self.dist_threshold_to_stop


    def _quat_to_xy_heading_vector(self, quat):
        return heading_vector

    def get_relative_heading(self, posA, rotA, posB):
        direction_vector = np.array([0, 0, -1])
        heading_vector = quaternion_rotate_vector(rotA, direction_vector)
        target_vector = np.array(posA) - np.array(posB)

        angle = self._angle_between(
            heading_vector,
            target_vector
        )

        return angle

    def get_relative_elevation(self, posA, posB):
        return 0

    def act(self, observations, goal):
        action = ""
        action_args = {}
        navigable_locations = observations["adjacentViewpoints"]
        posA = navigable_locations[0]["start_position"]
        rotA = navigable_locations[0]["start_rotation"]

        step_size = np.pi/6.0 # default step in R2R
        # Check if the goal is visible
        rel_heading = 0.0 # this is the relative heading
        rel_elevation = 0.0 # this is the relative elevation or altitute

        if rel_heading > step_size:
              action = "TURN_RIGHT" # Turn right
              action_args = {"num_steps": abs(int(rel_heading / step_size))}
        elif rel_heading < -step_size:
              action = "TURN_LEFT" # Turn left
              action_args = {"num_steps": abs(int(rel_heading / step_size))}
        elif rel_elevation > step_size:
              action = "LOOK_UP" # Look up
              action_args = {"num_steps": abs(int(rel_elevation / step_size))}
        elif rel_elevation < -step_size:
              action = "LOOK_DOWN" # Look down
              action_args = {"num_steps": abs(int(rel_elevation / step_size))}
        else:
              action = "MOVE_FORWARD" # Move forward
        return {"action": action, "action_args": action_args}


class seq2seqAgent(habitat.Agent):
    def __init__(self, success_distance, goal_sensor_uuid, encoder, decoder):
        self.dist_threshold_to_stop = success_distance
        self.goal_sensor_uuid = goal_sensor_uuid
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = nn.CrossEntropyLoss()





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-config", type=str, default="configs/tasks/pointnav.yaml"
    )
    parser.add_argument(
        "--num-episodes", type=int, default=100
    )
    args = parser.parse_args()

    agent = RandomDiscreteAgent(3.0, "SPL")
    benchmark = VLNRandomBenchmark(args.task_config)
    metrics = benchmark.evaluate(agent, num_episodes=args.num_episodes)

    for k, v in metrics.items():
        print("{}: {:.3f}".format(k, v))


if __name__ == "__main__":
    main()
