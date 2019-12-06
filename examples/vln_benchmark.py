#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
                while not self._env.episode_over:
                    action = agent.act(
                        observations,
                        self._env._elapsed_steps,
                        self._env._sim.previous_step_collided,
                        )
                    action["action_args"].update(
                        {
                        "episode": self._env._current_episode
                        }
                    )
                    observations = self._env.step(action)


                metrics = self._env.get_metrics()
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
        elif elapsed_steps >= 5:
            # Stop action after 5 tries.
            action = "STOP"
        elif len(observations["adjacentViewpoints"]) > 1:
            # Turn right until we can go forward
            action = "TELEPORT"
            pos = observations["adjacentViewpoints"][1]["start_position"]
            rot = observations["adjacentViewpoints"][1]["start_rotation"]
            image_id = observations["adjacentViewpoints"][1]["image_id"]

            viewpoint = ViewpointData(
                image_id=image_id,
                view_point=AgentState(position=pos, rotation=rot)
            )
            action_args.update({"target": viewpoint})

        else:
            action = "TURN_RIGHT"
        return {"action": action, "action_args": action_args}

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
