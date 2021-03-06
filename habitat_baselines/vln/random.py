#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import habitat

from collections import defaultdict
from typing import Dict, Optional
from habitat.core.agent import Agent

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
                print(self._env.current_episode.episode_id)

                while not self._env.episode_over:
                    action = agent.act(
                        observations,
                        self._env._elapsed_steps,
                        self._env._sim.previous_step_collided()
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
        if elapsed_steps == 0:
            # Turn right (direction choosing)
            action = "TURN_RIGHT"
            num_steps = random.randint(0,11)
            if num_steps > 0:
                return {
                    "action": action,
                    "action_args": {"num_steps": num_steps}
                    }
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
        return {"action": action}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-config", type=str, default="configs/tasks/pointnav.yaml"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=100
    )
    args = parser.parse_args()

    agent = RandomAgent(3.0, "SPL")
    benchmark = VLNRandomBenchmark(args.task_config)
    metrics = benchmark.evaluate(agent, num_episodes=1000)

    for k, v in metrics.items():
        print("{}: {:.3f}".format(k, v))


if __name__ == "__main__":
    main()
