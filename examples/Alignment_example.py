import argparse
import torch
import habitat

from typing import Dict, Optional
from habitat.core.env import Env
from habitat_baselines.vln.config.default import get_config
from habitat_baselines.vln.agents.alignmentAgent import alignmentAgent


class VLNBenchmark(habitat.Benchmark):

    def __init__(self, config_paths: Optional[str] = None) -> None:
        config_env = get_config()
        self._env = Env(config=config_env.TASK_CONFIG)

    def train(
        self,
        agent,
        num_episodes: Optional[int] = None,
        feedback="teacher"
    ) -> Dict[str, float]:

        observations = self._env.reset()
        action = agent.act(
            observations,
            self._env._current_episode
        )
        observations = self._env.reset()
        action = agent.act(
            observations,
            self._env._current_episode
        )
        observations = self._env.reset()
        action = agent.act(
            observations,
            self._env._current_episode
        )
        observations = self._env.reset()
        action = agent.act(
            observations,
            self._env._current_episode
        )
        observations = self._env.reset()
        action = agent.act(
            observations,
            self._env._current_episode
        )
        observations = self._env.reset()
        action = agent.act(
            observations,
            self._env._current_episode
        )
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
    parser.add_argument(
            "--train", action='store_true'
    )
    parser.add_argument(
            "--val", action='store_true'
    )
    parser.add_argument(
        "--feedback", type=int, default=0
    )
    parser.add_argument(
        "--checkpoint_num", type=int, default=0
    )
    args = parser.parse_args()


    experiment_config = get_config()
    task_config = experiment_config.TASK_CONFIG
    agent = alignmentAgent(experiment_config)
    benchmark = VLNBenchmark()
    benchmark.train(agent)





if __name__ == "__main__":
    main()
