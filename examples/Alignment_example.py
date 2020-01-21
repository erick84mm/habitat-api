import argparse
import torch
import habitat

from typing import Dict, Optional
from habitat.core.env import Env
from collections import defaultdict
from habitat_baselines.vln.config.default import get_config
from habitat_baselines.vln.agents.alignmentAgent import alignmentAgent


class VLNBenchmark(habitat.Benchmark):

    def __init__(self, config_paths: Optional[str] = None) -> None:
        config_env = get_config()
        self._env = Env(config=config_env.TASK_CONFIG)
        self.losses = []
        self.batch_scores = []

    def train(
        self,
        agent,
        num_episodes: Optional[int] = None,
        feedback="teacher"
    ) -> Dict[str, float]:

        print("Training is running on device ", torch.cuda.current_device())
        agent.train()
        count_episodes = 0
        agg_metrics = defaultdict(float)

        while count_episodes < num_episodes:
            if count_episodes and count_episodes % 1000 == 0:
                agent.save("checkpoints/encoder_train_{}.check".format(count_episodes),
                "checkpoints/decoder_train_{}.check".format(count_episodes))
                print("{} episodes have been processed".format(count_episodes))
            agent.reset()
            observations = self._env.reset()

            while not self._env.episode_over:
                final_goal = self._env._current_episode.goals[-1].image_id
                episode = self._env._current_episode
                shortest_path = self._env._task.get_shortest_path_to_target(
                    episode.scan,
                    episode.curr_viewpoint.image_id,
                    final_goal
                )

                #print("shortest_path", shortest_path)
                if len(shortest_path) > 1:
                    goal_viewpoint = shortest_path[1]
                else:
                    #print("Shortest Path is not good!!!")
                    goal_viewpoint = final_goal

                action, loss, batch_score = agent.act(
                    observations,
                    self._env._current_episode,
                    goal_viewpoint
                )
                self.losses.append(loss)
                self.batch_scores.append(batch_score)

                action["action_args"].update(
                    {
                    "episode": self._env._current_episode
                    }
                )

                observations = self._env.step(action) # Step 1

            self._env._current_episode.reset()
            agent.train_step(count_episodes)

            count_episodes += 1
            metrics = self._env.get_metrics()
            for m, v in metrics.items():
                if m != "distance_to_goal":
                    agg_metrics[m] += v

        agent.reset()
        print(count_episodes)
        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}
        avg_metrics["losses"] = sum(self.losses) / len(self.losses)
        avg_metrics["batch_score"] = sum(self.batch_scores) / len(self.batch_scores)

        return avg_metrics




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
    train_metrics = benchmark.train(agent, num_episodes=100)
    
    for k, v in train_metrics.items():
        print("{0}: {1}".format(k, v))




if __name__ == "__main__":
    main()
