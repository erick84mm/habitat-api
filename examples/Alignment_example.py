import argparse
import torch
import habitat

from typing import Dict, Optional
from habitat.core.env import Env
from collections import defaultdict
from habitat_baselines.vln.config.default import get_config
from habitat_baselines.vln.agents.alignmentAgent import alignmentAgent
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


class VLNBenchmark(habitat.Benchmark):

    def __init__(self, name, config_paths: Optional[str] = None) -> None:
        config_env = get_config()
        self._env = Env(config=config_env.TASK_CONFIG)
        self.losses = []
        self.batch_scores = []
        self.episode_losses = []
        self.episode_batch_scores = []
        self._name = name

    def evaluate(
        self,
        agent
    ):
        return

    def train_batch(
        self,
        agent,
        num_episodes: Optional[int] = None,
        feedback="teacher",
        checkpoint_iter = 1000,
        batch_size = 4
    ):
        print("Training is running on device ", torch.cuda.current_device())
        agent.train()
        count_episodes = 0
        agg_metrics = defaultdict(float)
        steps = 0
        action_padding_idx = agent.mode_actions.index("<ignore>")
        rollout_observations = []
        while count_episodes < num_episodes:
            if count_episodes and count_episodes % checkpoint_iter == 0:
                agent.save("checkpoints/{}_train_{}.check".format(
                                                        self._name,
                                                        count_episodes
                                                    )
                )
                print("{} episodes have been processed".format(count_episodes))
            agent.reset(steps)
            observations = self._env.reset()
            observations = {
                            "rgb": observations["rgb"],
                            "adjacentViewpoints": observations["adjacentViewpoints"]
                            }
            episode_loss = []
            episode_batch_score = []
            action_sequence = [agent.model_actions.index("<start>")]
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


                # Adding observations to rollout
                target_action, action_args = \
                    agent._teacher_actions(observations, goal_viewpoint)
                action_idx = agent.model_actions.index(target_action)
                action_sequence.append(action_idx)
                observations["golden_action"] = action_idx
                action = {"action": target_action, "action_args": action_args}

                action["action_args"].update(
                    {
                    "episode": self._env._current_episode
                    }
                )
                # Adding tokens from episode
                action_tokens = action_sequence[-10:]
                padding = [action_padding_idx] * (10 - len(action_tokens))
                action_mask = [1] * len(action_tokens) + [0] * len(padding)
                action_tokens = action_tokens + padding
                action_segment_ids = [1] * len(action_tokens)

                # add padding at the end
                observations["actions"] = action_tokens
                observations["tokens"] = \
                    self._env._current_episode.instruction.tokens
                observations["mask"] = \
                    self._env._current_episode.instruction.mask + \
                    action_mask
                observations["segment"] = \
                    [0] * len(self._env._current_episode.instruction.tokens)+ \
                    action_segment_ids

                rollout_observations.append(observations)
                observations = self._env.step(action) # Step 1
                observations = {
                                "rgb": observations["rgb"],
                                "adjacentViewpoints": observations["adjacentViewpoints"]
                                }
                steps += 1
                if len(rollout_observations) == batch_size:

                    ## Act with batch
                    loss, batch_score = agent.act_batch(
                        rollout_observations
                    )
                    episode_loss.append(loss)
                    episode_batch_score.append(batch_score)
                    self.losses.append(loss)
                    self.batch_scores.append(batch_score)

                    agent.train_step(steps)
                    rollout_observations = []
                    self.episode_losses.append(sum(episode_loss) / len(episode_loss))
                    self.episode_batch_scores.append(sum(episode_batch_score) / len(episode_batch_score))
                    print("Episode loss", self.episode_losses[-1])
                    print("Episode Batch Score", self.episode_batch_scores[-1])
                    writer.add_scalar('episode_Loss/train', self.episode_losses[-1], count_episodes)
                    writer.add_scalar('episode_batch_scores/train', self.episode_batch_scores[-1], count_episodes)

            self._env._current_episode.reset()

            count_episodes += 1


            metrics = self._env.get_metrics()
            for m, v in metrics.items():
                if m != "distance_to_goal":
                    agg_metrics[m] += v

        agent.reset(steps)
        print(count_episodes)
        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}
        avg_metrics["losses"] = sum(self.losses) / len(self.losses)
        avg_metrics["batch_score"] = sum(self.batch_scores) / len(self.batch_scores)

        return avg_metrics




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
        steps = 0

        while count_episodes < num_episodes:
            if count_episodes and count_episodes % 1000 == 0:
                agent.save("checkpoints/{}_train_{}.check".format(
                                                        self._name,
                                                        count_episodes
                                                    )
                )
                print("{} episodes have been processed".format(count_episodes))
            agent.reset(steps)
            observations = self._env.reset()
            episode_loss = []
            episode_batch_score = []
            #action_sequence = []
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
                episode_loss.append(loss)
                episode_batch_score.append(batch_score)
                self.losses.append(loss)
                self.batch_scores.append(batch_score)

                action["action_args"].update(
                    {
                    "episode": self._env._current_episode
                    }
                )

                observations = self._env.step(action) # Step 1
                steps += 1
                agent.train_step(steps)

            self._env._current_episode.reset()

            count_episodes += 1

            self.episode_losses.append(sum(episode_loss) / len(episode_loss))
            self.episode_batch_scores.append(sum(episode_batch_score) / len(episode_batch_score))
            print("Episode loss", self.episode_losses[-1])
            print("Episode Batch Score", self.episode_batch_scores[-1])
            writer.add_scalar('episode_Loss/train', self.episode_losses[-1], count_episodes)
            writer.add_scalar('episode_batch_scores/train', self.episode_batch_scores[-1], count_episodes)
            metrics = self._env.get_metrics()
            for m, v in metrics.items():
                if m != "distance_to_goal":
                    agg_metrics[m] += v

        agent.reset(steps)
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
        "--agent-type", type=int, default=0
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
        "--checkpoint-num", type=int, default=0
    )

    parser.add_argument(
        "--experiment-name", type=str, default="exp"
    )

    parser.add_argument(
        "--batch-size", type=int, default=1
    )
    args = parser.parse_args()


    experiment_config = get_config()
    task_config = experiment_config.TASK_CONFIG
    agent = alignmentAgent(experiment_config, num_train_optimization_steps=10*args.num_episodes / args.batch_size )
    benchmark = VLNBenchmark(args.experiment_name)
    train_metrics = benchmark.train_batch(agent, num_episodes=args.num_episodes, batch_size=args.batch_size)

    for k, v in train_metrics.items():
        print("{0}: {1}".format(k, v))




if __name__ == "__main__":
    main()
