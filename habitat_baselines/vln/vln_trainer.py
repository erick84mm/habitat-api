import os
import time
from collections import deque
from typing import Dict, List

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger
from habitat_baselines.common.base_trainer import BaseTrainer
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import (
    batch_obs,
    generate_video,
    linear_decay,
)


@baseline_registry.register_trainer(name="vln")
class VLNTrainer(BaseTrainer):
    def __init__(self, config):
        #super().__init__(config)
        self.agent = None
        self.envs = None
        self.config = config
        #if config is not None:
        #    logger.info(f"config: {config}")

    def _collect_rollout_step(
        self, rollouts, current_episode_reward, episode_rewards, episode_counts
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()

        # sample actions
        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }

            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = self.actor_critic.act(
                step_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            )

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()

        outputs = self.envs.step([a[0].item() for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations)
        rewards = torch.tensor(rewards, dtype=torch.float)
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones], dtype=torch.float
        )

        current_episode_reward += rewards
        episode_rewards += (1 - masks) * current_episode_reward
        episode_counts += 1 - masks
        current_episode_reward *= masks

        rollouts.insert(
            batch,
            recurrent_hidden_states,
            actions,
            actions_log_probs,
            values,
            rewards,
            masks,
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs

    def train(self):
        # Get environments for training
        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )

        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        #logger.info(
        #    "agent number of parameters: {}".format(
        #        sum(param.numel() for param in self.agent.parameters())
        #    )
        #)

        # Change for the actual value
        cfg = self.config.RL.PPO

        rollouts = RolloutStorage(
            cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
            cfg.hidden_size,
        )
        rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(observations)
        for sensor in rollouts.observations:
            print(batch[sensor].shape)


        # Copy the information to the wrapper
        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        episode_rewards = torch.zeros(self.envs.num_envs, 1)
        episode_counts = torch.zeros(self.envs.num_envs, 1)
        #current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        #window_episode_reward = deque(maxlen=ppo_cfg.reward_window_size)
        #window_episode_counts = deque(maxlen=ppo_cfg.reward_window_size)

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for update in range(self.config.NUM_UPDATES):
                if cfg.use_linear_lr_decay:
                    lr_scheduler.step()

                if cfg.use_linear_clip_decay:
                    self.agent.clip_param = cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )
                # Main cycle until episode end or num_steps is reached
                for step in range(cfg.num_steps):
                    delta_pth_time, delta_env_time, delta_steps = self._collect_rollout_step(
                        rollouts,
                        current_episode_reward,
                        episode_rewards,
                        episode_counts,
                    )

                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps

                delta_pth_time, value_loss, action_loss, dist_entropy = self._update_agent(
                    ppo_cfg, rollouts
                )
                pth_time += delta_pth_time

                window_episode_reward.append(episode_rewards.clone())
                window_episode_counts.append(episode_counts.clone())

                losses = [value_loss, action_loss]
                stats = zip(
                    ["count", "reward"],
                    [window_episode_counts, window_episode_reward],
                )
                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item()
                        if len(v) > 1
                        else v[0].sum().item()
                    )
                    for k, v in stats
                }
                deltas["count"] = max(deltas["count"], 1.0)

                writer.add_scalar(
                    "reward", deltas["reward"] / deltas["count"], count_steps
                )

                writer.add_scalars(
                    "losses",
                    {k: l for l, k in zip(losses, ["value", "policy"])},
                    count_steps,
                )

                # log stats
                if update > 0 and update % self.config.LOG_INTERVAL == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}\t".format(
                            update, count_steps / (time.time() - t_start)
                        )
                    )

                    logger.info(
                        "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                        "frames: {}".format(
                            update, env_time, pth_time, count_steps
                        )
                    )

                    window_rewards = (
                        window_episode_reward[-1] - window_episode_reward[0]
                    ).sum()
                    window_counts = (
                        window_episode_counts[-1] - window_episode_counts[0]
                    ).sum()

                    if window_counts > 0:
                        logger.info(
                            "Average window size {} reward: {:3f}".format(
                                len(window_episode_reward),
                                (window_rewards / window_counts).item(),
                            )
                        )
                    else:
                        logger.info("No episodes finish in current window")

                # checkpoint model
                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(f"ckpt.{count_checkpoints}.pth")
                    count_checkpoints += 1

            self.envs.close()

    def save_checkpoint(self):
        ''' Snapshot models '''
        encoder_path = os.path.join(
            self.config.CHECKPOINT_FOLDER,
            "encoder.pth"
        )

        decoder_path = os.path.join(
            self.config.CHECKPOINT_FOLDER,
            "decoder.pth"
        )

        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs):
        ''' Loads parameters (but not training state) '''
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))
