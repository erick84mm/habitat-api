
from typing import Dict, List

import numpy as np
import torch

from habitat import Config, logger
from habitat_baselines.common.base_trainer import BaseTrainer
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class


@baseline_registry.register_trainer(name="vln")
class VLNTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        if config is not None:
            logger.info(f"config: {config}")

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

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

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

        # Copy the information to the wrapper
        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None


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
