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
from habitat_baselines.vln import vln_trainer



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
    args = parser.parse_args()
    feedback_options = ["teacher", "argmax", "sample"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    encoder = EncoderLSTM(1300, 256, 256, 0, 0.5, bidirectional=False, num_layers=2).to(device)
    decoder = AttnDecoderLSTM(8, 6, 32, 256, 0.5).to(device)

    agent = seq2seqAgent(3.0, "SPL", encoder, decoder)

    benchmark = Seq2SeqBenchmark(args.task_config)

    if args.train:
        assert 0 <= args.feedback <= 2, "Incorrect feedback option"
        print("Running training with feedback %s" % feedback_options[args.feedback])
        trainer = vln_trainer(args.task_config)
        trainer.train()


        #metrics = benchmark.train(agent, num_episodes=args.num_episodes, feedback=feedback_options[args.feedback])
        #for k, v in metrics.items():
        #    print("{0}: {1}".format(k, v))

    if args.val:

        count_episodes = 5001
        agent.load("checkpoints/encoder_train_{}.check".format(count_episodes),
        "checkpoints/decoder_train_{}.check".format(count_episodes))
        metrics = benchmark.evaluate(agent, num_episodes=args.num_episodes)
        for k, v in metrics.items():
            print("{0}: {1}".format(k, v))

if __name__ == "__main__":
    main()
