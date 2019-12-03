#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
from math import pi

import numpy as np

import habitat
from habitat.config.default import get_config
from habitat.sims.habitat_simulator.actions import HabitatSimActions


class RandomAgent(habitat.Agent):
    def __init__(self, success_distance, goal_sensor_uuid):
        self.dist_threshold_to_stop = success_distance
        self.goal_sensor_uuid = goal_sensor_uuid

    def reset(self):
        pass

    def is_goal_reached(self, observations):
        dist = observations[self.goal_sensor_uuid][0]
        return dist <= self.dist_threshold_to_stop

    def act(self, observations):
        if self._elapsed_steps == 0:
            # Turn right (direction choosing)
            action = "TURN_RIGHT"
            num_steps = random.randint(0,11)
            return {"action": action, "action_args": {"num_steps": num_steps}}
        elif self._sim.previous_step_collided:
            # Turn right until we can go forward
            action = "TURN_RIGHT"
        else:
            action = "MOVE_FORWARD"
        return {"action": action}
