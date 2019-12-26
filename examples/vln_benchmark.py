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
import cv2
from pprint import pprint
from PIL import Image

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
from habitat.tasks.utils import (
    quaternion_rotate_vector,
    quaternion_from_coeff,
    cartesian_to_polar,
)
from habitat.core.simulator import (
    AgentState,
)
from habitat.utils.geometry_utils import (
    angle_between_quaternions,
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
                #pprint(action_history)
                metrics = self._env.get_metrics()
                pprint(metrics)
                for m, v in metrics.items():
                    agg_metrics[m] += v
                count_episodes += 1
                print(count_episodes)

            avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

            return avg_metrics

class VLNShortestPathBenchmark(habitat.Benchmark):
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
            images = []
            while count_episodes < num_episodes:
                agent.reset()
                observations = self._env.reset()
                action_history = []
                gif_images = []
                if images:
                    print("writing file with images")
                    for im in images:
                        image = Image.fromarray(im[:,:, [2,1,0]])
                        gif_images.append(image)
                        #image =  image[:,:, [2,1,0]]
                        #cv2.imshow("RGB", image)
                        #cv2.waitKey(0)
                    im1 = gif_images[0]
                    im1.save("out.gif", save_all=True, append_images=gif_images[1:], duration=1000, loop=0)

                gif_images = []
                images = []
                print("*"*20 + "Starting new episode" + "*"*20,
                    self._env._current_episode.curr_viewpoint.image_id)
                if observations and "heading" in observations:
                    print("Episode heading: %s" % observations["heading"])

                elapsed_steps = 0
                goal_idx = 1
                last_goal_idx = len(self._env._current_episode.goals) - 1
                images.append(observations["rgb"][:,:,[2,1,0]])
                observations["images"] = images

                print("Target path ", [str(goal) for goal in self._env._current_episode.goals])
                while not self._env.episode_over:
                    goal_viewpoint = self._env._current_episode.goals[goal_idx]

                    action = agent.act(
                        observations,
                        goal_viewpoint,
                        )
                    action["action_args"].update(
                        {
                        "episode": self._env._current_episode
                        }
                    )

                    if action["action"] == "TELEPORT":
                        if goal_idx < last_goal_idx:
                            goal_idx += 1
                        else:
                            goal_idx = -1

                    prev_state = self._env._sim.get_agent_state()

                    prev_image_id = self._env._current_episode.curr_viewpoint.image_id
                    prev_heading = observations["heading"]
                    prev_nav_locations = observations["adjacentViewpoints"]
                    #print("Taking action %s from %s \n" % (action["action"], self._env._current_episode.curr_viewpoint.image_id))
                    observations = self._env.step(action)
                    #pprint(observations["adjacentViewpoints"])
                    images.append(observations["rgb"][:,:,[2,1,0]])
                    observations["images"] = images
                    #print("Result of Action in position %s\n" %  self._env._current_episode.curr_viewpoint.image_id)
                    state = self._env._sim.get_agent_state()
                    image_id = self._env._current_episode.curr_viewpoint.image_id
                    heading = observations["heading"]
                    nav_locations = observations["adjacentViewpoints"]
                    #print("Current position", state.position)
                    #print("Current rotation", state.rotation)


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

                #pprint(action_history)
                metrics = self._env.get_metrics()
                pprint(metrics)
                for m, v in metrics.items():
                    agg_metrics[m] += v
                count_episodes += 1
                print(count_episodes)

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
        elif elapsed_steps >= 5:
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
    def __init__(self, success_distance, goal_sensor_uuid, half_visible_angle=0):
        self.dist_threshold_to_stop = success_distance
        self.goal_sensor_uuid = goal_sensor_uuid
        self.half_visible_angle = half_visible_angle

    def reset(self):
        pass

    def is_goal_reached(self, observations):
        dist = observations[self.goal_sensor_uuid][0]
        return dist <= self.dist_threshold_to_stop

    def act(self, observations, goal):
        action = ""
        action_args = {}
        navigable_locations = observations["adjacentViewpoints"]
        if goal.image_id == navigable_locations[0]["image_id"]:
            action = "STOP"
        else:
            print("The goal is: %s", goal.image_id)
            step_size = np.pi/6.0
            goal_location = None
            for location in navigable_locations:
                if location["image_id"] == goal.image_id:
                    goal_location = location
                    break

            if goal_location:
                # default step in R2R
                # Check if the goal is visible
                rel_heading = goal_location["rel_heading"]
                # this is the relative elevation or altitute
                rel_elevation = goal_location["rel_elevation"]

                print("The relative heading to the goal is %s" % str(rel_heading))
                print("The relative elevation to the goal is %s" % str(rel_elevation))
                print("The heading is", observations["heading"])
                print("The elevation is", observations["elevation"])

                if rel_heading > step_size:
                    action = "TURN_RIGHT" # Turn right
                      #action_args = {"num_steps": abs(int(rel_heading / step_size))}
                elif rel_heading < -step_size:
                    action = "TURN_LEFT" # Turn left
                      #action_args = {"num_steps": abs(int(rel_heading / step_size))}
                elif rel_elevation > step_size:
                    action = "LOOK_UP" # Look up
                      #action_args = {"num_steps": abs(int(rel_elevation / step_size))}
                elif rel_elevation < -step_size:
                    action = "LOOK_DOWN" # Look down
                      #action_args = {"num_steps": abs(int(rel_elevation / step_size))}
                else:
                    if goal_location["Restricted"]:
                        print("WARNING: The target was not in the Field of view," +
                        " but the step action is going to be performed")
                    action = "TELEPORT" # Move forward
                    image_id = goal.image_id
                    posB = goal_location["start_position"]
                    rotA = navigable_locations[0]["start_rotation"]
                    viewpoint = ViewpointData(
                        image_id=image_id,
                        view_point=AgentState(position=posB, rotation=rotA)
                    )
                    action_args.update({"target": viewpoint})
            else:
                print("Target position %s not visible, This is an error in the system" % goal.image_id)
                print("The relative heading is %s\n" % str(rel_heading))
                print("The relative elevation is %s\n" % str(rel_elevation))
                pprint(navigable_locations)
                for ob in observations["images"]:
                    image = ob
                    image =  image[:,:, [2,1,0]]
                    cv2.imshow("RGB", image)
                    cv2.waitKey(0)

            print(action, action_args)
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

    #agent = RandomDiscreteAgent(3.0, "SPL")
    #benchmark = VLNRandomBenchmark(args.task_config)
    #metrics = benchmark.evaluate(agent, num_episodes=args.num_episodes)


    agent = ShortestPathAgent(3.0, "SPL", half_visible_angle=0.767945)
    benchmark = VLNShortestPathBenchmark(args.task_config)
    metrics = benchmark.evaluate(agent, num_episodes=args.num_episodes)



    for k, v in metrics.items():
        print("{}: {:.3f}".format(k, v))


if __name__ == "__main__":
    main()
