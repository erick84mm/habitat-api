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
from habitat.config.default import get_config
from habitat.core.env import Env
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

class VLNBenchmark(habitat.Benchmark):

    def __init__(self, config_paths: Optional[str] = None) -> None:
        self.action_history: Dict = defaultdict()
        self.agg_metrics: Dict = defaultdict(float)
        config_env = get_config(config_paths)
        self._env = Env(config=config_env)

    def save_json(self, filename, data):
        if data:
            with open(filename, "w+") as outfile:
                json.dump(data, outfile)
        else:
            print("Error: There are no data to write")

    def save_action_history(self, filename):
        self.save_json(filename, self.action_history)

    def save_evaluation_report(self, filename):
        self.save_json(filename, self.agg_metrics)

    def reset_benchmark(self):
        self.action_history: Dict = defaultdict()
        self.agg_metrics: Dict = defaultdict(float)

    def save_action_summary(self, filename):
        if self.action_history:
            action_summary = {}

            for elem in self.action_history:
                summary = {
                    "gold_path" : elem["gold_path"],
                    "actions" : elem["actions"],
                    "path": elem["path"]
                }

                if not (elem["scan"] in action_summary):
                    action_summary[elem["scan"]] = {}

                action_summary[elem["scan"]].update({
                    elem["path_id"] : summary
                })

            self.save_json(filename, action_summary)


class RandomBenchmark(VLNBenchmark):
    def evaluate(
            self, agent: Agent, num_episodes: Optional[int] = None
        ) -> Dict[str, float]:
            r"""..

            :param agent: agent to be evaluated in environment.
            :param num_episodes: count of number of episodes for which the
                evaluation should be run.
            :return: dict containing metrics tracked by environment.
            """
            self.reset_benchmark()

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

            count_episodes = 0
            while count_episodes < num_episodes:
                agent.reset()
                observations = self._env.reset()
                action_history = []

                while not self._env.episode_over:
                    action = agent.act(
                        observations,
                        self._env._sim.previous_step_collided,
                        )
                    action["action_args"].update(
                        {
                        "episode": self._env._current_episode
                        }
                    )
                    observations = self._env.step(action)

                metrics = self._env.get_metrics()
                pprint(metrics)
                for m, v in metrics.items():
                    if m != "distance_to_goal":
                        self.agg_metrics[m] += v
                count_episodes += 1
                print(count_episodes)

            avg_metrics = {k: v / count_episodes for k, v in self.agg_metrics.items()}

            return avg_metrics


class DiscreteRandomBenchmark(VLNBenchmark):
    def evaluate(
            self, agent: Agent, num_episodes: Optional[int] = None
        ) -> Dict[str, float]:
            r"""..

            :param agent: agent to be evaluated in environment.
            :param num_episodes: count of number of episodes for which the
                evaluation should be run.
            :return: dict containing metrics tracked by environment.
            """
            self.reset_benchmark()

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

            count_episodes = 0
            while count_episodes < num_episodes:
                agent.reset()
                observations = self._env.reset()
                action_history = []
                #print("*"*20 + "Starting new episode" + "*"*20,
                #    self._env._current_episode.curr_viewpoint.image_id)
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
                        #"nav_locations": nav_locations,
                        })

                #print("Target path ", [str(goal) for goal in self._env._current_episode.goals])

                #pprint(action_history)
                metrics = self._env.get_metrics()
                if np.isinf(metrics["navigationError"]):
                    pprint(action_history)
                    print("Target path ", [str(goal) for goal in self._env._current_episode.goals])
                pprint(metrics)
                for m, v in metrics.items():
                    if m != "distance_to_goal":
                        self.agg_metrics[m] += v
                count_episodes += 1
                print(count_episodes)

            avg_metrics = {k: v / count_episodes for k, v in self.agg_metrics.items()}

            return avg_metrics


class VLNShortestPathBenchmark(VLNBenchmark):
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
                #if images:
                    #print("writing file with images")
                    #for im in images:
                    #    image = Image.fromarray(im[:,:, [2,1,0]])
                    #    gif_images.append(image)
                        #image =  image[:,:, [2,1,0]]
                        #cv2.imshow("RGB", image)
                        #cv2.waitKey(0)
                    #im1 = gif_images[0]
                    #im1.save("out.gif", save_all=True, append_images=gif_images[1:], duration=1000, loop=0)

                gif_images = []
                images = []
                #print("*"*20 + "Starting new episode" + "*"*20,
                #    self._env._current_episode.curr_viewpoint.image_id)
                #if observations and "heading" in observations:
                #    print("Episode heading: %s" % observations["heading"])

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
                if "navigationError" in metrics and  metrics["navigationError"] > 0:
                    print("Scan %s" % self._env._current_episode.scan)
                    print("image_id %s" % self._env._current_episode.goals[0].image_id)

                for m, v in metrics.items():
                    if m != "distance_to_goal":
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

    def act(self, observations, previous_step_collided):
        action = ""
        action_args = {}
        visible_points = sum([1 for ob in observations["adjacentViewpoints"]
                                if not ob["restricted"]])
        prob = random.random()

        # 3% probability of stopping
        if prob <= 0.03:
            action = "STOP"

        # 37% probability to choice a direction
        elif prob <= 0.40:
            action = "TURN_RIGHT"

        elif previous_step_collided:
            # Turn right until we can go forward
            action = "TURN_RIGHT"
        else:
            action = "MOVE_FORWARD"
        return {"action": action, "action_args": action_args}


class DiscreteRandomAgent(habitat.Agent):
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
        visible_points = sum([1 for ob in observations["adjacentViewpoints"]
                                if not ob["restricted"]])

        if elapsed_steps == 0:
            # Turn right (direction choosing)
            action = "TURN_RIGHT"
            num_steps = random.randint(0,11)
            if num_steps > 0:
                action_args = {"num_steps": num_steps}

        # Stop after teleporting 6 times.
        elif elapsed_steps >= 5:
            action = "STOP"

        # Turn right until we can go forward
        elif visible_points > 0:
            for ob in  observations["adjacentViewpoints"]:
                if not ob["restricted"]:
                    goal = ob
                    action = "TELEPORT"
                    image_id = goal["image_id"]
                    pos = goal["start_position"]

                    # Keeping the same rotation as the previous step
                    rot = observations["adjacentViewpoints"][0]["start_rotation"]

                    viewpoint = ViewpointData(
                        image_id=image_id,
                        view_point=AgentState(position=pos, rotation=rot)
                    )
                    action_args.update({"target": viewpoint})
                    break
        else:
            action = "TURN_RIGHT"
        return {"action": action, "action_args": action_args}


class DiscreteShortestPathAgent(habitat.Agent):
    def __init__(self, success_distance, goal_sensor_uuid):
        self.dist_threshold_to_stop = success_distance
        self.goal_sensor_uuid = goal_sensor_uuid

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
            step_size = np.pi/6.0  # default step in R2R
            goal_location = None
            for location in navigable_locations:
                if location["image_id"] == goal.image_id:
                    goal_location = location
                    break
            # Check if the goal is visible
            if goal_location:

                rel_heading = goal_location["rel_heading"]
                rel_elevation = goal_location["rel_elevation"]

                if rel_heading > step_size:
                    action = "TURN_RIGHT"
                elif rel_heading < -step_size:
                    action = "TURN_LEFT"
                elif rel_elevation > step_size:
                    action = "LOOK_UP"
                elif rel_elevation < -step_size:
                    action = "LOOK_DOWN"
                else:
                    if goal_location["restricted"]:
                        print("WARNING: The target was not in the" +
                              " Field of view, but the step action " +
                              "is going to be performed")
                    action = "TELEPORT"  # Move forward
                    image_id = goal.image_id
                    posB = goal_location["start_position"]
                    rotA = navigable_locations[0]["start_rotation"]
                    viewpoint = ViewpointData(
                        image_id=image_id,
                        view_point=AgentState(position=posB, rotation=rotA)
                    )
                    action_args.update({"target": viewpoint})
            else:
                print("Target position %s not visible, " % goal.image_id +
                      "This is an error in the system")
                '''
                for ob in observations["images"]:
                    image = ob
                    image =  image[:,:, [2,1,0]]
                    cv2.imshow("RGB", image)
                    cv2.waitKey(0)
                '''
        return {"action": action, "action_args": action_args}


class seq2seqAgent(habitat.Agent):
    def __init__(self, success_distance, goal_sensor_uuid, encoder, decoder):

        self.model_actions = ['TURN_LEFT', 'TURN_RIGHT', 'LOOK_UP', 'LOOK_DOWN', 'MOVE_FORWARD', 'STOP', '<start>', '<ignore>']
        self.dist_threshold_to_stop = success_distance
        self.goal_sensor_uuid = goal_sensor_uuid
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = nn.CrossEntropyLoss()
        self.losses = []

    def reset(self):
        pass

    def is_goal_reached(self, observations):
        dist = observations[self.goal_sensor_uuid][0]
        return dist <= self.dist_threshold_to_stop

    def _teacher_actions(self):
        return []

    def act(self, observations, action, episode):

        # Initialization when the action is start
        batch_size = 1
        seq = episode.instruction.tokens
        seq_mask = episode.instruction.mask
        seq_length = episode.instruction.tokens_length

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        ctx,h_t,c_t = self.encoder(seq, seq_lengths)

        # Initial action
        a_t = Variable(torch.ones(batch_size).long() * self.model_actions.index('<start>'),
                    requires_grad=False).cuda()
        ended = np.array([False] * batch_size) # Indices match permuation of the model, not env

        # Training cycle until stop action is predicted.

        # Do a sequence rollout and calculate the loss
        self.loss = 0
        f_t = self._feature_variable(perm_obs) # Image features from obs
        h_t,c_t,alpha,logit = self.decoder(a_t.view(-1, 1), f_t, h_t, c_t, ctx, seq_mask)
            # Mask outputs where agent can't move forward
        for i, ob in enumerate(perm_obs):
            if len(ob['navigableLocations']) <= 1:
                logit[i, self.model_actions.index('MOVE_FORWARD')] = -float('inf')

            # Supervised training
            target = self._teacher_action(perm_obs, ended)
            self.loss += self.criterion(logit, target)

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                # teacher forcing
            elif self.feedback == 'argmax':
                _,a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
            elif self.feedback == 'sample':
                probs = F.softmax(logit, dim=1)
                m = D.Categorical(probs)
                a_t = m.sample()            # sampling an action from model
            else:
                sys.exit('Invalid feedback option')

            # Updated 'ended' list and make environment action
            for i,idx in enumerate(perm_idx):
                action_idx = a_t[i].item()
                if action_idx == self.model_actions.index('<end>'):
                    ended[i] = True
                env_action[idx] = self.env_actions[action_idx]

            obs = np.array(self.env.step(env_action))
            perm_obs = obs[perm_idx]

            # Save trajectory output
            for i,ob in enumerate(perm_obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))

            # Early exit if all ended
            if ended.all():
                break

        self.losses.append(self.loss.item() / self.episode_len)
        return traj

        action_args = {}
        return {"action": action, "action_args": action_args}


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
    args = parser.parse_args()

    if args.discrete and args.agent_type == 0:
        print("Running the Discrete Shortest Path Agent")
        agent = DiscreteShortestPathAgent(3.0, "SPL")
        benchmark = VLNShortestPathBenchmark(args.task_config)
    elif args.agent_type == 0:
        print("Running the Continous Shortest Path Agent")
        agent = ShortestPathAgent(3.0, "SPL")
        benchmark = VLNShortestPathBenchmark(args.task_config)
    elif args.discrete and args.agent_type == 1:
        print("Running the Discrete Random Agent")
        agent = DiscreteRandomAgent(3.0, "SPL")
        benchmark = DiscreteRandomBenchmark(args.task_config)
    elif args.agent_type == 1:
        print("Running the Continous Random Agent")
        agent = RandomAgent(3.0, "SPL")
        benchmark = RandomBenchmark(args.task_config)

    metrics = benchmark.evaluate(agent, num_episodes=args.num_episodes)

    for k, v in metrics.items():
        print("{}: {:.3f}".format(k, v))

if __name__ == "__main__":
    main()
