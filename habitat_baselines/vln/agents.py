#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
from math import pi

import numpy as np
from PIL import Image

import habitat
from habitat.config.default import get_config
from habitat.sims.habitat_simulator.actions import HabitatSimActions

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.autograd import Variable


class seq2seqAgent(habitat.Agent):
    def __init__(self, success_distance, goal_sensor_uuid, encoder, decoder):

        self.model_actions = ['TURN_LEFT', 'TURN_RIGHT', 'LOOK_UP', 'LOOK_DOWN', 'TELEPORT', 'STOP', '<start>', '<ignore>']
        self.dist_threshold_to_stop = success_distance
        self.goal_sensor_uuid = goal_sensor_uuid
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = nn.CrossEntropyLoss()
        self.losses = []
        self.loss = 0
        self.previous_action = '<start>'

        # Initializing resnet152 model
        self.image_model = models.resnet152(pretrained=True)
        self.image_model=nn.Sequential(*list(self.image_model.children())[:-1])
        for p in self.image_model.parameters():
            p.requires_grad = False

    def reset(self):
        pass

    def is_goal_reached(self, observations):
        dist = observations[self.goal_sensor_uuid][0]
        return dist <= self.dist_threshold_to_stop

    def _get_image_features(self, im):
        input_image = Image.fromarray(im)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.image_model.to('cuda')

        with torch.no_grad():
            output = self.image_model(input_batch)

        return output.data.squeeze().unsqueeze(0)

    def _teacher_actions(self, observations, goal):
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
                # Episode Failure
                action = 'STOP'
                print("Target position %s not visible, " % goal.image_id +
                      "This is an error in the system")
                '''
                for ob in observations["images"]:
                    image = ob
                    image =  image[:,:, [2,1,0]]
                    cv2.imshow("RGB", image)
                    cv2.waitKey(0)
                '''
        return action, action_args

    def act(self, observations, episode, goal):
        # Initialization when the action is start
        batch_size = 1
        # should be a tensor of logits
        seq = torch.LongTensor([episode.instruction.tokens]).cuda()
        seq_lengths = torch.LongTensor([episode.instruction.tokens_length]).cuda()
        seq_mask = torch.tensor(np.array([False] * episode.instruction.tokens_length))
        seq_mask = seq_mask.unsqueeze(0).cuda()

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        ctx,h_t,c_t = self.encoder(seq, seq_lengths)
        im = observations["rgb"][:,:,[2,1,0]]
        f_t = self._get_image_features(im)

        a_t = Variable(torch.ones(batch_size).long() * \
                self.model_actions.index(self.previous_action),
                    requires_grad=False).unsqueeze(0).cuda()
        ended = np.array([False] * batch_size) # Indices match permuation of the model, not env

        print(f_t.shape, a_t.shape, h_t.shape, c_t.shape, ctx.shape)
        # Training cycle until stop action is predicted.

        # Do a sequence rollout and calculate the loss
        self.loss = 0

        h_t,c_t,alpha,logit = self.decoder(a_t.view(-1, 1), f_t, h_t, c_t, ctx, seq_mask)
            # Mask outputs where agent can't move forward

        visible_points = sum([1 for ob in observations["adjacentViewpoints"]
                                if not ob["restricted"]])

        if visible_points == 0:
            logit[0, self.model_actions.index('TELEPORT')] = -float('inf')

        # Supervised training
        target_action, action_args = self._teacher_action(observations, goal)
        target = torch.LongTensor([self.model_actions.index(target_action)])
        target = Variable(target, requires_grad=False).cuda()
        self.loss += self.criterion(logit, target)

        print(self.loss)
'''
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
'''
