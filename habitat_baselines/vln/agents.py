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
from habitat.tasks.vln.vln import ViewpointData
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.core.simulator import (
    AgentState,
)

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.distributions as D

from torch import optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()


class seq2seqAgent(habitat.Agent):
    def __init__(self, success_distance, goal_sensor_uuid, encoder, decoder, episode_len=20):

        # Constants, may change depending on configuration
        self.model_actions = ['TURN_LEFT', 'TURN_RIGHT', 'LOOK_UP', 'LOOK_DOWN', 'TELEPORT', 'STOP', '<start>', '<ignore>']
        self.feedback_options = ['teacher', 'argmax', 'sample']
        self.dist_threshold_to_stop = success_distance
        self.goal_sensor_uuid = goal_sensor_uuid

        # AI variables
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = nn.CrossEntropyLoss().to('cuda')
        self.losses = []
        self.loss = 0
        self.predicted_actions = []
        self.accuracy = 0
        print(self.encoder)
        print(self.decoder)
        print(self.criterion)
        # Other configurations
        self.previous_action = '<start>'
        self.feedback = 'teacher'
        self.episode_len = episode_len

        # Initializing resnet152 model
        self.image_model = models.resnet152(pretrained=True)
        self.image_model = nn.Sequential(*list(self.image_model.children())[:-1])
        for p in self.image_model.parameters():
            p.requires_grad = False

    def reset(self):
        self.previous_action = '<start>'
        self.ctx = None
        self.h_t = None
        self.c_t = None
        self.a_t = None
        self.seq_mask = None
        print("Agent Reset")



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

        if goal == navigable_locations[0][1]:  # image_id
            action = "STOP"
        else:
            step_size = np.pi/6.0  # default step in R2R
            goal_location = None
            for location in navigable_locations:
                if location[1] == goal:  # image_id
                    goal_location = location
                    break
            # Check if the goal is visible
            if goal_location:

                rel_heading = goal_location[2]  # rel_heading
                rel_elevation = goal_location[3]  #rel_elevation

                if rel_heading > step_size:
                    action = "TURN_RIGHT"
                elif rel_heading < -step_size:
                    action = "TURN_LEFT"
                elif rel_elevation > step_size:
                    action = "LOOK_UP"
                elif rel_elevation < -step_size:
                    action = "LOOK_DOWN"
                else:
                    if goal_location[0]:  # restricted
                        print("WARNING: The target was not in the" +
                              " Field of view, but the step action " +
                              "is going to be performed")
                    action = "TELEPORT"  # Move forward
                    image_id = goal
                    posB = goal_location[4:7]  # start_position
                    rotA = navigable_locations[0][14:18]  # camera_rotation
                    viewpoint = ViewpointData(
                        image_id=image_id,
                        view_point=AgentState(position=posB, rotation=rotA)
                    )
                    action_args.update({"target": viewpoint})
            else:
                # Episode Failure
                action = 'STOP'
                print("Target position %s not visible, " % goal +
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

        if self.previous_action == "<start>":
            # should be a tensor of logits
            seq = torch.LongTensor([episode.instruction.tokens]).to('cuda')
            seq_lengths = torch.LongTensor([episode.instruction.tokens_length]).to('cuda')
            seq_mask = torch.tensor(np.array([False] * episode.instruction.tokens_length))
            self.seq_mask = seq_mask.unsqueeze(0).to('cuda')

            # Forward through encoder, giving initial hidden state and memory cell for decoder
            self.ctx, self.h_t, self.c_t = self.encoder(seq, seq_lengths)
            self.a_t = torch.ones(batch_size).long() * \
                    self.model_actions.index(self.previous_action)
            self.a_t = self.a_t.unsqueeze(0).to('cuda')

        im = observations["rgb"][:,:,[2,1,0]]
        f_t = self._get_image_features(im) #.to('cuda')

        ended = np.array([False] * batch_size) # Indices match permuation of the model, not env

        # Do a sequence rollout and calculate the loss
        self.h_t,self.c_t,alpha,logit = self.decoder(
                                    self.a_t.view(-1, 1),
                                    f_t,
                                    self.h_t,
                                    self.c_t,
                                    self.ctx,
                                    self.seq_mask
                            )
        # Mask outputs where agent can't move forward

        visible_points = sum([1 - ob[0] for ob in observations["adjacentViewpoints"]])

        if visible_points == 0:
            logit[0, self.model_actions.index('TELEPORT')] = -float('inf')

        # Supervised training
        target_action, action_args = self._teacher_actions(observations, goal)
        target = torch.LongTensor(1)
        target[0] = self.model_actions.index(target_action)
        target = target.to('cuda')
        self.loss += self.criterion(logit, target)
        #print(logit)
        # Determine next model inputs
        if self.feedback == 'teacher':
            self.a_t = target                # teacher forcing
            action = target_action
        elif self.feedback == 'argmax':
            _,self.a_t = logit.max(1)        # student forcing - argmax
            self.a_t = self.a_t.detach()
            action = self.model_actions[self.a_t.item()]
            action_args = {}  # What happens if you need to teleport? How to choose?
        elif self.feedback == 'sample':
            probs = F.softmax(logit, dim=1)
            m = D.Categorical(probs)
            self.a_t = m.sample()            # sampling an action from model
            action = self.model_actions[self.a_t.item()]
            action_args = {}
        else:
            sys.exit('Invalid feedback option')

        # Teleport to the next locaiton
        if action == "TELEPORT" and self.feedback != 'teacher':
            for ob in observations["adjacentViewpoints"][1:]:
                if not ob[0]: # restricted
                    next_location = ob
                    action = "TELEPORT"
                    image_id = next_location[1]
                    pos = next_location[4:7]  # agent_position

                    # Keeping the same rotation as the previous step
                    # camera rotation
                    rot = observations["adjacentViewpoints"][0][14:18]
                    print("Teleporting to ",image_id, pos, rot, ob)
                    viewpoint = ViewpointData(
                        image_id=image_id,
                        view_point=AgentState(position=pos, rotation=rot)
                    )
                    action_args = {"target": viewpoint}
                    break
        print(action, target_action, self.loss.item())
        #self.predicted_actions.append(action)
        self.previous_action = action

        return {"action": action, "action_args": action_args}

    def train(self, learning_rate=0.0001, weight_decay=0.0005, feedback='teacher'):
        assert feedback in self.feedback_options
        self.feedback = feedback
        self.encoder.train()
        self.decoder.train()
        self.encoder_optimizer = optim.Adam(
                                self.encoder.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay
                            )
        self.decoder_optimizer = optim.Adam(
                                self.decoder.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay
                            )

    def train_step(self, n_iter):
        if (
            self.encoder_optimizer and
            self.decoder_optimizer and
            n_iter and
            n_iter % 10 == 0
        ):
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            self.loss.backward()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            print("The resulting loss is ", self.loss.item() / self.episode_len / 10)
            writer.add_scalar('Loss/train', self.loss.item() / self.episode_len / 10 , n_iter)
            if self.loss:
                self.losses.append(self.loss.item() / self.episode_len / 10 )
            self.loss = 0
            #writer.add_text('Predicted_Actions', ','.join([str(a) for a in self.predicted_actions]), n_iter)
        #else:
            #print("Please call train first")

    def test(self, use_dropout=False, feedback="argmax"):
        self.feedback = feedback
        if use_dropout:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

    def save(self, encoder_path, decoder_path):
        ''' Snapshot models '''
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

    def load(self, encoder_path, decoder_path):
        ''' Loads parameters (but not training state) '''
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))



class alignmentAgent(habitat.Agent):
    def __init__(self):
        return

    def _get_image_features(self):
        return

    def reset(self):
        pass

    def act(self):
        return {"action": action, "action_args": action_args}
