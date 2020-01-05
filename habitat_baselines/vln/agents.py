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


class seq2seqAgent(habitat.Agent):
    def __init__(self, success_distance, goal_sensor_uuid, encoder, decoder):

        self.model_actions = ['TURN_LEFT', 'TURN_RIGHT', 'LOOK_UP', 'LOOK_DOWN', 'MOVE_FORWARD', 'STOP', '<start>', '<ignore>']
        self.dist_threshold_to_stop = success_distance
        self.goal_sensor_uuid = goal_sensor_uuid
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = nn.CrossEntropyLoss()
        self.losses = []

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
        input_image = Image.Image.fromarray(im)
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


        with torch.no_grad():
            output = self.image_model(input_batch)
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        print(output.data)
        return output.data

    def _teacher_actions(self):
        return []

    def act(self, observations, episode):
        # Initialization when the action is start
        batch_size = 1
        # should be a tensor of logits
        seq = torch.LongTensor([episode.instruction.tokens]).cuda()
        seq_lengths = torch.LongTensor([episode.instruction.tokens_length]).cuda()
        #seq_mask = torch.LongTensor([episode.instruction.mask])

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        ctx,h_t,c_t = self.encoder(seq, seq_lengths)
        im = observations["rgb"][:,:,[2,1,0]]
        im_features = self._get_image_features(self, im)

'''
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
'''
