#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import habitat
from models.vilbert import VILBertForVLTasks, BertConfig
import caffe

class alignmentAgent(habitat.Agent):

    model_actions = ['TURN_LEFT', 'TURN_RIGHT', 'LOOK_UP', 'LOOK_DOWN', 'TELEPORT', 'STOP', '<start>', '<ignore>']
    base_path = '/home/aa5944/Research/bottom-up-attention/'
    weights = base_path + 'data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel'
    prototxt = base_path + 'models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt'

    def __init__(self, config):
        # Load vilBert config
        self.vilbert_config = BertConfig.from_json_file(config.BERT_CONFIG)
        self.pre_trained_model = config.BERT_PRE_TRAINED_MODEL
        self.model = VILBertForVLTasks.from_pretrained(
            self.pre_trained_model,
            self.vilbert_config,
            num_labels=len(self.model_actions) - 2, # number of predicted actions 6
            default_gpu=0
            )
        caffe.set_device(0)
        caffe.set_mode_gpu()

        self.image_model = caffe.net(
            self.prototxt,
            caffe.TEST,
            weights=self.weights
        )

    def _get_image_features(self):
        return

    def reset(self):
        pass

    def act(self):
        return {"action": action, "action_args": action_args}
