#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import habitat
import caffe
import cv2
import numpy as np

from habitat_baselines.vln.models.vilbert import VILBertForVLTasks, BertConfig
from fast_rcnn.config import cfg, cfg_from_file


class alignmentAgent(habitat.Agent):

    model_actions = ['TURN_LEFT', 'TURN_RIGHT', 'LOOK_UP', 'LOOK_DOWN', 'TELEPORT', 'STOP', '<start>', '<ignore>']
    base_path = '/home/aa5944/Research/bottom-up-attention/'
    weights = base_path + 'data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel'
    prototxt = base_path + 'models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt'
    caffe_cfg_file = base_path + 'experiments/cfgs/habitat_navigation.yml'

    def __init__(self, config):
        # Load vilBert config
        print("Loading ViLBERT model configuration")
        self.vilbert_config = BertConfig.from_json_file(config.BERT_CONFIG)
        self.pre_trained_model = config.BERT_PRE_TRAINED_MODEL

        print("Loading ViLBERT model")
        self.model = VILBertForVLTasks.from_pretrained(
            self.pre_trained_model,
            self.vilbert_config,
            num_labels=len(self.model_actions) - 2, # number of predicted actions 6
            default_gpu=0
            )

        caffe.set_device(0)
        caffe.set_mode_gpu()
        cfg_from_file(self.caffe_cfg_file)

        print("Loading Caffe model")
        self.image_model = caffe.Net(
            self.prototxt,
            caffe.TEST,
            weights=self.weights
        )

    def _get_image_features(self, im):
        im_file = '/home/aa5944/Research/bottom-up-attention/data/demo/000542.jpg'
        im_orig = im - cfg.PIXEL_MEANS

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in cfg.TEST.SCALES:
            im_scale = float(target_size) / float(im_size_min)

            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
                im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
            img = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(img)

            print(img.shape)


        print("_get_image_features")
        return

    def reset(self):
        pass

    def act(self):
        return {"action": action, "action_args": action_args}
