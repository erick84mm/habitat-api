#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import habitat
import caffe
import cv2
import numpy as np
import torch

from habitat_baselines.vln.models.vilbert import VILBertForVLTasks, BertConfig
from fast_rcnn.config import cfg, cfg_from_file


class alignmentAgent(habitat.Agent):

    model_actions = ['TURN_LEFT', 'TURN_RIGHT', 'LOOK_UP', 'LOOK_DOWN', 'TELEPORT', 'STOP', '<start>', '<ignore>']

    weights = 'data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel'
    prototxt = 'models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt'
    caffe_cfg_file = 'experiments/cfgs/habitat_navigation.yml'

    def __init__(self, config):
        #Load vilBert config
        #print("Loading ViLBERT model configuration")
        #self.vilbert_config = BertConfig.from_json_file(config.BERT_CONFIG)
        #self.pre_trained_model = config.BERT_PRE_TRAINED_MODEL
        #self.bert_gpu = config.BERT_GPU
        self.caffe_gpu = config.CAFFE_GPU
        #self.bert_gpu_device = torch.device(self.bert_gpu)
        self.caffe_gpu_device = torch.device(self.caffe_gpu)

        #print("Loading ViLBERT model on gpu {}".format(self.bert_gpu))
        #self.model = VILBertForVLTasks.from_pretrained(
        #    self.pre_trained_model,
        #    self.vilbert_config,
        #    num_labels=len(self.model_actions) - 2, # number of predicted actions 6
        #    )
        #self.model.to(self.bert_gpu_device)
        #print("ViLBERT loaded on GPU {}".format(self.bert_gpu))

        caffe.set_device(self.caffe_gpu)
        caffe.set_mode_gpu()
        self.base_path = config.CAFFE_BASE_PATH

        cfg_from_file(self.base_path + self.caffe_cfg_file)

        print("Loading Caffe model on gpu {}".format(self.caffe_gpu))
        print("The current device is ", torch.cuda.current_device())
        self.caffe_default_img_shape = config.CAFFE_DEFAULT_IMG_SHAPE
        self.caffe_default_info_shape = config.CAFFE_DEFAULT_INFO_SHAPE
        self.image_model = caffe.Net(
            self.base_path + self.prototxt,
            caffe.TEST,
            weights=self.base_path + self.weights
        )

        ## Modifying the network to be the network
        self.image_model.blobs["data"].reshape(*(self.caffe_default_img_shape))
        self.image_model.blobs["im_info"].reshape(*(self.caffe_default_info_shape))


    def im_list_to_blob(self, ims):
        """Convert a list of images into a network input.

        Assumes images are already prepared (means subtracted, BGR order, ...).
        """
        max_shape = np.array([im.shape for im in ims]).max(axis=0)
        num_images = len(ims)
        blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                        dtype=np.float32)
        for i in range(num_images):
            im = ims[i]
            blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
        # Move channels (axis 3) to axis 1
        # Axis order will become: (batch elem, channel, height, width)
        channel_swap = (0, 3, 1, 2)
        blob = blob.transpose(channel_swap)
        return blob

    def _get_image_features(self, im):

        print("_get_image_features")
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

        blob = self.im_list_to_blob(processed_ims)
        im_scales = np.array(im_scale_factors)
        im_info = np.array([[
            blob.shape[2],
            blob.shape[3],
            im_scales[0]
        ]], dtype=np.float32)

        forward_kwargs = {
            "data": blob.astype(np.float32, copy=False),
            "im_info": im_info.astype(np.float32, copy=False)
        }
        print("_get_image_features forward_kwargs creted")
        print("The current device in im features is ", torch.cuda.current_device())
        output = self.image_model.forward(**forward_kwargs)
        boxes = self.image_model.blobs["rois"].data.copy()
        return output, boxes

    def reset(self):
        pass

    def act(self, observations, episode):
        # Observations come in Caffe GPU
        im = observations["rgb"].numpy()#.to('cpu')
        print(torch.cuda.current_device())
        im_features, boxes = self._get_image_features(im) #.to(self.bert_gpu_device)
        print("features")

        action = "TURN_LEFT"

        action_args = {}


        return {"action": action, "action_args": action_args}
