#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import habitat
import cv2
import numpy as np
import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from habitat_baselines.vln.models.vilbert import VILBertForVLTasks, BertConfig


class alignmentAgent(habitat.Agent):

    model_actions = ['TURN_LEFT', 'TURN_RIGHT', 'LOOK_UP', 'LOOK_DOWN', 'TELEPORT', 'STOP', '<start>', '<ignore>']

    detectron2_checkpoints = {
                    "CD_R_50_C4_1x": "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml",
                    "CD_R_50_DC5_1x": "COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml",
                    "CD_R_50_FPN_1x": "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml",
                    "CD_R_50_C4_3x": "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml",
                    "CD_R_50_DC5_3x": "COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml",
                    "CD_R_50_FPN_3x": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
                    "CD_R_101_C4_3x": "COCO-Detection/faster_rcnn_R_101_C4_3x.yaml",
                    "CD_R_101_DC5_3x": "COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml",
                    "CD_R_101_FPN_3x": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
                    "CD_X_101_FPN_3x": "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
                    "CD_R_50_1x": "COCO-Detection/retinanet_R_50_FPN_1x.yaml",
                    "CD_R_50_3x": "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
                    "CD_R_101_3x": "COCO-Detection/retinanet_R_101_FPN_3x.yaml",
                    "CD_RPN_R_50_C4_1x": "COCO-Detection/rpn_R_50_C4_1x.yaml",
                    "CD_RPN_R_50_FPN_1x": "COCO-Detection/rpn_R_50_FPN_1x.yaml",
                    "CD_FAST_RCNN_R_50_FPN_1x": "COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml",
                    "IS_R_50_C4_1x": "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml",
                    "IS_R_50_DC5_1x": "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml",
                    "IS_R_50_FPN_1x": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
                    "IS_R_50_C4_3x": "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml",
                    "IS_R_50_DC5_3x": "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml",
                    "IS_R_50_FPN_3x": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                    "IS_R_101_C4_3x": "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml",
                    "IS_R_101_DC5_3x": "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml",
                    "IS_R_101_FPN_3x": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
                    "IS_X_101_FPN_3x": "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
    }

    def __init__(self, config):
        #Load vilBert config
        print("Loading ViLBERT model configuration")
        self.vilbert_config = BertConfig.from_json_file(config.BERT_CONFIG)
        self.pre_trained_model = config.BERT_PRE_TRAINED_MODEL
        self.bert_gpu = config.BERT_GPU
        self.detectron2_gpu = config.DETECTRON2_GPU
        self.bert_gpu_device = torch.device(self.bert_gpu)
        self.detectron2_gpu_device = torch.device(self.detectron2_gpu)

        print("Loading ViLBERT model on gpu {}".format(self.bert_gpu))
        self.model = VILBertForVLTasks.from_pretrained(
            self.pre_trained_model,
            self.vilbert_config,
            num_labels=len(self.model_actions) - 2, # number of predicted actions 6
            )
        self.model.to(self.bert_gpu_device)
        print("ViLBERT loaded on GPU {}".format(self.bert_gpu))

        print("Loading Detectron2 predictor on GPU {}".format(self.detectron2_gpu))
        detectron2_cfg = self.create_detectron2_cfg(config)
        self.image_predictor = DefaultPredictor(detectron2_cfg)
        print("Detectron2 loaded")

    def create_detectron2_cfg(self, config):
        cfg = get_cfg()
        checkpoint = detectron2_checkpoints[config.DETECTRON2_MODEL]
        cfg.merge_from_file(model_zoo.get_config_file(checkpoint))
        cfg.MODEL.DEVICE = "cuda:"+self.detectron2_gpu
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. Download model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint)
        return cfg


    def reset(self):
        pass

    def act(self, observations, episode):
        # Observations come in Caffe GPU
        im = observations["rgb"].to(self.detectron2_gpu_device)
        outputs = self.image_predictor(im)
        print(torch.cuda.current_device())
        #im_features, boxes = self._get_image_features(im) #.to(self.bert_gpu_device)
        print("features")

        action = "TURN_LEFT"

        action_args = {}


        return {"action": action, "action_args": action_args}
