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
from detectron2.modeling.roi_heads.fast_rcnn import(
    FastRCNNOutputLayers,
    FastRCNNOutputs
)
from habitat_baselines.vln.models.vilbert import VILBertForVLTasks, BertConfig
from torchvision.ops import nms
from detectron2.structures import Boxes, Instances
from detectron2.modeling.postprocessing import detector_postprocess


# We need the indices of the features to keep
def fast_rcnn_inference_single_image(
        boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image, device
    ):
        scores = scores[:, :-1]
        num_bbox_reg_classes = boxes.shape[1] // 4
        # Convert to Boxes to use the `clip` function ...
        boxes = Boxes(boxes.reshape(-1, 4))
        boxes.clip(image_shape)
        boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

        # Select max scores
        max_scores, max_classes = scores.max(1)       # R x C --> R
        num_objs = boxes.size(0)
        boxes = boxes.view(-1, 4)
        idxs = torch.arange(num_objs).cuda(device) * num_bbox_reg_classes + max_classes
        max_boxes = boxes[idxs]     # Select max boxes according to the max scores.

        # Apply NMS
        keep = nms(max_boxes, max_scores, nms_thresh)
        if topk_per_image >= 0:
            keep = keep[:topk_per_image]
        boxes, scores = max_boxes[keep], max_scores[keep]

        result = Instances(image_shape)
        result.pred_boxes = Boxes(boxes)
        result.scores = scores
        result.pred_classes = max_classes[keep]

        return result, keep

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
                    "BUA_R_101_C4":"VG-Detection/faster_rcnn_R_101_C4_caffe.yaml",
                    "BUA_R_101_C4_MAX":"VG-Detection/faster_rcnn_R_101_C4_caffemaxpool.yaml"
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
        self.detector = DefaultPredictor(detectron2_cfg)
        print("Detectron2 loaded")
        self._max_region_num = 36,
        self._max_seq_length = 128

    def create_detectron2_cfg(self, config):
        cfg = get_cfg()
        checkpoint = self.detectron2_checkpoints[config.DETECTRON2_MODEL]
        cfg.merge_from_file(model_zoo.get_config_file(checkpoint))
        cfg.MODEL.DEVICE = "cuda:" + str(self.detectron2_gpu)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. Download model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint)
        return cfg

    def reset(self):
        pass


    def _get_image_features(self, imgs, score_thresh=0.2, topk_per_image=36):
        # imgs tensor(batch, H, W, C)
        inputs = []
        for img in imgs:
            raw_img = img.permute(2,0,1)
            raw_img = raw_img.to(self.detectron2_gpu_device)
            (_, height, width) = raw_img.shape
            inputs.append({"image": raw_img, "height": height, "width": width})

        # Normalize the image by substracting mean
        # Moves the image to device (already in device)
        images = self.detector.model.preprocess_image(inputs)

        # Features from the backbone
        features = self.detector.model.backbone(images.tensor)

        # Get RPN proposals
        # proposal_generator inputs are the images, features, gt_instances
        # since is detect we don't need the gt instances
        proposals, _ = self.detector.model.proposal_generator(
                            images,
                            features,
                            None
                        )

        # The C4 model uses Res5ROIHeads where pooled feature can be extracted
        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in self.detector.model.roi_heads.in_features]
        box_features = self.detector.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        # Pooled features to use in the agent
        feature_pooled = box_features.mean(dim=[2, 3])

        # Predict classes and boxes for each proposal.
        pred_class_logits, pred_proposal_deltas = \
            self.detector.model.roi_heads.box_predictor(feature_pooled)

        rcnn_outputs = FastRCNNOutputs(
            self.detector.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.detector.model.roi_heads.smooth_l1_beta,
        )

        # Filter proposals using Non-Maximum Suppression (NMS)
        instances_list, ids_list = [], []
        probs_list = rcnn_outputs.predict_probs()
        boxes_list = rcnn_outputs.predict_boxes()
        image_shapes = [x.image_size for x in proposals]

        for probs, boxes, image_size in zip(probs_list, boxes_list, image_shapes):

            # We need to get topk_per_image boxes so we gradually increase
            # the tolerance of the nms_thresh if we don't have enough boxes
            for nms_thresh in np.arange(0.3, 1.0, 0.1):
                instances, ids = fast_rcnn_inference_single_image(
                    boxes,
                    probs,
                    image_size,
                    score_thresh=score_thresh,
                    nms_thresh=nms_thresh,
                    topk_per_image=topk_per_image,
                    device=self.detectron2_gpu_device
                )
                #
                if len(ids) >= topk_per_image:
                    break
            instances_list.append(instances)
            ids_list.append(ids)

        # Post processing for features
        features_list = feature_pooled.split(rcnn_outputs.num_preds_per_image) # (sum_proposals, 2048) --> [(p1, 2048), (p2, 2048), ..., (pn, 2048)]
        roi_features_list = []
        for ids, features in zip(ids_list, features_list):
            roi_features_list.append(features[ids].detach())

        # Post processing for bounding boxes (rescale to raw_image)
        raw_instances_list = []
        for instances, input_per_image, image_size in zip(
                instances_list, inputs, images.image_sizes
            ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            raw_instances = detector_postprocess(instances, height, width)
            raw_instances_list.append(raw_instances)
        print(raw_instances_list[0])
        # features, boxes, image_mask
        return roi_features_list, raw_instances_list, None

    def act(self, observations, episode):

        # Observations come in Caffe GPU
        im = observations["rgb"]
        features, boxes, image_mask = self._get_image_features([im])
        instruction = torch.tensor(episode.instruction.tokens)
        input_mask = torch.tensor(episode.instruction.mask)
        segment_ids = torch.tensor([1 - i for i in input_mask])
        co_attention_mask = torch.zeros((
                                self._max_region_num,
                                self._max_seq_length
                            ))

        #vil_prediction, vil_logit, vil_binary_prediction, vision_prediction, \
        #vision_logit, linguisic_prediction, linguisic_logit = \
        #self.model(
        #    instruction,
        #    features,
        #    spatials,
        #    segment_ids,
        #    input_mask,
        #    image_mask,
        #    co_attention_mask
        #)

        #im_features, boxes = self._get_image_features(im) #.to(self.bert_gpu_device)
        print("features ", len(features), len(features[0]), features[0].shape)

        action = "TURN_LEFT"

        action_args = {}


        return {"action": action, "action_args": action_args}
