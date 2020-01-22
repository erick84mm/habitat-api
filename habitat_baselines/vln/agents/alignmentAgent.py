#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import habitat
import cv2
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import detectron2
from detectron2.utils.logger import setup_logger
from habitat.tasks.vln.vln import ViewpointData
from habitat.core.simulator import (
    AgentState,
)
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
from habitat_baselines.vln.models.optimization import Adam


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
    categorical_model_actions = ["TURN", "ELEVATION", "TELEPORT", "STOP"]
    reduce_model_actions = ["SEARCH", "TELEPORT", "STOP"]

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


    def __init__(self, config, num_train_optimization_steps=3100):
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
        #self.detector.eval()
        print("Detectron2 loaded")
        self._max_region_num = 36
        self._max_seq_length = 128
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.loss = 0
        self.learning_rate = 1e-4
        self.vision_scratch = False
        self.max_steps = 30
        self.grad_accumulation = 10
        optimizer_grouped_parameters = []
        lr = 1e-4
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

        for key, value in dict(self.model.named_parameters()).items():
            if value.requires_grad:
                if 'vil_prediction' in key:
                    lr = 1e-4
                if any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [
                        {"params": [value], "lr": lr, "weight_decay": 0.01}
                    ]
                if not any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [
                        {"params": [value], "lr": lr, "weight_decay": 0.0}
                    ]

        self.optimizer = Adam(
                            optimizer_grouped_parameters,
                            lr=self.learning_rate,
                            warmup=0.1,
                            t_total=num_train_optimization_steps,
                            schedule='warmup_constant'
                        )

        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, \
                        mode='max',
                        factor=0.2,
                        patience=10,
                        cooldown=4,
                        threshold=0.001)

    def create_detectron2_cfg(self, config):
        cfg = get_cfg()
        checkpoint = self.detectron2_checkpoints[config.DETECTRON2_MODEL]
        cfg.merge_from_file(model_zoo.get_config_file(checkpoint))
        cfg.MODEL.DEVICE = "cuda:" + str(self.detectron2_gpu)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. Download model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint)
        return cfg

    def reset(self, steps):
        self.loss = None
        pass


    def train_step(self, steps):
        self.loss = self.loss / self.grad_accumulation
        self.loss.backward()
        self.loss = None

        if steps and steps % self.grad_accumulation == 0:
            self.optimizer.step()
            self.model.zero_grad()

    def _get_image_features(self, imgs, score_thresh=0.2, min_num_image=10, max_regions=36):
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
        num_boxes = []
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
                    topk_per_image=max_regions,
                    device=self.detectron2_gpu_device
                )
                #
                if len(ids) >= min_num_image:
                    break
            num_boxes.append(len(ids)+1)
            instances_list.append(instances)
            ids_list.append(ids)

        # Post processing for features
        features_list = feature_pooled.split(rcnn_outputs.num_preds_per_image) # (sum_proposals, 2048) --> [(p1, 2048), (p2, 2048), ..., (pn, 2048)]
        roi_features_list = []
        for ids, features in zip(ids_list, features_list):
            head_box = torch.sum(features[ids], axis=0) / \
                       len(features[ids])
            head_box = head_box.unsqueeze(0)
            roi_features_list.append(torch.cat((head_box, features[ids]), 0))

        # Post processing for bounding boxes (rescale to raw_image)
        boxes = []
        for instances, input_per_image, image_size in zip(
                instances_list, inputs, images.image_sizes
            ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            raw_instances = detector_postprocess(instances, height, width)

            box = torch.zeros(
                    (len(raw_instances)+1, 5),
                    device=self.detectron2_gpu_device
                 )
            box[0] = torch.tensor(
                        [[0,0,1,1,1]],
                        device=self.detectron2_gpu_device
                       ).float()
            box[1:,:4] = raw_instances.pred_boxes.tensor
            box[:,0] /= float(width)
            box[:,1] /= float(width)
            box[:,2] /= float(width)
            box[:,3] /= float(width)
            box[:,4] = (box[:,3] - box[:,1]) * (box[:,2] - box[:,0]) / \
                (float(height) * float(width))

            boxes.append(box)
        # features, boxes, image_mask
        return roi_features_list, boxes, num_boxes

    def train(self):
        self.model.train()

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
                    if goal_location[0] == 1:  # restricted
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

    def _get_target_onehot(self, observations, goals):
        target_action, args = self._teacher_actions(observations, goals)
        idx = self.model_actions.index(target_action)
        one_hot = torch.zeros((1,6), device=self.bert_gpu_device)
        category_one_hot = torch.zeros((1,3), device=self.bert_gpu_device)
        one_hot[0][idx] = 1
        if idx < 4:
            category_one_hot[0][0] = 1
        else:
            category_one_hot[0][idx-3] = 1
        return category_one_hot, one_hot, target_action, args



    def act(self, observations, episode, goals):

        # Observations come in Caffe GPU
        batch_size = 1
        category_target, target, target_action, action_args = self._get_target_onehot(
                                                            observations,
                                                            goals
                                            )
        im = observations["rgb"]
        features, boxes, num_boxes = self._get_image_features([im])
        max_regions = self._max_region_num + 1

        # The following have to be done outside in the rollouts
        instruction = torch.tensor(episode.instruction.tokens).to(self.bert_gpu_device)
        input_mask = torch.tensor(episode.instruction.mask).to(self.bert_gpu_device)
        segment_ids = torch.tensor([1 - i for i in input_mask]).to(self.bert_gpu_device)
        co_attention_mask = torch.zeros((
                                max_regions,
                                self._max_seq_length
                            ), dtype=torch.float).to(self.bert_gpu_device)

        mix_num_boxes = min(int(num_boxes[0]), max_regions)
        mix_boxes_pad = torch.zeros((max_regions, 5))
        mix_features_pad = torch.zeros((max_regions, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < max_regions:
            image_mask.append(0)

        mix_boxes_pad[:mix_num_boxes] = boxes[0][:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[0][:mix_num_boxes]

        features = mix_features_pad.float().to(self.bert_gpu_device)
        image_mask = torch.tensor(image_mask).long().to(self.bert_gpu_device)
        spatials = mix_boxes_pad.float().to(self.bert_gpu_device)

        vil_prediction, vil_logit, vil_binary_prediction, vision_prediction, \
        vision_logit, linguisic_prediction, linguisic_logit = \
        self.model(
            instruction.unsqueeze(0),
            features.unsqueeze(0),
            spatials.unsqueeze(0),
            segment_ids.unsqueeze(0),
            input_mask.unsqueeze(0),
            image_mask.unsqueeze(0),
            co_attention_mask.unsqueeze(0)
        )

        instruction = None
        features = None
        spatials = None
        segment_ids = None
        input_mask = None
        image_mask = None
        co_attention_mask = None


        #reduced_probs = torch.zeros((1, 3),
        #                        device=self.bert_gpu_device,
        #                        dtype=torch.float,
        #                        requires_grad=True
        #                        )
        #reduced_probs[:,0] += torch.sum(vil_prediction[:,:4], dim=1)
        #reduced_probs[:,1:] += vil_prediction[:,4:]
        reduced_probs = torch.cat((torch.sum(vil_prediction[:,:4], dim=-1, keepdims=True),
                                    vil_prediction[:,4:]), dim=1)


        self.loss = self.criterion(reduced_probs, category_target) + self.criterion(vil_prediction, target)
        #self.finegrained_loss = self.criterion(vil_prediction, target)
        self.loss = self.loss.mean() * target.size(1)
        batch_score = self.compute_score_with_logits(vil_prediction, target).sum() / float(batch_size)

        #im_features, boxes = self._get_image_features(im) #.to(self.bert_gpu_device)
        print("Target action ", target_action)
        return {"action": target_action, "action_args": action_args}, self.loss.item(), batch_score.item()

    def compute_score_with_logits(self, logits, labels):
        logits = torch.max(logits, 1)[1].data  # argmax
        one_hots = torch.zeros(*labels.size(), device=self.bert_gpu_device)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = one_hots * labels
        idx = torch.argmax(one_hots, dim=1).item()
        print("Predicted action", self.model_actions[idx])
        return scores


    def save(self, path):
        ''' Snapshot models '''
        torch.save(self.model.state_dict(), path)

    def load(self, encoder_path, decoder_path):
        ''' Loads parameters (but not training state) '''
        self.model.load_state_dict(torch.load(path))
