#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import habitat
import cv2
import os
import json
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
from detectron2.data import MetadataCatalog
from transformers import BertTokenizer

def get_image_labels2(classes, pred_class_logits, keep):
    labels = []
    for c, i in zip(pred_class_logits, keep):
        labels.append((classes[c], c, i))
    return labels

# We need the indices of the features to keep
def fast_rcnn_inference_single_image(
        boxes,
        scores,
        image_shape,
        score_thresh,
        nms_thresh,
        topk_per_image,
        device,
        preferred_labels = [],
        tokens = [],
        tokenizer = None
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
        # calculate the closes tokens
        words = get_image_labels2(preferred_labels, max_classes[keep].tolist(), keep.tolist())
        relevant = []
        others = []
        class_list = []

        for word, c, i in words:
            tok = tokenizer.vocab.get(word, tokenizer.vocab["[UNK]"])
            ## inserting the relevant first
            if tok in tokens:
                relevant.append(i)
            ## repeated predictions go last.
            elif c in class_list:
                class_list.append(c)
                others.append(i)
            ## Inserting varied predictions first
            else:
                class_list.append(c)
                others.insert(i, 0)

        keep = torch.tensor(relevant+others, device=device)

        #remove duplicate classes......


        if topk_per_image >= 0:
            keep = keep[:topk_per_image]
            keep = keep[torch.randperm(keep.size()[0])]
        boxes, scores = max_boxes[keep], max_scores[keep]

        result = Instances(image_shape)
        result.pred_boxes = Boxes(boxes)
        result.scores = scores
        result.pred_classes = max_classes[keep]

        return result, keep

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


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
                    "BUA_R_101_C4": "VG-Detection/faster_rcnn_R_101_C4_caffe.yaml",
                    "BUA_R_101_C4_MAX": "VG-Detection/faster_rcnn_R_101_C4_caffemaxpool.yaml"
    }


    def __init__(self, config, num_train_optimization_steps=3100, include_actions=True):
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
        new_voc_size = self.vilbert_config.vocab_size + 8
        self.model.resize_token_embeddings(new_voc_size)
        self.model.to(self.bert_gpu_device)
        print("ViLBERT loaded on GPU {}".format(self.bert_gpu))

        print("Loading Detectron2 predictor on GPU {}".format(self.detectron2_gpu))
        detectron2_cfg = self.create_detectron2_cfg(config)
        self.detector = DefaultPredictor(detectron2_cfg)
        #self.detector.eval()
        print("Detectron2 loaded")
        self._max_region_num = 36
        self._max_seq_length = 128
        #if include_actions:
            #self._max_seq_length = 128 + 10
        self.tokenizer = BertTokenizer.from_pretrained(
                             "bert-base-uncased",
                             do_lower_case=True,
                             do_basic_tokenize=True
                         )
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.loss = 0
        self.learning_rate = 3e-6
        self.vision_scratch = False
        self.max_steps = 30
        self.grad_accumulation = 1 #00
        self.action_history = []
        self.loss_weight = {
                "a": 0.1,
                "b": 0.1,
                "c": 0.8,
                "a_loss": [],
                "b_loss": [],
                "c_loss": [],
        }
        self.save_example = {
            "path_id":"",
            "images": [],
            "boxes": [],
            "box_probs": [],
            "text": [],
            "actions": [],
            "box_one_hots": [],
            "box_labels": []
        }
        optimizer_grouped_parameters = []
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

        for key, value in dict(self.model.named_parameters()).items():
            if value.requires_grad:
                if any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [
                        {"params": [value], "lr": self.learning_rate , "weight_decay": 0.01}
                    ]
                if not any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [
                        {"params": [value], "lr": self.learning_rate , "weight_decay": 0.0}
                    ]

        print(len(list(self.model.named_parameters())), len(optimizer_grouped_parameters))

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

        data_path = '/home/erick/Research/vln/libs/habitat/habitat-api/habitat_baselines/vln/data/genome/1600-400-20/'

        vg_classes = []
        with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
            for object in f.readlines():
                vg_classes.append(object.split(',')[0].lower().strip())

        MetadataCatalog.get("vg").thing_classes = vg_classes
        self.class_names = vg_classes
        cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe.pkl"
        #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint)
        return cfg

    def adjust_weights(self):
        num = 100
        if len(self.loss_weight["c_loss"]) > (num - 1):
            self.loss_weight["a_loss"] = self.loss_weight["a_loss"][-num:]
            self.loss_weight["b_loss"] = self.loss_weight["b_loss"][-num:]
            self.loss_weight["c_loss"] = self.loss_weight["c_loss"][-num:]
            a_avg = sum(self.loss_weight["a_loss"]) / num
            b_avg = sum(self.loss_weight["b_loss"]) / num
            c_avg = sum(self.loss_weight["c_loss"]) / num
            if c_avg > 0.92:
                self.loss_weight["c"] = 0.1
                if b_avg > 0.92:
                    self.loss_weight["a"] = 0.8
                    self.loss_weight["b"] = 0.1
                else:
                    self.loss_weight["a"] = 0.1
                    self.loss_weight["b"] = 0.8
            else:
                self.loss_weight["a"] = 0.1
                self.loss_weight["b"] = 0.1
                self.loss_weight["c"] = 0.80

            self.loss_weight["a_loss"] = self.loss_weight["a_loss"][-int(num/2):]
            self.loss_weight["b_loss"] = self.loss_weight["b_loss"][-int(num/2):]
            self.loss_weight["c_loss"] = self.loss_weight["c_loss"][-int(num/2):]
            print("Weights adjusted to ",
                    bcolors.OKBLUE + "A: " + str(self.loss_weight["a"]) + bcolors.ENDC ,
                    a_avg,
                    bcolors.OKBLUE + "B: " + str(self.loss_weight["b"]) + bcolors.ENDC ,
                    b_avg,
                    bcolors.OKBLUE + "C: " + str(self.loss_weight["c"]) + bcolors.ENDC ,
                    c_avg
                )

    def get_word_token(self, word):
        token = self.tokenizer.vocab.get(word, self.tokenizer.vocab["[UNK]"])
        return token

    def reset(self, steps):
        self.loss = None
        self.action_history = []
        self.adjust_weights()
        self.save_example = {
            "path_id":"",
            "images": [],
            "boxes": [],
            "box_probs": [],
            "text": [],
            "actions": [],
            "box_one_hots": [],
            "box_labels": []
        }
        #pass
    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def train_step(self, steps):
        self.loss = self.loss / self.grad_accumulation
        self.loss.backward()
        self.loss = None

        if (steps + 1) % self.grad_accumulation == 0:
            self.optimizer.step()
            self.model.zero_grad()

    def get_image_labels(self, pred_class_logits):
        labels = []
        classes = self.class_names
        for c in pred_class_logits:
            labels.append(classes[c])
        #print(labels)
        return labels

    def get_image_target_onehot(self, classes, instr_tokens):
        one_hots = [[0]]
        class_labels = self.get_image_labels(classes)
        chosen_labels = []
        for cl , cl_id in zip(class_labels, classes):
            token_id = self.get_word_token(cl)
            if token_id in instr_tokens:
                one_hots.append([1])
                chosen_labels.append((cl, cl_id))
            else:
                one_hots.append([0])
                chosen_labels.append(("", -1))

        one_hots = torch.tensor(
                        one_hots,
                        dtype=torch.float,
                        device=self.bert_gpu_device
                 ).unsqueeze(0)
        #print(chosen_labels)
        #print(instr_tokens)

        return one_hots, chosen_labels

    def _get_image_features(self, imgs, score_thresh=0.2, min_num_image=10, max_regions=36, tokens=[]):
        # imgs tensor(batch, H, W, C)
        with torch.no_grad():
            inputs = []
            for img in imgs:
                raw_img = img.permute(2,0,1)
                raw_img = raw_img.to(self.detectron2_gpu_device)
                (_, height, width) = raw_img.shape
                inputs.append({"image": raw_img, "height": height, "width": width})

            # Normalize the image by substracting mean
            # Moves the image to device (already in device)
            images = self.detector.model.preprocess_image(inputs)
            sizes = images.image_sizes

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
            images = None
            # The C4 model uses Res5ROIHeads where pooled feature can be extracted
            proposal_boxes = [x.proposal_boxes for x in proposals]
            features = [features[f] for f in self.detector.model.roi_heads.in_features]
            box_features = self.detector.model.roi_heads._shared_roi_transform(
                features, proposal_boxes
            )
            features = None
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
            proposals = None

            # Filter proposals using Non-Maximum Suppression (NMS)
            instances_list, ids_list = [], []
            probs_list = rcnn_outputs.predict_probs()
            boxes_list = rcnn_outputs.predict_boxes()
            #image_shapes = [x.image_size for x in proposals]
            num_boxes = []
            for probs, boxes, image_size in zip(probs_list, boxes_list, sizes):

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
                        device=self.detectron2_gpu_device,
                        preferred_labels=self.class_names,
                        tokens=tokens,
                        tokenizer=self.tokenizer
                    )
                    #
                    if len(ids) >= min_num_image:
                        break
                num_boxes.append(len(ids)+1)
                instances_list.append(instances)
                ids_list.append(ids)

            # Post processing for features
            features_list = feature_pooled.split(rcnn_outputs.num_preds_per_image) # (sum_proposals, 2048) --> [(p1, 2048), (p2, 2048), ..., (pn, 2048)]
            feature_pooled = None
            roi_features_list = []
            for ids, features in zip(ids_list, features_list):
                head_box = torch.sum(features[ids], axis=0) / \
                           len(features[ids])
                head_box = head_box.unsqueeze(0)
                roi_features_list.append(torch.cat((head_box, features[ids]), 0))

            # Post processing for bounding boxes (rescale to raw_image)
            boxes = []
            classes = []
            for instances, input_per_image, image_size in zip(
                    instances_list, inputs, sizes
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
                box[:,1] /= float(height)
                box[:,2] /= float(width)
                box[:,3] /= float(height)
                box[:,4] = (box[:,3] - box[:,1]) * (box[:,2] - box[:,0]) / \
                    (float(height) * float(width))

                boxes.append(box)
                classes.append(raw_instances.pred_classes)
            # features, boxes, image_mask
            return roi_features_list, boxes, num_boxes, classes#, pred_proposal_deltas

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

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

    def _teleport_target(self, observations):
        action = ""
        action_args = {}
        navigable_locations = observations["adjacentViewpoints"]

        for location in navigable_locations[1:]:
            if location[0] == 1: # location is restricted
                continue
            elif location[0] == 0: # Non restricted location
                action = "TELEPORT"
                image_id = location[1]
                posB = location[4:7]  # start_position
                rotA = navigable_locations[0][14:18]  # camera_rotation
                viewpoint = ViewpointData(
                    image_id=image_id,
                    view_point=AgentState(position=posB, rotation=rotA)
                )
                action_args = {"target": viewpoint}
                #print("the target is ", location)
                return {"action": action, "action_args": action_args}

        return {"action": action, "action_args": action_args}

    def _get_target_onehot(self, observations, goals):
        target_action, args = self._teacher_actions(observations, goals)
        idx = self.model_actions.index(target_action)
        one_hot = torch.zeros((1,6), device=self.bert_gpu_device)
        category_one_hot = torch.zeros((1,3), device=self.bert_gpu_device)
        stop_one_hot = torch.zeros((1,2), device=self.bert_gpu_device)
        one_hot[0][idx] = 1
        if idx < 4:
            category_one_hot[0][0] = 1
        else:
            category_one_hot[0][idx-3] = 1
        if idx == 5:
            stop_one_hot[0][1] = 1
        else:
            stop_one_hot[0][0] = 1


        return category_one_hot, one_hot, stop_one_hot, target_action, args

    def _get_batch_target_onehot(self, observations):
            batch_size = len(observations)
            one_hot = torch.zeros((batch_size, 6), device=self.bert_gpu_device)
            category_one_hot = torch.zeros((batch_size, 3), device=self.bert_gpu_device)
            stop_one_hot = torch.zeros((batch_size, 2), device=self.bert_gpu_device)
            for i, ob in enumerate(observations):
                idx = ob["golden_action"]
                one_hot[i][idx] = 1
                if idx < 4:
                    category_one_hot[i][0] = 1
                else:
                    category_one_hot[i][idx-3] = 1
                if idx == 5:
                    stop_one_hot[i][1] = 1
                else:
                    stop_one_hot[i][0] = 1


            return category_one_hot, one_hot, stop_one_hot

    def _get_tensor_image_features(self, im, max_regions=37, tokens=[]):
        features, boxes, num_boxes, pred_class_logits = \
            self._get_image_features([im], min_num_image=36, tokens=tokens)
        #print(self.get_image_labels(pred_class_logits[0].tolist()))
        mix_num_boxes = min(int(num_boxes[0]), max_regions)
        mix_boxes_pad = torch.zeros((max_regions, 5)
                                    , dtype=torch.float
                                    , device=self.bert_gpu_device)
        mix_features_pad = torch.zeros((max_regions, 2048)
                                    , dtype=torch.float
                                    , device=self.bert_gpu_device)

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < max_regions:
            image_mask.append(0)

        mix_boxes_pad[:mix_num_boxes] = boxes[0][:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[0][:mix_num_boxes]

        feat = mix_features_pad.unsqueeze(0)
        img_mask = torch.tensor(image_mask
                                    , dtype=torch.long
                                    , device=self.bert_gpu_device
                                ).unsqueeze(0)
        spat = mix_boxes_pad.unsqueeze(0)
        return feat, img_mask, spat, pred_class_logits[0].tolist()

    def tensorize(self, observations):
        batch_size = len(observations)
        max_regions = self._max_region_num + 1
        instructions = []
        masks = []
        segments_ids = []
        co_attention_masks = []
        tensor_features = []
        spatials = []
        image_masks = []
        target_tokens = []
        image_logits = []
        image_labels = []
        for ob in observations:
            im = ob["rgb"]
            feat, img_mask, spat, labels = self._get_tensor_image_features(im,  tokens=ob["tokens"])
            img_logit, img_label = self.get_image_target_onehot(labels, ob["tokens"])
            tensor_features.append(feat)
            spatials.append(spat)
            image_masks.append(img_mask)
            image_logits.append(img_logit)
            image_labels.append(img_label)

            instruction = torch.tensor(
                                ob["tokens"],
                                dtype=torch.long,
                                device=self.bert_gpu_device
                          ).unsqueeze(0)

            t_tokens = torch.tensor(
                                ob["target_tokens"],
                                dtype=torch.long,
                                device=self.bert_gpu_device
                          ).unsqueeze(0)

            #print(ob["mask"])
            input_mask = torch.tensor(
                                ob["mask"],
                                dtype=torch.long,
                                device=self.bert_gpu_device
                         ).unsqueeze(0)

            #print(ob["segment"])
            segment = torch.tensor(
                                    ob["segment"],
                                    dtype=torch.long,
                                    device=self.bert_gpu_device
                        ).unsqueeze(0)
            co_attention_mask = torch.zeros((
                                    max_regions,
                                    self._max_seq_length
                                ), dtype=torch.long
                                , device=self.bert_gpu_device
                        ).unsqueeze(0)

            instructions.append(instruction)
            masks.append(input_mask)
            segments_ids.append(segment)
            co_attention_masks.append(co_attention_mask)
            target_tokens.append(t_tokens)

        instructions = torch.cat(instructions, dim=0)
        masks = torch.cat(masks, dim=0)
        segments_ids = torch.cat(segments_ids, dim=0)
        co_attention_masks = torch.cat(co_attention_masks, dim=0)
        tensor_features = torch.cat(tensor_features, dim=0)
        spatials = torch.cat(spatials, dim=0)
        image_masks = torch.cat(image_masks, dim=0)
        image_logits = torch.cat(image_logits, dim=0)
        target_tokens = torch.cat(target_tokens, dim=0)

        return instructions, masks, segments_ids, co_attention_masks, tensor_features, spatials, image_masks, image_logits, image_labels, target_tokens

    def act_batch(self, observations):
        batch_size = len(observations)
        instructions, input_masks, segment_ids,  \
        co_attention_masks, features, spatials, image_masks, \
        image_one_hots, _ , target_tokens = \
            self.tensorize(observations)
        category_target, target, stop_target = \
            self._get_batch_target_onehot(
                        observations
                    )
        vil_prediction, vil_logit, vil_binary_prediction, vision_prediction, \
        vision_logit, linguisic_prediction, linguisic_logit = \
        self.model(
            instructions,
            features,
            spatials,
            segment_ids,
            input_masks,
            image_masks,
            co_attention_masks
        )

        for i, ob in enumerate(observations):
            if not any(True for obs in ob['adjacentViewpoints'] if obs[0] == 0):
                teleport_idx = self.model_actions.index("TELEPORT")
                vil_prediction[i][teleport_idx] = -100.0

        instructions = None
        previous_actions = None
        features = None
        spatials = None
        segment_ids = None
        input_masks = None
        image_masks = None
        co_attention_masks = None
        #print("vision_prediction", vision_prediction.shape)
        #print("linguisic_prediction", linguisic_prediction.shape)
        #print("linguisic_logit", linguisic_logit.shape)

        linguistic_tokens = torch.max(linguisic_prediction, 1)[1].data  # argmax
        #print(linguisic_prediction.shape, linguisic_logit.shape)
        #print(linguistic_tokens[:,-10:])

        reduced_probs = torch.cat((torch.sum(vil_prediction[:,:4], dim=-1, keepdims=True),
                                    vil_prediction[:,4:]), dim=1)
        stop_probs = torch.cat((torch.sum(vil_prediction[:,:-1], dim=-1, keepdims=True),
                                    vil_prediction[:,-1:]), dim=-1)

        #self.loss = self.loss_weight["b"] * self.criterion(reduced_probs, category_target) + \
        #    self.loss_weight["a"] * self.criterion(vil_prediction, target) + \
        #    self.loss_weight["c"] * self.criterion(stop_probs, stop_target)
        self.loss = self.criterion(vil_prediction, target) + \
                    self.criterion(vision_logit, image_one_hots) #+ \
                    #self.criterion(linguisic_logit, )

        self.loss = self.loss.mean() * target.size(1)
        scores, reduce_scores, stop_scores = self.compute_all_scores_with_logits(vil_prediction, target)

        scores = scores.sum() / float(batch_size)
        reduce_scores = reduce_scores.sum() / float(batch_size)
        stop_scores = stop_scores.sum() / float(batch_size)
        self.loss_weight["a_loss"].append(scores.item())
        self.loss_weight["b_loss"].append(reduce_scores.item())
        self.loss_weight["c_loss"].append(stop_scores.item())

        #im_features, boxes = self._get_image_features(im) #.to(self.bert_gpu_device)
        #print("Target action ", target_action)

        return  self.loss.item(), scores.item()

    def act(self, observations, episode, goals):

        # Observations come in Caffe GPU
        batch_size = 1
        category_target, target, stop_target, target_action, \
        action_args = self._get_target_onehot(
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

        reduced_probs = torch.cat((torch.sum(vil_prediction[:,:4], dim=-1, keepdims=True),
                                    vil_prediction[:,4:]), dim=1)
        stop_probs = torch.cat((torch.sum(vil_prediction[:,:-1], dim=-1, keepdims=True),
                                    vil_prediction[:,-1:]), dim=1)

        self.loss = 0.25 * self.criterion(reduced_probs, category_target) + \
            0.35 * self.criterion(vil_prediction, target) + \
            0.4 * self.criterion(stop_probs, stop_target)

        self.loss = self.loss.mean() * target.size(1)
        batch_score = self.compute_mistakes(stop_probs, reduced_probs, vil_prediction, target).sum() / float(batch_size)

        #im_features, boxes = self._get_image_features(im) #.to(self.bert_gpu_device)
        print("Target action ", target_action)
        return {"action": target_action, "action_args": action_args}, self.loss.item(), batch_score.item()

    def act_eval(self, observations):
        action = "<start>"
        action_args = {}

        instructions, input_masks, segment_ids,  \
        co_attention_masks, features, spatials, image_masks, \
        image_one_hots, image_labels, target_tokens = \
            self.tensorize(observations)

        vil_prediction, vil_logit, vil_binary_prediction, vision_prediction, \
        vision_logit, linguisic_prediction, linguisic_logit = \
        self.model(
            instructions,
            features,
            spatials,
            segment_ids,
            input_masks,
            image_masks,
            co_attention_masks
        )

        '''
        self.save_example = {
            "path_id":""
            "images": [],
            "boxes": [],
            "box_probs": [],
            "text": [],
            "actions": []
        }
        '''
        ob = observations[0]
        #print(linguisic_prediction.shape)
        linguistic_tokens = torch.max(linguisic_prediction, -1)[1].data  # argmax
        selected_images = vision_logit.tolist()
        if not any(True for obs in ob['adjacentViewpoints'] if obs[0] == 0):
            teleport_idx = self.model_actions.index("TELEPORT")
            vil_prediction[0][teleport_idx] = -float('inf')
        #if self.mode == "argmax":
        logit = torch.max(vil_prediction, 1)[1].data  # argmax
        #elif self.mode == "sample":
        action = self.model_actions[logit]
        self.save_example["path_id"] = ob["path_id"]
        self.save_example["images"].append(ob["rgb"].tolist())
        self.save_example["boxes"].append(spatials[0].tolist())
        self.save_example["box_probs"].append(vision_logit.tolist())
        self.save_example["text"].append(linguistic_tokens.tolist())
        self.save_example["actions"].append(action)
        self.save_example["box_one_hots"].append(image_one_hots.tolist())
        self.save_example["box_labels"].append(image_labels)
        #print(vision_logit.tolist()) #, linguistic_tokens.tolist(), spatials[0].tolist())

        next_action = {"action": action, "action_args": action_args}
        #print(action)
        if action == "TELEPORT":
            next_action = self._teleport_target(ob)
        #print("next action", next_action)
        return next_action

    def act_eval_batch(self, observations):
        batch_size = len(observations)
        instructions, input_masks, segment_ids,  \
        co_attention_masks, features, spatials, image_masks, \
        image_one_hots, image_labels , target_tokens = \
            self.tensorize(observations)
        category_target, target, stop_target = \
            self._get_batch_target_onehot(
                        observations
                    )
        vil_prediction, vil_logit, vil_binary_prediction, vision_prediction, \
        vision_logit, linguisic_prediction, linguisic_logit = \
        self.model(
            instructions,
            features,
            spatials,
            segment_ids,
            input_masks,
            image_masks,
            co_attention_masks
        )

        for i, ob in enumerate(observations):
            if not any(True for obs in ob['adjacentViewpoints'] if obs[0] == 0):
                teleport_idx = self.model_actions.index("TELEPORT")
                vil_prediction[i][teleport_idx] = -100.0


        #print("vision_prediction", vision_prediction.shape)
        #print("linguisic_prediction", linguisic_prediction.shape)
        #print("linguisic_logit", linguisic_logit.shape)

        linguistic_tokens = torch.max(linguisic_prediction, 1)[1].data  # argmax
        #print(linguisic_prediction.shape, linguisic_logit.shape)
        #print(linguistic_tokens[:,-10:])

        reduced_probs = torch.cat((torch.sum(vil_prediction[:,:4], dim=-1, keepdims=True),
                                    vil_prediction[:,4:]), dim=1)
        stop_probs = torch.cat((torch.sum(vil_prediction[:,:-1], dim=-1, keepdims=True),
                                    vil_prediction[:,-1:]), dim=-1)

        #self.loss = self.loss_weight["b"] * self.criterion(reduced_probs, category_target) + \
        #    self.loss_weight["a"] * self.criterion(vil_prediction, target) + \
        #    self.loss_weight["c"] * self.criterion(stop_probs, stop_target)
        logit = torch.max(vil_prediction, 1)[1].data  # argmax

        action = self.model_actions[logit]
        self.save_example["path_id"] = ob["path_id"]
        self.save_example["images"].append(ob["rgb"].tolist())
        self.save_example["boxes"].append(spatials[0].tolist())
        self.save_example["box_probs"].append(vision_logit.tolist())
        self.save_example["text"].append(linguistic_tokens.tolist())
        self.save_example["actions"].append(action)
        self.save_example["box_one_hots"].append(image_one_hots.tolist())
        self.save_example["box_labels"].append(image_labels)

        instructions = None
        previous_actions = None
        features = None
        spatials = None
        segment_ids = None
        input_masks = None
        image_masks = None
        co_attention_masks = None

        scores, reduce_scores, stop_scores = self.compute_all_scores_with_logits(vil_prediction, target)
        vision_scores = self.compute_vision_score(vision_logit, image_one_hots)

        scores = scores.sum() / float(batch_size)
        reduce_scores = reduce_scores.sum() / float(batch_size)
        stop_scores = stop_scores.sum() / float(batch_size)

        #im_features, boxes = self._get_image_features(im) #.to(self.bert_gpu_device)
        #print("Target action ", target_action)

        return  scores.item(), vision_scores

    def save_example_to_file(self):
        PATH = "/home/erick/Research/vln/examples/"
        path_id = self.save_example["path_id"] + ".json"

        if len(self.save_example["actions"]) == 0 or self.save_example["actions"][-1] != "STOP":
            return

        with open(os.path.join(PATH, path_id), "w+") as outfile:
            json.dump(self.save_example, outfile)

    def compute_mistakes(self, stop_probs, category_probs, logits, labels):
        logits = torch.max(logits, 1)[1].data  # argmax
        one_hots = torch.zeros(*labels.size(), device=self.bert_gpu_device)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        idx = torch.argmax(one_hots, dim=1).item()
        gold_idx = torch.argmax(labels, dim=1).item()

        color = bcolors.FAIL
        if idx == gold_idx:
            color = bcolors.OKGREEN

        if idx < 4:
            print(color + " Prediction: NON_STOP, SEARCH, " + self.model_actions[idx] + bcolors.ENDC)
        elif idx == 4:
            print(color + " NON_STOP, " + self.model_actions[idx] + bcolors.ENDC)
        else:
            print(color + self.model_actions[idx] + bcolors.ENDC)

        category_one_hots = torch.zeros(*category_probs.size(), device=self.bert_gpu_device)
        stop_one_hots = torch.zeros(*stop_probs.size(), device=self.bert_gpu_device)
        scores = one_hots * labels
        return scores

    def compute_vision_score(self, logits, labels):
        logits_one_hots = (logits > 0).long()
        tp = logits_one_hots * labels # and
        fn = labels - tp #
        fp = logits_one_hots - tp
        tn = torch.ones((logits.size()),
        device=self.bert_gpu_device).long() - ( fn + fp + tp)

        tp = torch.sum(tp).item()
        fn = torch.sum(fn).item()
        fp = torch.sum(fp).item()
        tn = torch.sum(tn).item()
        print("tp", tp)
        print("fn", fn)
        print("fp", fp)
        print("tn", tn)
        e = 0.0000000001

        precision = tp / (tp + fp + e)
        recall = tp / (tp + fn + e)
        accuracy = (tp + tn) / (tp + tn + fp + fn + e)

        return precision, recall, accuracy

    def compute_score_with_logits(self, logits, labels):
        logits = torch.max(logits, 1)[1].data  # argmax
        one_hots = torch.zeros(*labels.size(), device=self.bert_gpu_device)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = one_hots * labels
        #idx = torch.argmax(one_hots, dim=1).item()
        #print("Predicted action", self.model_actions[idx])
        return scores

    def compute_all_scores_with_logits(self, logits, labels):


        reduced_labels = torch.cat((torch.sum(labels[:,:4], dim=-1, keepdims=True),
                                        labels[:,4:]), dim=1)
        stop_labels = torch.cat((torch.sum(labels[:,:-1], dim=-1, keepdims=True),
                                        labels[:,-1:]), dim=-1)

        logits = torch.max(logits, 1)[1].data  # argmax
        one_hots = torch.zeros(*labels.size(), device=self.bert_gpu_device)
        one_hots.scatter_(1, logits.view(-1, 1), 1)


        reduced_one_hots = torch.cat((torch.sum(one_hots[:,:4], dim=-1, keepdims=True),
                                        one_hots[:,4:]), dim=1)
        stop_one_hots = torch.cat((torch.sum(one_hots[:,:-1], dim=-1, keepdims=True),
                                        one_hots[:,-1:]), dim=-1)
        scores = one_hots * labels
        reduce_scores = reduced_one_hots * reduced_labels
        stop_scores = stop_one_hots * stop_labels

        return scores, reduce_scores, stop_scores


    def save(self, path):
        ''' Snapshot models '''
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        self.model.load_state_dict(torch.load(path))
        print("model loaded from path " + path)
