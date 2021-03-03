# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import time
import numpy as np
import ncnn
from .model_store import get_model_file
from ..utils.objects import Detect_Object
from ..utils.functional import *
import cv2


class NanoDet:
    def __init__(
        self,
        target_size=320,
        prob_threshold=0.4,
        nms_threshold=0.3,
        num_threads=1,
        use_gpu=False,
    ):
        self.target_size = target_size
        self.prob_threshold = prob_threshold
        self.nms_threshold = nms_threshold
        self.num_threads = num_threads
        self.use_gpu = use_gpu

        self.mean_vals = [103.53, 116.28, 123.675]
        self.norm_vals = [0.017429, 0.017507, 0.017125]

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = self.use_gpu
        self.net.opt.num_threads = self.num_threads

        # original pretrained model from https://github.com/RangiLyu/nanodet
        # the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
        self.net.load_param(get_model_file("nanodet_m.param"))
        self.net.load_model(get_model_file("nanodet_m.bin"))

        self.reg_max = 7
        self.strides = [8, 16, 32]
        self.num_candidate = 1000
        self.top_k = -1

        self.class_names = [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]

    def __del__(self):
        self.net = None

    def __call__(self, img):
        img_w = img.shape[1]
        img_h = img.shape[0]

        w = img_w
        h = img_h
        scale = 1.0
        if w > h:
            scale = float(self.target_size) / w
            w = self.target_size
            h = int(h * scale)
        else:
            scale = float(self.target_size) / h
            h = self.target_size
            w = int(w * scale)

        mat_in = ncnn.Mat.from_pixels_resize(
            img, ncnn.Mat.PixelType.PIXEL_BGR, img_w, img_h, w, h
        )

        # pad to target_size rectangle
        wpad = (w + 31) // 32 * 32 - w
        hpad = (h + 31) // 32 * 32 - h
        mat_in_pad = ncnn.copy_make_border(
            mat_in,
            hpad // 2,
            hpad - hpad // 2,
            wpad // 2,
            wpad - wpad // 2,
            ncnn.BorderType.BORDER_CONSTANT,
            0,
        )

        mat_in_pad.substract_mean_normalize(self.mean_vals, self.norm_vals)

        ex = self.net.create_extractor()
        ex.input("input.1", mat_in_pad)

        score_out_name = ["792", "814", "836"]
        scores = [ex.extract(x)[1] for x in score_out_name]
        scores = [np.reshape(x, (-1, 80)) for x in scores]

        boxes_out_name = ["795", "817", "839"]
        raw_boxes = [ex.extract(x)[1] for x in boxes_out_name]
        raw_boxes = [np.reshape(x, (-1, 32)) for x in raw_boxes]

        # generate centers
        decode_boxes = []
        select_scores = []
        for stride, box_distribute, score in zip(self.strides, raw_boxes, scores):
            # centers
            if mat_in_pad.w > mat_in_pad.h:
                fm_w = mat_in_pad.w // stride
                fm_h = score.shape[0] // fm_w
            else:
                fm_h = mat_in_pad.h // stride
                fm_w = score.shape[1] // fm_h
            h_range = np.arange(fm_h)
            w_range = np.arange(fm_w)
            ww, hh = np.meshgrid(w_range, h_range)
            ct_row = (hh.flatten() + 0.5) * stride
            ct_col = (ww.flatten() + 0.5) * stride
            center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)

            # box distribution to distance
            reg_range = np.arange(self.reg_max + 1)
            box_distance = box_distribute.reshape((-1, self.reg_max + 1))
            box_distance = softmax(box_distance)
            box_distance = box_distance * np.expand_dims(reg_range, axis=0)
            box_distance = np.sum(box_distance, axis=1).reshape((-1, 4))
            box_distance = box_distance * stride

            # top K candidate
            topk_idx = np.argsort(score.max(axis=1))[::-1]
            topk_idx = topk_idx[: self.num_candidate]
            center = center[topk_idx]
            score = score[topk_idx]
            box_distance = box_distance[topk_idx]

            # decode box
            decode_box = center + [-1, -1, 1, 1] * box_distance

            select_scores.append(score)
            decode_boxes.append(decode_box)

        # nms
        bboxes = np.concatenate(decode_boxes, axis=0)
        confidences = np.concatenate(select_scores, axis=0)
        picked_box = []
        picked_probs = []
        picked_labels = []
        for class_index in range(0, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > self.prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = bboxes[mask, :]
            picked = nms(
                subset_boxes,
                probs,
                iou_threshold=self.nms_threshold,
                top_k=self.top_k,
            )
            picked_box.append(subset_boxes[picked])
            picked_probs.append(probs[picked])
            picked_labels.extend([class_index] * len(picked))

        if not picked_box:
            return []

        picked_box = np.concatenate(picked_box)
        picked_probs = np.concatenate(picked_probs)

        # result with clip
        objects = [
            Detect_Object(
                label,
                score,
                (bbox[0] - wpad / 2) / scale if bbox[0] > 0 else 0,
                (bbox[1] - hpad / 2) / scale if bbox[1] > 0 else 0,
                (bbox[2] - bbox[0]) / scale
                if bbox[2] < mat_in_pad.w
                else (mat_in_pad.w - bbox[0]) / scale,
                (bbox[3] - bbox[1]) / scale
                if bbox[3] < mat_in_pad.h
                else (mat_in_pad.h - bbox[1]) / scale,
            )
            for label, score, bbox in zip(picked_labels, picked_probs, picked_box)
        ]

        return objects
