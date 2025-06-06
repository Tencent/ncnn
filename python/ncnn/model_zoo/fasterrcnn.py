# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

import numpy as np
import ncnn
from .model_store import get_model_file
from ..utils.objects import Detect_Object


class Faster_RCNN:
    def __init__(
        self,
        img_width=600,
        img_height=600,
        num_threads=1,
        use_gpu=False,
        max_per_image=100,
        confidence_thresh=0.05,
        nms_threshold=0.3,
    ):
        self.img_width = img_width
        self.img_height = img_height
        self.num_threads = num_threads
        self.use_gpu = use_gpu

        self.mean_vals = [102.9801, 115.9465, 122.7717]
        self.norm_vals = []

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = self.use_gpu

        # original pretrained model from https://github.com/rbgirshick/py-faster-rcnn
        # py-faster-rcnn/models/pascal_voc/ZF/faster_rcnn_alt_opt/faster_rcnn_test.pt
        # https://dl.dropboxusercontent.com/s/o6ii098bu51d139/faster_rcnn_models.tgz?dl=0
        # ZF_faster_rcnn_final.caffemodel
        # the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
        self.net.load_param(get_model_file("ZF_faster_rcnn_final.param"))
        self.net.load_model(get_model_file("ZF_faster_rcnn_final.bin"))

        self.max_per_image = max_per_image
        self.confidence_thresh = confidence_thresh
        self.nms_threshold = nms_threshold

        self.class_names = [
            "background",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]

    def __del__(self):
        self.net = None

    def __call__(self, img):
        # scale to target detect size
        h = img.shape[0]
        w = img.shape[1]
        scale = 1.0
        if w < h:
            scale = float(self.img_width) / w
            w = self.img_width
            h = int(h * scale)
        else:
            scale = float(self.img_height) / h
            h = self.img_height
            w = int(w * scale)

        mat_in = ncnn.Mat.from_pixels_resize(
            img, ncnn.Mat.PixelType.PIXEL_BGR, img.shape[1], img.shape[0], w, h
        )
        mat_in.substract_mean_normalize(self.mean_vals, self.norm_vals)

        # method 1 use numpy to Mat interface
        # im_info = ncnn.Mat(np.array([h, w, scale], dtype=np.float32))

        # method 2 use ncnn.Mat interface
        im_info = ncnn.Mat(3)
        im_info[0] = h
        im_info[1] = w
        im_info[2] = scale

        ex1 = self.net.create_extractor()
        ex1.set_num_threads(self.num_threads)

        ex1.input("data", mat_in)
        ex1.input("im_info", im_info)

        ret1, conv5_relu5 = ex1.extract("conv5_relu5")
        ret2, rois = ex1.extract("rois")

        class_candidates = []
        for i in range(rois.c):
            ex2 = self.net.create_extractor()

            roi = rois.channel(i)  # get single roi
            ex2.input("conv5_relu5", conv5_relu5)
            ex2.input("rois", roi)

            ret1, bbox_pred = ex2.extract("bbox_pred")
            ret2, cls_prob = ex2.extract("cls_prob")

            num_class = cls_prob.w
            while len(class_candidates) < num_class:
                class_candidates.append([])

            # find class id with highest score
            label = 0
            score = 0.0
            for j in range(num_class):
                class_score = cls_prob[j]
                if class_score > score:
                    label = j
                    score = class_score

            # ignore background or low score
            if label == 0 or score <= self.confidence_thresh:
                continue

            # fprintf(stderr, "%d = %f\n", label, score);

            # unscale to image size
            x1 = roi[0] / scale
            y1 = roi[1] / scale
            x2 = roi[2] / scale
            y2 = roi[3] / scale

            pb_w = x2 - x1 + 1
            pb_h = y2 - y1 + 1

            # apply bbox regression
            dx = bbox_pred[label * 4]
            dy = bbox_pred[label * 4 + 1]
            dw = bbox_pred[label * 4 + 2]
            dh = bbox_pred[label * 4 + 3]

            cx = x1 + pb_w * 0.5
            cy = y1 + pb_h * 0.5

            obj_cx = cx + pb_w * dx
            obj_cy = cy + pb_h * dy

            obj_w = pb_w * np.exp(dw)
            obj_h = pb_h * np.exp(dh)

            obj_x1 = obj_cx - obj_w * 0.5
            obj_y1 = obj_cy - obj_h * 0.5
            obj_x2 = obj_cx + obj_w * 0.5
            obj_y2 = obj_cy + obj_h * 0.5

            # clip
            obj_x1 = np.maximum(np.minimum(obj_x1, float(img.shape[1] - 1)), 0.0)
            obj_y1 = np.maximum(np.minimum(obj_y1, float(img.shape[0] - 1)), 0.0)
            obj_x2 = np.maximum(np.minimum(obj_x2, float(img.shape[1] - 1)), 0.0)
            obj_y2 = np.maximum(np.minimum(obj_y2, float(img.shape[0] - 1)), 0.0)

            # append object
            obj = Detect_Object()
            obj.rect.x = obj_x1
            obj.rect.y = obj_y1
            obj.rect.w = obj_x2 - obj_x1 + 1
            obj.rect.h = obj_y2 - obj_y1 + 1
            obj.label = label
            obj.prob = score

            class_candidates[label].append(obj)

        # post process
        objects = []
        for candidates in class_candidates:
            if len(candidates) == 0:
                continue

            candidates.sort(key=lambda obj: obj.prob, reverse=True)

            picked = self.nms_sorted_bboxes(candidates, self.nms_threshold)

            for j in range(len(picked)):
                z = picked[j]
                objects.append(candidates[z])

        objects.sort(key=lambda obj: obj.prob, reverse=True)

        objects = objects[: self.max_per_image]

        return objects

    def nms_sorted_bboxes(self, objects, nms_threshold):
        picked = []

        n = len(objects)

        areas = np.zeros((n,), dtype=np.float32)
        for i in range(n):
            areas[i] = objects[i].rect.area()

        for i in range(n):
            a = objects[i]

            keep = True
            for j in range(len(picked)):
                b = objects[picked[j]]

                # intersection over union
                inter_area = a.rect.intersection_area(b.rect)
                union_area = areas[i] + areas[picked[j]] - inter_area
                # float IoU = inter_area / union_area
                if inter_area / union_area > nms_threshold:
                    keep = False

            if keep:
                picked.append(i)

        return picked
