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

from math import sqrt
import numpy as np
import cv2
import ncnn
from .model_store import get_model_file
from ..utils.objects import Detect_Object
from ..utils.functional import sigmoid, nms


class Yolact:
    def __init__(
        self,
        target_size=550,
        confidence_threshold=0.05,
        nms_threshold=0.5,
        keep_top_k=200,
        num_threads=1,
        use_gpu=False,
    ):
        self.target_size = target_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.keep_top_k = keep_top_k
        self.num_threads = num_threads
        self.use_gpu = use_gpu

        self.mean_vals = [123.68, 116.78, 103.94]
        self.norm_vals = [1.0 / 58.40, 1.0 / 57.12, 1.0 / 57.38]

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = self.use_gpu
        self.net.opt.num_threads = self.num_threads

        # original model converted from https://github.com/dbolya/yolact
        # yolact_resnet50_54_800000.pth
        # the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
        self.net.load_param(get_model_file("yolact.param"))
        self.net.load_model(get_model_file("yolact.bin"))

        self.conv_ws = [69, 35, 18, 9, 5]
        self.conv_hs = [69, 35, 18, 9, 5]
        self.aspect_ratios = [1, 0.5, 2]
        self.scales = [24, 48, 96, 192, 384]

        self.priors = None
        self.last_img_size = None

        self.make_priors()

        self.class_names = [
            "background",
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
        img_h = img.shape[0]
        img_w = img.shape[1]

        mat_in = ncnn.Mat.from_pixels_resize(
            img,
            ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            img_w,
            img_h,
            self.target_size,
            self.target_size,
        )
        mat_in.substract_mean_normalize(self.mean_vals, self.norm_vals)

        ex = self.net.create_extractor()
        ex.input("input.1", mat_in)

        ret1, proto_data = ex.extract("619")  # 138x138 x 32
        ret2, loc_data = ex.extract("816")  # 4 x 19248
        ret3, mask_data = ex.extract("818")  # maskdim 32 x 19248
        ret4, conf_data = ex.extract("820")  # 81 x 19248

        proto_data = np.array(proto_data)
        loc_data = np.array(loc_data)
        mask_data = np.array(mask_data)
        conf_data = np.array(conf_data)
        prior_data = self.make_priors()

        # decoded_boxes = self.decode(loc_data, prior_data)
        boxes, masks, classes, scores = self.detect(
            conf_data, loc_data, prior_data, mask_data, img_w, img_h
        )

        # generate mask
        masks = proto_data.transpose(1, 2, 0) @ masks.T
        masks = sigmoid(masks)

        # Scale masks up to the full image
        masks = cv2.resize(masks, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

        # transpose into the correct output shape [num_dets, proto_h, proto_w]
        masks = masks.transpose(2, 0, 1)

        masks = masks > 0.5

        return boxes, masks, classes, scores

    def make_priors(self):
        """ Note that priors are [x,y,width,height] where (x,y) is the center of the box. """
        if self.last_img_size != (self.target_size, self.target_size):
            prior_data = []

            for conv_w, conv_h, scale in zip(self.conv_ws, self.conv_hs, self.scales):
                for i in range(conv_h):
                    for j in range(conv_w):
                        # +0.5 because priors are in center-size notation
                        cx = (j + 0.5) / conv_w
                        cy = (i + 0.5) / conv_h

                        for ar in self.aspect_ratios:
                            ar = sqrt(ar)

                            w = scale * ar / self.target_size
                            h = scale / ar / self.target_size

                            # This is for backward compatibility with a bug where I made everything square by accident
                            h = w

                            prior_data += [cx, cy, w, h]

            self.priors = np.array(prior_data).reshape(-1, 4)
            self.last_img_size = (self.target_size, self.target_size)

        return self.priors

    def decode(self, loc, priors, img_w, img_h):
        """
        Decode predicted bbox coordinates using the same scheme
        employed by Yolov2: https://arxiv.org/pdf/1612.08242.pdf

            b_x = (sigmoid(pred_x) - .5) / conv_w + prior_x
            b_y = (sigmoid(pred_y) - .5) / conv_h + prior_y
            b_w = prior_w * exp(loc_w)
            b_h = prior_h * exp(loc_h)

        Note that loc is inputed as [(s(x)-.5)/conv_w, (s(y)-.5)/conv_h, w, h]
        while priors are inputed as [x, y, w, h] where each coordinate
        is relative to size of the image (even sigmoid(x)). We do this
        in the network by dividing by the 'cell size', which is just
        the size of the convouts.

        Also note that prior_x and prior_y are center coordinates which
        is why we have to subtract .5 from sigmoid(pred_x and pred_y).

        Args:
            - loc:    The predicted bounding boxes of size [num_priors, 4]
            - priors: The priorbox coords with size [num_priors, 4]

        Returns: A tensor of decoded relative coordinates in point form
                form with size [num_priors, 4(x, y, w, h)]
        """

        variances = [0.1, 0.2]

        boxes = np.concatenate(
            (
                priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                priors[:, 2:] * np.exp(loc[:, 2:] * variances[1]),
            ),
            1,
        )
        boxes[:, :2] -= boxes[:, 2:] / 2
        # boxes[:, 2:] += boxes[:, :2]

        # crop
        np.where(boxes[:, 0] < 0, 0, boxes[:, 0])
        np.where(boxes[:, 1] < 0, 0, boxes[:, 1])
        np.where(boxes[:, 2] > 1, 1, boxes[:, 2])
        np.where(boxes[:, 3] > 1, 1, boxes[:, 3])

        # decode to img size
        boxes[:, 0] *= img_w
        boxes[:, 1] *= img_h
        boxes[:, 2] = boxes[:, 2] * img_w + 1
        boxes[:, 3] = boxes[:, 3] * img_h + 1

        return boxes

    def detect(self, conf_preds, loc_data, prior_data, mask_data, img_w, img_h):
        """ Perform nms for only the max scoring class that isn't background (class 0) """
        cur_scores = conf_preds[:, 1:]
        num_class = cur_scores.shape[1]

        classes = np.argmax(cur_scores, axis=1)
        conf_scores = cur_scores[range(cur_scores.shape[0]), classes]

        # filte by confidence_threshold
        keep = conf_scores > self.confidence_threshold
        conf_scores = conf_scores[keep]
        classes = classes[keep]
        loc_data = loc_data[keep, :]
        prior_data = prior_data[keep, :]
        masks = mask_data[keep, :]

        # decode x, y, w, h
        boxes = self.decode(loc_data, prior_data, img_w, img_h)

        # nms for every class
        boxes_result = []
        masks_result = []
        classes_result = []
        conf_scores_result = []
        for i in range(num_class):
            where = np.where(classes == i)
            if len(where) == 0:
                continue

            boxes_tmp = boxes[where]
            masks_tmp = masks[where]
            classes_tmp = classes[where]
            conf_scores_tmp = conf_scores[where]

            score_mask = conf_scores_tmp > self.confidence_threshold
            boxes_tmp = boxes_tmp[score_mask]
            masks_tmp = masks_tmp[score_mask]
            classes_tmp = classes_tmp[score_mask]
            conf_scores_tmp = conf_scores_tmp[score_mask]

            indexes = nms(
                boxes_tmp,
                conf_scores_tmp,
                iou_threshold=self.nms_threshold,
                top_k=self.keep_top_k,
            )

            for index in indexes:
                boxes_result.append(boxes_tmp[index])
                masks_result.append(masks_tmp[index])
                classes_result.append(classes_tmp[index] + 1)
                conf_scores_result.append(conf_scores_tmp[index])

        # keep top k
        if len(conf_scores_result) > self.keep_top_k:
            indexes = np.argsort(conf_scores_result)
            indexes = indexes[: self.keep_top_k]

            boxes_result = boxes_result[indexes]
            masks_result = masks_result[indexes]
            classes_result = classes_result[indexes]
            conf_scores_result = conf_scores_result[indexes]

        return (
            np.array(boxes_result),
            np.array(masks_result),
            np.array(classes_result),
            np.array(conf_scores_result),
        )
