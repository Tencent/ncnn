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

import time
import numpy as np
import ncnn
from .model_store import get_model_file
from ..utils.objects import Detect_Object
from ..utils.functional import *


class YoloV5Focus(ncnn.Layer):
    yolov5FocusLayers = []

    def __init__(self):
        ncnn.Layer.__init__(self)
        self.one_blob_only = True

        self.yolov5FocusLayers.append(self)

    def forward(self, bottom_blob, top_blob, opt):
        x = np.array(bottom_blob)
        x = np.concatenate(
            [
                x[..., ::2, ::2],
                x[..., 1::2, ::2],
                x[..., ::2, 1::2],
                x[..., 1::2, 1::2],
            ]
        )

        top_blob.clone_from(ncnn.Mat(x), opt.blob_allocator)
        if top_blob.empty():
            return -100

        return 0


def YoloV5Focus_layer_creator():
    return YoloV5Focus()


def YoloV5Focus_layer_destroyer(layer):
    for i in range(len(YoloV5Focus.yolov5FocusLayers)):
        if YoloV5Focus.yolov5FocusLayers[i] == layer:
            del YoloV5Focus.yolov5FocusLayers[i]
            break


class YoloV5s:
    def __init__(
        self,
        target_size=640,
        prob_threshold=0.25,
        nms_threshold=0.45,
        num_threads=1,
        use_gpu=False,
    ):
        self.target_size = target_size
        self.prob_threshold = prob_threshold
        self.nms_threshold = nms_threshold
        self.num_threads = num_threads
        self.use_gpu = use_gpu

        self.mean_vals = []
        self.norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = self.use_gpu
        self.net.opt.num_threads = self.num_threads

        self.net.register_custom_layer(
            "YoloV5Focus", YoloV5Focus_layer_creator, YoloV5Focus_layer_destroyer
        )

        # original pretrained model from https://github.com/ultralytics/yolov5
        # the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
        self.net.load_param(get_model_file("yolov5s.param"))
        self.net.load_model(get_model_file("yolov5s.bin"))

        self.grid = [make_grid(10, 6), make_grid(20, 12), make_grid(40, 24)]
        self.stride = np.array([32, 16, 8])
        self.anchor_grid = np.array(
            [
                [116, 90, 156, 198, 373, 326],
                [30, 61, 62, 45, 59, 119],
                [10, 13, 16, 30, 33, 23],
            ]
        ).reshape((3, 1, 3, 1, 1, 2))

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
            img, ncnn.Mat.PixelType.PIXEL_BGR2RGB, img_w, img_h, w, h
        )
        # pad to target_size rectangle
        # yolov5/utils/datasets.py letterbox
        wpad = (w + 31) // 32 * 32 - w
        hpad = (h + 31) // 32 * 32 - h
        mat_in_pad = ncnn.copy_make_border(
            mat_in,
            hpad // 2,
            hpad - hpad // 2,
            wpad // 2,
            wpad - wpad // 2,
            ncnn.BorderType.BORDER_CONSTANT,
            114.0,
        )

        mat_in_pad.substract_mean_normalize(self.mean_vals, self.norm_vals)

        ex = self.net.create_extractor()
        ex.input("images", mat_in_pad)

        # anchor setting from yolov5/models/yolov5s.yaml
        ret1, mat_out1 = ex.extract("output")  # stride 8
        ret2, mat_out2 = ex.extract("781")  # stride 16
        ret3, mat_out3 = ex.extract("801")  # stride 32

        pred = [np.array(mat_out3), np.array(mat_out2), np.array(mat_out1)]
        z = []
        for i in range(len(pred)):
            num_grid = pred[i].shape[1]
            if mat_in_pad.w > mat_in_pad.h:
                num_grid_x = mat_in_pad.w // self.stride[i]
                num_grid_y = num_grid // num_grid_x
            else:
                num_grid_y = mat_in_pad.h // self.stride[i]
                num_grid_x = num_grid // num_grid_y
            if (
                self.grid[i].shape[0] != num_grid_x
                or self.grid[i].shape[1] != num_grid_y
            ):
                self.grid[i] = make_grid(num_grid_x, num_grid_y)

            y = sigmoid(pred[i])
            y = y.reshape(pred[i].shape[0], num_grid_y, num_grid_x, pred[i].shape[2])
            y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[
                i
            ]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.reshape(1, -1, y.shape[-1]))
        pred = np.concatenate(z, 1)

        result = self.non_max_suppression(
            pred, self.prob_threshold, self.nms_threshold
        )[0]

        objects = [
            Detect_Object(
                obj[5],
                obj[4],
                obj[0] / scale,
                obj[1] / scale,
                (obj[2] - obj[0]) / scale,
                (obj[3] - obj[1]) / scale,
            )
            for obj in result
        ]

        return objects

    def non_max_suppression(
        self,
        prediction,
        conf_thres=0.1,
        iou_thres=0.6,
        merge=False,
        classes=None,
        agnostic=False,
    ):
        """Performs Non-Maximum Suppression (NMS) on inference results

        Returns:
            detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
        """
        nc = prediction[0].shape[1] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_det = 300  # maximum number of detections per image
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

        t = time.time()
        output = [None] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero()
                x = np.concatenate(
                    (box[i], x[i, j + 5, None], j[:, None].astype(np.float32)), axis=1
                )
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = np.concatenate((box, conf, j.float()), axis=1)[
                    conf.view(-1) > conf_thres
                ]

            # Filter by class
            if classes:
                x = x[(x[:, 5:6] == np.array(classes)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # If none remain process next image
            n = x.shape[0]  # number of boxes
            if not n:
                continue

            # Sort by confidence
            # x = x[x[:, 4].argsort(descending=True)]

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = nms(boxes, scores, iou_threshold=iou_thres)
            if len(i) > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
                try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                    iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                    weights = iou * scores[None]  # box weights
                    x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                        1, keepdim=True
                    )  # merged boxes
                    if redundant:
                        i = i[iou.sum(1) > 1]  # require redundancy
                except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                    print(x, i, x.shape, i.shape)
                    pass

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                break  # time limit exceeded

        return output
