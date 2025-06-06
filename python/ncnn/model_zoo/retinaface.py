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
from ..utils.objects import Point, Face_Object


class RetinaFace:
    def __init__(
        self, prob_threshold=0.8, nms_threshold=0.4, num_threads=1, use_gpu=False
    ):
        self.prob_threshold = prob_threshold
        self.nms_threshold = nms_threshold
        self.num_threads = num_threads
        self.use_gpu = use_gpu

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = self.use_gpu

        # model is converted from
        # https://github.com/deepinsight/insightface/tree/master/RetinaFace#retinaface-pretrained-models
        # https://github.com/deepinsight/insightface/issues/669
        # the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
        self.net.load_param(get_model_file("mnet.25-opt.param"))
        self.net.load_model(get_model_file("mnet.25-opt.bin"))

    def __del__(self):
        self.net = None

    def __call__(self, img):
        img_h = img.shape[0]
        img_w = img.shape[1]

        mat_in = ncnn.Mat.from_pixels(
            img, ncnn.Mat.PixelType.PIXEL_BGR2RGB, img_w, img_h
        )

        ex = self.net.create_extractor()
        ex.set_num_threads(self.num_threads)

        ex.input("data", mat_in)

        faceobjects32 = self.detect_stride32(ex)
        faceobjects16 = self.detect_stride16(ex)
        faceobjects8 = self.detect_stride8(ex)

        faceproposals = [*faceobjects32, *faceobjects16, *faceobjects8]

        # sort all proposals by score from highest to lowest
        faceproposals.sort(key=lambda obj: obj.prob, reverse=True)

        # apply nms with nms_threshold
        picked = self.nms_sorted_bboxes(faceproposals, self.nms_threshold)

        face_count = len(picked)

        faceobjects = []
        for i in range(face_count):
            faceobjects.append(faceproposals[picked[i]])

            # clip to image size
            x0 = faceobjects[i].rect.x
            y0 = faceobjects[i].rect.y
            x1 = x0 + faceobjects[i].rect.w
            y1 = y0 + faceobjects[i].rect.h

            x0 = np.maximum(np.minimum(x0, float(img_w) - 1), 0.0)
            y0 = np.maximum(np.minimum(y0, float(img_h) - 1), 0.0)
            x1 = np.maximum(np.minimum(x1, float(img_w) - 1), 0.0)
            y1 = np.maximum(np.minimum(y1, float(img_h) - 1), 0.0)

            faceobjects[i].rect.x = x0
            faceobjects[i].rect.y = y0
            faceobjects[i].rect.w = x1 - x0
            faceobjects[i].rect.h = y1 - y0

        return faceobjects

    def detect_stride32(self, ex):
        ret1, score_blob = ex.extract("face_rpn_cls_prob_reshape_stride32")
        ret2, bbox_blob = ex.extract("face_rpn_bbox_pred_stride32")
        ret3, landmark_blob = ex.extract("face_rpn_landmark_pred_stride32")

        base_size = 16
        feat_stride = 32
        ratios = ncnn.Mat(1)
        ratios[0] = 1.0
        scales = ncnn.Mat(2)
        scales[0] = 32.0
        scales[1] = 16.0
        anchors = self.generate_anchors(base_size, ratios, scales)

        faceobjects32 = self.generate_proposals(
            anchors,
            feat_stride,
            score_blob,
            bbox_blob,
            landmark_blob,
            self.prob_threshold,
        )

        return faceobjects32

    def detect_stride16(self, ex):
        ret1, score_blob = ex.extract("face_rpn_cls_prob_reshape_stride16")
        ret2, bbox_blob = ex.extract("face_rpn_bbox_pred_stride16")
        ret3, landmark_blob = ex.extract("face_rpn_landmark_pred_stride16")

        base_size = 16
        feat_stride = 16
        ratios = ncnn.Mat(1)
        ratios[0] = 1.0
        scales = ncnn.Mat(2)
        scales[0] = 8.0
        scales[1] = 4.0
        anchors = self.generate_anchors(base_size, ratios, scales)

        faceobjects16 = self.generate_proposals(
            anchors,
            feat_stride,
            score_blob,
            bbox_blob,
            landmark_blob,
            self.prob_threshold,
        )

        return faceobjects16

    def detect_stride8(self, ex):
        ret1, score_blob = ex.extract("face_rpn_cls_prob_reshape_stride8")
        ret2, bbox_blob = ex.extract("face_rpn_bbox_pred_stride8")
        ret3, landmark_blob = ex.extract("face_rpn_landmark_pred_stride8")

        base_size = 16
        feat_stride = 8
        ratios = ncnn.Mat(1)
        ratios[0] = 1.0
        scales = ncnn.Mat(2)
        scales[0] = 2.0
        scales[1] = 1.0
        anchors = self.generate_anchors(base_size, ratios, scales)

        faceobjects8 = self.generate_proposals(
            anchors,
            feat_stride,
            score_blob,
            bbox_blob,
            landmark_blob,
            self.prob_threshold,
        )

        return faceobjects8

    def generate_anchors(self, base_size, ratios, scales):
        num_ratio = ratios.w
        num_scale = scales.w

        # anchors = ncnn.Mat()
        # anchors.create(w=4, h=num_ratio * num_scale)

        anchors_np = np.zeros((2, 4), dtype=np.float32)

        cx = base_size * 0.5
        cy = base_size * 0.5

        for i in range(num_ratio):
            ar = ratios[i]

            r_w = np.round(base_size / np.sqrt(ar))
            r_h = np.round(r_w * ar)  # round(base_size * np.sqrt(ar))

            for j in range(num_scale):
                scale = scales[j]

                rs_w = r_w * scale
                rs_h = r_h * scale

                anchor = anchors_np[i * num_scale + j]

                anchor[0] = cx - rs_w * 0.5
                anchor[1] = cy - rs_h * 0.5
                anchor[2] = cx + rs_w * 0.5
                anchor[3] = cy + rs_h * 0.5

        anchors = ncnn.Mat(anchors_np)
        return anchors

    def generate_proposals(
        self, anchors, feat_stride, score_blob, bbox_blob, landmark_blob, prob_threshold
    ):
        faceobjects = []

        w = score_blob.w
        h = score_blob.h

        # generate face proposal from bbox deltas and shifted anchors
        num_anchors = anchors.h

        for q in range(num_anchors):
            anchor = anchors.row(q)

            score = score_blob.channel(q + num_anchors)
            bbox = bbox_blob.channel_range(q * 4, 4)
            landmark = landmark_blob.channel_range(q * 10, 10)

            # shifted anchor
            anchor_y = anchor[1]

            anchor_w = anchor[2] - anchor[0]
            anchor_h = anchor[3] - anchor[1]

            for i in range(h):
                anchor_x = anchor[0]

                for j in range(w):
                    index = i * w + j

                    prob = score[index]

                    if prob >= prob_threshold:
                        # apply center size
                        dx = bbox.channel(0)[index]
                        dy = bbox.channel(1)[index]
                        dw = bbox.channel(2)[index]
                        dh = bbox.channel(3)[index]

                        cx = anchor_x + anchor_w * 0.5
                        cy = anchor_y + anchor_h * 0.5

                        pb_cx = cx + anchor_w * dx
                        pb_cy = cy + anchor_h * dy

                        pb_w = anchor_w * np.exp(dw)
                        pb_h = anchor_h * np.exp(dh)

                        x0 = pb_cx - pb_w * 0.5
                        y0 = pb_cy - pb_h * 0.5
                        x1 = pb_cx + pb_w * 0.5
                        y1 = pb_cy + pb_h * 0.5

                        obj = Face_Object()
                        obj.rect.x = x0
                        obj.rect.y = y0
                        obj.rect.w = x1 - x0 + 1
                        obj.rect.h = y1 - y0 + 1
                        obj.landmark = [Point(), Point(), Point(), Point(), Point()]
                        obj.landmark[0].x = (
                            cx + (anchor_w + 1) * landmark.channel(0)[index]
                        )
                        obj.landmark[0].y = (
                            cy + (anchor_h + 1) * landmark.channel(1)[index]
                        )
                        obj.landmark[1].x = (
                            cx + (anchor_w + 1) * landmark.channel(2)[index]
                        )
                        obj.landmark[1].y = (
                            cy + (anchor_h + 1) * landmark.channel(3)[index]
                        )
                        obj.landmark[2].x = (
                            cx + (anchor_w + 1) * landmark.channel(4)[index]
                        )
                        obj.landmark[2].y = (
                            cy + (anchor_h + 1) * landmark.channel(5)[index]
                        )
                        obj.landmark[3].x = (
                            cx + (anchor_w + 1) * landmark.channel(6)[index]
                        )
                        obj.landmark[3].y = (
                            cy + (anchor_h + 1) * landmark.channel(7)[index]
                        )
                        obj.landmark[4].x = (
                            cx + (anchor_w + 1) * landmark.channel(8)[index]
                        )
                        obj.landmark[4].y = (
                            cy + (anchor_h + 1) * landmark.channel(9)[index]
                        )
                        obj.prob = prob

                        faceobjects.append(obj)

                    anchor_x += feat_stride

                anchor_y += feat_stride

        return faceobjects

    def nms_sorted_bboxes(self, faceobjects, nms_threshold):
        picked = []

        n = len(faceobjects)

        areas = []
        for i in range(n):
            areas.append(faceobjects[i].rect.area())

        for i in range(n):
            a = faceobjects[i]

            keep = True
            for j in range(len(picked)):
                b = faceobjects[picked[j]]

                # intersection over union
                inter_area = a.rect.intersection_area(b.rect)
                union_area = areas[i] + areas[picked[j]] - inter_area
                # float IoU = inter_area / union_area
                if inter_area / union_area > nms_threshold:
                    keep = False

            if keep:
                picked.append(i)

        return picked
