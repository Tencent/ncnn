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
import cv2
from .objects import Detect_Object, Face_Object


def draw_detection_objects(image, class_names, objects, min_prob=0.0):
    for obj in objects:
        if obj.prob < min_prob:
            continue

        print(
            "%d = %.5f at %.2f %.2f %.2f x %.2f\n"
            % (obj.label, obj.prob, obj.rect.x, obj.rect.y, obj.rect.w, obj.rect.h)
        )

        cv2.rectangle(
            image,
            (int(obj.rect.x), int(obj.rect.y)),
            (int(obj.rect.x + obj.rect.w), int(obj.rect.y + obj.rect.h)),
            (255, 0, 0),
        )

        text = "%s %.1f%%" % (class_names[int(obj.label)], obj.prob * 100)

        label_size, baseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        x = obj.rect.x
        y = obj.rect.y - label_size[1] - baseLine
        if y < 0:
            y = 0
        if x + label_size[0] > image.shape[1]:
            x = image.shape[1] - label_size[0]

        cv2.rectangle(
            image,
            (int(x), int(y)),
            (int(x + label_size[0]), int(y + label_size[1] + baseLine)),
            (255, 255, 255),
            -1,
        )

        cv2.putText(
            image,
            text,
            (int(x), int(y + label_size[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
        )

    cv2.imshow("image", image)
    cv2.waitKey(0)


def print_topk(cls_scores, topk):
    indexes = np.argsort(cls_scores)[::-1][0:topk]
    scores = cls_scores[indexes]

    for index, score in zip(indexes, scores):
        print("%d=%f" % (index, score))


def draw_faceobjects(image, faceobjects):
    for obj in faceobjects:
        print(
            "%.5f at %.2f %.2f %.2f x %.2f"
            % (obj.prob, obj.rect.x, obj.rect.y, obj.rect.w, obj.rect.h)
        )

        cv2.rectangle(
            image,
            (int(obj.rect.x), int(obj.rect.y)),
            (int(obj.rect.x + obj.rect.w), int(obj.rect.y + obj.rect.h)),
            (255, 0, 0),
        )

        cv2.circle(
            image,
            (int(obj.landmark[0].x), int(obj.landmark[0].y)),
            2,
            (0, 255, 255),
            -1,
        )
        cv2.circle(
            image,
            (int(obj.landmark[1].x), int(obj.landmark[1].y)),
            2,
            (0, 255, 255),
            -1,
        )
        cv2.circle(
            image,
            (int(obj.landmark[2].x), int(obj.landmark[2].y)),
            2,
            (0, 255, 255),
            -1,
        )
        cv2.circle(
            image,
            (int(obj.landmark[3].x), int(obj.landmark[3].y)),
            2,
            (0, 255, 255),
            -1,
        )
        cv2.circle(
            image,
            (int(obj.landmark[4].x), int(obj.landmark[4].y)),
            2,
            (0, 255, 255),
            -1,
        )

        text = "%.1f%%" % (obj.prob * 100)

        label_size, baseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        x = obj.rect.x
        y = obj.rect.y - label_size[1] - baseLine
        if y < 0:
            y = 0
        if x + label_size[0] > image.shape[1]:
            x = image.shape[1] - label_size[0]

        cv2.rectangle(
            image,
            (int(x), int(y)),
            (int(x + label_size[0]), int(y + label_size[1] + baseLine)),
            (255, 255, 255),
            -1,
        )

        cv2.putText(
            image,
            text,
            (int(x), int(y + label_size[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
        )

    cv2.imshow("image", image)
    cv2.waitKey(0)


def draw_pose(image, keypoints):
    # draw bone
    joint_pairs = [
        (0, 1),
        (1, 3),
        (0, 2),
        (2, 4),
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (12, 14),
        (13, 15),
        (14, 16),
    ]

    for i in range(16):
        p1 = keypoints[joint_pairs[i][0]]
        p2 = keypoints[joint_pairs[i][1]]

        if p1.prob < 0.2 or p2.prob < 0.2:
            continue

        cv2.line(
            image,
            (int(p1.p.x), int(p1.p.y)),
            (int(p2.p.x), int(p2.p.y)),
            (255, 0, 0),
            2,
        )

    # draw joint
    for keypoint in keypoints:
        print("%.2f %.2f = %.5f" % (keypoint.p.x, keypoint.p.y, keypoint.prob))

        if keypoint.prob < 0.2:
            continue

        cv2.circle(image, (int(keypoint.p.x), int(keypoint.p.y)), 3, (0, 255, 0), -1)

    cv2.imshow("image", image)
    cv2.waitKey(0)
