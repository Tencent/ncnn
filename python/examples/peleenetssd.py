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

import sys
import cv2
import numpy as np
import ncnn
from ncnn.model_zoo import get_model


def draw_detection_objects_seg(image, class_names, objects, mat_map):
    color = [128, 255, 128, 244, 35, 232]
    color_count = len(color)

    for obj in objects:
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

    width = mat_map.w
    height = mat_map.h
    size = mat_map.c
    img_index2 = 0
    threshold = 0.45
    ptr2 = np.array(mat_map)
    for i in range(height):
        ptr1 = image[i].flatten()
        img_index1 = 0
        for j in range(width):
            maxima = threshold
            index = -1
            for c in range(size):
                # const float* ptr3 = ptr2 + c*width*height
                ptr3 = ptr2[c].flatten()
                if ptr3[img_index2] > maxima:
                    maxima = ptr3[img_index2]
                    index = c

            if index > -1:
                color_index = (index) * 3
                if color_index < color_count:
                    b = color[color_index]
                    g = color[color_index + 1]
                    r = color[color_index + 2]
                    ptr1[img_index1] = b / 2 + ptr1[img_index1] / 2
                    ptr1[img_index1 + 1] = g / 2 + ptr1[img_index1 + 1] / 2
                    ptr1[img_index1 + 2] = r / 2 + ptr1[img_index1 + 2] / 2

            img_index1 += 3
            img_index2 += 1

        image[i] = ptr1.reshape(image[i].shape)

    cv2.imshow("image", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: %s [imagepath]\n" % (sys.argv[0]))
        sys.exit(0)

    imagepath = sys.argv[1]

    m = cv2.imread(imagepath)
    if m is None:
        print("cv2.imread %s failed\n" % (imagepath))
        sys.exit(0)

    net = get_model("peleenet_ssd", num_threads=4, use_gpu=True)

    objects, seg_out = net(m)

    draw_detection_objects_seg(m, net.class_names, objects, seg_out)
