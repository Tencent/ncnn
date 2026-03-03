# Copyright 2020 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import sys
import cv2
from ncnn.model_zoo import get_model
from ncnn.utils import draw_detection_objects

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: %s [imagepath]\n" % (sys.argv[0]))
        sys.exit(0)

    imagepath = sys.argv[1]

    m = cv2.imread(imagepath)
    if m is None:
        print("cv2.imread %s failed\n" % (imagepath))
        sys.exit(0)

    net = get_model(
        "yolov8s",
        target_size=640,
        prob_threshold=0.25,
        nms_threshold=0.45,
        num_threads=4,
        use_gpu=True,
    )

    objects = net(m)

    draw_detection_objects(m, net.class_names, objects)
