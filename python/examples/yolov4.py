# Copyright 2020 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import sys
import cv2
from ncnn.model_zoo import get_model
from ncnn.utils import draw_detection_objects

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: %s [v4l input device or image]\n" % (sys.argv[0]))
        sys.exit(0)

    devicepath = sys.argv[1]

    net = get_model("yolov4_tiny", num_threads=4, use_gpu=True)
    # net = get_model("yolov4", num_threads=4, use_gpu=True)

    if devicepath.find("/dev/video") == -1:
        m = cv2.imread(devicepath)
        if m is None:
            print("cv2.imread %s failed\n" % (devicepath))
            sys.exit(0)

        objects = net(m)

        draw_detection_objects(m, net.class_names, objects)
    else:
        cap = cv2.VideoCapture(devicepath)

        if cap.isOpened() == False:
            print("Failed to open %s" % (devicepath))
            sys.exit(0)

        while True:
            ret, frame = cap.read()

            objects = net(frame)

            draw_detection_objects(frame, net.class_names, objects)
