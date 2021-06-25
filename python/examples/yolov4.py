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
