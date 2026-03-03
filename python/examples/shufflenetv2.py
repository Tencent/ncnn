# Copyright 2020 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import sys
import cv2
from ncnn.model_zoo import get_model
from ncnn.utils import print_topk

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: %s [imagepath]\n" % (sys.argv[0]))
        sys.exit(0)

    imagepath = sys.argv[1]

    m = cv2.imread(imagepath)
    if m is None:
        print("cv2.imread %s failed\n" % (imagepath))
        sys.exit(0)

    net = get_model("shufflenetv2", num_threads=4, use_gpu=True)

    cls_scores = net(m)

    print_topk(cls_scores, 3)
