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


class Point(object):
    def __init__(self):
        self.x = 0.0
        self.y = 0.0


class Rect(object):
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.w = 0.0
        self.h = 0.0

    def area(self):
        return self.w * self.h

    def intersection_area(self, b):
        x1 = np.maximum(self.x, b.x)
        y1 = np.maximum(self.y, b.y)
        x2 = np.minimum(self.x + self.w, b.x + b.w)
        y2 = np.minimum(self.y + self.h, b.y + b.h)
        return np.abs(x1 - x2) * np.abs(y1 - y2)


class Detect_Object(object):
    def __init__(self):
        self.label = 0
        self.prob = 0.0
        self.rect = Rect()


class Face_Object(object):
    def __init__(self):
        self.prob = 0.0
        self.rect = Rect()
        self.landmark = []


class KeyPoint(object):
    def __init__(self):
        self.p = Point()
        self.prob = 0.0
