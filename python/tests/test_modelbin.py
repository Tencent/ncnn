# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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
import pytest

import ncnn

def test_modelbin():

    class CustomModelBin(ncnn.ModelBin):
        def __init__(self, weights):
            ncnn.ModelBin.__init__(self)
            self.weights = weights
            self.index = 0

        def load(self, w, type):
            m = ncnn.Mat(self.weights[self.index])
            self.index = self.index + 1
            return m

    weights = [
        np.array([123], dtype=np.float32),
        np.array([233, 456], dtype=np.float32)
    ]

    mb = CustomModelBin(weights)

    b0 = np.array(mb.load(1, 1))
    b1 = np.array(mb.load(2, 1))

    assert b0 == weights[0] and b1.all() == weights[1].all()
