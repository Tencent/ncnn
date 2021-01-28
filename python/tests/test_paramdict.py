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

import pytest

import ncnn


def test_paramdict():
    pd = ncnn.ParamDict()
    assert pd.type(0) == 0
    assert pd.get(0, -1) == -1

    pd.set(1, 1)
    assert pd.type(1) == 2 and pd.get(1, -1) == 1

    pd.set(2, 2.0)
    assert pd.type(2) == 3 and pd.get(2, -2.0) == 2.0

    mat = ncnn.Mat(1)
    pd.set(3, mat)
    assert pd.type(3) == 4 and pd.get(3, ncnn.Mat()).dims == mat.dims
