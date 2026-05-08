# Copyright 2020 Tencent
# SPDX-License-Identifier: BSD-3-Clause

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
