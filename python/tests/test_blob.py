# Copyright 2020 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import ncnn


def test_blob():
    blob = ncnn.Blob()

    blob.name = "myblob"
    assert blob.name == "myblob"

    blob.producer = 0
    assert blob.producer == 0

    blob.consumer = 0
    assert blob.consumer == 0

    blob.shape = ncnn.Mat(1)
    assert blob.shape.dims == 1 and blob.shape.w == 1
