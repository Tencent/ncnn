# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import ncnn


def check_gpuinfo(gpuinfo):
    assert gpuinfo.api_version() > 0
    assert gpuinfo.driver_version() > 0
    assert gpuinfo.vendor_id() > 0
    assert gpuinfo.device_id() > 0
    assert gpuinfo.pipeline_cache_uuid() is not None
    assert gpuinfo.type() >= 0


def test_gpu_api():
    if not hasattr(ncnn, "get_gpu_count"):
        return

    assert ncnn.create_gpu_instance() == 0
    assert ncnn.get_gpu_count() > 0
    assert ncnn.get_default_gpu_index() >= 0

    gpuinfo = ncnn.get_gpu_info(0)
    check_gpuinfo(gpuinfo)

    vkdev = ncnn.get_gpu_device(0)
    assert vkdev is not None
    gpuinfo = vkdev.info()
    check_gpuinfo(gpuinfo)

    ncnn.destroy_gpu_instance()


def test_vulkan_device():
    if not hasattr(ncnn, "get_gpu_count"):
        return

    vkdev = ncnn.VulkanDevice(0)
    assert vkdev is not None
    gpuinfo = vkdev.info()
    check_gpuinfo(gpuinfo)
