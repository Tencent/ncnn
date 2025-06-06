# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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
