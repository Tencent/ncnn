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
import time
import ncnn

param_root = "../benchmark/"

g_warmup_loop_count = 8
g_loop_count = 4
g_enable_cooling_down = True

g_vkdev = None
g_blob_vkallocator = None
g_staging_vkallocator = None

g_blob_pool_allocator = ncnn.UnlockedPoolAllocator()
g_workspace_pool_allocator = ncnn.PoolAllocator()


def benchmark(comment, _in, opt):
    _in.fill(0.01)

    g_blob_pool_allocator.clear()
    g_workspace_pool_allocator.clear()

    if opt.use_vulkan_compute:
        g_blob_vkallocator.clear()
        g_staging_vkallocator.clear()

    net = ncnn.Net()
    net.opt = opt

    if net.opt.use_vulkan_compute:
        net.set_vulkan_device(g_vkdev)

    net.load_param(param_root + comment + ".param")

    dr = ncnn.DataReaderFromEmpty()
    net.load_model(dr)

    input_names = net.input_names()
    output_names = net.output_names()

    if g_enable_cooling_down:
        time.sleep(10)

    # warm up
    for i in range(g_warmup_loop_count):
        # test with statement
        with net.create_extractor() as ex:
            ex.input(input_names[0], _in)
            ex.extract(output_names[0])

    time_min = sys.float_info.max
    time_max = -sys.float_info.max
    time_avg = 0.0

    for i in range(g_loop_count):
        start = time.time()

        # test net keep alive until ex freed
        ex = net.create_extractor()
        ex.input(input_names[0], _in)
        ex.extract(output_names[0])

        end = time.time()

        timespan = end - start

        time_min = timespan if timespan < time_min else time_min
        time_max = timespan if timespan > time_max else time_max
        time_avg += timespan

    time_avg /= g_loop_count

    print(
        "%20s  min = %7.2f  max = %7.2f  avg = %7.2f"
        % (comment, time_min * 1000, time_max * 1000, time_avg * 1000)
    )


if __name__ == "__main__":
    loop_count = 4
    num_threads = ncnn.get_cpu_count()
    powersave = 0
    gpu_device = -1
    cooling_down = 1

    argc = len(sys.argv)
    if argc >= 2:
        loop_count = int(sys.argv[1])
    if argc >= 3:
        num_threads = int(sys.argv[2])
    if argc >= 4:
        powersave = int(sys.argv[3])
    if argc >= 5:
        gpu_device = int(sys.argv[4])
    if argc >= 6:
        cooling_down = int(sys.argv[5])

    use_vulkan_compute = gpu_device != -1

    g_enable_cooling_down = cooling_down != 0

    g_loop_count = loop_count

    g_blob_pool_allocator.set_size_compare_ratio(0.0)
    g_workspace_pool_allocator.set_size_compare_ratio(0.5)

    if use_vulkan_compute:
        g_warmup_loop_count = 10

        g_vkdev = ncnn.get_gpu_device(gpu_device)

        g_blob_vkallocator = ncnn.VkBlobAllocator(g_vkdev)
        g_staging_vkallocator = ncnn.VkStagingAllocator(g_vkdev)

    opt = ncnn.Option()
    opt.lightmode = True
    opt.num_threads = num_threads
    opt.blob_allocator = g_blob_pool_allocator
    opt.workspace_allocator = g_workspace_pool_allocator
    if use_vulkan_compute:
        opt.blob_vkallocator = g_blob_vkallocator
        opt.workspace_vkallocator = g_blob_vkallocator
        opt.staging_vkallocator = g_staging_vkallocator
    opt.use_winograd_convolution = True
    opt.use_sgemm_convolution = True
    opt.use_int8_inference = True
    opt.use_vulkan_compute = use_vulkan_compute
    opt.use_fp16_packed = True
    opt.use_fp16_storage = True
    opt.use_fp16_arithmetic = True
    opt.use_int8_storage = True
    opt.use_int8_arithmetic = True
    opt.use_packing_layout = True
    opt.use_shader_pack8 = False
    opt.use_image_storage = False

    ncnn.set_cpu_powersave(powersave)
    ncnn.set_omp_dynamic(0)
    ncnn.set_omp_num_threads(num_threads)

    print("loop_count =", loop_count)
    print("num_threads =", num_threads)
    print("powersave =", ncnn.get_cpu_powersave())
    print("gpu_device =", gpu_device)
    print("cooling_down =", g_enable_cooling_down)

    benchmark("squeezenet", ncnn.Mat((227, 227, 3)), opt)
    benchmark("squeezenet_int8", ncnn.Mat((227, 227, 3)), opt)
    benchmark("mobilenet", ncnn.Mat((224, 224, 3)), opt)
    benchmark("mobilenet_int8", ncnn.Mat((224, 224, 3)), opt)
    benchmark("mobilenet_v2", ncnn.Mat((224, 224, 3)), opt)
    # benchmark("mobilenet_v2_int8", ncnn.Mat(w=224, h=224, c=3), opt)
    benchmark("mobilenet_v3", ncnn.Mat((224, 224, 3)), opt)
    benchmark("shufflenet", ncnn.Mat((224, 224, 3)), opt)
    benchmark("shufflenet_v2", ncnn.Mat((224, 224, 3)), opt)
    benchmark("mnasnet", ncnn.Mat((224, 224, 3)), opt)
    benchmark("proxylessnasnet", ncnn.Mat((224, 224, 3)), opt)
    benchmark("efficientnet_b0", ncnn.Mat((224, 224, 3)), opt)
    benchmark("regnety_400m", ncnn.Mat((224, 224, 3)), opt)
    benchmark("blazeface", ncnn.Mat((128, 128, 3)), opt)
    benchmark("googlenet", ncnn.Mat((224, 224, 3)), opt)
    benchmark("googlenet_int8", ncnn.Mat((224, 224, 3)), opt)
    benchmark("resnet18", ncnn.Mat((224, 224, 3)), opt)
    benchmark("resnet18_int8", ncnn.Mat((224, 224, 3)), opt)
    benchmark("alexnet", ncnn.Mat((227, 227, 3)), opt)
    benchmark("vgg16", ncnn.Mat((224, 224, 3)), opt)
    benchmark("vgg16_int8", ncnn.Mat((224, 224, 3)), opt)
    benchmark("resnet50", ncnn.Mat((224, 224, 3)), opt)
    benchmark("resnet50_int8", ncnn.Mat((224, 224, 3)), opt)
    benchmark("squeezenet_ssd", ncnn.Mat((300, 300, 3)), opt)
    benchmark("squeezenet_ssd_int8", ncnn.Mat((300, 300, 3)), opt)
    benchmark("mobilenet_ssd", ncnn.Mat((300, 300, 3)), opt)
    benchmark("mobilenet_ssd_int8", ncnn.Mat((300, 300, 3)), opt)
    benchmark("mobilenet_yolo", ncnn.Mat((416, 416, 3)), opt)
    benchmark("mobilenetv2_yolov3", ncnn.Mat((352, 352, 3)), opt)
    benchmark("yolov4-tiny", ncnn.Mat((416, 416, 3)), opt)
