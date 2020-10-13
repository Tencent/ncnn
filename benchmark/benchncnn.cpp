// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <float.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <algorithm>
#include <windows.h> // Sleep()
#else
#include <unistd.h> // sleep()
#endif

#include "benchmark.h"
#include "cpu.h"
#include "datareader.h"
#include "net.h"
#include "gpu.h"

class DataReaderFromEmpty : public ncnn::DataReader
{
public:
    virtual int scan(const char* format, void* p) const
    {
        return 0;
    }
    virtual size_t read(void* buf, size_t size) const
    {
        memset(buf, 0, size);
        return size;
    }
};

static int g_warmup_loop_count = 8;
static int g_loop_count = 4;
static bool g_enable_cooling_down = true;

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

#if NCNN_VULKAN
static ncnn::VulkanDevice* g_vkdev = 0;
static ncnn::VkAllocator* g_blob_vkallocator = 0;
static ncnn::VkAllocator* g_staging_vkallocator = 0;
#endif // NCNN_VULKAN

void benchmark(const char* comment, const ncnn::Mat& _in, const ncnn::Option& opt)
{
    ncnn::Mat in = _in;
    in.fill(0.01f);

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();

#if NCNN_VULKAN
    if (opt.use_vulkan_compute)
    {
        g_blob_vkallocator->clear();
        g_staging_vkallocator->clear();
    }
#endif // NCNN_VULKAN

    ncnn::Net net;

    net.opt = opt;

#if NCNN_VULKAN
    if (net.opt.use_vulkan_compute)
    {
        net.set_vulkan_device(g_vkdev);
    }
#endif // NCNN_VULKAN

    char parampath[256];
    sprintf(parampath, "%s.param", comment);
    net.load_param(parampath);

    DataReaderFromEmpty dr;
    net.load_model(dr);

    if (g_enable_cooling_down)
    {
        // sleep 10 seconds for cooling down SOC  :(
#ifdef _WIN32
        Sleep(10 * 1000);
#elif defined(__unix__) || defined(__APPLE__)
        sleep(10);
#elif _POSIX_TIMERS
        struct timespec ts;
        ts.tv_sec = 10;
        ts.tv_nsec = 0;
        nanosleep(&ts, &ts);
#else
        // TODO How to handle it ?
#endif
    }

    ncnn::Mat out;

    // warm up
    for (int i = 0; i < g_warmup_loop_count; i++)
    {
        ncnn::Extractor ex = net.create_extractor();
        ex.input("data", in);
        ex.extract("output", out);
    }

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;

    for (int i = 0; i < g_loop_count; i++)
    {
        double start = ncnn::get_current_time();

        {
            ncnn::Extractor ex = net.create_extractor();
            ex.input("data", in);
            ex.extract("output", out);
        }

        double end = ncnn::get_current_time();

        double time = end - start;

        time_min = std::min(time_min, time);
        time_max = std::max(time_max, time);
        time_avg += time;
    }

    time_avg /= g_loop_count;

    fprintf(stderr, "%20s  min = %7.2f  max = %7.2f  avg = %7.2f\n", comment, time_min, time_max, time_avg);
}

int main(int argc, char** argv)
{
    int loop_count = 4;
    int num_threads = ncnn::get_cpu_count();
    int powersave = 0;
    int gpu_device = -1;
    int cooling_down = 1;

    if (argc >= 2)
    {
        loop_count = atoi(argv[1]);
    }
    if (argc >= 3)
    {
        num_threads = atoi(argv[2]);
    }
    if (argc >= 4)
    {
        powersave = atoi(argv[3]);
    }
    if (argc >= 5)
    {
        gpu_device = atoi(argv[4]);
    }
    if (argc >= 6)
    {
        cooling_down = atoi(argv[5]);
    }

    bool use_vulkan_compute = gpu_device != -1;

    g_enable_cooling_down = cooling_down != 0;

    g_loop_count = loop_count;

    g_blob_pool_allocator.set_size_compare_ratio(0.0f);
    g_workspace_pool_allocator.set_size_compare_ratio(0.5f);

#if NCNN_VULKAN
    if (use_vulkan_compute)
    {
        g_warmup_loop_count = 10;

        g_vkdev = ncnn::get_gpu_device(gpu_device);

        g_blob_vkallocator = new ncnn::VkBlobAllocator(g_vkdev);
        g_staging_vkallocator = new ncnn::VkStagingAllocator(g_vkdev);
    }
#endif // NCNN_VULKAN

    // default option
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = num_threads;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
#if NCNN_VULKAN
    opt.blob_vkallocator = g_blob_vkallocator;
    opt.workspace_vkallocator = g_blob_vkallocator;
    opt.staging_vkallocator = g_staging_vkallocator;
#endif // NCNN_VULKAN
    opt.use_winograd_convolution = true;
    opt.use_sgemm_convolution = true;
    opt.use_int8_inference = true;
    opt.use_vulkan_compute = use_vulkan_compute;
    opt.use_fp16_packed = true;
    opt.use_fp16_storage = true;
    opt.use_fp16_arithmetic = true;
    opt.use_int8_storage = true;
    opt.use_int8_arithmetic = true;
    opt.use_packing_layout = true;
    opt.use_shader_pack8 = false;
    opt.use_image_storage = false;

    ncnn::set_cpu_powersave(powersave);

    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);

    fprintf(stderr, "loop_count = %d\n", g_loop_count);
    fprintf(stderr, "num_threads = %d\n", num_threads);
    fprintf(stderr, "powersave = %d\n", ncnn::get_cpu_powersave());
    fprintf(stderr, "gpu_device = %d\n", gpu_device);
    fprintf(stderr, "cooling_down = %d\n", (int)g_enable_cooling_down);

    // run
    benchmark("squeezenet", ncnn::Mat(227, 227, 3), opt);

    benchmark("squeezenet_int8", ncnn::Mat(227, 227, 3), opt);

    benchmark("mobilenet", ncnn::Mat(224, 224, 3), opt);

    benchmark("mobilenet_int8", ncnn::Mat(224, 224, 3), opt);

    benchmark("mobilenet_v2", ncnn::Mat(224, 224, 3), opt);

    // benchmark("mobilenet_v2_int8", ncnn::Mat(224, 224, 3), opt);

    benchmark("mobilenet_v3", ncnn::Mat(224, 224, 3), opt);

    benchmark("shufflenet", ncnn::Mat(224, 224, 3), opt);

    benchmark("shufflenet_v2", ncnn::Mat(224, 224, 3), opt);

    benchmark("mnasnet", ncnn::Mat(224, 224, 3), opt);

    benchmark("proxylessnasnet", ncnn::Mat(224, 224, 3), opt);

    benchmark("efficientnet_b0", ncnn::Mat(224, 224, 3), opt);

    benchmark("regnety_400m", ncnn::Mat(224, 224, 3), opt);

    benchmark("blazeface", ncnn::Mat(128, 128, 3), opt);

    benchmark("googlenet", ncnn::Mat(224, 224, 3), opt);

    benchmark("googlenet_int8", ncnn::Mat(224, 224, 3), opt);

    benchmark("resnet18", ncnn::Mat(224, 224, 3), opt);

    benchmark("resnet18_int8", ncnn::Mat(224, 224, 3), opt);

    benchmark("alexnet", ncnn::Mat(227, 227, 3), opt);

    benchmark("vgg16", ncnn::Mat(224, 224, 3), opt);

    benchmark("vgg16_int8", ncnn::Mat(224, 224, 3), opt);

    benchmark("resnet50", ncnn::Mat(224, 224, 3), opt);

    benchmark("resnet50_int8", ncnn::Mat(224, 224, 3), opt);

    benchmark("squeezenet_ssd", ncnn::Mat(300, 300, 3), opt);

    benchmark("squeezenet_ssd_int8", ncnn::Mat(300, 300, 3), opt);

    benchmark("mobilenet_ssd", ncnn::Mat(300, 300, 3), opt);

    benchmark("mobilenet_ssd_int8", ncnn::Mat(300, 300, 3), opt);

    benchmark("mobilenet_yolo", ncnn::Mat(416, 416, 3), opt);

    benchmark("mobilenetv2_yolov3", ncnn::Mat(352, 352, 3), opt);

    benchmark("yolov4-tiny", ncnn::Mat(416, 416, 3), opt);

#if NCNN_VULKAN
    delete g_blob_vkallocator;
    delete g_staging_vkallocator;
#endif // NCNN_VULKAN

    return 0;
}
