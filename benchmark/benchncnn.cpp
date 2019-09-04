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

#ifdef _WIN32
#include <algorithm>
#include <windows.h> // Sleep()
#else
#include <unistd.h> // sleep()
#endif

#include "benchmark.h"
#include "cpu.h"
#include "net.h"

#if NCNN_VULKAN
#include "gpu.h"

class GlobalGpuInstance
{
public:
    GlobalGpuInstance() { ncnn::create_gpu_instance(); }
    ~GlobalGpuInstance() { ncnn::destroy_gpu_instance(); }
};
// initialize vulkan runtime before main()
GlobalGpuInstance g_global_gpu_instance;
#endif // NCNN_VULKAN

namespace ncnn {

// always return empty weights
class ModelBinFromEmpty : public ModelBin
{
public:
    virtual Mat load(int w, int /*type*/) const { return Mat(w); }
};

class BenchNet : public Net
{
public:
    int load_model()
    {
        // load file
        int ret = 0;

        ModelBinFromEmpty mb;
        for (size_t i=0; i<layers.size(); i++)
        {
            Layer* layer = layers[i];

            int lret = layer->load_model(mb);
            if (lret != 0)
            {
                fprintf(stderr, "layer load_model %d failed\n", (int)i);
                ret = -1;
                break;
            }

            int cret = layer->create_pipeline(opt);
            if (cret != 0)
            {
                fprintf(stderr, "layer create_pipeline %d failed\n", (int)i);
                ret = -1;
                break;
            }
        }

#if NCNN_VULKAN
        if (opt.use_vulkan_compute)
        {
            upload_model();

            create_pipeline();
        }
#endif // NCNN_VULKAN

        fuse_network();

        return ret;
    }
};

} // namespace ncnn

static int g_warmup_loop_count = 3;
static int g_loop_count = 4;

static ncnn::Option g_default_option;

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

#if NCNN_VULKAN
static ncnn::VulkanDevice* g_vkdev = 0;
static ncnn::VkAllocator* g_blob_vkallocator = 0;
static ncnn::VkAllocator* g_staging_vkallocator = 0;
#endif // NCNN_VULKAN

void benchmark(const char* comment, const ncnn::Mat& in)
{
    ncnn::BenchNet net;

    net.opt = g_default_option;

#if NCNN_VULKAN
    if (net.opt.use_vulkan_compute)
    {
        net.set_vulkan_device(g_vkdev);
    }
#endif // NCNN_VULKAN

    char parampath[256];
    sprintf(parampath, "%s.param", comment);
    net.load_param(parampath);

    net.load_model();

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();

#if NCNN_VULKAN
    if (net.opt.use_vulkan_compute)
    {
        g_blob_vkallocator->clear();
        g_staging_vkallocator->clear();
    }
#endif // NCNN_VULKAN

    // sleep 10 seconds for cooling down SOC  :(
#ifdef _WIN32
    Sleep(10 * 1000);
#else
//     sleep(10);
#endif

    ncnn::Mat out;

    // warm up
    for (int i=0; i<g_warmup_loop_count; i++)
    {
        ncnn::Extractor ex = net.create_extractor();
        ex.input("data", in);
        ex.extract("output", out);
    }

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;

    for (int i=0; i<g_loop_count; i++)
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

    bool use_vulkan_compute = gpu_device != -1;

    g_loop_count = loop_count;

    g_blob_pool_allocator.set_size_compare_ratio(0.0f);
    g_workspace_pool_allocator.set_size_compare_ratio(0.5f);

#if NCNN_VULKAN
    if (use_vulkan_compute)
    {
        g_warmup_loop_count = 10;

        g_vkdev = ncnn::get_gpu_device(gpu_device);

        g_blob_vkallocator = new ncnn::VkBlobBufferAllocator(g_vkdev);
        g_staging_vkallocator = new ncnn::VkStagingBufferAllocator(g_vkdev);
    }
#endif // NCNN_VULKAN

    // default option
    g_default_option.lightmode = true;
    g_default_option.num_threads = num_threads;
    g_default_option.blob_allocator = &g_blob_pool_allocator;
    g_default_option.workspace_allocator = &g_workspace_pool_allocator;
#if NCNN_VULKAN
    g_default_option.blob_vkallocator = g_blob_vkallocator;
    g_default_option.workspace_vkallocator = g_blob_vkallocator;
    g_default_option.staging_vkallocator = g_staging_vkallocator;
#endif // NCNN_VULKAN
    g_default_option.use_winograd_convolution = true;
    g_default_option.use_sgemm_convolution = true;
    g_default_option.use_int8_inference = true;
    g_default_option.use_vulkan_compute = use_vulkan_compute;
    g_default_option.use_fp16_packed = true;
    g_default_option.use_fp16_storage = true;
    g_default_option.use_fp16_arithmetic = true;
    g_default_option.use_int8_storage = true;
    g_default_option.use_int8_arithmetic = true;

    ncnn::set_cpu_powersave(powersave);

    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);

    fprintf(stderr, "loop_count = %d\n", g_loop_count);
    fprintf(stderr, "num_threads = %d\n", num_threads);
    fprintf(stderr, "powersave = %d\n", ncnn::get_cpu_powersave());
    fprintf(stderr, "gpu_device = %d\n", gpu_device);

    // run
    benchmark("squeezenet", ncnn::Mat(227, 227, 3));

#if NCNN_VULKAN
    if (!use_vulkan_compute)
#endif // NCNN_VULKAN
    benchmark("squeezenet_int8", ncnn::Mat(227, 227, 3));

    benchmark("mobilenet", ncnn::Mat(224, 224, 3));

#if NCNN_VULKAN
    if (!use_vulkan_compute)
#endif // NCNN_VULKAN
    benchmark("mobilenet_int8", ncnn::Mat(224, 224, 3));

    benchmark("mobilenet_v2", ncnn::Mat(224, 224, 3));

// #if NCNN_VULKAN
//     if (!use_vulkan_compute)
// #endif // NCNN_VULKAN
//     benchmark("mobilenet_v2_int8", ncnn::Mat(224, 224, 3));

    benchmark("mobilenet_v3", ncnn::Mat(224, 224, 3));

    benchmark("shufflenet", ncnn::Mat(224, 224, 3));

    benchmark("mnasnet", ncnn::Mat(224, 224, 3));

    benchmark("proxylessnasnet", ncnn::Mat(224, 224, 3));

    benchmark("googlenet", ncnn::Mat(224, 224, 3));

#if NCNN_VULKAN
    if (!use_vulkan_compute)
#endif // NCNN_VULKAN
    benchmark("googlenet_int8", ncnn::Mat(224, 224, 3));

    benchmark("resnet18", ncnn::Mat(224, 224, 3));

#if NCNN_VULKAN
    if (!use_vulkan_compute)
#endif // NCNN_VULKAN
    benchmark("resnet18_int8", ncnn::Mat(224, 224, 3));

    benchmark("alexnet", ncnn::Mat(227, 227, 3));

    benchmark("vgg16", ncnn::Mat(224, 224, 3));

#if NCNN_VULKAN
    if (!use_vulkan_compute)
#endif // NCNN_VULKAN
    benchmark("vgg16_int8", ncnn::Mat(224, 224, 3));

    benchmark("resnet50", ncnn::Mat(224, 224, 3));

#if NCNN_VULKAN
    if (!use_vulkan_compute)
#endif // NCNN_VULKAN
    benchmark("resnet50_int8", ncnn::Mat(224, 224, 3));

    benchmark("squeezenet_ssd", ncnn::Mat(300, 300, 3));

#if NCNN_VULKAN
    if (!use_vulkan_compute)
#endif // NCNN_VULKAN
    benchmark("squeezenet_ssd_int8", ncnn::Mat(300, 300, 3));

    benchmark("mobilenet_ssd", ncnn::Mat(300, 300, 3));

#if NCNN_VULKAN
    if (!use_vulkan_compute)
#endif // NCNN_VULKAN
    benchmark("mobilenet_ssd_int8", ncnn::Mat(300, 300, 3));

    benchmark("mobilenet_yolo", ncnn::Mat(416, 416, 3));

    benchmark("mobilenetv2_yolov3", ncnn::Mat(352, 352, 3));

#if NCNN_VULKAN
    delete g_blob_vkallocator;
    delete g_staging_vkallocator;
#endif // NCNN_VULKAN

    return 0;
}
