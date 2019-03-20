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
#define NOMINMAX
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
        }

#if NCNN_VULKAN
        if (use_vulkan_compute)
        {
            upload_model();

            create_pipeline();
        }
#endif // NCNN_VULKAN

        return ret;
    }
};

} // namespace ncnn

static int g_loop_count = 4;

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

#if NCNN_VULKAN
static bool g_use_vulkan_compute = false;

static ncnn::VulkanDevice* g_vkdev = 0;
static ncnn::VkAllocator* g_blob_vkallocator = 0;
static ncnn::VkAllocator* g_staging_vkallocator = 0;
#endif // NCNN_VULKAN

void benchmark(const char* comment, void (*init)(ncnn::Net&), void (*run)(const ncnn::Net&))
{
    ncnn::BenchNet net;

#if NCNN_VULKAN
    if (g_use_vulkan_compute)
    {
        net.use_vulkan_compute = g_use_vulkan_compute;

        net.set_vulkan_device(g_vkdev);
    }
#endif // NCNN_VULKAN

    init(net);

    net.load_model();

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();

#if NCNN_VULKAN
    if (g_use_vulkan_compute)
    {
        g_blob_vkallocator->clear();
        g_staging_vkallocator->clear();
    }
#endif // NCNN_VULKAN

    // sleep 10 seconds for cooling down SOC  :(
#ifdef _WIN32
    Sleep(10 * 1000);
#else
    sleep(10);
#endif

    // warm up
    run(net);
    run(net);
    run(net);

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;

    for (int i=0; i<g_loop_count; i++)
    {
        double start = ncnn::get_current_time();

        run(net);

        double end = ncnn::get_current_time();

        double time = end - start;

        time_min = std::min(time_min, time);
        time_max = std::max(time_max, time);
        time_avg += time;
    }

    time_avg /= g_loop_count;

    fprintf(stderr, "%20s  min = %7.2f  max = %7.2f  avg = %7.2f\n", comment, time_min, time_max, time_avg);
}

void squeezenet_init(ncnn::Net& net)
{
    net.load_param("squeezenet.param");
}

void squeezenet_int8_init(ncnn::Net& net)
{
    net.load_param("squeezenet_int8.param");
}

void squeezenet_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(227, 227, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);
}

void mobilenet_init(ncnn::Net& net)
{
    net.load_param("mobilenet.param");
}

void mobilenet_int8_init(ncnn::Net& net)
{
    net.load_param("mobilenet_int8.param");
}

void mobilenet_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(224, 224, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);
}

void mobilenet_v2_init(ncnn::Net& net)
{
    net.load_param("mobilenet_v2.param");
}

void mobilenet_v2_int8_init(ncnn::Net& net)
{
    net.load_param("mobilenet_v2_int8.param");
}

void mobilenet_v2_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(224, 224, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);
}

void shufflenet_init(ncnn::Net& net)
{
    net.load_param("shufflenet.param");
}

void shufflenet_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(224, 224, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("fc1000", out);
}

void mnasnet_init(ncnn::Net& net)
{
    net.load_param("mnasnet.param");
}

void mnasnet_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(224, 224, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);
}

void proxylessnasnet_init(ncnn::Net& net)
{
    net.load_param("proxylessnasnet.param");
}

void proxylessnasnet_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(224, 224, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);
}

void googlenet_init(ncnn::Net& net)
{
    net.load_param("googlenet.param");
}

void googlenet_int8_init(ncnn::Net& net)
{
    net.load_param("googlenet_int8.param");
}

void googlenet_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(224, 224, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);
}

void resnet18_init(ncnn::Net& net)
{
    net.load_param("resnet18.param");
}

void resnet18_int8_init(ncnn::Net& net)
{
    net.load_param("resnet18_int8.param");
}

void resnet18_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(224, 224, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);
}

void alexnet_init(ncnn::Net& net)
{
    net.load_param("alexnet.param");
}

void alexnet_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(227, 227, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);
}

void vgg16_init(ncnn::Net& net)
{
    net.load_param("vgg16.param");
}

void vgg16_int8_init(ncnn::Net& net)
{
    net.load_param("vgg16_int8.param");
}

void vgg16_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(224, 224, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);
}

void resnet50_init(ncnn::Net& net)
{
    net.load_param("resnet50.param");
}

void resnet50_int8_init(ncnn::Net& net)
{
    net.load_param("resnet50_int8.param");
}

void resnet50_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(224, 224, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);
}

void squeezenet_ssd_init(ncnn::Net& net)
{
    net.load_param("squeezenet_ssd.param");
}

void squeezenet_ssd_int8_init(ncnn::Net& net)
{
    net.load_param("squeezenet_ssd_int8.param");
}

void squeezenet_ssd_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(300, 300, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("detection_out", out);
}

void mobilenet_ssd_init(ncnn::Net& net)
{
    net.load_param("mobilenet_ssd.param");
}

void mobilenet_ssd_int8_init(ncnn::Net& net)
{
    net.load_param("mobilenet_ssd_int8.param");
}

void mobilenet_ssd_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(300, 300, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("detection_out", out);
}

void mobilenet_yolo_init(ncnn::Net& net)
{
    net.load_param("mobilenet_yolo.param");
}

void mobilenet_yolo_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(416, 416, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("detection_out", out);
}

void mobilenet_yolov3_init(ncnn::Net& net)
{
    net.load_param("mobilenet_yolov3.param");
}

void mobilenet_yolov3_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(416, 416, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("detection_out", out);
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

    g_loop_count = loop_count;

    g_blob_pool_allocator.set_size_compare_ratio(0.0f);
    g_workspace_pool_allocator.set_size_compare_ratio(0.5f);

#if NCNN_VULKAN
    g_use_vulkan_compute = gpu_device != -1;
    if (g_use_vulkan_compute)
    {
        g_vkdev = new ncnn::VulkanDevice(gpu_device);

        g_blob_vkallocator = new ncnn::VkUnlockedBlobBufferAllocator(g_vkdev);
        g_staging_vkallocator = new ncnn::VkUnlockedStagingBufferAllocator(g_vkdev);
    }
#endif // NCNN_VULKAN

    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = num_threads;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;

#if NCNN_VULKAN
    opt.vulkan_compute = g_use_vulkan_compute;
    opt.blob_vkallocator = g_blob_vkallocator;
    opt.workspace_vkallocator = g_blob_vkallocator;
    opt.staging_vkallocator = g_staging_vkallocator;
#endif // NCNN_VULKAN

    ncnn::set_default_option(opt);

    ncnn::set_cpu_powersave(powersave);

    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);

    fprintf(stderr, "loop_count = %d\n", g_loop_count);
    fprintf(stderr, "num_threads = %d\n", num_threads);
    fprintf(stderr, "powersave = %d\n", ncnn::get_cpu_powersave());
    fprintf(stderr, "gpu_device = %d\n", gpu_device);

    // run
    benchmark("squeezenet", squeezenet_init, squeezenet_run);

#if NCNN_VULKAN
    if (!g_use_vulkan_compute)
#endif // NCNN_VULKAN
    benchmark("squeezenet-int8", squeezenet_int8_init, squeezenet_run);

    benchmark("mobilenet", mobilenet_init, mobilenet_run);

#if NCNN_VULKAN
    if (!g_use_vulkan_compute)
#endif // NCNN_VULKAN
    benchmark("mobilenet-int8", mobilenet_int8_init, mobilenet_run);

    benchmark("mobilenet_v2", mobilenet_v2_init, mobilenet_v2_run);

// #if NCNN_VULKAN
//     if (!g_use_vulkan_compute)
// #endif // NCNN_VULKAN
//     benchmark("mobilenet_v2-int8", mobilenet_v2_int8_init, mobilenet_v2_run);

    benchmark("shufflenet", shufflenet_init, shufflenet_run);

    benchmark("mnasnet", mnasnet_init, mnasnet_run);

    benchmark("proxylessnasnet", proxylessnasnet_init, proxylessnasnet_run);

    benchmark("googlenet", googlenet_init, googlenet_run);

#if NCNN_VULKAN
    if (!g_use_vulkan_compute)
#endif // NCNN_VULKAN
    benchmark("googlenet-int8", googlenet_int8_init, googlenet_run);

    benchmark("resnet18", resnet18_init, resnet18_run);

#if NCNN_VULKAN
    if (!g_use_vulkan_compute)
#endif // NCNN_VULKAN
    benchmark("resnet18-int8", resnet18_int8_init, resnet18_run);

    benchmark("alexnet", alexnet_init, alexnet_run);

    benchmark("vgg16", vgg16_init, vgg16_run);

    benchmark("resnet50", resnet50_init, resnet50_run);

#if NCNN_VULKAN
    if (!g_use_vulkan_compute)
#endif // NCNN_VULKAN
    benchmark("resnet50-int8", resnet50_int8_init, resnet50_run);

    benchmark("squeezenet-ssd", squeezenet_ssd_init, squeezenet_ssd_run);

#if NCNN_VULKAN
    if (!g_use_vulkan_compute)
#endif // NCNN_VULKAN
    benchmark("squeezenet-ssd-int8", squeezenet_ssd_int8_init, squeezenet_ssd_run);

    benchmark("mobilenet-ssd", mobilenet_ssd_init, mobilenet_ssd_run);

#if NCNN_VULKAN
    if (!g_use_vulkan_compute)
#endif // NCNN_VULKAN
    benchmark("mobilenet-ssd-int8", mobilenet_ssd_int8_init, mobilenet_ssd_run);

    benchmark("mobilenet-yolo", mobilenet_yolo_init, mobilenet_yolo_run);

    benchmark("mobilenet-yolov3", mobilenet_yolov3_init, mobilenet_yolov3_run);

#if NCNN_VULKAN
    delete g_blob_vkallocator;
    delete g_staging_vkallocator;

    delete g_vkdev;
#endif // NCNN_VULKAN

    return 0;
}
