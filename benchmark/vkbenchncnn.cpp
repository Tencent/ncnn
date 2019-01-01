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

#include "benchmark.h"
#include "cpu.h"
#include "gpu.h"
#include "net.h"

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

        mb.vk_model_loader = 0;
        if (use_vulkan_compute)
        {
            mb.vk_model_loader = new VkCompute(vkdev);
            mb.weight_vkallocator = weight_vkallocator;
            mb.staging_vkallocator = staging_vkallocator;
            mb.vk_model_loader->begin();
        }

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

        if (use_vulkan_compute)
        {
            mb.vk_model_loader->end();
            mb.vk_model_loader->submit();

            mb.vk_model_loader->wait();

            delete mb.vk_model_loader;

            for (size_t i=0; i<layers.size(); i++)
            {
                Layer* layer = layers[i];

                if (layer->support_vulkan)
                {
                    layer->create_vulkan_pipeline();
                }
            }

        }

        return ret;
    }
};

} // namespace ncnn

int main(int argc, char** argv)
{
    ncnn::create_gpu_instance();

    {

    ncnn::VulkanDevice vkdev;

    {

    ncnn::VkAllocator g_weight_vkallocator(&vkdev, 0);
    ncnn::VkAllocator g_blob_vkallocator(&vkdev, 0);
    ncnn::VkAllocator g_workspace_vkallocator(&vkdev, 0);
    ncnn::VkAllocator g_staging_vkallocator(&vkdev, 1);

    ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
    ncnn::PoolAllocator g_workspace_pool_allocator;

    int g_loop_count = 40;

    g_blob_pool_allocator.set_size_compare_ratio(0.0f);
    g_workspace_pool_allocator.set_size_compare_ratio(0.5f);

    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 1;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;

    opt.blob_vkallocator = &g_blob_vkallocator;
    opt.workspace_vkallocator = &g_workspace_vkallocator;
    opt.staging_vkallocator = &g_staging_vkallocator;

    ncnn::set_default_option(opt);

//     ncnn::set_cpu_powersave(powersave);
//     ncnn::set_omp_dynamic(0);
//     ncnn::set_omp_num_threads(8);

    fprintf(stderr, "loop_count = %d\n", g_loop_count);
    fprintf(stderr, "num_threads = %d\n", opt.num_threads);
    fprintf(stderr, "powersave = %d\n", ncnn::get_cpu_powersave());

    ncnn::BenchNet net;

    net.set_vulkan_device(&vkdev);
    net.set_weight_vkallocator(&g_weight_vkallocator);
    net.set_staging_vkallocator(&g_staging_vkallocator);

//     net.load_param("vgg16.param");
    net.load_param("mobilenet_v2.param");
    net.load_model();

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;

    for (int i=0; i<g_loop_count; i++)
    {
        double start = ncnn::get_current_time();

        {
            ncnn::Extractor ex = net.create_extractor();

            ncnn::Mat in(224, 224, 3);
            in.fill(0.5f);
            ex.input("data", in);

            ncnn::Mat prob;
            ex.extract("prob", prob);
        }

        double end = ncnn::get_current_time();

        double time = end - start;

        time_min = std::min(time_min, time);
        time_max = std::max(time_max, time);
        time_avg += time;
    }

    time_avg /= g_loop_count;

    fprintf(stderr, "min = %7.2f  max = %7.2f  avg = %7.2f\n", time_min, time_max, time_avg);

    g_weight_vkallocator.clear();
    g_blob_vkallocator.clear();
    g_workspace_vkallocator.clear();
    g_staging_vkallocator.clear();

    }

    }

    ncnn::destroy_gpu_instance();

    return 0;
}
