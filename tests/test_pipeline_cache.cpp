// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gpu.h"
#include "mat.h"
#include "net.h"
#include "pipelinecache.h"
#include "pipeline.h"
#include "testutil.h"

#include <stdio.h>

#if NCNN_STDIO
#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif
#endif

static const char* test_param = "7767517\n"
                                "2 2\n"
                                "Input    input0    0   1   input0\n"
                                "Sigmoid  sigmoid0  1   1   input0    output0\n";

static int test_pipeline_cache_memory()
{
    ncnn::Mat input = RandomMat(16, 16);
    ncnn::Mat output0;
    ncnn::Mat output1;

    std::vector<unsigned char> cache_data;

    {
        ncnn::Net net;
        net.opt.use_vulkan_compute = true;

        if (net.load_param_mem(test_param) != 0)
        {
            fprintf(stderr, "load_param_mem failed\n");
            return -1;
        }

        ncnn::PipelineCache pipeline_cache(net.vulkan_device());
        net.opt.pipeline_cache = &pipeline_cache;

        static const unsigned int empty_model_data[1] = {0};
        net.load_model((const unsigned char*)empty_model_data);

        ncnn::Extractor ex = net.create_extractor();
        ex.input("input0", input);
        ex.extract("output0", output0);

        net.opt.pipeline_cache = 0;

        if (output0.empty())
        {
            fprintf(stderr, "extract output failed\n");
            return -1;
        }

        if (pipeline_cache.save_cache(cache_data) != 0)
        {
            fprintf(stderr, "save_cache to memory failed\n");
            return -1;
        }
    }

    if (cache_data.empty())
    {
        fprintf(stderr, "cache data is empty\n");
        return -1;
    }

    {
        ncnn::Net net;
        net.opt.use_vulkan_compute = true;

        if (net.load_param_mem(test_param) != 0)
        {
            fprintf(stderr, "load_param_mem failed\n");
            return -1;
        }

        ncnn::PipelineCache pipeline_cache(net.vulkan_device());
        if (pipeline_cache.load_cache(cache_data) != 0)
        {
            fprintf(stderr, "load_cache from memory failed\n");
            return -1;
        }

        // reject empty and truncated input without losing the loaded cache
        const std::vector<unsigned char> empty_cache;
        if (pipeline_cache.load_cache(empty_cache) == 0
                || pipeline_cache.load_cache((const unsigned char*)0, 0) == 0
                || pipeline_cache.load_cache(cache_data.data(), 0) == 0
                || pipeline_cache.load_cache(cache_data.data(), cache_data.size() - 1) == 0)
        {
            fprintf(stderr, "load_cache accepted empty or truncated data\n");
            return -1;
        }

        std::vector<unsigned char> trailing_data = cache_data;
        trailing_data.push_back(0);
        if (pipeline_cache.load_cache(trailing_data) == 0)
        {
            fprintf(stderr, "load_cache accepted trailing data\n");
            return -1;
        }

        // loading the same cache again merges existing pipeline and spirv entries
        if (pipeline_cache.load_cache(cache_data) != 0)
        {
            fprintf(stderr, "reload_cache from memory failed\n");
            return -1;
        }

        net.opt.pipeline_cache = &pipeline_cache;

        static const unsigned int empty_model_data[1] = {0};
        net.load_model((const unsigned char*)empty_model_data);

        ncnn::Extractor ex = net.create_extractor();
        ex.input("input0", input);
        ex.extract("output0", output1);

        net.opt.pipeline_cache = 0;

        // cover both populated and already-cleared caches without relying on layer test counts
        pipeline_cache.clear();
        if (pipeline_cache.size() != 0)
        {
            fprintf(stderr, "pipeline cache is not empty after clear\n");
            return -1;
        }
        pipeline_cache.clear();
    }

    if (CompareMat(output0, output1, 0.001) != 0)
    {
        fprintf(stderr, "pipeline cache output mismatch\n");
        return -1;
    }

    std::vector<unsigned char> corrupted = cache_data;
    corrupted[0] ^= 0xff;
    {
        ncnn::Net net;
        net.opt.use_vulkan_compute = true;
        net.load_param_mem(test_param);

        ncnn::PipelineCache pipeline_cache(net.vulkan_device());
        if (pipeline_cache.load_cache(corrupted) == 0)
        {
            fprintf(stderr, "load_cache accepted corrupted header\n");
            return -1;
        }
    }

    corrupted = cache_data;
    corrupted[corrupted.size() - 1] ^= 0xff;
    {
        ncnn::Net net;
        net.opt.use_vulkan_compute = true;
        net.load_param_mem(test_param);

        ncnn::PipelineCache pipeline_cache(net.vulkan_device());
        if (pipeline_cache.load_cache(corrupted) == 0)
        {
            fprintf(stderr, "load_cache accepted corrupted payload\n");
            return -1;
        }
    }

    return 0;
}

#if NCNN_STDIO
static int test_pipeline_cache_file()
{
    char cache_path[256];
#if defined(_WIN32)
    snprintf(cache_path, sizeof(cache_path), "test_pipeline_cache.%u.bin", (unsigned int)_getpid());
#else
    snprintf(cache_path, sizeof(cache_path), "test_pipeline_cache.%u.bin", (unsigned int)getpid());
#endif

    ncnn::Mat input = RandomMat(8, 8);
    ncnn::Mat output0;
    ncnn::Mat output1;

    {
        ncnn::Net net;
        net.opt.use_vulkan_compute = true;

        if (net.load_param_mem(test_param) != 0)
        {
            fprintf(stderr, "load_param_mem failed\n");
            return -1;
        }

        ncnn::PipelineCache pipeline_cache(net.vulkan_device());
        net.opt.pipeline_cache = &pipeline_cache;

        static const unsigned int empty_model_data[1] = {0};
        net.load_model((const unsigned char*)empty_model_data);

        ncnn::Extractor ex = net.create_extractor();
        ex.input("input0", input);
        ex.extract("output0", output0);

        net.opt.pipeline_cache = 0;

        if (pipeline_cache.save_cache(cache_path) != 0)
        {
            fprintf(stderr, "save_cache to file failed\n");
            return -1;
        }
    }

    {
        ncnn::Net net;
        net.opt.use_vulkan_compute = true;

        if (net.load_param_mem(test_param) != 0)
        {
            fprintf(stderr, "load_param_mem failed\n");
            remove(cache_path);
            return -1;
        }

        ncnn::PipelineCache pipeline_cache(net.vulkan_device());
        if (pipeline_cache.load_cache(cache_path) != 0)
        {
            fprintf(stderr, "load_cache from file failed\n");
            remove(cache_path);
            return -1;
        }

        net.opt.pipeline_cache = &pipeline_cache;

        static const unsigned int empty_model_data[1] = {0};
        net.load_model((const unsigned char*)empty_model_data);

        ncnn::Extractor ex = net.create_extractor();
        ex.input("input0", input);
        ex.extract("output0", output1);

        net.opt.pipeline_cache = 0;
    }

    remove(cache_path);

    if (CompareMat(output0, output1, 0.001) != 0)
    {
        fprintf(stderr, "file pipeline cache output mismatch\n");
        return -1;
    }

    return 0;
}
#endif // NCNN_STDIO

static int test_pipeline_local_size()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::Pipeline pipeline(vkdev);
    const int sizes[][3] = {{3, 1, 1}, {1, 3, 1}, {1, 1, 3}, {3, 1, 3}, {1, 3, 3}, {3, 3, 3}};
    for (int i = 0; i < 6; i++)
    {
        // rounding preserves singleton axes and includes a complete subgroup
        pipeline.set_local_size_xyz(sizes[i][0], sizes[i][1], sizes[i][2]);
        const unsigned int x = pipeline.local_size_x();
        const unsigned int y = pipeline.local_size_y();
        const unsigned int z = pipeline.local_size_z();
        if (x < (unsigned int)sizes[i][0] || y < (unsigned int)sizes[i][1] || z < (unsigned int)sizes[i][2]
                || (sizes[i][1] == 1 && y != 1) || (sizes[i][2] == 1 && z != 1)
                || x * y * z % vkdev->info.subgroup_size() != 0)
        {
            fprintf(stderr, "invalid local size %u %u %u for %d %d %d\n", x, y, z, sizes[i][0], sizes[i][1], sizes[i][2]);
            return -1;
        }
    }

    if (vkdev->info.min_subgroup_size() != 0 && vkdev->info.max_subgroup_size() >= vkdev->info.min_subgroup_size())
    {
        // requests outside the supported range clamp to the device limits
        pipeline.set_subgroup_size(0);
        pipeline.set_local_size_xyz(1, 1, 1);
        if (pipeline.local_size_x() * pipeline.local_size_y() * pipeline.local_size_z() != vkdev->info.min_subgroup_size())
            return -1;

        pipeline.set_subgroup_size(vkdev->info.max_subgroup_size() + 1);
        pipeline.set_local_size_xyz(1, 1, 1);
        if (pipeline.local_size_x() * pipeline.local_size_y() * pipeline.local_size_z() != vkdev->info.max_subgroup_size())
            return -1;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    if (ncnn::get_gpu_count() == 0)
        return 0;

    int ret = test_pipeline_local_size();
    if (ret != 0)
        return ret;

    ret = test_pipeline_cache_memory();
    if (ret != 0)
        return ret;

#if NCNN_STDIO
    ret = test_pipeline_cache_file();
    if (ret != 0)
        return ret;
#endif

    return 0;
}
