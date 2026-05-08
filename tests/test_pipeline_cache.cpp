// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gpu.h"
#include "mat.h"
#include "net.h"
#include "pipelinecache.h"
#include "testutil.h"

#include <stdio.h>
#include <vector>

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

        net.opt.pipeline_cache = &pipeline_cache;

        static const unsigned int empty_model_data[1] = {0};
        net.load_model((const unsigned char*)empty_model_data);

        ncnn::Extractor ex = net.create_extractor();
        ex.input("input0", input);
        ex.extract("output0", output1);

        net.opt.pipeline_cache = 0;
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

int main()
{
    SRAND(7767517);

    if (ncnn::get_gpu_count() == 0)
        return 0;

    int ret = test_pipeline_cache_memory();
    if (ret != 0)
        return ret;

#if NCNN_STDIO
    ret = test_pipeline_cache_file();
    if (ret != 0)
        return ret;
#endif

    return 0;
}
