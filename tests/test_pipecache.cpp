// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "datareader.h"
#include "gpu.h"
#include "mat.h"
#include "net.h"
#include "pipelinecache.h"
#include "testutil.h"
#include "benchmark.h"

#include <cstdio>
#include <vector>
#include <cstring>

class DataReaderFromEmpty : public ncnn::DataReader
{
public:
    virtual int scan(const char* format, void* p) const
    {
        (void)format; // unused
        (void)p;      // unused
        return 0;
    }
    virtual size_t read(void* buf, size_t size) const
    {
        memset(buf, 0, size);
        return size;
    }
};

static int warmup_gpu_pipecache()
{
    printf("==================================================\n");
    printf("           Warmup: Testing Basic Cache IO         \n");
    printf("==================================================\n");

    ncnn::Net net;
    net.opt.use_vulkan_compute = true;

    net.load_param_mem("7767517\n2 2\nInput    input0    0   1   input0\nSigmoid  sigmoid0  1   1   input0    output0");
    net.load_model((unsigned char*)"");

    ncnn::Mat input0 = RandomMat(224, 224);
    ncnn::Mat output0;
    {
        ncnn::Extractor ex = net.create_extractor();
        ex.input("input0", input0);
        ex.extract("output0", output0);
    }

    if (output0.empty())
    {
        fprintf(stderr, "Warmup failed: initial extraction failed.\n");
        return -1;
    }

    const char* cache_path = "./sigmoid_pipecache.bin";
    if (net.opt.pipeline_cache->save_cache(cache_path) != 0)
    {
        fprintf(stderr, "Warmup failed: could not save pipeline cache to %s\n", cache_path);
        return -1;
    }
    printf("Warmup: Pipeline cache saved successfully.\n");

    ncnn::Net net2;
    net2.opt.use_vulkan_compute = true;
    net2.opt.pipeline_cache = new ncnn::PipelineCache(net.vulkan_device());

    net2.load_param_mem("7767517\n2 2\nInput    input0    0   1   input0\nSigmoid  sigmoid0  1   1   input0    output0");
    if (net2.opt.pipeline_cache->load_cache(cache_path) != 0)
    {
        fprintf(stderr, "Warmup failed: could not load pipeline cache from %s\n", cache_path);
        return -1;
    }
    printf("Warmup: Pipeline cache loaded successfully.\n");
    net2.load_model((unsigned char*)"");

    ncnn::Mat output0_2;
    {
        ncnn::Extractor ex2 = net2.create_extractor();
        ex2.input("input0", input0);
        ex2.extract("output0", output0_2);
    }

    if (output0_2.empty())
    {
        fprintf(stderr, "Warmup failed: extraction after loading cache failed.\n");
        return -1;
    }

    if (CompareMat(output0, output0_2, 0.001) != 0)
    {
        fprintf(stderr, "Warmup failed: output mismatch after loading cache.\n");
        return -1;
    }

    printf("Warmup PASSED: Outputs are identical.\n");
    return 0;
}

static int test_gpu_pipecache_performance()
{
    ncnn::Mat output_no_cache;
    double time_no_cache = 0;

    const char* cache_path = "./mobilenet_pipecache.bin";
    DataReaderFromEmpty dr;
    ncnn::Mat input = RandomMat(224, 224, 3);

#ifdef __EMSCRIPTEN__
#define MODEL_DIR "/working"
#else
#define MODEL_DIR "../../benchmark"
#endif

    // -------------------------------------------------
    // 1. Without cache
    // -------------------------------------------------
    printf("\n==================================================\n");
    printf("       Performance Test: Without Pipeline Cache   \n");
    printf("==================================================\n");
    {
        ncnn::Net net_no_cache;
        net_no_cache.opt.use_vulkan_compute = true;

        auto start = ncnn::get_current_time();

        net_no_cache.load_param(MODEL_DIR "/mobilenet_v3.param");
        net_no_cache.load_model(dr);

        auto end = ncnn::get_current_time();
        time_no_cache = end - start;
        printf("Model loading time without cache: %lf ms\n", time_no_cache);

        ncnn::Extractor ex = net_no_cache.create_extractor();
        ex.input("data", input);
        ex.extract("output", output_no_cache);

        if (output_no_cache.empty())
        {
            fprintf(stderr, "Test failed: extraction without cache failed.\n");
            return -1;
        }

        // save cache
        if (net_no_cache.opt.pipeline_cache->save_cache(cache_path) != 0)
        {
            fprintf(stderr, "Test failed: could not save pipeline cache to %s\n", cache_path);
            return -1;
        }
        printf("Pipeline cache generated and saved to %s\n", cache_path);
    }

    // -------------------------------------------------
    // 2. With Cache
    // -------------------------------------------------
    ncnn::Mat output_with_cache;
    double time_with_cache = 0;
    printf("\n==================================================\n");
    printf("        Performance Test: With Pipeline Cache     \n");
    printf("==================================================\n");
    {
        ncnn::Net net_with_cache;
        net_with_cache.opt.pipeline_cache = new ncnn::PipelineCache(ncnn::get_gpu_device());
        net_with_cache.opt.use_vulkan_compute = true;

        auto start = ncnn::get_current_time();

        // load from cache
        if (net_with_cache.opt.pipeline_cache->load_cache(cache_path) != 0)
        {
            fprintf(stderr, "Test failed: could not load pipeline cache from %s\n", cache_path);
            return -1;
        }
        net_with_cache.load_param(MODEL_DIR "/mobilenet_v3.param");
        net_with_cache.load_model(dr);

        auto end = ncnn::get_current_time();
        time_with_cache = end - start;
        printf("Model loading time with cache: %lf ms\n", time_with_cache);

        ncnn::Extractor ex2 = net_with_cache.create_extractor();
        ex2.input("data", input);
        ex2.extract("output", output_with_cache);

        if (output_with_cache.empty())
        {
            fprintf(stderr, "Test failed: extraction with cache failed.\n");
            return -1;
        }
    }

    // -------------------------------------------------
    // 3. Verification
    // -------------------------------------------------
    printf("\n==================================================\n");
    printf("              Verification and Summary            \n");
    printf("==================================================\n");

    bool is_output_same = (CompareMat(output_no_cache, output_with_cache, 0.001) == 0);

    printf("Output verification: %s\n", (is_output_same ? "SUCCESS" : "FAILURE"));
    printf("--------------------------------------------------\n");
    printf("Performance Summary:\n");
    printf("  - Without Cache: %f ms\n", time_no_cache);
    printf("  - With Cache:    %f ms\n", time_with_cache);

    if (time_no_cache > 0)
    {
        double speedup = (time_no_cache - time_with_cache) / time_no_cache * 100;
        printf("  - Speedup:       %f%%\n", speedup);
    }

    if (!is_output_same)
    {
        fprintf(stderr, "\nTest FAILED due to output mismatch.\n");
        return -1;
    }

    printf("\nTest PASSED.\n");
    return 0;
}

int main()
{
    // warming up
    if (warmup_gpu_pipecache() != 0)
    {
        return -1;
    }

    return test_gpu_pipecache_performance();
}
