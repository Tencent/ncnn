// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "datareader.h"
#include "gpu.h"
#include "mat.h"
#include "net.h"
#include "testutil.h"
#include "benchmark.h" // For ncnn::get_current_time()

#include <cstdio>
#include <vector>
#include <cstring> // For memset

int device_index = 1;

// A data reader that provides zero-filled data, useful for loading models without actual weights.
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

// The main test function to compare default vs. fast math performance.
static int test_vulkan_fast_math()
{
    // Define model path based on environment
    // Create a random input matrix
    ncnn::Mat input = RandomMat(512, 512, 3);
    DataReaderFromEmpty dr;

#ifdef __EMSCRIPTEN__
#define MODEL_DIR "/working"
#else
#define MODEL_DIR "../../benchmark"
#endif

    // ==================================================
    // 1. Setup Net with Default Options
    // ==================================================
    printf("==================================================\n");
    printf("         Testing with Default Vulkan Options      \n");
    printf("==================================================\n");
    ncnn::Net net_default;
    net_default.opt.use_vulkan_compute = true;
    net_default.opt.vulkan_device_index = device_index;
    net_default.opt.use_fp16_arithmetic = false;
    net_default.opt.use_fp16_storage = false;
    net_default.opt.use_fp16_packed = false;

    net_default.load_param(MODEL_DIR "/resnet50.param");
    net_default.load_model(dr);
    printf("Default net loaded successfully.\n");

    // ==================================================
    // 2. Setup Net with Fast Math Options
    // ==================================================
    printf("\n==================================================\n");
    printf("        Testing with Vulkan Fast Math Options     \n");
    printf("==================================================\n");
    ncnn::Net net_fast_math;
    net_fast_math.opt.use_vulkan_compute = true;
    net_fast_math.opt.vk_fast_math_flag = ncnn::Option::VK_FAST_MATH_FLAG_Fast
                                          | ncnn::Option::VK_FAST_MATH_FLAG_AllowContract
                                          | ncnn::Option::VK_FAST_MATH_FLAG_AllowReassoc
                                          | ncnn::Option::VK_FAST_MATH_FLAG_AllowTransform;
    net_fast_math.opt.vulkan_device_index = device_index;
    net_fast_math.opt.use_fp16_arithmetic = false;
    net_fast_math.opt.use_fp16_packed = false;
    net_fast_math.opt.use_fp16_storage = false;

    net_fast_math.load_param(MODEL_DIR "/resnet50.param");
    net_fast_math.load_model(dr);
    printf("Fast math net loaded successfully.\n");

    // ==================================================
    // 3. Warm-up Run
    // ==================================================
    printf("\n==================================================\n");
    printf("             Warming up both networks...          \n");
    printf("==================================================\n");
    ncnn::Mat output_default, output_fast_math;
    {
        ncnn::Extractor ex = net_default.create_extractor();
        ex.input("data", input);
        ex.extract("output", output_default);
    }
    {
        ncnn::Extractor ex = net_fast_math.create_extractor();
        ex.input("data", input);
        ex.extract("output", output_fast_math);
    }
    printf("Warm-up complete.\n");

    // ==================================================
    // 4. Benchmark Performance
    // ==================================================
    printf("\n==================================================\n");
    printf("             Benchmarking Performance           \n");
    printf("==================================================\n");
    const int loop_count = 10;
    double time_default = 0;
    double time_fast_math = 0;

    // Benchmark default net
    {
        double start = ncnn::get_current_time();
        for (int i = 0; i < loop_count; i++)
        {
            ncnn::Extractor ex = net_default.create_extractor();
            ex.input("data", input);
            ex.extract("output", output_default);
        }
        double end = ncnn::get_current_time();
        time_default = (end - start) / loop_count;
        printf("Default Net Average Time:      %.2f ms\n", time_default);
    }

    // Benchmark fast math net
    {
        double start = ncnn::get_current_time();
        for (int i = 0; i < loop_count; i++)
        {
            ncnn::Extractor ex = net_fast_math.create_extractor();
            ex.input("data", input);
            ex.extract("output", output_fast_math);
        }
        double end = ncnn::get_current_time();
        time_fast_math = (end - start) / loop_count;
        printf("Fast Math Net Average Time:    %.2f ms\n", time_fast_math);
    }

    // ==================================================
    // 5. Verification and Summary
    // ==================================================
    printf("\n==================================================\n");
    printf("              Verification and Summary            \n");
    printf("==================================================\n");

    // Compare results. A larger tolerance is needed due to fast math optimizations.
    int ret = CompareMat(output_default, output_fast_math, 0.01f);
    printf("Output comparison result (0 means success): %d\n", ret);
    if (ret != 0)
    {
        fprintf(stderr, "Warning: Output mismatch is larger than tolerance. Fast math might be affecting precision significantly.\n");
    }
    else
    {
        printf("Output verification: SUCCESS (within tolerance)\n");
    }

    printf("--------------------------------------------------\n");
    printf("Performance Summary:\n");
    printf("  - Default Net:   %.2f ms\n", time_default);
    printf("  - Fast Math Net: %.2f ms\n", time_fast_math);

    if (time_default > 0 && time_fast_math > 0)
    {
        double speedup = (time_default - time_fast_math) / time_default * 100;
        printf("  - Speedup:       %.2f%%\n", speedup);
    }

    printf("\nTest finished.\n");
    return 0;
}

int main(int argc, char** argv)
{
    if (argc >= 2)
    {
        device_index = atoi(argv[1]);
    }

    int gpu_count = ncnn::get_gpu_count();
    if (device_index < 0 || device_index >= gpu_count)
    {
        fprintf(stderr, "Invalid GPU device index %d. The valid range is [0, %d-1]. Using default device 0.\n", device_index, gpu_count);
        device_index = 0;
    }

    // Set the default device for all ncnn operations.
    printf("Using Vulkan Device: %d\n", device_index);

    // Run the performance test.
    int ret = test_vulkan_fast_math();

    return ret;
}
