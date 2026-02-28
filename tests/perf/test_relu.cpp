// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perftestutil.h"

#include <stdio.h>

static void perf_relu(const ncnn::Mat& a, float slope, const ncnn::Option& opt, const char* env_tag)
{
    ncnn::ParamDict pd;
    pd.set(0, slope); // slope

    std::vector<ncnn::Mat> weights(0);

    PerfResult result;
    int ret = perf_layer_cpu("ReLU", pd, weights, a, opt, PERF_WARMUP_COUNT, PERF_RUN_COUNT, result);
    if (ret != 0)
        return;

    char shape_buf[64];
    char tag[256];
    format_shape(shape_buf, sizeof(shape_buf), a);
    snprintf(tag, sizeof(tag), "ReLU  slope=%.1f  %s  threads=%d  %s",
             slope, shape_buf, opt.num_threads, env_tag);
    print_perf_result(tag, result);
}

#if NCNN_VULKAN
static void perf_relu_gpu(const ncnn::Mat& a, float slope, const ncnn::Option& opt,
                          ncnn::VulkanDevice* vkdev)
{
    ncnn::ParamDict pd;
    pd.set(0, slope);

    std::vector<ncnn::Mat> weights(0);

    PerfResult result;
    int ret = perf_layer_gpu("ReLU", pd, weights, a, opt, vkdev, PERF_WARMUP_COUNT, PERF_RUN_COUNT, result);
    if (ret != 0)
        return;

    char shape_buf[64];
    char tag[256];
    format_shape(shape_buf, sizeof(shape_buf), a);
    snprintf(tag, sizeof(tag), "ReLU  slope=%.1f  %s  GPU", slope, shape_buf);
    print_perf_result(tag, result);
}
#endif // NCNN_VULKAN

int main()
{
    fprintf(stdout, "=== ReLU Performance Test ===\n\n");
    fflush(stdout);

    int max_threads = ncnn::get_physical_big_cpu_count();
    if (max_threads < 1) max_threads = 1;

    // --- vary shapes, fixed: slope=0, threads=1, fp32, all-core ---
    {
        ncnn::Option opt = make_perf_option(1, true, false, false);
        const int shapes[][4] = {
            {1, 10000, 0, 0},
            {1, 100000, 0, 0},
            {1, 1000000, 0, 0},
            {3, 56, 56, 64},
            {3, 28, 28, 128},
            {3, 14, 14, 256},
            {3, 7, 7, 512},
            {3, 112, 112, 32},
            {3, 224, 224, 3},
        };
        for (int i = 0; i < 9; i++)
        {
            ncnn::Mat input;
            if (shapes[i][0] == 1)
                input = PerfMat(shapes[i][1]);
            else
                input = PerfMat(shapes[i][1], shapes[i][2], shapes[i][3]);
            perf_relu(input, 0.f, opt, "fp32  all-core");
        }
    }

    fprintf(stdout, "\n");

    // --- vary slope, fixed: shape=(56x56x64), threads=1, fp32 ---
    {
        ncnn::Option opt = make_perf_option(1, true, false, false);
        ncnn::Mat input = PerfMat(56, 56, 64);
        perf_relu(input, 0.0f, opt, "fp32  all-core");
        perf_relu(input, 0.1f, opt, "fp32  all-core");
    }

    fprintf(stdout, "\n");

    // --- vary threads, fixed: shape=(56x56x64), slope=0, fp32 ---
    {
        ncnn::Mat input = PerfMat(56, 56, 64);
        int threads[] = {1, 2, 4};
        for (int i = 0; i < 3; i++)
        {
            if (threads[i] > max_threads) continue;
            ncnn::Option opt = make_perf_option(threads[i], true, false, false);
            perf_relu(input, 0.f, opt, "fp32  all-core");
        }
    }

    fprintf(stdout, "\n");

    // --- vary precision, fixed: shape=(56x56x64), slope=0, threads=1 ---
    {
        ncnn::Mat input = PerfMat(56, 56, 64);
        perf_relu(input, 0.f, make_perf_option(1, true, false, false), "fp32  all-core");
        perf_relu(input, 0.f, make_perf_option(1, true, true, false), "fp16  all-core");
#if NCNN_BF16
        perf_relu(input, 0.f, make_perf_option(1, true, false, true), "bf16  all-core");
#endif
    }

    fprintf(stdout, "\n");

    // --- vary powersave, fixed: shape=(56x56x64), slope=0, threads=2, fp32 ---
    {
        ncnn::Mat input = PerfMat(56, 56, 64);
        int ps_modes[] = {0, 1, 2};
        const char* ps_names[] = {"all-core", "little", "big"};
        for (int i = 0; i < 3; i++)
        {
            ncnn::set_cpu_powersave(ps_modes[i]);
            int nthreads = 2;
            if (nthreads > max_threads) nthreads = max_threads;
            ncnn::Option opt = make_perf_option(nthreads, true, false, false);
            char env_tag[64];
            snprintf(env_tag, sizeof(env_tag), "fp32  %s", ps_names[i]);
            perf_relu(input, 0.f, opt, env_tag);
        }
        ncnn::set_cpu_powersave(0);
    }

#if NCNN_VULKAN
    fprintf(stdout, "\n");
    {
        ncnn::create_gpu_instance();
        int gpu_count = ncnn::get_gpu_count();
        for (int gpu_id = 0; gpu_id < gpu_count; gpu_id++)
        {
            ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_id);
            if (!vkdev) continue;

            fprintf(stdout, "--- GPU %d ---\n", gpu_id);
            fflush(stdout);

            ncnn::Option opt = make_perf_option(1, true, true, false);
            ncnn::Mat shapes_3d[] = {PerfMat(56, 56, 64), PerfMat(14, 14, 256), PerfMat(224, 224, 3)};
            for (int i = 0; i < 3; i++)
            {
                perf_relu_gpu(shapes_3d[i], 0.f, opt, vkdev);
            }
        }
        ncnn::destroy_gpu_instance();
    }
#endif // NCNN_VULKAN

    return 0;
}
