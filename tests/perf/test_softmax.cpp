// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perftestutil.h"

#include <stdio.h>

static void perf_softmax(const ncnn::Mat& a, int axis, const ncnn::Option& opt, const char* env_tag)
{
    ncnn::ParamDict pd;
    pd.set(0, axis); // axis
    pd.set(1, 1);    // fixbug0

    std::vector<ncnn::Mat> weights(0);

    PerfResult result;
    int ret = perf_layer_cpu("Softmax", pd, weights, a, opt, PERF_WARMUP_COUNT, PERF_RUN_COUNT, result);
    if (ret != 0)
        return;

    char shape_buf[64];
    char tag[256];
    format_shape(shape_buf, sizeof(shape_buf), a);
    snprintf(tag, sizeof(tag), "Softmax  %s  axis=%d  threads=%d  %s",
             shape_buf, axis, opt.num_threads, env_tag);
    print_perf_result(tag, result);
}

#if NCNN_VULKAN
static void perf_softmax_gpu(const ncnn::Mat& a, int axis, const ncnn::Option& opt,
                             ncnn::VulkanDevice* vkdev)
{
    ncnn::ParamDict pd;
    pd.set(0, axis);
    pd.set(1, 1);

    std::vector<ncnn::Mat> weights(0);

    PerfResult result;
    int ret = perf_layer_gpu("Softmax", pd, weights, a, opt, vkdev, PERF_WARMUP_COUNT, PERF_RUN_COUNT, result);
    if (ret != 0)
        return;

    char shape_buf[64];
    char tag[256];
    format_shape(shape_buf, sizeof(shape_buf), a);
    snprintf(tag, sizeof(tag), "Softmax  %s  axis=%d  GPU", shape_buf, axis);
    print_perf_result(tag, result);
}
#endif // NCNN_VULKAN

struct SoftmaxConfig
{
    int w, h, c, axis;
};

int main()
{
    fprintf(stdout, "=== Softmax Performance Test ===\n\n");
    fflush(stdout);

    int max_threads = ncnn::get_physical_big_cpu_count();
    if (max_threads < 1) max_threads = 1;

    // --- vary shapes, fixed: axis=0, threads=1, fp32, all-core ---
    {
        ncnn::Option opt = make_perf_option(1, true, false, false);
        perf_softmax(PerfMat(1000), 0, opt, "fp32  all-core");
        perf_softmax(PerfMat(100000), 0, opt, "fp32  all-core");
        perf_softmax(PerfMat(1, 1, 1000), 2, opt, "fp32  all-core");
        perf_softmax(PerfMat(56, 56, 64), 0, opt, "fp32  all-core");
    }

    fprintf(stdout, "\n");

    // --- vary axis, fixed: shape=(56x56x64), threads=1, fp32 ---
    {
        ncnn::Option opt = make_perf_option(1, true, false, false);
        ncnn::Mat input = PerfMat(56, 56, 64);
        perf_softmax(input, 0, opt, "fp32  all-core");
        perf_softmax(input, 1, opt, "fp32  all-core");
        perf_softmax(input, 2, opt, "fp32  all-core");
    }

    fprintf(stdout, "\n");

    // --- vary threads, fixed: shape=(1x1x1000) axis=2, fp32 ---
    {
        ncnn::Mat input = PerfMat(1, 1, 1000);
        int threads[] = {1, 2, 4};
        for (int i = 0; i < 3; i++)
        {
            if (threads[i] > max_threads) continue;
            ncnn::Option opt = make_perf_option(threads[i], true, false, false);
            perf_softmax(input, 2, opt, "fp32  all-core");
        }
    }

    fprintf(stdout, "\n");

    // --- vary precision, fixed: shape=(1x1x1000) axis=2, threads=1 ---
    {
        ncnn::Mat input = PerfMat(1, 1, 1000);
        perf_softmax(input, 2, make_perf_option(1, true, false, false), "fp32  all-core");
        perf_softmax(input, 2, make_perf_option(1, true, true, false), "fp16  all-core");
#if NCNN_BF16
        perf_softmax(input, 2, make_perf_option(1, true, false, true), "bf16  all-core");
#endif
    }

    fprintf(stdout, "\n");

    // --- vary powersave, fixed: shape=(1x1x1000) axis=2, threads=2, fp32 ---
    {
        ncnn::Mat input = PerfMat(1, 1, 1000);
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
            perf_softmax(input, 2, opt, env_tag);
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
            perf_softmax_gpu(PerfMat(1, 1, 1000), 2, opt, vkdev);
            perf_softmax_gpu(PerfMat(56, 56, 64), 0, opt, vkdev);
        }
        ncnn::destroy_gpu_instance();
    }
#endif // NCNN_VULKAN

    return 0;
}
