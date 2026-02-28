// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perftestutil.h"

#include <stdio.h>

static void perf_pooling(int w, int h, int c, int pooling_type, int kernel, int stride, int pad,
                         int global_pooling, const ncnn::Option& opt, const char* env_tag)
{
    ncnn::Mat input = PerfMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, pooling_type); // pooling_type: 0=max, 1=avg
    pd.set(1, kernel);       // kernel_w
    pd.set(2, stride);       // stride_w
    pd.set(3, pad);          // pad_w
    pd.set(4, global_pooling);
    pd.set(5, 0); // pad_mode
    pd.set(6, 1); // avgpool_count_include_pad

    std::vector<ncnn::Mat> weights(0);

    PerfResult result;
    int ret = perf_layer_cpu("Pooling", pd, weights, input, opt, PERF_WARMUP_COUNT, PERF_RUN_COUNT, result);
    if (ret != 0)
        return;

    const char* pool_name = (pooling_type == 0) ? "MaxPool" : "AvgPool";
    char shape_buf[64];
    char tag[256];
    format_shape(shape_buf, sizeof(shape_buf), input);
    if (global_pooling)
    {
        snprintf(tag, sizeof(tag), "%-8s %s global          threads=%d  %s",
                 pool_name, shape_buf, opt.num_threads, env_tag);
    }
    else
    {
        snprintf(tag, sizeof(tag), "%-8s %s k=%d s=%d p=%d  threads=%d  %s",
                 pool_name, shape_buf, kernel, stride, pad, opt.num_threads, env_tag);
    }
    print_perf_result(tag, result);
}

#if NCNN_VULKAN
static void perf_pooling_gpu(int w, int h, int c, int pooling_type, int kernel, int stride, int pad,
                             int global_pooling, const ncnn::Option& opt, ncnn::VulkanDevice* vkdev)
{
    ncnn::Mat input = PerfMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, pooling_type);
    pd.set(1, kernel);
    pd.set(2, stride);
    pd.set(3, pad);
    pd.set(4, global_pooling);
    pd.set(5, 0);
    pd.set(6, 1);

    std::vector<ncnn::Mat> weights(0);

    PerfResult result;
    int ret = perf_layer_gpu("Pooling", pd, weights, input, opt, vkdev, PERF_WARMUP_COUNT, PERF_RUN_COUNT, result);
    if (ret != 0)
        return;

    const char* pool_name = (pooling_type == 0) ? "MaxPool" : "AvgPool";
    char shape_buf[64];
    char tag[256];
    format_shape(shape_buf, sizeof(shape_buf), input);
    if (global_pooling)
    {
        snprintf(tag, sizeof(tag), "%-8s %s global          GPU", pool_name, shape_buf);
    }
    else
    {
        snprintf(tag, sizeof(tag), "%-8s %s k=%d s=%d p=%d  GPU",
                 pool_name, shape_buf, kernel, stride, pad);
    }
    print_perf_result(tag, result);
}
#endif // NCNN_VULKAN

struct PoolConfig
{
    int w, h, c, pooling_type, kernel, stride, pad, global_pooling;
};

int main()
{
    fprintf(stdout, "=== Pooling Performance Test ===\n\n");
    fflush(stdout);

    int max_threads = ncnn::get_physical_big_cpu_count();
    if (max_threads < 1) max_threads = 1;

    // --- vary shapes, fixed: MaxPool k=3 s=2 p=1, threads=1, fp32, all-core ---
    {
        ncnn::Option opt = make_perf_option(1, true, false, false);
        perf_pooling(112, 112, 64, 0, 3, 2, 1, 0, opt, "fp32  all-core");
        perf_pooling(56, 56, 128, 0, 3, 2, 1, 0, opt, "fp32  all-core");
        perf_pooling(28, 28, 256, 0, 3, 2, 1, 0, opt, "fp32  all-core");
        perf_pooling(14, 14, 512, 0, 3, 2, 1, 0, opt, "fp32  all-core");
    }

    fprintf(stdout, "\n");

    // --- vary pool type, fixed: shape=(56x56x128), threads=1, fp32 ---
    {
        ncnn::Option opt = make_perf_option(1, true, false, false);
        perf_pooling(56, 56, 128, 0, 3, 2, 1, 0, opt, "fp32  all-core"); // MaxPool k=3 s=2
        perf_pooling(56, 56, 128, 1, 3, 2, 1, 0, opt, "fp32  all-core"); // AvgPool k=3 s=2
        perf_pooling(7, 7, 512, 1, 0, 0, 0, 1, opt, "fp32  all-core");   // Global AvgPool
    }

    fprintf(stdout, "\n");

    // --- vary threads, fixed: shape=(56x56x128) MaxPool k=3 s=2 p=1, fp32 ---
    {
        int threads[] = {1, 2, 4};
        for (int i = 0; i < 3; i++)
        {
            if (threads[i] > max_threads) continue;
            ncnn::Option opt = make_perf_option(threads[i], true, false, false);
            perf_pooling(56, 56, 128, 0, 3, 2, 1, 0, opt, "fp32  all-core");
        }
    }

    fprintf(stdout, "\n");

    // --- vary precision, fixed: shape=(56x56x128) MaxPool k=3 s=2 p=1, threads=1 ---
    {
        perf_pooling(56, 56, 128, 0, 3, 2, 1, 0, make_perf_option(1, true, false, false), "fp32  all-core");
        perf_pooling(56, 56, 128, 0, 3, 2, 1, 0, make_perf_option(1, true, true, false), "fp16  all-core");
#if NCNN_BF16
        perf_pooling(56, 56, 128, 0, 3, 2, 1, 0, make_perf_option(1, true, false, true), "bf16  all-core");
#endif
    }

    fprintf(stdout, "\n");

    // --- vary powersave, fixed: shape=(56x56x128) MaxPool k=3 s=2 p=1, threads=2, fp32 ---
    {
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
            perf_pooling(56, 56, 128, 0, 3, 2, 1, 0, opt, env_tag);
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
            perf_pooling_gpu(56, 56, 128, 0, 3, 2, 1, 0, opt, vkdev);
            perf_pooling_gpu(14, 14, 512, 0, 2, 2, 0, 0, opt, vkdev);
            perf_pooling_gpu(7, 7, 512, 1, 0, 0, 0, 1, opt, vkdev);
        }
        ncnn::destroy_gpu_instance();
    }
#endif // NCNN_VULKAN

    return 0;
}
