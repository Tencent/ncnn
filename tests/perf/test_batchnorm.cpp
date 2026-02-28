// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perftestutil.h"

#include <stdio.h>

static void perf_batchnorm(int w, int h, int c, const ncnn::Option& opt, const char* env_tag)
{
    ncnn::Mat input = PerfMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, c); // channels

    // BatchNorm weights: slope, mean, variance, bias
    std::vector<ncnn::Mat> weights(4);
    weights[0] = PerfMat(c, 1.0f); // slope (gamma)
    weights[1] = PerfMat(c, 0.0f); // mean
    weights[2] = PerfMat(c, 1.0f); // variance
    weights[3] = PerfMat(c, 0.0f); // bias (beta)

    PerfResult result;
    int ret = perf_layer_cpu("BatchNorm", pd, weights, input, opt, PERF_WARMUP_COUNT, PERF_RUN_COUNT, result);
    if (ret != 0)
        return;

    char shape_buf[64];
    char tag[256];
    format_shape(shape_buf, sizeof(shape_buf), input);
    snprintf(tag, sizeof(tag), "BatchNorm  %s threads=%d  %s",
             shape_buf, opt.num_threads, env_tag);
    print_perf_result(tag, result);
}

#if NCNN_VULKAN
static void perf_batchnorm_gpu(int w, int h, int c,
                               const ncnn::Option& opt, ncnn::VulkanDevice* vkdev)
{
    ncnn::Mat input = PerfMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, c);

    std::vector<ncnn::Mat> weights(4);
    weights[0] = PerfMat(c, 1.0f);
    weights[1] = PerfMat(c, 0.0f);
    weights[2] = PerfMat(c, 1.0f);
    weights[3] = PerfMat(c, 0.0f);

    PerfResult result;
    int ret = perf_layer_gpu("BatchNorm", pd, weights, input, opt, vkdev, PERF_WARMUP_COUNT, PERF_RUN_COUNT, result);
    if (ret != 0)
        return;

    char shape_buf[64];
    char tag[256];
    format_shape(shape_buf, sizeof(shape_buf), input);
    snprintf(tag, sizeof(tag), "BatchNorm  %s GPU", shape_buf);
    print_perf_result(tag, result);
}
#endif // NCNN_VULKAN

struct BNConfig
{
    int w, h, c;
};

int main()
{
    fprintf(stdout, "=== BatchNorm Performance Test ===\n\n");
    fflush(stdout);

    int max_threads = ncnn::get_physical_big_cpu_count();
    if (max_threads < 1) max_threads = 1;

    // --- vary shapes, fixed: threads=1, fp32, all-core ---
    {
        ncnn::Option opt = make_perf_option(1, true, false, false);
        perf_batchnorm(224, 224, 3, opt, "fp32  all-core");
        perf_batchnorm(112, 112, 32, opt, "fp32  all-core");
        perf_batchnorm(56, 56, 64, opt, "fp32  all-core");
        perf_batchnorm(28, 28, 128, opt, "fp32  all-core");
        perf_batchnorm(14, 14, 256, opt, "fp32  all-core");
    }

    fprintf(stdout, "\n");

    // --- vary threads, fixed: shape=(56x56x64), fp32 ---
    {
        int threads[] = {1, 2, 4};
        for (int i = 0; i < 3; i++)
        {
            if (threads[i] > max_threads) continue;
            ncnn::Option opt = make_perf_option(threads[i], true, false, false);
            perf_batchnorm(56, 56, 64, opt, "fp32  all-core");
        }
    }

    fprintf(stdout, "\n");

    // --- vary precision, fixed: shape=(56x56x64), threads=1 ---
    {
        perf_batchnorm(56, 56, 64, make_perf_option(1, true, false, false), "fp32  all-core");
        perf_batchnorm(56, 56, 64, make_perf_option(1, true, true, false), "fp16  all-core");
#if NCNN_BF16
        perf_batchnorm(56, 56, 64, make_perf_option(1, true, false, true), "bf16  all-core");
#endif
    }

    fprintf(stdout, "\n");

    // --- vary powersave, fixed: shape=(56x56x64), threads=2, fp32 ---
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
            perf_batchnorm(56, 56, 64, opt, env_tag);
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
            perf_batchnorm_gpu(56, 56, 64, opt, vkdev);
            perf_batchnorm_gpu(14, 14, 256, opt, vkdev);
            perf_batchnorm_gpu(7, 7, 512, opt, vkdev);
        }
        ncnn::destroy_gpu_instance();
    }
#endif // NCNN_VULKAN

    return 0;
}
