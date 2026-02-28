// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perftestutil.h"

#include <stdio.h>
#include <stdlib.h>

static void perf_convolution(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad,
                             const ncnn::Option& opt, const char* env_tag)
{
    ncnn::Mat input = PerfMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, outch);                      // num_output
    pd.set(1, kernel);                     // kernel_w
    pd.set(2, dilation);                   // dilation_w
    pd.set(3, stride);                     // stride_w
    pd.set(4, pad);                        // pad_w
    pd.set(5, 1);                          // bias_term
    pd.set(6, outch * c * kernel * kernel); // weight_data_size

    std::vector<ncnn::Mat> weights(2);
    weights[0] = PerfMat(outch * c * kernel * kernel);
    weights[1] = PerfMat(outch);

    PerfResult result;
    int ret = perf_layer_cpu("Convolution", pd, weights, input, opt, PERF_WARMUP_COUNT, PERF_RUN_COUNT, result);
    if (ret != 0)
        return;

    char shape_buf[64];
    char tag[256];
    format_shape(shape_buf, sizeof(shape_buf), input);
    snprintf(tag, sizeof(tag), "Conv  %s out=%-4d k=%d d=%d s=%d p=%-2d  threads=%d  %s",
             shape_buf, outch, kernel, dilation, stride, pad, opt.num_threads, env_tag);
    print_perf_result(tag, result);
}

#if NCNN_VULKAN
static void perf_convolution_gpu(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad,
                                 const ncnn::Option& opt, ncnn::VulkanDevice* vkdev)
{
    ncnn::Mat input = PerfMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, kernel);
    pd.set(2, dilation);
    pd.set(3, stride);
    pd.set(4, pad);
    pd.set(5, 1);
    pd.set(6, outch * c * kernel * kernel);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = PerfMat(outch * c * kernel * kernel);
    weights[1] = PerfMat(outch);

    PerfResult result;
    int ret = perf_layer_gpu("Convolution", pd, weights, input, opt, vkdev, PERF_WARMUP_COUNT, PERF_RUN_COUNT, result);
    if (ret != 0)
        return;

    char shape_buf[64];
    char tag[256];
    format_shape(shape_buf, sizeof(shape_buf), input);
    snprintf(tag, sizeof(tag), "Conv  %s out=%-4d k=%d d=%d s=%d p=%-2d  GPU",
             shape_buf, outch, kernel, dilation, stride, pad);
    print_perf_result(tag, result);
}
#endif // NCNN_VULKAN

// typical convolution configurations from common networks
struct ConvConfig
{
    int w, h, c, outch, kernel, dilation, stride, pad;
};

int main()
{
    fprintf(stdout, "=== Convolution Performance Test ===\n\n");
    fflush(stdout);

    int max_threads = ncnn::get_physical_big_cpu_count();
    if (max_threads < 1) max_threads = 1;

    // --- vary shapes, fixed: out=64 k=3 d=1 s=1 p=1, threads=1, fp32, all-core ---
    {
        ncnn::Option opt = make_perf_option(1, true, false, false);
        static const ConvConfig shapes[] = {
            {224, 224, 3, 64, 7, 1, 2, 3},   // ResNet first conv
            {56, 56, 64, 64, 3, 1, 1, 1},    // ResNet block
            {28, 28, 128, 128, 3, 1, 1, 1},
            {14, 14, 256, 256, 3, 1, 1, 1},
            {7, 7, 512, 512, 3, 1, 1, 1},
        };
        for (int i = 0; i < 5; i++)
        {
            const ConvConfig& cfg = shapes[i];
            perf_convolution(cfg.w, cfg.h, cfg.c, cfg.outch, cfg.kernel, cfg.dilation, cfg.stride, cfg.pad, opt, "fp32  all-core");
        }
    }

    fprintf(stdout, "\n");

    // --- vary conv params, fixed: shape=(56x56x64), threads=1, fp32 ---
    {
        ncnn::Option opt = make_perf_option(1, true, false, false);
        perf_convolution(56, 56, 64, 64, 3, 1, 1, 1, opt, "fp32  all-core");   // k=3 s=1
        perf_convolution(56, 56, 64, 128, 3, 1, 2, 1, opt, "fp32  all-core");  // k=3 s=2 downsample
        perf_convolution(56, 56, 64, 256, 1, 1, 1, 0, opt, "fp32  all-core");  // k=1 pointwise
        perf_convolution(56, 56, 64, 64, 5, 1, 1, 2, opt, "fp32  all-core");   // k=5 s=1
    }

    fprintf(stdout, "\n");

    // --- vary threads, fixed: shape=(56x56x64) out=64 k=3 d=1 s=1 p=1, fp32 ---
    {
        int threads[] = {1, 2, 4};
        for (int i = 0; i < 3; i++)
        {
            if (threads[i] > max_threads) continue;
            ncnn::Option opt = make_perf_option(threads[i], true, false, false);
            perf_convolution(56, 56, 64, 64, 3, 1, 1, 1, opt, "fp32  all-core");
        }
    }

    fprintf(stdout, "\n");

    // --- vary precision, fixed: shape=(56x56x64) out=64 k=3 d=1 s=1 p=1, threads=1 ---
    {
        perf_convolution(56, 56, 64, 64, 3, 1, 1, 1, make_perf_option(1, true, false, false), "fp32  all-core");
        perf_convolution(56, 56, 64, 64, 3, 1, 1, 1, make_perf_option(1, true, true, false), "fp16  all-core");
#if NCNN_BF16
        perf_convolution(56, 56, 64, 64, 3, 1, 1, 1, make_perf_option(1, true, false, true), "bf16  all-core");
#endif
    }

    fprintf(stdout, "\n");

    // --- vary powersave, fixed: shape=(56x56x64) out=64 k=3 d=1 s=1 p=1, threads=2, fp32 ---
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
            perf_convolution(56, 56, 64, 64, 3, 1, 1, 1, opt, env_tag);
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
            perf_convolution_gpu(56, 56, 64, 64, 3, 1, 1, 1, opt, vkdev);
            perf_convolution_gpu(14, 14, 256, 256, 3, 1, 1, 1, opt, vkdev);
            perf_convolution_gpu(7, 7, 512, 512, 3, 1, 1, 1, opt, vkdev);
        }
        ncnn::destroy_gpu_instance();
    }
#endif // NCNN_VULKAN

    return 0;
}
