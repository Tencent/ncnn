// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perftestutil.h"

#include <stdio.h>

static void perf_convolutiondepthwise(int w, int h, int c, int kernel, int dilation, int stride, int pad, int group,
                                      const ncnn::Option& opt, const char* env_tag)
{
    ncnn::Mat input = PerfMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, c);                           // num_output (same as input for depthwise)
    pd.set(1, kernel);                      // kernel_w
    pd.set(2, dilation);                    // dilation_w
    pd.set(3, stride);                      // stride_w
    pd.set(4, pad);                         // pad_w
    pd.set(5, 1);                           // bias_term
    pd.set(6, c * kernel * kernel / group); // weight_data_size per group * num_output
    pd.set(7, group);                       // group

    // For depthwise, weight_data_size = num_output / group * c / group * kernel * kernel * group
    // When group == c == num_output, that simplifies to kernel * kernel * c
    int weight_size = c * kernel * kernel;

    std::vector<ncnn::Mat> weights(2);
    weights[0] = PerfMat(weight_size);
    weights[1] = PerfMat(c);

    PerfResult result;
    int ret = perf_layer_cpu("ConvolutionDepthWise", pd, weights, input, opt, PERF_WARMUP_COUNT, PERF_RUN_COUNT, result);
    if (ret != 0)
        return;

    char shape_buf[64];
    char tag[256];
    format_shape(shape_buf, sizeof(shape_buf), input);
    snprintf(tag, sizeof(tag), "DWConv  %s k=%d s=%d g=%-3d  threads=%d  %s",
             shape_buf, kernel, stride, group, opt.num_threads, env_tag);
    print_perf_result(tag, result);
}

#if NCNN_VULKAN
static void perf_convolutiondepthwise_gpu(int w, int h, int c, int kernel, int dilation, int stride, int pad, int group,
        const ncnn::Option& opt, ncnn::VulkanDevice* vkdev)
{
    ncnn::Mat input = PerfMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, c);
    pd.set(1, kernel);
    pd.set(2, dilation);
    pd.set(3, stride);
    pd.set(4, pad);
    pd.set(5, 1);
    pd.set(6, c * kernel * kernel);
    pd.set(7, group);

    int weight_size = c * kernel * kernel;

    std::vector<ncnn::Mat> weights(2);
    weights[0] = PerfMat(weight_size);
    weights[1] = PerfMat(c);

    PerfResult result;
    int ret = perf_layer_gpu("ConvolutionDepthWise", pd, weights, input, opt, vkdev, PERF_WARMUP_COUNT, PERF_RUN_COUNT, result);
    if (ret != 0)
        return;

    char shape_buf[64];
    char tag[256];
    format_shape(shape_buf, sizeof(shape_buf), input);
    snprintf(tag, sizeof(tag), "DWConv  %s k=%d s=%d g=%-3d  GPU",
             shape_buf, kernel, stride, group);
    print_perf_result(tag, result);
}
#endif // NCNN_VULKAN

struct DWConvConfig
{
    int w, h, c, kernel, dilation, stride, pad, group;
};

int main()
{
    fprintf(stdout, "=== ConvolutionDepthWise Performance Test ===\n\n");
    fflush(stdout);

    int max_threads = ncnn::get_physical_big_cpu_count();
    if (max_threads < 1) max_threads = 1;

    // --- vary shapes, fixed: k=3 d=1 s=1 p=1 g=c, threads=1, fp32, all-core ---
    {
        ncnn::Option opt = make_perf_option(1, true, false, false);
        perf_convolutiondepthwise(112, 112, 32, 3, 1, 1, 1, 32, opt, "fp32  all-core");
        perf_convolutiondepthwise(56, 56, 64, 3, 1, 1, 1, 64, opt, "fp32  all-core");
        perf_convolutiondepthwise(28, 28, 128, 3, 1, 1, 1, 128, opt, "fp32  all-core");
        perf_convolutiondepthwise(14, 14, 256, 3, 1, 1, 1, 256, opt, "fp32  all-core");
        perf_convolutiondepthwise(7, 7, 512, 3, 1, 1, 1, 512, opt, "fp32  all-core");
    }

    fprintf(stdout, "\n");

    // --- vary kernel, fixed: shape=(56x56x64) g=64, threads=1, fp32 ---
    {
        ncnn::Option opt = make_perf_option(1, true, false, false);
        perf_convolutiondepthwise(56, 56, 64, 3, 1, 1, 1, 64, opt, "fp32  all-core"); // 3x3
        perf_convolutiondepthwise(56, 56, 64, 5, 1, 1, 2, 64, opt, "fp32  all-core"); // 5x5
    }

    fprintf(stdout, "\n");

    // --- vary threads, fixed: shape=(56x56x64) k=3 s=1 g=64, fp32 ---
    {
        int threads[] = {1, 2, 4};
        for (int i = 0; i < 3; i++)
        {
            if (threads[i] > max_threads) continue;
            ncnn::Option opt = make_perf_option(threads[i], true, false, false);
            perf_convolutiondepthwise(56, 56, 64, 3, 1, 1, 1, 64, opt, "fp32  all-core");
        }
    }

    fprintf(stdout, "\n");

    // --- vary precision, fixed: shape=(56x56x64) k=3 s=1 g=64, threads=1 ---
    {
        perf_convolutiondepthwise(56, 56, 64, 3, 1, 1, 1, 64, make_perf_option(1, true, false, false), "fp32  all-core");
        perf_convolutiondepthwise(56, 56, 64, 3, 1, 1, 1, 64, make_perf_option(1, true, true, false), "fp16  all-core");
#if NCNN_BF16
        perf_convolutiondepthwise(56, 56, 64, 3, 1, 1, 1, 64, make_perf_option(1, true, false, true), "bf16  all-core");
#endif
    }

    fprintf(stdout, "\n");

    // --- vary powersave, fixed: shape=(56x56x64) k=3 s=1 g=64, threads=2, fp32 ---
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
            perf_convolutiondepthwise(56, 56, 64, 3, 1, 1, 1, 64, opt, env_tag);
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
            perf_convolutiondepthwise_gpu(56, 56, 64, 3, 1, 1, 1, 64, opt, vkdev);
            perf_convolutiondepthwise_gpu(14, 14, 256, 3, 1, 1, 1, 256, opt, vkdev);
            perf_convolutiondepthwise_gpu(7, 7, 512, 3, 1, 1, 1, 512, opt, vkdev);
        }
        ncnn::destroy_gpu_instance();
    }
#endif // NCNN_VULKAN

    return 0;
}
