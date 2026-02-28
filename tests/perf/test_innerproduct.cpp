// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perftestutil.h"

#include <stdio.h>

static void perf_innerproduct(int w, int h, int c, int outch, int bias,
                              const ncnn::Option& opt, const char* env_tag)
{
    ncnn::Mat input;
    int weight_data_size;
    if (h == 0 && c == 0)
    {
        input = PerfMat(w);
        weight_data_size = outch * w;
    }
    else if (c == 0)
    {
        input = PerfMat(w, h);
        weight_data_size = outch * w * h;
    }
    else
    {
        input = PerfMat(w, h, c);
        weight_data_size = outch * w * h * c;
    }

    ncnn::ParamDict pd;
    pd.set(0, outch);           // num_output
    pd.set(1, bias);            // bias_term
    pd.set(2, weight_data_size); // weight_data_size

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = PerfMat(weight_data_size);
    if (bias)
        weights[1] = PerfMat(outch);

    PerfResult result;
    int ret = perf_layer_cpu("InnerProduct", pd, weights, input, opt, PERF_WARMUP_COUNT, PERF_RUN_COUNT, result);
    if (ret != 0)
        return;

    char shape_buf[64];
    format_shape(shape_buf, sizeof(shape_buf), input);
    char tag[256];
    snprintf(tag, sizeof(tag), "FC  in=%s  out=%d  bias=%d  threads=%d  %s",
             shape_buf, outch, bias, opt.num_threads, env_tag);
    print_perf_result(tag, result);
}

#if NCNN_VULKAN
static void perf_innerproduct_gpu(int w, int h, int c, int outch, int bias,
                                  const ncnn::Option& opt, ncnn::VulkanDevice* vkdev)
{
    ncnn::Mat input;
    int weight_data_size;
    if (h == 0 && c == 0)
    {
        input = PerfMat(w);
        weight_data_size = outch * w;
    }
    else if (c == 0)
    {
        input = PerfMat(w, h);
        weight_data_size = outch * w * h;
    }
    else
    {
        input = PerfMat(w, h, c);
        weight_data_size = outch * w * h * c;
    }

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, bias);
    pd.set(2, weight_data_size);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = PerfMat(weight_data_size);
    if (bias)
        weights[1] = PerfMat(outch);

    PerfResult result;
    int ret = perf_layer_gpu("InnerProduct", pd, weights, input, opt, vkdev, PERF_WARMUP_COUNT, PERF_RUN_COUNT, result);
    if (ret != 0)
        return;

    char shape_buf[64];
    format_shape(shape_buf, sizeof(shape_buf), input);
    char tag[256];
    snprintf(tag, sizeof(tag), "FC  in=%s  out=%d  bias=%d  GPU", shape_buf, outch, bias);
    print_perf_result(tag, result);
}
#endif // NCNN_VULKAN

struct FCConfig
{
    int w, h, c, outch, bias;
};

int main()
{
    fprintf(stdout, "=== InnerProduct (Fully Connected) Performance Test ===\n\n");
    fflush(stdout);

    int max_threads = ncnn::get_physical_big_cpu_count();
    if (max_threads < 1) max_threads = 1;

    // --- vary shapes, fixed: out=4096 bias=1, threads=1, fp32, all-core ---
    {
        ncnn::Option opt = make_perf_option(1, true, false, false);
        perf_innerproduct(25088, 0, 0, 4096, 1, opt, "fp32  all-core");  // VGG-style
        perf_innerproduct(4096, 0, 0, 1000, 1, opt, "fp32  all-core");
        perf_innerproduct(2048, 0, 0, 1000, 1, opt, "fp32  all-core");   // ResNet final
        perf_innerproduct(7, 7, 512, 4096, 1, opt, "fp32  all-core");    // 3D input
    }

    fprintf(stdout, "\n");

    // --- vary output size, fixed: input=(25088) bias=1, threads=1, fp32 ---
    {
        ncnn::Option opt = make_perf_option(1, true, false, false);
        perf_innerproduct(25088, 0, 0, 1000, 1, opt, "fp32  all-core");
        perf_innerproduct(25088, 0, 0, 4096, 1, opt, "fp32  all-core");
        perf_innerproduct(25088, 0, 0, 256, 1, opt, "fp32  all-core");
    }

    fprintf(stdout, "\n");

    // --- vary threads, fixed: input=(25088) out=4096 bias=1, fp32 ---
    {
        int threads[] = {1, 2, 4};
        for (int i = 0; i < 3; i++)
        {
            if (threads[i] > max_threads) continue;
            ncnn::Option opt = make_perf_option(threads[i], true, false, false);
            perf_innerproduct(25088, 0, 0, 4096, 1, opt, "fp32  all-core");
        }
    }

    fprintf(stdout, "\n");

    // --- vary precision, fixed: input=(25088) out=4096 bias=1, threads=1 ---
    {
        perf_innerproduct(25088, 0, 0, 4096, 1, make_perf_option(1, true, false, false), "fp32  all-core");
        perf_innerproduct(25088, 0, 0, 4096, 1, make_perf_option(1, true, true, false), "fp16  all-core");
#if NCNN_BF16
        perf_innerproduct(25088, 0, 0, 4096, 1, make_perf_option(1, true, false, true), "bf16  all-core");
#endif
    }

    fprintf(stdout, "\n");

    // --- vary powersave, fixed: input=(25088) out=4096 bias=1, threads=2, fp32 ---
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
            perf_innerproduct(25088, 0, 0, 4096, 1, opt, env_tag);
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
            perf_innerproduct_gpu(25088, 0, 0, 4096, 1, opt, vkdev);
            perf_innerproduct_gpu(4096, 0, 0, 1000, 1, opt, vkdev);
        }
        ncnn::destroy_gpu_instance();
    }
#endif // NCNN_VULKAN

    return 0;
}
