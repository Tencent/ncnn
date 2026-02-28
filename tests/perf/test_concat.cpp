// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perftestutil.h"

#include <stdio.h>

static void perf_concat(int axis, const std::vector<ncnn::Mat>& inputs,
                        const ncnn::Option& opt, const char* env_tag)
{
    ncnn::ParamDict pd;
    pd.set(0, axis); // axis

    std::vector<ncnn::Mat> weights(0);

    PerfResult result;
    int ret = perf_layer_cpu("Concat", pd, weights, inputs, opt, PERF_WARMUP_COUNT, PERF_RUN_COUNT, result);
    if (ret != 0)
        return;

    char shape_buf[64];
    format_shape(shape_buf, sizeof(shape_buf), inputs[0]);
    char tag[256];
    snprintf(tag, sizeof(tag), "Concat  x%d  %s  axis=%d  threads=%d  %s",
             (int)inputs.size(), shape_buf, axis, opt.num_threads, env_tag);
    print_perf_result(tag, result);
}

#if NCNN_VULKAN
static void perf_concat_gpu(int axis, const std::vector<ncnn::Mat>& inputs,
                            const ncnn::Option& opt, ncnn::VulkanDevice* vkdev)
{
    ncnn::ParamDict pd;
    pd.set(0, axis);

    std::vector<ncnn::Mat> weights(0);

    PerfResult result;
    int ret = perf_layer_gpu("Concat", pd, weights, inputs, opt, vkdev, PERF_WARMUP_COUNT, PERF_RUN_COUNT, result);
    if (ret != 0)
        return;

    char shape_buf[64];
    format_shape(shape_buf, sizeof(shape_buf), inputs[0]);
    char tag[256];
    snprintf(tag, sizeof(tag), "Concat  x%d  %s  axis=%d  GPU",
             (int)inputs.size(), shape_buf, axis);
    print_perf_result(tag, result);
}
#endif // NCNN_VULKAN

int main()
{
    fprintf(stdout, "=== Concat Performance Test ===\n\n");
    fflush(stdout);

    int max_threads = ncnn::get_physical_big_cpu_count();
    if (max_threads < 1) max_threads = 1;

    // --- vary shapes, fixed: x2 axis=0, threads=1, fp32, all-core ---
    {
        ncnn::Option opt = make_perf_option(1, true, false, false);

        {
            std::vector<ncnn::Mat> inputs(2);
            inputs[0] = PerfMat(56, 56, 64);
            inputs[1] = PerfMat(56, 56, 64);
            perf_concat(0, inputs, opt, "fp32  all-core");
        }
        {
            std::vector<ncnn::Mat> inputs(2);
            inputs[0] = PerfMat(28, 28, 128);
            inputs[1] = PerfMat(28, 28, 128);
            perf_concat(0, inputs, opt, "fp32  all-core");
        }
        {
            std::vector<ncnn::Mat> inputs(2);
            inputs[0] = PerfMat(14, 14, 256);
            inputs[1] = PerfMat(14, 14, 256);
            perf_concat(0, inputs, opt, "fp32  all-core");
        }
    }

    fprintf(stdout, "\n");

    // --- vary num_inputs, fixed: shape=(56x56x64) axis=0, threads=1, fp32 ---
    {
        ncnn::Option opt = make_perf_option(1, true, false, false);

        {
            std::vector<ncnn::Mat> inputs(2);
            for (int i = 0; i < 2; i++) inputs[i] = PerfMat(56, 56, 64);
            perf_concat(0, inputs, opt, "fp32  all-core");
        }
        {
            std::vector<ncnn::Mat> inputs(4);
            for (int i = 0; i < 4; i++) inputs[i] = PerfMat(56, 56, 64);
            perf_concat(0, inputs, opt, "fp32  all-core");
        }
    }

    fprintf(stdout, "\n");

    // --- vary threads, fixed: shape=(56x56x64) x2 axis=0, fp32 ---
    {
        std::vector<ncnn::Mat> inputs(2);
        inputs[0] = PerfMat(56, 56, 64);
        inputs[1] = PerfMat(56, 56, 64);
        int threads[] = {1, 2, 4};
        for (int i = 0; i < 3; i++)
        {
            if (threads[i] > max_threads) continue;
            ncnn::Option opt = make_perf_option(threads[i], true, false, false);
            perf_concat(0, inputs, opt, "fp32  all-core");
        }
    }

    fprintf(stdout, "\n");

    // --- vary precision, fixed: shape=(56x56x64) x2 axis=0, threads=1 ---
    {
        std::vector<ncnn::Mat> inputs(2);
        inputs[0] = PerfMat(56, 56, 64);
        inputs[1] = PerfMat(56, 56, 64);
        perf_concat(0, inputs, make_perf_option(1, true, false, false), "fp32  all-core");
        perf_concat(0, inputs, make_perf_option(1, true, true, false), "fp16  all-core");
#if NCNN_BF16
        perf_concat(0, inputs, make_perf_option(1, true, false, true), "bf16  all-core");
#endif
    }

    fprintf(stdout, "\n");

    // --- vary powersave, fixed: shape=(56x56x64) x2 axis=0, threads=2, fp32 ---
    {
        std::vector<ncnn::Mat> inputs(2);
        inputs[0] = PerfMat(56, 56, 64);
        inputs[1] = PerfMat(56, 56, 64);
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
            perf_concat(0, inputs, opt, env_tag);
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

            {
                std::vector<ncnn::Mat> inputs(2);
                inputs[0] = PerfMat(56, 56, 64);
                inputs[1] = PerfMat(56, 56, 64);
                perf_concat_gpu(0, inputs, opt, vkdev);
            }
            {
                std::vector<ncnn::Mat> inputs(2);
                inputs[0] = PerfMat(14, 14, 256);
                inputs[1] = PerfMat(14, 14, 256);
                perf_concat_gpu(0, inputs, opt, vkdev);
            }
        }
        ncnn::destroy_gpu_instance();
    }
#endif // NCNN_VULKAN

    return 0;
}
