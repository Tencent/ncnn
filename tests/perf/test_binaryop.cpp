// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perftestutil.h"

#include <stdio.h>

// BinaryOp operation types (from ncnn layer definition):
// 0=Add, 1=Sub, 2=Mul, 3=Div, 4=Max, 5=Min, 6=Pow, 7=RSub, 8=RDiv, 9=RPow, 10=Atan2, 11=RAtan2

static const char* op_type_name(int op_type)
{
    static const char* names[] = {"Add", "Sub", "Mul", "Div", "Max", "Min", "Pow", "RSub", "RDiv", "RPow", "Atan2", "RAtan2"};
    if (op_type >= 0 && op_type < 12)
        return names[op_type];
    return "Unknown";
}

static void perf_binaryop(const ncnn::Mat& a, const ncnn::Mat& b, int op_type,
                          const ncnn::Option& opt, const char* env_tag)
{
    ncnn::ParamDict pd;
    pd.set(0, op_type); // op_type
    pd.set(1, 0);       // with_scalar
    pd.set(2, 0.f);     // b

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> inputs(2);
    inputs[0] = a;
    inputs[1] = b;

    PerfResult result;
    int ret = perf_layer_cpu("BinaryOp", pd, weights, inputs, opt, PERF_WARMUP_COUNT, PERF_RUN_COUNT, result);
    if (ret != 0)
        return;

    char shape_a[64], shape_b[64];
    format_shape(shape_a, sizeof(shape_a), a);
    format_shape(shape_b, sizeof(shape_b), b);
    char tag[256];
    snprintf(tag, sizeof(tag), "BinaryOp(%s)  a=%s  b=%s  threads=%d  %s",
             op_type_name(op_type), shape_a, shape_b, opt.num_threads, env_tag);
    print_perf_result(tag, result);
}

static void perf_binaryop_scalar(const ncnn::Mat& a, int op_type, float scalar_b,
                                 const ncnn::Option& opt, const char* env_tag)
{
    ncnn::ParamDict pd;
    pd.set(0, op_type);
    pd.set(1, 1);        // with_scalar
    pd.set(2, scalar_b); // b

    std::vector<ncnn::Mat> weights(0);

    PerfResult result;
    int ret = perf_layer_cpu("BinaryOp", pd, weights, a, opt, PERF_WARMUP_COUNT, PERF_RUN_COUNT, result);
    if (ret != 0)
        return;

    char shape_a[64];
    format_shape(shape_a, sizeof(shape_a), a);
    char tag[256];
    snprintf(tag, sizeof(tag), "BinaryOp(%s)  a=%s  b=scalar  threads=%d  %s",
             op_type_name(op_type), shape_a, opt.num_threads, env_tag);
    print_perf_result(tag, result);
}

#if NCNN_VULKAN
static void perf_binaryop_gpu(const ncnn::Mat& a, const ncnn::Mat& b, int op_type,
                              const ncnn::Option& opt, ncnn::VulkanDevice* vkdev)
{
    ncnn::ParamDict pd;
    pd.set(0, op_type);
    pd.set(1, 0);
    pd.set(2, 0.f);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> inputs(2);
    inputs[0] = a;
    inputs[1] = b;

    PerfResult result;
    int ret = perf_layer_gpu("BinaryOp", pd, weights, inputs, opt, vkdev, PERF_WARMUP_COUNT, PERF_RUN_COUNT, result);
    if (ret != 0)
        return;

    char shape_a[64], shape_b[64];
    format_shape(shape_a, sizeof(shape_a), a);
    format_shape(shape_b, sizeof(shape_b), b);
    char tag[256];
    snprintf(tag, sizeof(tag), "BinaryOp(%s)  a=%s  b=%s  GPU",
             op_type_name(op_type), shape_a, shape_b);
    print_perf_result(tag, result);
}
#endif // NCNN_VULKAN

int main()
{
    fprintf(stdout, "=== BinaryOp Performance Test ===\n\n");
    fflush(stdout);

    int max_threads = ncnn::get_physical_big_cpu_count();
    if (max_threads < 1) max_threads = 1;

    // --- vary shapes, fixed: Add, threads=1, fp32, all-core ---
    {
        ncnn::Option opt = make_perf_option(1, true, false, false);
        ncnn::Mat s1 = PerfMat(100000);
        perf_binaryop(s1, s1, 0, opt, "fp32  all-core");
        ncnn::Mat s2 = PerfMat(56, 56, 64);
        perf_binaryop(s2, s2, 0, opt, "fp32  all-core");
        ncnn::Mat s3 = PerfMat(28, 28, 128);
        perf_binaryop(s3, s3, 0, opt, "fp32  all-core");
        ncnn::Mat s4 = PerfMat(14, 14, 256);
        perf_binaryop(s4, s4, 0, opt, "fp32  all-core");
    }

    fprintf(stdout, "\n");

    // --- vary op type, fixed: shape=(56x56x64), threads=1, fp32 ---
    {
        ncnn::Option opt = make_perf_option(1, true, false, false);
        ncnn::Mat input = PerfMat(56, 56, 64);
        perf_binaryop(input, input, 0, opt, "fp32  all-core");       // Add
        perf_binaryop(input, input, 2, opt, "fp32  all-core");       // Mul
        perf_binaryop(input, input, 1, opt, "fp32  all-core");       // Sub
        perf_binaryop_scalar(input, 0, 0.5f, opt, "fp32  all-core"); // Add scalar
    }

    fprintf(stdout, "\n");

    // --- vary threads, fixed: shape=(56x56x64) Add, fp32 ---
    {
        ncnn::Mat input = PerfMat(56, 56, 64);
        int threads[] = {1, 2, 4};
        for (int i = 0; i < 3; i++)
        {
            if (threads[i] > max_threads) continue;
            ncnn::Option opt = make_perf_option(threads[i], true, false, false);
            perf_binaryop(input, input, 0, opt, "fp32  all-core");
        }
    }

    fprintf(stdout, "\n");

    // --- vary precision, fixed: shape=(56x56x64) Add, threads=1 ---
    {
        ncnn::Mat input = PerfMat(56, 56, 64);
        perf_binaryop(input, input, 0, make_perf_option(1, true, false, false), "fp32  all-core");
        perf_binaryop(input, input, 0, make_perf_option(1, true, true, false), "fp16  all-core");
#if NCNN_BF16
        perf_binaryop(input, input, 0, make_perf_option(1, true, false, true), "bf16  all-core");
#endif
    }

    fprintf(stdout, "\n");

    // --- vary powersave, fixed: shape=(56x56x64) Add, threads=2, fp32 ---
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
            perf_binaryop(input, input, 0, opt, env_tag);
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
            ncnn::Mat g1 = PerfMat(56, 56, 64);
            perf_binaryop_gpu(g1, g1, 0, opt, vkdev);
            ncnn::Mat g2 = PerfMat(14, 14, 256);
            perf_binaryop_gpu(g2, g2, 0, opt, vkdev);
        }
        ncnn::destroy_gpu_instance();
    }
#endif // NCNN_VULKAN

    return 0;
}
