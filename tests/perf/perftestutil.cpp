// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perftestutil.h"

#include "layer.h"
#include "modelbin.h"

#include <stdio.h>
#include <string.h>

#if NCNN_VULKAN
#include "command.h"
#include "gpu.h"
#endif // NCNN_VULKAN

void FillMat(ncnn::Mat& m, float v)
{
    for (size_t i = 0; i < m.total(); i++)
    {
        m[i] = v;
    }
}

ncnn::Mat PerfMat(int w, float v)
{
    ncnn::Mat m(w);
    FillMat(m, v);
    return m;
}

ncnn::Mat PerfMat(int w, int h, float v)
{
    ncnn::Mat m(w, h);
    FillMat(m, v);
    return m;
}

ncnn::Mat PerfMat(int w, int h, int c, float v)
{
    ncnn::Mat m(w, h, c);
    FillMat(m, v);
    return m;
}

ncnn::Mat PerfMat(int w, int h, int d, int c, float v)
{
    ncnn::Mat m(w, h, d, c);
    FillMat(m, v);
    return m;
}

void sort_doubles(double* arr, int n)
{
    for (int i = 1; i < n; i++)
    {
        double key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key)
        {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// convert input mat to the optimal layout that the layer expects
// this handles fp16/bf16 cast and packing conversion
// so that these costs are excluded from timing
static void convert_input_layout(const ncnn::Mat& src, ncnn::Mat& dst, const ncnn::Option& opt, const ncnn::Layer* op)
{
    ncnn::Mat casted;

    // clang-format off
    // *INDENT-OFF*
#if NCNN_ARM82
    if (opt.use_fp16_storage && ncnn::cpu_support_arm_asimdhp() && op->support_fp16_storage)
    {
        ncnn::cast_float32_to_float16(src, casted, opt);
    }
    else
#endif // NCNN_ARM82
#if NCNN_VFPV4
    if (opt.use_fp16_storage && !opt.use_bf16_storage && ncnn::cpu_support_arm_vfpv4() && op->support_fp16_storage)
    {
        ncnn::cast_float32_to_float16(src, casted, opt);
    }
    else
#endif // NCNN_VFPV4
#if NCNN_ZFH
    if (opt.use_fp16_storage && (ncnn::cpu_support_riscv_zvfh() || (!ncnn::cpu_support_riscv_v() && ncnn::cpu_support_riscv_zfh())) && op->support_fp16_storage)
    {
        ncnn::cast_float32_to_float16(src, casted, opt);
    }
    else
#endif // NCNN_ZFH
#if NCNN_BF16
    if (opt.use_bf16_storage && op->support_bf16_storage)
    {
        ncnn::cast_float32_to_bfloat16(src, casted, opt);
    }
    else
#endif // NCNN_BF16
    if (opt.use_fp16_storage && op->support_fp16_storage)
    {
        ncnn::cast_float32_to_float16(src, casted, opt);
    }
    else
    {
        casted = src;
    }
    // *INDENT-ON*
    // clang-format on

    // packing conversion
    if (opt.use_packing_layout && op->support_packing)
    {
        int dims = casted.dims;
        int elemcount = 0;
        if (dims == 1) elemcount = casted.elempack * casted.w;
        if (dims == 2) elemcount = casted.elempack * casted.h;
        if (dims == 3 || dims == 4) elemcount = casted.elempack * casted.c;

        int elembits = casted.elembits();
        int dst_elempack = 1;

        if (elembits == 32)
        {
#if NCNN_AVX512
            if (elemcount % 16 == 0 && ncnn::cpu_support_x86_avx512())
                dst_elempack = 16;
            else if (elemcount % 8 == 0 && ncnn::cpu_support_x86_avx())
                dst_elempack = 8;
            else if (elemcount % 4 == 0)
                dst_elempack = 4;
#elif NCNN_AVX
            if (elemcount % 8 == 0 && ncnn::cpu_support_x86_avx())
                dst_elempack = 8;
            else if (elemcount % 4 == 0)
                dst_elempack = 4;
#elif NCNN_RVV || NCNN_XTHEADVECTOR
            const int packn = ncnn::cpu_riscv_vlenb() / 4;
            if (elemcount % packn == 0)
                dst_elempack = packn;
#else
            if (elemcount % 4 == 0)
                dst_elempack = 4;
#endif
        }
        if (elembits == 16)
        {
#if NCNN_ARM82
            if (elemcount % 8 == 0 && ncnn::cpu_support_arm_asimdhp() && opt.use_fp16_arithmetic && op->support_fp16_storage)
                dst_elempack = 8;
            else if (elemcount % 4 == 0)
                dst_elempack = 4;
#elif NCNN_RVV || NCNN_XTHEADVECTOR
            const int packn = ncnn::cpu_riscv_vlenb() / 2;
            if (elemcount % packn == 0)
                dst_elempack = packn;
#else
            if (elemcount % 4 == 0)
                dst_elempack = 4;
#endif
        }
        if (elembits == 8)
        {
#if NCNN_RVV || NCNN_XTHEADVECTOR
            const int packn = ncnn::cpu_riscv_vlenb() / 1;
            if (elemcount % packn == 0)
                dst_elempack = packn;
#else
            if (elemcount % 8 == 0)
                dst_elempack = 8;
#endif
        }

        ncnn::convert_packing(casted, dst, dst_elempack, opt);
    }
    else
    {
        dst = casted;
    }
}

// run a single forward pass (pure compute, no conversion)
static int run_layer_forward_cpu(ncnn::Layer* op,
                                 const std::vector<ncnn::Mat>& converted_inputs,
                                 const ncnn::Option& opt)
{
    if (op->one_blob_only)
    {
        if (op->support_inplace)
        {
            ncnn::Mat blob = converted_inputs[0].clone();
            return op->forward_inplace(blob, opt);
        }
        else
        {
            ncnn::Mat out;
            return op->forward(converted_inputs[0], out, opt);
        }
    }
    else
    {
        if (op->support_inplace)
        {
            std::vector<ncnn::Mat> blobs(converted_inputs.size());
            for (size_t i = 0; i < converted_inputs.size(); i++)
            {
                blobs[i] = converted_inputs[i].clone();
            }
            return op->forward_inplace(blobs, opt);
        }
        else
        {
            std::vector<ncnn::Mat> outputs;
            return op->forward(converted_inputs, outputs, opt);
        }
    }
}

// calibrate inner loop count so that total time per iteration >= PERF_MIN_TOTAL_MS
static int calibrate_loop_count(ncnn::Layer* op,
                                const std::vector<ncnn::Mat>& converted_inputs,
                                const ncnn::Option& opt)
{
    // time a single forward to estimate duration
    double t0 = ncnn::get_current_time();
    run_layer_forward_cpu(op, converted_inputs, opt);
    double t1 = ncnn::get_current_time();
    double single_ms = t1 - t0;

    if (single_ms >= PERF_MIN_TOTAL_MS)
        return 1;

    if (single_ms < 0.001)
        single_ms = 0.001;

    int loops = (int)(PERF_MIN_TOTAL_MS / single_ms) + 1;
    if (loops < 1)
        loops = 1;

    return loops;
}

int perf_layer_cpu(const char* layer_type, const ncnn::ParamDict& pd,
                   const std::vector<ncnn::Mat>& weights,
                   const std::vector<ncnn::Mat>& inputs,
                   const ncnn::Option& opt,
                   int warmup_count, int run_count,
                   PerfResult& result)
{
    ncnn::Layer* op = ncnn::create_layer_cpu(layer_type);
    if (!op)
    {
        fprintf(stderr, "perf_layer_cpu: create_layer_cpu(%s) failed\n", layer_type);
        return -1;
    }

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());
    op->load_model(mb);

    op->create_pipeline(opt);

    // pre-convert inputs to the layout the layer expects
    // so that cast/packing cost is excluded from timing
    std::vector<ncnn::Mat> converted(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
    {
        convert_input_layout(inputs[i], converted[i], opt, op);
    }

    // warmup
    for (int i = 0; i < warmup_count; i++)
    {
        int ret = run_layer_forward_cpu(op, converted, opt);
        if (ret != 0)
        {
            fprintf(stderr, "perf_layer_cpu: warmup forward failed ret=%d\n", ret);
            op->destroy_pipeline(opt);
            delete op;
            return -1;
        }
    }

    // calibrate inner loop count for short ops
    int inner_loops = calibrate_loop_count(op, converted, opt);

    double* times = new double[run_count];
    double time_sum = 0;
    double time_min_val = DBL_MAX;
    double time_max_val = -DBL_MAX;

    for (int i = 0; i < run_count; i++)
    {
        double start = ncnn::get_current_time();

        for (int k = 0; k < inner_loops; k++)
        {
            run_layer_forward_cpu(op, converted, opt);
        }

        double end = ncnn::get_current_time();
        double t = (end - start) / inner_loops;

        times[i] = t;
        time_sum += t;
        if (t < time_min_val) time_min_val = t;
        if (t > time_max_val) time_max_val = t;
    }

    sort_doubles(times, run_count);
    double time_median_val;
    if (run_count % 2 == 0)
        time_median_val = (times[run_count / 2 - 1] + times[run_count / 2]) / 2.0;
    else
        time_median_val = times[run_count / 2];

    result.time_min = time_min_val;
    result.time_max = time_max_val;
    result.time_avg = time_sum / run_count;
    result.time_median = time_median_val;
    result.loop_count = inner_loops;

    delete[] times;

    op->destroy_pipeline(opt);
    delete op;

    return 0;
}

int perf_layer_cpu(const char* layer_type, const ncnn::ParamDict& pd,
                   const std::vector<ncnn::Mat>& weights,
                   const ncnn::Mat& input,
                   const ncnn::Option& opt,
                   int warmup_count, int run_count,
                   PerfResult& result)
{
    std::vector<ncnn::Mat> inputs(1);
    inputs[0] = input;
    return perf_layer_cpu(layer_type, pd, weights, inputs, opt, warmup_count, run_count, result);
}

#if NCNN_VULKAN

static void run_layer_forward_gpu(ncnn::Layer* op,
                                  const std::vector<ncnn::VkMat>& vk_inputs,
                                  ncnn::VkCompute& cmd, const ncnn::Option& opt)
{
    if (op->one_blob_only)
    {
        if (op->support_inplace)
        {
            ncnn::VkMat vk_in = vk_inputs[0];
            op->forward_inplace(vk_in, cmd, opt);
        }
        else
        {
            ncnn::VkMat vk_out;
            op->forward(vk_inputs[0], vk_out, cmd, opt);
        }
    }
    else
    {
        if (op->support_inplace)
        {
            std::vector<ncnn::VkMat> vk_in(vk_inputs);
            op->forward_inplace(vk_in, cmd, opt);
        }
        else
        {
            std::vector<ncnn::VkMat> vk_out(1);
            op->forward(vk_inputs, vk_out, cmd, opt);
        }
    }
}

int perf_layer_gpu(const char* layer_type, const ncnn::ParamDict& pd,
                   const std::vector<ncnn::Mat>& weights,
                   const std::vector<ncnn::Mat>& inputs,
                   const ncnn::Option& _opt,
                   ncnn::VulkanDevice* vkdev,
                   int warmup_count, int run_count,
                   PerfResult& result)
{
    ncnn::Layer* op = ncnn::create_layer_vulkan(layer_type);
    if (!op)
    {
        return -1;
    }

    op->load_param(pd);

    if (!op->support_vulkan)
    {
        delete op;
        return -1;
    }

    op->vkdev = vkdev;

    ncnn::Option opt = _opt;
    opt.use_vulkan_compute = true;

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    // respect device capabilities
    if (!vkdev->info.support_fp16_packed()) opt.use_fp16_packed = false;
    if (!vkdev->info.support_fp16_storage()) opt.use_fp16_storage = false;
    if (!vkdev->info.support_fp16_uniform()) opt.use_fp16_uniform = false;
    if (!vkdev->info.support_fp16_arithmetic()) opt.use_fp16_arithmetic = false;
    if (!vkdev->info.support_int8_packed()) opt.use_int8_packed = false;
    if (!vkdev->info.support_int8_storage()) opt.use_int8_storage = false;
    if (!vkdev->info.support_int8_uniform()) opt.use_int8_uniform = false;
    if (!vkdev->info.support_int8_arithmetic()) opt.use_int8_arithmetic = false;
    if (!vkdev->info.support_cooperative_matrix()) opt.use_cooperative_matrix = false;
    if (!vkdev->info.support_subgroup_ops()) opt.use_subgroup_ops = false;

    ncnn::VkWeightAllocator weight_vkallocator(vkdev);
    ncnn::VkWeightStagingAllocator weight_staging_vkallocator(vkdev);

    ncnn::ModelBinFromMatArray mb(weights.data());
    op->load_model(mb);

    op->create_pipeline(opt);

    if (!op->support_vulkan)
    {
        op->destroy_pipeline(opt);
        delete op;
        vkdev->reclaim_blob_allocator(blob_vkallocator);
        vkdev->reclaim_staging_allocator(staging_vkallocator);
        return -1;
    }

    // upload model weights
    {
        ncnn::VkTransfer cmd(vkdev);

        ncnn::Option opt_upload = opt;
        opt_upload.blob_vkallocator = &weight_vkallocator;
        opt_upload.workspace_vkallocator = &weight_vkallocator;
        opt_upload.staging_vkallocator = &weight_staging_vkallocator;

        op->upload_model(cmd, opt_upload);
        cmd.submit_and_wait();
    }

    // pre-upload and pack inputs on GPU, exclude from timing
    std::vector<ncnn::VkMat> vk_inputs(inputs.size());
    {
        ncnn::VkCompute cmd(vkdev);
        for (size_t j = 0; j < inputs.size(); j++)
        {
            cmd.record_upload(inputs[j], vk_inputs[j], opt);
        }

        // convert packing on gpu if needed
        if (op->support_vulkan_packing)
        {
            for (size_t j = 0; j < vk_inputs.size(); j++)
            {
                int elemcount = 0;
                int dims = inputs[j].dims;
                if (dims == 1) elemcount = inputs[j].elempack * inputs[j].w;
                if (dims == 2) elemcount = inputs[j].elempack * inputs[j].h;
                if (dims == 3 || dims == 4) elemcount = inputs[j].elempack * inputs[j].c;

                int dst_elempack = elemcount % 4 == 0 ? 4 : 1;
                if (vk_inputs[j].elempack != dst_elempack)
                {
                    ncnn::VkMat vk_packed;
                    vkdev->convert_packing(vk_inputs[j], vk_packed, dst_elempack, cmd, opt);
                    vk_inputs[j] = vk_packed;
                }
            }
        }

        cmd.submit_and_wait();
    }

    // warmup (forward only, inputs already on device)
    for (int i = 0; i < warmup_count; i++)
    {
        ncnn::VkCompute cmd(vkdev);
        run_layer_forward_gpu(op, vk_inputs, cmd, opt);
        cmd.submit_and_wait();
    }

    // calibrate inner loops for fast GPU ops
    int inner_loops = 1;
    {
        double t0 = ncnn::get_current_time();
        {
            ncnn::VkCompute cmd(vkdev);
            run_layer_forward_gpu(op, vk_inputs, cmd, opt);
            cmd.submit_and_wait();
        }
        double single_ms = ncnn::get_current_time() - t0;
        if (single_ms < PERF_MIN_TOTAL_MS)
        {
            if (single_ms < 0.001) single_ms = 0.001;
            inner_loops = (int)(PERF_MIN_TOTAL_MS / single_ms) + 1;
            if (inner_loops < 1) inner_loops = 1;
        }
    }

    // benchmark
    double* times = new double[run_count];
    double time_sum = 0;
    double time_min_val = DBL_MAX;
    double time_max_val = -DBL_MAX;

    for (int i = 0; i < run_count; i++)
    {
        double start = ncnn::get_current_time();

        for (int k = 0; k < inner_loops; k++)
        {
            ncnn::VkCompute cmd(vkdev);
            run_layer_forward_gpu(op, vk_inputs, cmd, opt);
            cmd.submit_and_wait();
        }

        double end = ncnn::get_current_time();
        double t = (end - start) / inner_loops;

        times[i] = t;
        time_sum += t;
        if (t < time_min_val) time_min_val = t;
        if (t > time_max_val) time_max_val = t;
    }

    sort_doubles(times, run_count);
    double time_median_val;
    if (run_count % 2 == 0)
        time_median_val = (times[run_count / 2 - 1] + times[run_count / 2]) / 2.0;
    else
        time_median_val = times[run_count / 2];

    result.time_min = time_min_val;
    result.time_max = time_max_val;
    result.time_avg = time_sum / run_count;
    result.time_median = time_median_val;
    result.loop_count = inner_loops;

    delete[] times;

    op->destroy_pipeline(opt);
    delete op;

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    return 0;
}

int perf_layer_gpu(const char* layer_type, const ncnn::ParamDict& pd,
                   const std::vector<ncnn::Mat>& weights,
                   const ncnn::Mat& input,
                   const ncnn::Option& opt,
                   ncnn::VulkanDevice* vkdev,
                   int warmup_count, int run_count,
                   PerfResult& result)
{
    std::vector<ncnn::Mat> inputs(1);
    inputs[0] = input;
    return perf_layer_gpu(layer_type, pd, weights, inputs, opt, vkdev, warmup_count, run_count, result);
}
#endif // NCNN_VULKAN

void print_perf_result(const char* tag, const PerfResult& result)
{
    if (result.loop_count > 1)
    {
        fprintf(stdout, "%-72s  min=%7.3f  max=%7.3f  avg=%7.3f  median=%7.3f ms  (x%d)\n",
                tag, result.time_min, result.time_max, result.time_avg, result.time_median, result.loop_count);
    }
    else
    {
        fprintf(stdout, "%-72s  min=%7.3f  max=%7.3f  avg=%7.3f  median=%7.3f ms\n",
                tag, result.time_min, result.time_max, result.time_avg, result.time_median);
    }
    fflush(stdout);
}

char* format_shape(char* buf, int bufsize, const ncnn::Mat& m)
{
    char tmp[64];
    if (m.dims == 1)
        snprintf(tmp, sizeof(tmp), "(%d)", m.w);
    else if (m.dims == 2)
        snprintf(tmp, sizeof(tmp), "(%dx%d)", m.w, m.h);
    else if (m.dims == 3)
        snprintf(tmp, sizeof(tmp), "(%dx%dx%d)", m.w, m.h, m.c);
    else if (m.dims == 4)
        snprintf(tmp, sizeof(tmp), "(%dx%dx%dx%d)", m.w, m.h, m.d, m.c);
    else
        snprintf(tmp, sizeof(tmp), "(empty)");
    snprintf(buf, bufsize, "%-16s", tmp);
    return buf;
}

ncnn::Option make_perf_option(int num_threads, bool use_packing, bool use_fp16, bool use_bf16)
{
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = num_threads;
    opt.use_packing_layout = use_packing;
    opt.use_fp16_packed = use_fp16;
    opt.use_fp16_storage = use_fp16;
    opt.use_fp16_arithmetic = use_fp16;
    opt.use_bf16_storage = use_bf16;
    opt.use_vulkan_compute = false;
    opt.use_winograd_convolution = true;
    opt.use_sgemm_convolution = true;
    opt.use_int8_inference = true;
    return opt;
}
