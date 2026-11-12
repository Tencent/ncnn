// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#if NCNN_VULKAN
#include "command.h"
#include "gpu.h"
#endif

#include <limits.h>

#define OP_TYPE_MAX 11

static int op_type = 0;

static std::vector<int> IntArray(int a0)
{
    std::vector<int> m(1);
    m[0] = a0;
    return m;
}

static std::vector<int> IntArray(int a0, int a1)
{
    std::vector<int> m(2);
    m[0] = a0;
    m[1] = a1;
    return m;
}

static void print_int_array(const std::vector<int>& a)
{
    fprintf(stderr, "[");
    for (size_t i = 0; i < a.size(); i++)
    {
        fprintf(stderr, " %d", a[i]);
    }
    fprintf(stderr, " ]");
}

static int test_reduction(const ncnn::Mat& _a, float coeff, int keepdims)
{
    ncnn::Mat a = _a;
    if (op_type == 9)
    {
        // value must be positive for logsum
        Randomize(a, 0.001f, 2.f);
    }
    if (op_type == 10)
    {
        // exp overflows above 88 unless the max is subtracted
        Randomize(a, -100.f, 100.f);
    }

    ncnn::ParamDict pd;
    pd.set(0, op_type);
    pd.set(1, 1); // reduce_all
    pd.set(2, coeff);
    pd.set(4, keepdims);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Reduction", pd, weights, a, 0.001);
    if (ret != 0)
    {
        fprintf(stderr, "test_reduction failed a.dims=%d a=(%d %d %d %d) op_type=%d coeff=%f keepdims=%d reduce_all=1\n", a.dims, a.w, a.h, a.d, a.c, op_type, coeff, keepdims);
    }

    return ret;
}

static int test_reduction(const ncnn::Mat& _a, float coeff, int keepdims, const std::vector<int>& axes_array)
{
    ncnn::Mat a = _a;
    if (op_type == 9)
    {
        // value must be positive for logsum
        Randomize(a, 0.001f, 2.f);
    }
    if (op_type == 10)
    {
        // exp overflows above 88 unless the max is subtracted
        Randomize(a, -100.f, 100.f);
    }

    ncnn::Mat axes(axes_array.size());
    {
        int* p = axes;
        for (size_t i = 0; i < axes_array.size(); i++)
        {
            p[i] = axes_array[i];
        }
    }

    ncnn::ParamDict pd;
    pd.set(0, op_type);
    pd.set(1, 0); // reduce_all
    pd.set(2, coeff);
    pd.set(3, axes);
    pd.set(4, keepdims);
    pd.set(5, 1); // fixbug0

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Reduction", pd, weights, a, 0.001);
    if (ret != 0)
    {
        fprintf(stderr, "test_reduction failed a.dims=%d a=(%d %d %d %d) op_type=%d coeff=%f keepdims=%d", a.dims, a.w, a.h, a.d, a.c, op_type, coeff, keepdims);
        fprintf(stderr, " axes=");
        print_int_array(axes_array);
        fprintf(stderr, "\n");
    }

    return ret;
}

static int test_reduction_axes(const ncnn::Mat& a)
{
    // axis geometry and output rank are independent of coefficient scaling
    int ret = test_reduction(a, 1.f, 0) || test_reduction(a, 2.f, 1);
    if (ret != 0)
        return ret;
    const int axes1[][4] = {{0, -1, -1, -1}};
    const int axes2[][4] = {{0, -1, -1, -1}, {1, -1, -1, -1}, {0, 1, -1, -1}};
    const int axes3[][4] = {
        {0, -1, -1, -1}, {1, -1, -1, -1}, {2, -1, -1, -1}, {0, 1, -1, -1}, {0, 2, -1, -1}, {1, 2, -1, -1}, {0, 1, 2, -1}
    };
    const int axes4[][4] = {
        {0, -1, -1, -1}, {1, -1, -1, -1}, {2, -1, -1, -1}, {3, -1, -1, -1}, {0, 1, -1, -1}, {0, 2, -1, -1}, {0, 3, -1, -1}, {1, 2, -1, -1}, {1, 3, -1, -1}, {2, 3, -1, -1}, {0, 1, 2, -1}, {0, 1, 3, -1}, {0, 2, 3, -1}, {1, 2, 3, -1}, {0, 1, 2, 3}
    };
    const int(*axis_sets)[4] = a.dims == 4 ? axes4 : a.dims == 3 ? axes3 : a.dims == 2 ? axes2 : axes1;
    const int counts[] = {1, 3, 7, 15};
    for (int i = 0; i < counts[a.dims - 1]; i++)
    {
        std::vector<int> axes;
        for (int j = 0; j < 4 && axis_sets[i][j] >= 0; j++)
            axes.push_back(axis_sets[i][j]);
        ret = test_reduction(a, 1.f, 0, axes) || test_reduction(a, 2.f, 1, axes);
        if (ret != 0)
            return ret;
    }
    return 0;
}

static int test_reduction_arithmetic()
{
    // each operation exercises every axis kernel and both output layouts
    return test_reduction_axes(RandomMat(127))
           || test_reduction_axes(RandomMat(19, 15))
           || test_reduction_axes(RandomMat(7, 9, 12))
           || test_reduction_axes(RandomMat(7, 8, 9, 12))
           || test_reduction(RandomMat(3, 1, 5, 1), 2.f, 0, IntArray(1, 3));
}

static int test_reduction_vector_boundaries()
{
    // lengths below, at and above SIMD widths retain vector and scalar loops
    const int sizes[] = {1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 128};
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++)
    {
        ncnn::Mat a = RandomMat(sizes[i]);
        ncnn::Mat b = RandomMat(sizes[i], 3);
        int ret = test_reduction(a, 1.f, 0)
                  || test_reduction(a, 2.f, 1)
                  || test_reduction(b, 1.f, 0)
                  || test_reduction(b, 2.f, 1, IntArray(1));
        if (ret != 0)
            return ret;
    }
    return 0;
}

static int test_reduction_geometry()
{
    // singleton ranks, packed transfers and a large workgroup reduction
    return test_reduction_axes(RandomMat(1, 3, 1, 5))
           || test_reduction_axes(RandomMat(3, 1, 5, 1))
           || test_reduction_axes(RandomMat(1, 1, 1, 1))
           || test_reduction_axes(RandomMat(1, 3, 5))
           || test_reduction_axes(RandomMat(3, 1, 5))
           || test_reduction_axes(RandomMat(1, 5))
           || test_reduction_axes(RandomMat(5, 1))
           || test_reduction_axes(RandomMat(3, 4, 5, 13))
           || test_reduction(RandomMat(5, 6, 7, 24), 1.f, 0)
           || test_reduction(RandomMat(17, 12), 1.f, 0)
           || test_reduction(RandomMat(5, 7, 24), 2.f, 1, IntArray(2));
}

static int test_reduction_numeric_boundaries()
{
    // zero L2 results exercise the subnormal square-root guard
    op_type = 8;
    ncnn::Mat a(33);
    ncnn::Mat b(3, 2, 4);
    a.fill(0.f);
    b.fill(0.f);
    int ret = test_reduction(a, 1.f, 0)
              || test_reduction(b, 2.f, 1, IntArray(2));
    if (ret != 0)
        return ret;

    // negative axes preserve each operation and both output layouts
    for (op_type = 0; op_type < OP_TYPE_MAX; op_type++)
    {
        ret = test_reduction(RandomMat(3, 2, 4, 5), 1.f, 0, IntArray(-1))
              || test_reduction(RandomMat(3, 2, 4, 5), 2.f, 1, IntArray(-4, -2));
        if (ret != 0)
            return ret;
    }
    return 0;
}

static int test_reduction_logsumexp_shift()
{
    // exp overflows at 1000 and underflows at -1000 unless the max is subtracted
    const float offsets[2] = {1000.f, -1000.f};
    for (int i = 0; i < 2; i++)
    {
        ncnn::Mat a(7, 5);
        float sum = 0.f;
        for (int j = 0; j < 35; j++)
        {
            a[j] = offsets[i] - j * 0.25f;
            sum += expf(-j * 0.25f);
        }

        const float expected = offsets[i] + logf(sum);

        ncnn::ParamDict pd;
        pd.set(0, 10);
        pd.set(1, 1); // reduce_all

        std::vector<ncnn::Mat> weights(0);
        ncnn::Mat b;

        int ret = test_layer_naive(ncnn::LayerType::Reduction, pd, weights, a, b, 0);
        if (ret == 0 && (b.empty() || !NearlyEqual(b[0], expected, 0.001)))
            ret = -1;

        if (ret == 0)
            ret = test_layer("Reduction", pd, weights, a, 0.001);

        if (ret != 0)
        {
            fprintf(stderr, "test_reduction_logsumexp_shift failed offset=%f expected=%f\n", offsets[i], expected);
            return ret;
        }
    }

    return 0;
}

static ncnn::Mat param_int_array(int size, int value)
{
    ncnn::Mat m(size);
    int* p = m;
    for (int i = 0; i < size; i++)
        p[i] = value;

    return m;
}

#if NCNN_VALIDATION
static int test_reduction_load_param()
{
    ncnn::ParamDict base;
    base.set(5, 1); // fixbug0
    base.set(3, param_int_array(1, 0));
    if (test_layer_param(ncnn::LayerType::Reduction, base, 0)
            || test_layer_param(ncnn::LayerType::Reduction, base, 3, ncnn::Mat(0), 0))
        return -1;

    ncnn::Mat missing_data(0);
    missing_data.w = 1;

    return 0
           || test_layer_param(ncnn::LayerType::Reduction, base, 3, missing_data, -1)
           || test_layer_param(ncnn::LayerType::Reduction, base, 3, ncnn::Mat(1, (size_t)1u), -1)
           || test_layer_param(ncnn::LayerType::Reduction, base, 3, ncnn::Mat(1, 2), -1)
           || test_layer_param(ncnn::LayerType::Reduction, base, 3, 1.f, -1)
           || test_layer_param(ncnn::LayerType::Reduction, base, 3, param_int_array(5, 1), -1)
           || test_layer_param(ncnn::LayerType::Reduction, base, 3, param_int_array(1, INT_MIN), -1);
}

static int test_reduction_load_param_type()
{
    ncnn::ParamDict base;
    if (test_layer_param(ncnn::LayerType::Reduction, base, 0) != 0)
        return -1;

    for (int i = 0; i <= 10; i++)
    {
        if (test_layer_param(ncnn::LayerType::Reduction, base, 0, i, 0) != 0)
            return -1;
    }

    const int invalid[] = {-1, 11, INT_MIN, INT_MAX};
    for (int i = 0; i < 4; i++)
    {
        if (test_layer_param(ncnn::LayerType::Reduction, base, 0, invalid[i], -1) != 0)
            return -1;
    }

    return 0;
}

static int test_reduction_load_param_text()
{
#if NCNN_STRING
    const char* params[] = {"5=1 -23303=1,0", "5=1 -23303=1,0.0"};
    for (int i = 0; i < 2; i++)
    {
        TestParamDict pd;
        if (pd.load_param(params[i]) != 0 || pd.type(3) != 5 + i)
            return -1;

        if (test_layer_param(ncnn::LayerType::Reduction, pd, i == 0 ? 0 : -1) != 0)
            return -1;
    }
#endif
    return 0;
}
#endif // NCNN_VALIDATION

static int test_reduction_depth_padding_boundaries()
{
    // reducing depth isolates input channel padding from output depth indexing
    ncnn::Mat a(1, 1, 2, 1);
    a[0] = 1.f;
    a[1] = 2.f;

    // reducing height preserves two output depths with independent channel tail padding
    ncnn::Mat b(1, 4, 2, 1);
    for (int i = 0; i < 4; i++)
    {
        b[i] = 1.f;
        b[4 + i] = 2.f;
    }

    op_type = 0;
    return test_reduction(a, 1.f, 1, IntArray(1))
           || test_reduction(a, 1.f, 1, IntArray(-3))
           || test_reduction(b, 1.f, 1, IntArray(0, 2))
           || test_reduction(b, 1.f, 1, IntArray(-4, -2));
}

#if NCNN_VULKAN
static int test_reduction_gpu_dispatch(const ncnn::Mat& a, int keepdims)
{
    ncnn::ParamDict pd;
    pd.set(0, op_type);
    pd.set(1, 0); // reduce_all
    pd.set(3, param_int_array(1, 0));
    pd.set(4, keepdims);
    pd.set(5, 1); // fixbug0

    std::vector<ncnn::Mat> weights(0);
    ncnn::Mat expected;
    int ret = test_layer_naive(ncnn::LayerType::Reduction, pd, weights, a, expected, 0);
    if (ret != 0)
        return ret;

    for (int fp16 = 0; fp16 < 2; fp16++)
    {
        for (int subgroup = 0; subgroup < 2; subgroup++)
        {
            ncnn::Option opt;
            opt.use_fp16_packed = fp16 != 0;
            opt.use_fp16_storage = fp16 != 0;
            opt.use_fp16_arithmetic = false;
            opt.use_bf16_packed = false;
            opt.use_bf16_storage = false;
            opt.use_subgroup_ops = subgroup != 0;

            ncnn::Mat b;
            ret = test_layer_gpu(ncnn::LayerType::Reduction, pd, weights, opt, a, b, ncnn::Mat(), 0);
            if (ret != 233 && (ret != 0 || CompareMat(expected, b, 0.001f) != 0))
            {
                fprintf(stderr, "test_reduction_gpu_dispatch failed a=(%d %d %d) op_type=%d keepdims=%d fp16=%d subgroup=%d ret=%d\n", a.w, a.h, a.c, op_type, keepdims, fp16, subgroup, ret);
                return -1;
            }
        }
    }

    return 0;
}

static int test_reduction_gpu_dispatch()
{
    if (ncnn::get_gpu_count() == 0)
        return 0;

    // output counts around the 65535-workgroup Vulkan minimum Y dispatch limit
    // 65537 outputs leave a padded tail workgroup once split across x and y
    const int shapes[3][2] = {{257, 255}, {256, 256}, {65537, 1}};

    // sum mean logsumexp, logsumexp takes a separate barrier path in the shader
    static const int op_types[3] = {0, 3, 10};

    for (int i = 0; i < 3; i++)
    {
        ncnn::Mat a(shapes[i][0], shapes[i][1], 3);
        for (int q = 0; q < a.c; q++)
        {
            float* ptr = a.channel(q);
            for (int j = 0; j < a.w * a.h; j++)
                ptr[j] = (j % 13 - 6) * 0.25f + q * 0.5f;
        }

        for (int j = 0; j < 3; j++)
        {
            op_type = op_types[j];

            if (test_reduction_gpu_dispatch(a, 0)
                    || test_reduction_gpu_dispatch(a, 1))
                return -1;
        }
    }

    return 0;
}

static int test_reduction_gpu_fp16a(const ncnn::Mat& a, float coeff, const std::vector<int>& axes_array, bool use_subgroup_ops)
{
    ncnn::ParamDict pd;
    pd.set(0, op_type);
    pd.set(2, coeff);
    pd.set(4, 0); // keepdims
    if (axes_array.empty())
    {
        pd.set(1, 1); // reduce_all
    }
    else
    {
        ncnn::Mat axes((int)axes_array.size());
        int* p = axes;
        for (size_t i = 0; i < axes_array.size(); i++)
        {
            p[i] = axes_array[i];
        }

        pd.set(1, 0); // reduce_all
        pd.set(3, axes);
        pd.set(5, 1); // fixbug0
    }

    std::vector<ncnn::Mat> weights(0);

    ncnn::Mat expected;
    int ret = test_layer_naive(ncnn::LayerType::Reduction, pd, weights, a, expected, 0);
    if (ret != 0)
        return ret;

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_subgroup_ops = use_subgroup_ops;
    opt.use_fp16_packed = true;
    opt.use_fp16_storage = true;
    opt.use_fp16_arithmetic = true;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;
    opt.use_int8_storage = false;
    opt.use_int8_arithmetic = false;

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev->info.support_subgroup_ops()) opt.use_subgroup_ops = false;

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    ncnn::Layer* op = ncnn::create_layer_vulkan("Reduction");
    op->vkdev = vkdev;
    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());
    op->load_model(mb);
    ret = op->create_pipeline(opt);

    ncnn::Mat b;
    if (ret == 0)
    {
        // exercise fp16 arithmetic directly because test_layer_gpu disables it
        ncnn::VkCompute cmd(vkdev);
        ncnn::VkMat a_gpu;
        cmd.record_upload(a, a_gpu, opt);

        // Reduction_vulkan takes elempack 1
        ncnn::VkMat a_gpu_unpacked;
        vkdev->convert_packing(a_gpu, a_gpu_unpacked, 1, cmd, opt);

        ncnn::VkMat b_gpu;
        ret = op->forward(a_gpu_unpacked, b_gpu, cmd, opt);
        if (ret == 0)
        {
            cmd.record_download(b_gpu, b, opt);
            ret = cmd.submit_and_wait();
        }
    }

    op->destroy_pipeline(opt);
    delete op;

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    if (ret != 0 || CompareMat(expected, b, 0.001f) != 0)
    {
        fprintf(stderr, "test_reduction_gpu_fp16a failed a.dims=%d a=(%d %d %d %d) op_type=%d coeff=%f subgroup=%d ret=%d", a.dims, a.w, a.h, a.d, a.c, op_type, coeff, use_subgroup_ops, ret);
        fprintf(stderr, " axes=");
        print_int_array(axes_array);
        fprintf(stderr, "\n");
        return -1;
    }

    return 0;
}

static int test_reduction_gpu_fp16a()
{
    if (ncnn::get_gpu_count() == 0)
        return 0;

    const ncnn::GpuInfo& info = ncnn::get_gpu_info();
    if (!info.support_fp16_storage() || !info.support_fp16_arithmetic())
        return 0;

    // the sums exceed the fp16 max 65504 while every result fits in fp16
    ncnn::Mat a(20000);
    for (int i = 0; i < a.w; i++)
    {
        a[i] = 3.5f + (i % 4) * 0.5f;
    }

    ncnn::Mat b(128, 128, 4);
    for (int q = 0; q < b.c; q++)
    {
        float* ptr = b.channel(q);
        for (int i = 0; i < b.w * b.h; i++)
        {
            ptr[i] = 3.5f + ((i + q) % 4) * 0.5f;
        }
    }

    // sum asum sumsq mean l1 l2 logsum
    static const int op_types[7] = {0, 1, 2, 3, 7, 8, 9};
    static const float coeffs[7] = {1 / 256.f, 1 / 256.f, 1 / 256.f, 1.f, 1 / 256.f, 1.f, 1.f};

    for (int i = 0; i < 7; i++)
    {
        op_type = op_types[i];

        int ret = 0
                  || test_reduction_gpu_fp16a(a, coeffs[i], std::vector<int>(), false)
                  || test_reduction_gpu_fp16a(a, coeffs[i], std::vector<int>(), true)
                  || test_reduction_gpu_fp16a(b, coeffs[i], IntArray(1, 2), false)
                  || test_reduction_gpu_fp16a(b, coeffs[i], IntArray(1, 2), true);

        if (ret != 0)
            return ret;
    }

    return 0;
}
#endif // NCNN_VULKAN

int main()
{
    SRAND(7767517);

    for (op_type = 0; op_type < OP_TYPE_MAX; op_type++)
    {
        int ret = 0
                  || test_reduction_arithmetic()
                  || test_reduction_vector_boundaries();

        if (ret != 0)
            return ret;
    }

    op_type = 0;
    return test_reduction_geometry()
           || test_reduction_numeric_boundaries()
           || test_reduction_logsumexp_shift()
#if NCNN_VALIDATION
           || test_reduction_load_param()
           || test_reduction_load_param_type()
           || test_reduction_load_param_text()
#endif // NCNN_VALIDATION
#if NCNN_VULKAN
           || test_reduction_gpu_dispatch()
           || test_reduction_gpu_fp16a()
#endif // NCNN_VULKAN
           || test_reduction_depth_padding_boundaries();
}
