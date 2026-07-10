// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#if NCNN_SIMPLESTL
static const float TEST_INF = 1.f / 0.f;
static const float TEST_NAN = 0.f / 0.f;
#define INFINITY TEST_INF
#define NAN      TEST_NAN
#else
#include <algorithm>
#include <math.h>
#endif

// Unified runner: want_indices=false → top_blobs(1), else top_blobs(2).
static int run_topk(const ncnn::Mat& a, int axis, int k, int largest, int sorted,
                    bool want_indices, ncnn::Mat& values, ncnn::Mat& indices)
{
    ncnn::ParamDict pd;
    pd.set(0, axis);
    pd.set(1, largest);
    pd.set(2, sorted);
    pd.set(3, k);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer_cpu("TopK");
    if (!op)
        return -1;

    op->load_param(pd);

    std::vector<ncnn::Mat> weights(0);
    ncnn::ModelBinFromMatArray mb(weights.data());
    op->load_model(mb);
    op->create_pipeline(opt);

    std::vector<ncnn::Mat> bottom_blobs(1);
    bottom_blobs[0] = a;
    std::vector<ncnn::Mat> top_blobs(want_indices ? 2 : 1);
    int ret = op->forward(bottom_blobs, top_blobs, opt);

    op->destroy_pipeline(opt);
    delete op;

    if (ret != 0)
        return ret;

    values = top_blobs[0];
    if (want_indices)
        indices = top_blobs[1];
    return 0;
}

static int test_topk(const ncnn::Mat& a, int axis, int k, int largest, int sorted)
{
    ncnn::ParamDict pd;
    pd.set(0, axis);
    pd.set(1, largest);
    pd.set(2, sorted);
    pd.set(3, k);

    std::vector<ncnn::Mat> weights(0);
    std::vector<ncnn::Mat> a0(1);
    a0[0] = a;

    int ret = test_layer("TopK", pd, weights, a0, 2, 0.01f, TEST_LAYER_DISABLE_AUTO_INPUT_CASTING);
    if (ret != 0)
    {
        fprintf(stderr, "test_topk failed a.dims=%d a=(%d %d %d %d) axis=%d k=%d largest=%d sorted=%d\n",
                a.dims, a.w, a.h, a.d, a.c, axis, k, largest, sorted);
    }
    return ret;
}

static int test_topk_0()
{
    ncnn::Mat a = RandomMat(13);

    return 0
           || test_topk(a, 0, 1, 1, 1)
           || test_topk(a, 0, 5, 1, 1)
           || test_topk(a, 0, 1, 0, 0)
           || test_topk(a, -1, 7, 0, 1)
           || test_topk(a, 0, 4, 1, 0)
           || test_topk(a, 0, 9, 1, 1);
}

static int test_topk_1()
{
    ncnn::Mat a = RandomMat(12, 17);

    return 0
           || test_topk(a, 0, 1, 1, 1)
           || test_topk(a, 0, 5, 1, 1)
           || test_topk(a, 1, 3, 1, 1)
           || test_topk(a, -1, 8, 0, 1)
           || test_topk(a, 1, 6, 0, 0)
           || test_topk(a, -2, 7, 1, 1);
}

static int test_topk_2()
{
    ncnn::Mat a = RandomMat(8, 9, 11);

    return 0
           || test_topk(a, 0, 3, 1, 1)
           || test_topk(a, 1, 4, 1, 1)
           || test_topk(a, 2, 2, 0, 1)
           || test_topk(a, 2, 5, 1, 0)
           || test_topk(a, -1, 6, 1, 1)
           || test_topk(a, -2, 5, 0, 1)
           || test_topk(a, -3, 7, 1, 1);
}

static int test_topk_3()
{
    ncnn::Mat a = RandomMat(5, 7, 9, 10);

    return 0
           || test_topk(a, 0, 2, 1, 1)
           || test_topk(a, 1, 3, 0, 1)
           || test_topk(a, 2, 4, 1, 1)
           || test_topk(a, 3, 4, 0, 0)
           || test_topk(a, 3, 5, 1, 1)
           || test_topk(a, -1, 6, 0, 1)
           || test_topk(a, -2, 3, 1, 1)
           || test_topk(a, -3, 4, 0, 1)
           || test_topk(a, -4, 2, 1, 1);
}

static int test_topk_inf_order()
{
    ncnn::Mat a(6);
    float* ptr = a;
    ptr[0] = 1.f;
    ptr[1] = INFINITY;
    ptr[2] = -2.f;
    ptr[3] = -INFINITY;
    ptr[4] = 0.5f;
    ptr[5] = 3.f;

    ncnn::Mat values, indices;

    if (run_topk(a, 0, 2, 1, 1, true, values, indices) != 0)
    {
        fprintf(stderr, "test_topk_inf_order largest failed\n");
        return -1;
    }
    const float* vptr = values;
    const int* iptr = (const int*)(const void*)indices;
    if (values.w != 2 || indices.w != 2 || vptr[0] != INFINITY || vptr[1] != 3.f || iptr[0] != 1 || iptr[1] != 5)
    {
        fprintf(stderr, "test_topk_inf_order largest mismatch\n");
        return -1;
    }

    if (run_topk(a, 0, 2, 0, 1, true, values, indices) != 0)
    {
        fprintf(stderr, "test_topk_inf_order smallest failed\n");
        return -1;
    }
    vptr = values;
    iptr = (const int*)(const void*)indices;
    if (values.w != 2 || indices.w != 2 || vptr[0] != -INFINITY || vptr[1] != -2.f || iptr[0] != 3 || iptr[1] != 2)
    {
        fprintf(stderr, "test_topk_inf_order smallest mismatch\n");
        return -1;
    }

    return 0;
}

static int test_topk_nan_robust()
{
    // NaN mid-array: [1, NaN, 2, -1], k=2, largest → {2@2, 1@0}
    ncnn::Mat a(4);
    float* ptr = a;
    ptr[0] = 1.f;
    ptr[1] = NAN;
    ptr[2] = 2.f;
    ptr[3] = -1.f;

    ncnn::Mat values, indices;

    if (run_topk(a, 0, 2, 1, 1, true, values, indices) != 0)
    {
        fprintf(stderr, "test_topk_nan_robust sorted failed\n");
        return -1;
    }
    const float* vptr = values;
    const int* iptr = (const int*)(const void*)indices;
    if (values.w != 2 || vptr[0] != 2.f || vptr[1] != 1.f || iptr[0] != 2 || iptr[1] != 0)
    {
        fprintf(stderr, "test_topk_nan_robust sorted largest mismatch\n");
        return -1;
    }

    if (run_topk(a, 0, 2, 0, 1, true, values, indices) != 0)
    {
        fprintf(stderr, "test_topk_nan_robust sorted smallest failed\n");
        return -1;
    }
    vptr = values;
    iptr = (const int*)(const void*)indices;
    if (values.w != 2 || vptr[0] != -1.f || vptr[1] != 1.f || iptr[0] != 3 || iptr[1] != 0)
    {
        fprintf(stderr, "test_topk_nan_robust sorted smallest mismatch\n");
        return -1;
    }

    if (run_topk(a, 0, 2, 1, 0, true, values, indices) != 0)
    {
        fprintf(stderr, "test_topk_nan_robust unsorted failed\n");
        return -1;
    }
    iptr = (const int*)(const void*)indices;
    if (iptr[0] < 0 || iptr[0] >= 4 || iptr[1] < 0 || iptr[1] >= 4)
    {
        fprintf(stderr, "test_topk_nan_robust unsorted invalid indices\n");
        return -1;
    }

    return 0;
}

// NaN at index 0 — exercises `has_nan = topk_isnan(best_value)` at the top of
// the k=1 scalar fast path; without this, the fast loop is entered with a NaN
// as the running best and comparisons are silently wrong.
static int test_topk_nan_first_element()
{
    ncnn::Mat a(5);
    float* ptr = a;
    ptr[0] = NAN;
    ptr[1] = 3.f;
    ptr[2] = 1.f;
    ptr[3] = 5.f;
    ptr[4] = 2.f;

    ncnn::Mat values, indices;

    // k=1 largest: best is 5@3
    if (run_topk(a, 0, 1, 1, 1, true, values, indices) != 0)
    {
        fprintf(stderr, "test_topk_nan_first_element k1 failed\n");
        return -1;
    }
    const float* vp = values;
    const int* ip = (const int*)(const void*)indices;
    if (values.w != 1 || vp[0] != 5.f || ip[0] != 3)
    {
        fprintf(stderr, "test_topk_nan_first_element k1 mismatch v=%f i=%d\n", vp[0], ip[0]);
        return -1;
    }

    // k=2 smallest sorted: {1@2, 2@4}
    if (run_topk(a, 0, 2, 0, 1, true, values, indices) != 0)
    {
        fprintf(stderr, "test_topk_nan_first_element k2 failed\n");
        return -1;
    }
    vp = values;
    ip = (const int*)(const void*)indices;
    if (values.w != 2 || vp[0] != 1.f || vp[1] != 2.f || ip[0] != 2 || ip[1] != 4)
    {
        fprintf(stderr, "test_topk_nan_first_element k2 mismatch\n");
        return -1;
    }

    return 0;
}

// Multiple NaN values — exercises NaN eviction from the k-buffer in the k≤4 path.
static int test_topk_multiple_nans()
{
    ncnn::Mat a(7);
    float* ptr = a;
    ptr[0] = NAN;
    ptr[1] = 2.f;
    ptr[2] = NAN;
    ptr[3] = 5.f;
    ptr[4] = NAN;
    ptr[5] = 1.f;
    ptr[6] = NAN;

    ncnn::Mat values, indices;

    // k=2, largest, sorted: {5@3, 2@1}
    if (run_topk(a, 0, 2, 1, 1, true, values, indices) != 0)
    {
        fprintf(stderr, "test_topk_multiple_nans failed\n");
        return -1;
    }
    const float* vp = values;
    const int* ip = (const int*)(const void*)indices;
    if (values.w != 2 || vp[0] != 5.f || vp[1] != 2.f || ip[0] != 3 || ip[1] != 1)
    {
        fprintf(stderr, "test_topk_multiple_nans mismatch v=[%f,%f] i=[%d,%d]\n",
                vp[0], vp[1], ip[0], ip[1]);
        return -1;
    }

    // k=3, smallest, sorted: {1@5, 2@1, 5@3}
    if (run_topk(a, 0, 3, 0, 1, true, values, indices) != 0)
    {
        fprintf(stderr, "test_topk_multiple_nans k3 failed\n");
        return -1;
    }
    vp = values;
    ip = (const int*)(const void*)indices;
    if (values.w != 3 || vp[0] != 1.f || vp[1] != 2.f || vp[2] != 5.f
            || ip[0] != 5 || ip[1] != 1 || ip[2] != 3)
    {
        fprintf(stderr, "test_topk_multiple_nans k3 mismatch\n");
        return -1;
    }

    return 0;
}

// sorted=0 must return the same SET of top-k values as sorted=1.
static int test_topk_sorted0_vs_sorted1()
{
    ncnn::Mat a(8);
    float* ptr = a;
    ptr[0] = 3.f;
    ptr[1] = 1.f;
    ptr[2] = 4.f;
    ptr[3] = 1.f;
    ptr[4] = 5.f;
    ptr[5] = 9.f;
    ptr[6] = 2.f;
    ptr[7] = 6.f;

    ncnn::Mat sv, uv, dummy;

    // k=3, largest
    if (run_topk(a, 0, 3, 1, 1, false, sv, dummy) != 0
            || run_topk(a, 0, 3, 1, 0, false, uv, dummy) != 0)
    {
        fprintf(stderr, "test_topk_sorted0_vs_sorted1: forward failed\n");
        return -1;
    }
    {
        float s[3], u[3];
        const float* sp = sv;
        const float* up = uv;
        for (int i = 0; i < 3; i++)
        {
            s[i] = sp[i];
            u[i] = up[i];
        }
        std::sort(s, s + 3);
        std::sort(u, u + 3);
        for (int i = 0; i < 3; i++)
        {
            if (s[i] != u[i])
            {
                fprintf(stderr, "test_topk_sorted0_vs_sorted1 largest: value set mismatch at %d: sorted=%f unsorted=%f\n",
                        i, s[i], u[i]);
                return -1;
            }
        }
    }

    // k=4, smallest
    if (run_topk(a, 0, 4, 0, 1, false, sv, dummy) != 0
            || run_topk(a, 0, 4, 0, 0, false, uv, dummy) != 0)
    {
        fprintf(stderr, "test_topk_sorted0_vs_sorted1: smallest forward failed\n");
        return -1;
    }
    {
        float s[4], u[4];
        const float* sp = sv;
        const float* up = uv;
        for (int i = 0; i < 4; i++)
        {
            s[i] = sp[i];
            u[i] = up[i];
        }
        std::sort(s, s + 4);
        std::sort(u, u + 4);
        for (int i = 0; i < 4; i++)
        {
            if (s[i] != u[i])
            {
                fprintf(stderr, "test_topk_sorted0_vs_sorted1 smallest: value set mismatch at %d\n", i);
                return -1;
            }
        }
    }

    return 0;
}

// Equal values → lower original index wins as tiebreak.
static int test_topk_tie_breaking()
{
    ncnn::Mat a(5);
    float* ptr = a;
    ptr[0] = 5.f;
    ptr[1] = 5.f;
    ptr[2] = 3.f;
    ptr[3] = 5.f;
    ptr[4] = 1.f;

    ncnn::Mat values, indices;

    // Top-2 largest: 5@0, 5@1 (lower indices win)
    if (run_topk(a, 0, 2, 1, 1, true, values, indices) != 0)
    {
        fprintf(stderr, "test_topk_tie_breaking: forward failed\n");
        return -1;
    }
    const float* vp = values;
    const int* ip = (const int*)(const void*)indices;
    if (values.w != 2 || vp[0] != 5.f || vp[1] != 5.f || ip[0] != 0 || ip[1] != 1)
    {
        fprintf(stderr, "test_topk_tie_breaking largest: got v=[%f,%f] i=[%d,%d]\n",
                vp[0], vp[1], ip[0], ip[1]);
        return -1;
    }

    // Top-2 smallest: 1@4, 3@2
    if (run_topk(a, 0, 2, 0, 1, true, values, indices) != 0)
    {
        fprintf(stderr, "test_topk_tie_breaking: smallest forward failed\n");
        return -1;
    }
    vp = values;
    ip = (const int*)(const void*)indices;
    if (values.w != 2 || vp[0] != 1.f || vp[1] != 3.f || ip[0] != 4 || ip[1] != 2)
    {
        fprintf(stderr, "test_topk_tie_breaking smallest: got v=[%f,%f] i=[%d,%d]\n",
                vp[0], vp[1], ip[0], ip[1]);
        return -1;
    }

    return 0;
}

// k=0 must produce empty output without crashing.
static int test_topk_k_zero()
{
    ncnn::Mat a(6);
    float* ptr = a;
    for (int i = 0; i < 6; i++) ptr[i] = (float)i;

    ncnn::Mat values, indices;
    if (run_topk(a, 0, 0, 1, 1, true, values, indices) != 0)
    {
        fprintf(stderr, "test_topk_k_zero: forward failed\n");
        return -1;
    }
    if (values.total() != 0 || indices.total() != 0)
    {
        fprintf(stderr, "test_topk_k_zero: expected empty output, got values=%d indices=%d\n",
                (int)values.total(), (int)indices.total());
        return -1;
    }
    return 0;
}

// k > axis_size must be clamped to axis_size.
static int test_topk_k_clamp()
{
    ncnn::Mat a(4);
    float* ptr = a;
    ptr[0] = 1.f;
    ptr[1] = 4.f;
    ptr[2] = 3.f;
    ptr[3] = 2.f;

    ncnn::Mat values, indices;
    if (run_topk(a, 0, 10, 1, 1, true, values, indices) != 0)
    {
        fprintf(stderr, "test_topk_k_clamp: forward failed\n");
        return -1;
    }
    const float* vp = values;
    const int* ip = (const int*)(const void*)indices;
    // clamped to k=4, sorted largest: 4@1, 3@2, 2@3, 1@0
    if ((int)values.total() != 4 || vp[0] != 4.f || vp[1] != 3.f || vp[2] != 2.f || vp[3] != 1.f
            || ip[0] != 1 || ip[1] != 2 || ip[2] != 3 || ip[3] != 0)
    {
        fprintf(stderr, "test_topk_k_clamp: mismatch\n");
        return -1;
    }
    return 0;
}

static int test_topk_values_only_fastpaths()
{
    ncnn::Mat a(5);
    float* ptr = a;
    ptr[0] = 1.f;
    ptr[1] = -2.f;
    ptr[2] = 4.f;
    ptr[3] = 3.f;
    ptr[4] = 0.f;

    ncnn::Mat values, dummy;

    // k=1, values-only (triggers NEON path on ARM when axis_size >= 4)
    if (run_topk(a, 0, 1, 1, 0, false, values, dummy) != 0)
    {
        fprintf(stderr, "test_topk_values_only_fastpaths k1 failed\n");
        return -1;
    }
    if (values.w != 1 || ((const float*)values)[0] != 4.f)
    {
        fprintf(stderr, "test_topk_values_only_fastpaths k1 mismatch\n");
        return -1;
    }

    // k=full, values-only (copy-all fast path)
    if (run_topk(a, 0, 5, 1, 0, false, values, dummy) != 0)
    {
        fprintf(stderr, "test_topk_values_only_fastpaths fullk failed\n");
        return -1;
    }
    if (values.w != 5)
    {
        fprintf(stderr, "test_topk_values_only_fastpaths fullk shape mismatch\n");
        return -1;
    }
    const float* vptr = values;
    for (int i = 0; i < 5; i++)
    {
        if (vptr[i] != ptr[i])
        {
            fprintf(stderr, "test_topk_values_only_fastpaths fullk value mismatch at %d\n", i);
            return -1;
        }
    }

    // k=1, values-only, smallest — exercises NEON min path
    if (run_topk(a, 0, 1, 0, 0, false, values, dummy) != 0)
    {
        fprintf(stderr, "test_topk_values_only_fastpaths k1_min failed\n");
        return -1;
    }
    if (values.w != 1 || ((const float*)values)[0] != -2.f)
    {
        fprintf(stderr, "test_topk_values_only_fastpaths k1_min mismatch: got %f\n",
                ((const float*)values)[0]);
        return -1;
    }

    return 0;
}

static int test_topk_full_k()
{
    ncnn::Mat a2d = RandomMat(8, 5);
    if (test_topk(a2d, 0, 5, 1, 1) != 0) return -1;
    if (test_topk(a2d, 0, 5, 0, 1) != 0) return -1;
    if (test_topk(a2d, 1, 8, 1, 1) != 0) return -1;

    ncnn::Mat a3d = RandomMat(6, 4, 3);
    if (test_topk(a3d, 0, 3, 1, 1) != 0) return -1;
    if (test_topk(a3d, 1, 4, 1, 1) != 0) return -1;
    if (test_topk(a3d, 2, 6, 1, 1) != 0) return -1;

    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_topk_0()
           || test_topk_1()
           || test_topk_2()
           || test_topk_3()
           || test_topk_inf_order()
           || test_topk_nan_robust()
           || test_topk_nan_first_element()
           || test_topk_multiple_nans()
           || test_topk_sorted0_vs_sorted1()
           || test_topk_tie_breaking()
           || test_topk_k_zero()
           || test_topk_k_clamp()
           || test_topk_values_only_fastpaths()
           || test_topk_full_k();
}
