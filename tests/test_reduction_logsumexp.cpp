// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include <math.h>
#include <limits>

static int test_logsumexp(const ncnn::Mat& a, int mask, int keepdims, float coeff, float value)
{
    const int shape[4] = {a.w, a.h, a.d, a.c};
    int reduce_size = 1;
    int axes_count = 0;
    for (int i = 0; i < a.dims; i++)
        if (mask & (1 << i)) axes_count++;

    ncnn::Mat axes(axes_count);
    int* axes_ptr = axes;
    int axis_index = 0;
    for (int i = 0; i < a.dims; i++)
    {
        if (!(mask & (1 << i))) continue;
        axes_ptr[axis_index++] = i - a.dims;
        const int physical_axis = a.dims - i - 1;
        reduce_size *= shape[a.dims == 3 && physical_axis == 2 ? 3 : physical_axis];
    }

    ncnn::ParamDict pd;
    pd.set(0, 10);
    pd.set(1, 0);
    pd.set(2, coeff);
    pd.set(3, axes);
    pd.set(4, keepdims);
    pd.set(5, 1);

    const float expected = (value + logf((float)reduce_size)) * coeff;
    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = false;
    for (int naive = 0; naive < 2; naive++)
    {
        ncnn::Layer* op = naive ? ncnn::create_layer_naive("Reduction") : ncnn::create_layer_cpu("Reduction");
        if (!op) return -1;
        int ret = op->load_param(pd);
        if (ret == 0) ret = op->create_pipeline(opt);
        ncnn::Mat out;
        if (ret == 0) ret = op->forward(a, out, opt);
        op->destroy_pipeline(opt);
        delete op;
        if (ret != 0 || out.empty()) return -1;

        for (int q = 0; q < out.c; q++)
        {
            const float* ptr = out.channel(q);
            for (int i = 0; i < out.w * out.h * out.d; i++)
            {
                if (!NearlyEqual(ptr[i], expected, 0.00001f))
                {
                    fprintf(stderr, "logsumexp mismatch dims=%d mask=%d keepdims=%d coeff=%f value=%f naive=%d got=%f expected=%f\n", a.dims, mask, keepdims, coeff, value, naive, ptr[i], expected);
                    return -1;
                }
            }
        }
    }

    std::vector<ncnn::Mat> weights;
    return test_layer("Reduction", pd, weights, a, 0.001f);
}

static int test_logsumexp_mixed_and_nonfinite()
{
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inputs[12][2] = {{1000.f, 999.f}, {-1000.f, -1001.f}, {1.f, 0.f}, {inf, 1.f}, {-inf, -inf}, {nan, 1.f}, {1.f, inf}, {inf, inf}, {-inf, 1.f}, {1.f, -inf}, {1.f, nan}, {nan, inf}};
    const float expected[12] = {1000.f + logf(1.f + expf(-1.f)), -1000.f + logf(1.f + expf(-1.f)), logf(expf(1.f) + 1.f), inf, -inf, nan, inf, inf, 1.f, 1.f, nan, nan};
    for (int t = 0; t < 12; t++)
    {
        ncnn::Mat a(2);
        a[0] = inputs[t][0];
        a[1] = inputs[t][1];
        ncnn::ParamDict pd;
        pd.set(0, 10);
        pd.set(1, 1);
        ncnn::Option opt;
        for (int naive = 0; naive < 2; naive++)
        {
            ncnn::Layer* op = naive ? ncnn::create_layer_naive("Reduction") : ncnn::create_layer_cpu("Reduction");
            if (!op) return -1;
            int ret = op->load_param(pd);
            if (ret == 0) ret = op->create_pipeline(opt);
            ncnn::Mat out;
            if (ret == 0) ret = op->forward(a, out, opt);
            op->destroy_pipeline(opt);
            delete op;
            if (ret != 0 || out.empty() || (expected[t] != expected[t] ? out[0] == out[0] : expected[t] == inf || expected[t] == -inf ? out[0] != expected[t] : !NearlyEqual(out[0], expected[t], 0.00001f)))
            {
                fprintf(stderr, "logsumexp mixed/nonfinite mismatch case=%d naive=%d\n", t, naive);
                return -1;
            }
        }
        std::vector<ncnn::Mat> weights;
        // The testutil comparison does not consider two NaNs equal.
        if (expected[t] == expected[t] && test_layer("Reduction", pd, weights, a, 0.001f) != 0) return -1;
    }
    return 0;
}

int main()
{
    const float values[3] = {1000.f, -1000.f, 1.f};
    const float coeffs[3] = {1.f, 0.5f, -2.f};
    for (int dims = 1; dims <= 4; dims++)
    {
        ncnn::Mat a;
        if (dims == 1) a.create(3);
        if (dims == 2) a.create(3, 2);
        if (dims == 3) a.create(3, 2, 3);
        if (dims == 4) a.create(3, 2, 2, 3);
        for (int v = 0; v < 3; v++)
        {
            a.fill(values[v]);
            for (int mask = 1; mask < (1 << dims); mask++)
                for (int keepdims = 0; keepdims < 2; keepdims++)
                    for (int c = 0; c < 3; c++)
                    {
                        if (test_logsumexp(a, mask, keepdims, coeffs[c], values[v]) != 0) return -1;
                    }
        }
    }
    ncnn::Mat large(65536);
    large.fill(-16.f);
    if (test_logsumexp(large, 1, 0, 1.f, -16.f) != 0) return -1;
    if (test_logsumexp_mixed_and_nonfinite() != 0) return -1;
    fprintf(stderr, "logsumexp: 962 public CPU checks and 478 backend comparisons passed\n");
    return 0;
}
