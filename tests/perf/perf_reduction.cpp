// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perfutil.h"

static ncnn::Mat IntArray(int a0)
{
    ncnn::Mat m(1, (size_t)4u);
    int* p = m;
    p[0] = a0;
    return m;
}

static ncnn::Mat IntArray(int a0, int a1)
{
    ncnn::Mat m(2, (size_t)4u);
    int* p = m;
    p[0] = a0;
    p[1] = a1;
    return m;
}

static ncnn::Mat IntArray(int a0, int a1, int a2)
{
    ncnn::Mat m(3, (size_t)4u);
    int* p = m;
    p[0] = a0;
    p[1] = a1;
    p[2] = a2;
    return m;
}

static const char* op_type_name(int op_type)
{
    static const char* names[] = {"sum", "asum", "sumsq", "mean", "max", "min", "prod", "l1", "l2", "logsum", "logsumexp"};
    if (op_type >= 0 && op_type < 11)
        return names[op_type];
    return "op";
}

static void perf_reduction(const ncnn::Mat& a, int op_type, int keepdims)
{
    ncnn::ParamDict pd;
    pd.set(0, op_type);
    pd.set(1, 1);
    pd.set(2, 1.f);
    pd.set(4, keepdims);

    std::vector<ncnn::Mat> weights(0);

    perf_layer("Reduction", pd, weights, a, "%s axes=all keep=%d", op_type_name(op_type), keepdims);
}

static void perf_reduction(const ncnn::Mat& a, int op_type, int keepdims, int axis)
{
    ncnn::ParamDict pd;
    pd.set(0, op_type);
    pd.set(1, 0);
    pd.set(2, 1.f);
    pd.set(3, IntArray(axis));
    pd.set(4, keepdims);
    pd.set(5, 1);

    std::vector<ncnn::Mat> weights(0);

    perf_layer("Reduction", pd, weights, a, "%s axis=%d keep=%d", op_type_name(op_type), axis, keepdims);
}

static void perf_reduction(const ncnn::Mat& a, int op_type, int keepdims, int axis0, int axis1)
{
    ncnn::ParamDict pd;
    pd.set(0, op_type);
    pd.set(1, 0);
    pd.set(2, 1.f);
    pd.set(3, IntArray(axis0, axis1));
    pd.set(4, keepdims);
    pd.set(5, 1);

    std::vector<ncnn::Mat> weights(0);

    perf_layer("Reduction", pd, weights, a, "%s axes=%d,%d keep=%d", op_type_name(op_type), axis0, axis1, keepdims);
}

static void perf_reduction(const ncnn::Mat& a, int op_type, int keepdims, int axis0, int axis1, int axis2)
{
    ncnn::ParamDict pd;
    pd.set(0, op_type);
    pd.set(1, 0);
    pd.set(2, 1.f);
    pd.set(3, IntArray(axis0, axis1, axis2));
    pd.set(4, keepdims);
    pd.set(5, 1);

    std::vector<ncnn::Mat> weights(0);

    perf_layer("Reduction", pd, weights, a, "%s axes=%d,%d,%d keep=%d", op_type_name(op_type), axis0, axis1, axis2, keepdims);
}

int main()
{
    perf_reduction(PerfMat(1048576), 0, 0);

    perf_reduction(PerfMat(1024, 1024), 0, 0, 1);
    perf_reduction(PerfMat(1024, 1024), 0, 0, 0);

    perf_reduction(PerfMat(56, 56, 64), 3, 1, 1, 2);
    perf_reduction(PerfMat(56, 56, 64), 4, 0, 0);

    perf_reduction(PerfMat(64, 8, 8, 32), 0, 0, 0, 1, 2);
    perf_reduction(PerfMat(16, 16, 8, 64), 8, 0, 0);
    perf_reduction(PerfMat(16, 16, 8, 64), 1, 0, 2, 3);
    perf_reduction(PerfMat(16, 16, 8, 64), 2, 1, 0, 2, 3);

    return 0;
}
