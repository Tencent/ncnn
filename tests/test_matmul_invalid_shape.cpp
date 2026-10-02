// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer.h"

static int test_matmul_shape(const ncnn::Mat& a, const ncnn::Mat& b, int transb, int expected_ret)
{
    // create via the registry so the platform-specific MatMul variant
    // registered in this build is the one being checked
    ncnn::Layer* op = ncnn::create_layer("MatMul");
    if (!op)
    {
        fprintf(stderr, "test_matmul_shape failed to create MatMul layer\n");
        return -1;
    }

    ncnn::ParamDict pd;
    pd.set(0, transb); // transB

    op->load_param(pd);

    ncnn::Option opt;
    opt.num_threads = 1;

    if (op->create_pipeline(opt) != 0)
    {
        fprintf(stderr, "test_matmul_shape failed to create pipeline\n");
        delete op;
        return -1;
    }

    std::vector<ncnn::Mat> bottoms(2);
    bottoms[0] = a;
    bottoms[1] = b;
    std::vector<ncnn::Mat> tops(1);

    int ret = op->forward(bottoms, tops, opt);

    op->destroy_pipeline(opt);
    delete op;

    if (ret != expected_ret)
    {
        fprintf(stderr, "test_matmul_shape failed a=(%d %d) b=(%d %d) transb=%d ret=%d expected=%d\n", a.w, a.h, b.w, b.h, transb, ret, expected_ret);
        return -1;
    }

    return 0;
}

static int test_matmul_invalid_shape()
{
    // reduction dim mismatch must be rejected instead of reading out of bounds
    // transb=0: A.w vs B.h ; transb=1: A.w vs B.w ; 1d dot: A.w vs B.w
    return 0
           || test_matmul_shape(RandomMat(8, 2), RandomMat(4, 3), 0, -1)
           || test_matmul_shape(RandomMat(8, 2), RandomMat(4, 3), 1, -1)
           || test_matmul_shape(RandomMat(8), RandomMat(4), 0, -1);
}

static int test_matmul_valid_shape_still_accepted()
{
    // matching reduction dims keep working
    return 0
           || test_matmul_shape(RandomMat(3, 2), RandomMat(4, 3), 0, 0)
           || test_matmul_shape(RandomMat(3, 2), RandomMat(3, 4), 1, 0)
           || test_matmul_shape(RandomMat(8), RandomMat(8), 0, 0);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_matmul_invalid_shape()
           || test_matmul_valid_shape_still_accepted();
}
