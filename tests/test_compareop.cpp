// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int op_type = 0;

static int test_compareop(const ncnn::Mat& _a, const ncnn::Mat& _b)
{
    ncnn::Mat a = _a;
    ncnn::Mat b = _b;

    ncnn::ParamDict pd;
    pd.set(0, op_type);
    pd.set(1, 0);
    pd.set(2, 0.f);

    std::vector<ncnn::Mat> weights(0);

    int typeindex = ncnn::layer_to_index("CompareOp");
    if (typeindex == -1)
    {
        fprintf(stderr, "layer_to_index CompareOp failed\n");
        return -1;
    }

    ncnn::Layer* op = ncnn::create_layer_cpu(typeindex);
    if (!op)
    {
        fprintf(stderr, "create_layer_cpu CompareOp failed\n");
        return -1;
    }

    op->load_param(pd);

    ncnn::Option opt;
    opt.num_threads = 1;
    op->create_pipeline(opt);

    std::vector<ncnn::Mat> ab(2);
    ab[0] = a;
    ab[1] = b;

    std::vector<ncnn::Mat> c(1);
    int ret = op->forward(ab, c, opt);
    if (ret != 0)
    {
        fprintf(stderr, "forward failed ret=%d a.dims=%d b.dims=%d op_type=%d\n", ret, a.dims, b.dims, op_type);
        op->destroy_pipeline(opt);
        delete op;
        return -1;
    }

    op->destroy_pipeline(opt);
    delete op;

    return 0;
}

static int test_compareop_scalar(const ncnn::Mat& _a, float scalar)
{
    ncnn::Mat a = _a;

    ncnn::ParamDict pd;
    pd.set(0, op_type);
    pd.set(1, 1);
    pd.set(2, scalar);

    std::vector<ncnn::Mat> weights(0);

    ncnn::Layer* op = ncnn::create_layer_cpu(ncnn::layer_to_index("CompareOp"));
    if (!op)
    {
        fprintf(stderr, "create_layer_cpu CompareOp failed\n");
        return -1;
    }

    op->load_param(pd);

    ncnn::Option opt;
    opt.num_threads = 1;
    op->create_pipeline(opt);

    ncnn::Mat c;
    int ret = op->forward(a, c, opt);
    if (ret != 0)
    {
        fprintf(stderr, "forward failed ret=%d\n", ret);
        op->destroy_pipeline(opt);
        delete op;
        return -1;
    }

    op->destroy_pipeline(opt);
    delete op;

    return 0;
}

static int test_compareop_1()
{
    ncnn::Mat a = RandomMat(16);
    ncnn::Mat b = RandomMat(16);

    int ret = test_compareop(a, b);
    if (ret != 0)
        return ret;

    ret = test_compareop_scalar(a, 0.5f);
    if (ret != 0)
        return ret;

    return 0;
}

static int test_compareop_2()
{
    ncnn::Mat a = RandomMat(16, 16);
    ncnn::Mat b = RandomMat(16, 16);

    int ret = test_compareop(a, b);
    if (ret != 0)
        return ret;

    ret = test_compareop_scalar(a, 0.5f);
    if (ret != 0)
        return ret;

    return 0;
}

static int test_compareop_3()
{
    ncnn::Mat a = RandomMat(8, 8, 8);
    ncnn::Mat b = RandomMat(8, 8, 8);

    int ret = test_compareop(a, b);
    if (ret != 0)
        return ret;

    ret = test_compareop_scalar(a, 0.5f);
    if (ret != 0)
        return ret;

    return 0;
}

static int test_compareop_4()
{
    ncnn::Mat a = RandomMat(4, 4, 4, 4);
    ncnn::Mat b = RandomMat(4, 4, 4, 4);

    int ret = test_compareop(a, b);
    if (ret != 0)
        return ret;

    ret = test_compareop_scalar(a, 0.5f);
    if (ret != 0)
        return ret;

    return 0;
}

int main()
{
    SRAND(7767517);

    for (op_type = 0; op_type < 6; op_type++)
    {
        int ret = 0
                  || test_compareop_1()
                  || test_compareop_2()
                  || test_compareop_3()
                  || test_compareop_4();

        if (ret != 0)
        {
            fprintf(stderr, "test_compareop failed op_type=%d\n", op_type);
            return ret;
        }
    }

    fprintf(stderr, "test_compareop passed all tests\n");

    return 0;
}
