// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_where(const ncnn::Mat& cond, const ncnn::Mat& a, const ncnn::Mat& b)
{
    ncnn::ParamDict pd;

    int typeindex = ncnn::layer_to_index("Where");
    if (typeindex == -1)
    {
        fprintf(stderr, "layer_to_index Where failed\n");
        return -1;
    }

    ncnn::Layer* op = ncnn::create_layer_cpu(typeindex);
    if (!op)
    {
        fprintf(stderr, "create_layer_cpu Where failed\n");
        return -1;
    }

    op->load_param(pd);

    ncnn::Option opt;
    opt.num_threads = 1;
    op->create_pipeline(opt);

    std::vector<ncnn::Mat> inputs(3);
    inputs[0] = cond;
    inputs[1] = a;
    inputs[2] = b;

    std::vector<ncnn::Mat> outputs(1);
    int ret = op->forward(inputs, outputs, opt);
    if (ret != 0)
    {
        fprintf(stderr, "forward failed ret=%d cond.dims=%d a.dims=%d b.dims=%d\n", ret, cond.dims, a.dims, b.dims);
        op->destroy_pipeline(opt);
        delete op;
        return -1;
    }

    op->destroy_pipeline(opt);
    delete op;

    return 0;
}

static int test_where_1()
{
    ncnn::Mat cond = RandomBoolMat(16);
    ncnn::Mat a = RandomMat(16);
    ncnn::Mat b = RandomMat(16);

    return test_where(cond, a, b);
}

static int test_where_2()
{
    ncnn::Mat cond = RandomBoolMat(16, 16);
    ncnn::Mat a = RandomMat(16, 16);
    ncnn::Mat b = RandomMat(16, 16);

    return test_where(cond, a, b);
}

static int test_where_3()
{
    ncnn::Mat cond = RandomBoolMat(8, 8, 8);
    ncnn::Mat a = RandomMat(8, 8, 8);
    ncnn::Mat b = RandomMat(8, 8, 8);

    return test_where(cond, a, b);
}

static int test_where_4()
{
    ncnn::Mat cond = RandomBoolMat(4, 4, 4, 4);
    ncnn::Mat a = RandomMat(4, 4, 4, 4);
    ncnn::Mat b = RandomMat(4, 4, 4, 4);

    return test_where(cond, a, b);
}

static int test_where_broadcast()
{
    ncnn::Mat cond = RandomBoolMat(1);
    ncnn::Mat a = RandomMat(16);
    ncnn::Mat b = RandomMat(16);

    return test_where(cond, a, b);
}

int main()
{
    SRAND(7767517);

    int ret = 0
              || test_where_1()
              || test_where_2()
              || test_where_3()
              || test_where_4()
              || test_where_broadcast();

    if (ret != 0)
    {
        fprintf(stderr, "test_where failed\n");
        return ret;
    }

    fprintf(stderr, "test_where passed all tests\n");

    return 0;
}
