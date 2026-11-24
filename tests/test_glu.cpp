// Copyright 2022 Xiaomi Corp.   (author: Fangjun Kuang)
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_glu(const ncnn::Mat& a, int axis, int flag = 0)
{
    ncnn::ParamDict pd;
    pd.set(0, axis);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("GLU", pd, weights, a, 0.001f, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_glu failed a.dims=%d a=(%d %d %d %d) axis=%d\n", a.dims, a.w, a.h, a.d, a.c, axis);
    }

    return ret;
}

static int test_glu_4d()
{
    return 0
           || test_glu(RandomMat(5, 7, 3, 8), 0)
           || test_glu(RandomMat(5, 7, 6, 3), 1)
           || test_glu(RandomMat(5, 8, 3, 3), 2)
           || test_glu(RandomMat(10, 7, 3, 3), 3)
           || test_glu(RandomMat(5, 7, 3, 8), -4)
           || test_glu(RandomMat(5, 7, 6, 3), -3)
           || test_glu(RandomMat(5, 8, 3, 3), -2)
           || test_glu(RandomMat(10, 7, 3, 3), -1)
           || test_glu(RandomMat(5, 7, 3, 12), 0)
           || test_glu(RandomMat(5, 7, 3, 48), -4);
}

static int test_glu_0()
{
    return 0
           || test_glu(RandomMat(6, 7, 24), 0)
           || test_glu(RandomMat(6, 8, 24), 1)
           || test_glu(RandomMat(6, 8, 24), 2)
           || test_glu(RandomMat(36, 7, 22), 0)
           || test_glu(RandomMat(5, 256, 23), -2)
           || test_glu(RandomMat(129, 9, 60), 2)
           || test_glu(RandomMat(129, 9, 30), -1)
           || test_glu(RandomMat(6, 7, 12), 0)
           || test_glu(RandomMat(5, 3, 20), 0)
           || test_glu(RandomMat(6, 7, 48), 0)
           || test_glu(RandomMat(2, 33, 16), -1)
           || test_glu(RandomMat(2, 33, 15), -1)
           || test_glu(RandomMat(7, 9, 5), 2);
}

static int test_glu_1()
{
    return 0
           || test_glu(RandomMat(10, 24), 0)
           || test_glu(RandomMat(7, 24), 1)
           || test_glu(RandomMat(128, 22), 0)
           || test_glu(RandomMat(128, 256), 1)
           || test_glu(RandomMat(17, 12), 0)
           || test_glu(RandomMat(17, 48), 0)
           || test_glu(RandomMat(3, 17), 1);
}

static int test_glu_2()
{
    return 0
           || test_glu(RandomMat(10), 0)
           || test_glu(RandomMat(20), 0)
           || test_glu(RandomMat(128), 0)
           || test_glu(RandomMat(12), 0)
           || test_glu(RandomMat(48), 0)
           || test_glu(RandomMat(7), 0);
}

static int test_glu_3()
{
    return 0
           || test_glu(RandomMat(2050, 3, 8), -1, TEST_LAYER_ENABLE_THREADING)
           || test_glu(RandomMat(2, 300, 16), -1, TEST_LAYER_ENABLE_THREADING)
           || test_glu(RandomMat(2, 300, 15), -1, TEST_LAYER_ENABLE_THREADING)
           || test_glu(RandomMat(40, 40, 12), 0, TEST_LAYER_ENABLE_THREADING)
           || test_glu(RandomMat(40, 40, 24), 0, TEST_LAYER_ENABLE_THREADING)
           || test_glu(RandomMat(40, 40, 48), 0, TEST_LAYER_ENABLE_THREADING)
           || test_glu(RandomMat(1500, 12), 0, TEST_LAYER_ENABLE_THREADING)
           || test_glu(RandomMat(4100), 0, TEST_LAYER_ENABLE_THREADING);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_glu_4d()
           || test_glu_0()
           || test_glu_1()
           || test_glu_2()
           || test_glu_3();
}
