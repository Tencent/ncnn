// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_bias(const ncnn::Mat& a)
{
    int channels = a.c;

    ncnn::ParamDict pd;
    pd.set(0, channels);

    std::vector<ncnn::Mat> weights(1);
    weights[0] = RandomMat(channels);

    int ret = test_layer("Bias", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_bias failed a.dims=%d a=(%d %d %d %d)\n", a.dims, a.w, a.h, a.d, a.c);
    }

    return ret;
}

static int test_bias_0()
{
    return 0
           || test_bias(RandomMat(5, 6, 7, 24))
           || test_bias(RandomMat(7, 8, 9, 12))
           || test_bias(RandomMat(3, 4, 5, 13));
}

static int test_bias_1()
{
    return 0
           || test_bias(RandomMat(5, 7, 24))
           || test_bias(RandomMat(7, 9, 12))
           || test_bias(RandomMat(3, 5, 13));
}

#if NCNN_VALIDATION
// forward_inplace() reads bias_data[q] for every channel q of the blob
static int test_bias_short(int naive, int bias_data_size, int channels, int expected_ret)
{
    ncnn::ParamDict pd;
    pd.set(0, bias_data_size);

    std::vector<ncnn::Mat> weights(1);
    weights[0] = RandomMat(bias_data_size);

    ncnn::Layer* op = naive ? ncnn::create_layer_naive("Bias") : ncnn::create_layer_cpu("Bias");

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = false;

    op->load_param(pd);
    ncnn::ModelBinFromMatArray mb(weights.data());
    op->load_model(mb);
    op->create_pipeline(opt);

    ncnn::Mat a = RandomMat(2, 2, channels);
    int ret = op->forward_inplace(a, opt);

    op->destroy_pipeline(opt);
    delete op;

    if (ret != expected_ret)
    {
        fprintf(stderr, "test_bias_short failed naive=%d bias_data_size=%d channels=%d ret=%d expected=%d\n", naive, bias_data_size, channels, ret, expected_ret);
        return -1;
    }

    return 0;
}

static int test_bias_2()
{
    return 0
           || test_bias_short(1, 256, 256, 0)
           || test_bias_short(0, 256, 256, 0)
           || test_bias_short(1, 1, 256, -1)
           || test_bias_short(0, 1, 256, -1)
           || test_bias_short(1, 255, 256, -1)
           || test_bias_short(0, 255, 256, -1);
}
#endif // NCNN_VALIDATION

int main()
{
    SRAND(7767517);

    return 0
           || test_bias_0()
           || test_bias_1()
#if NCNN_VALIDATION
           || test_bias_2()
#endif
           ;
}
