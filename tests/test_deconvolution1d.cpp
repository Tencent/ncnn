// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

static int test_deconvolution1d(int w, int h, int outh, int kernel, int dilation, int stride, int pad, int bias, int output_pad_right, int output_w)
{
    ncnn::Mat a = RandomMat(w, h);

    if (output_w > 0 && pad != -233 && pad != -234)
    {
        pad = -233;
    }

    ncnn::ParamDict pd;
    pd.set(0, outh);
    pd.set(1, kernel);
    pd.set(2, dilation);
    pd.set(3, stride);
    pd.set(4, pad);
    pd.set(5, bias);
    pd.set(6, outh * h * kernel);

    int activation_type = RAND() % 5; // 0 1 2 3 4
    ncnn::Mat activation_params(2);
    activation_params[0] = RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);  // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    pd.set(18, output_pad_right);
    pd.set(20, output_w);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(outh * h * kernel);
    weights[1] = RandomMat(outh);

    int ret = test_layer("Deconvolution1D", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_deconvolution1d failed w=%d h=%d outh=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d actparams=[%f,%f] output_pad_right=%d output_w=%d\n", w, h, outh, kernel, dilation, stride, pad, bias, activation_type, activation_params[0], activation_params[1], output_pad_right, output_w);
    }

    return ret;
}

static int test_deconvolution1d_0()
{
    static const int kdsp[16][4] = {
        {1, 1, 1, 0},
        {1, 1, 2, 0},
        {2, 1, 1, 1},
        {2, 1, 2, -233},
        {3, 1, 1, 1},
        {3, 1, 2, 1},
        {3, 2, 1, 1},
        {4, 1, 1, -233},
        {4, 1, 2, -234},
        {4, 2, 1, -234},
        {5, 1, 1, 2},
        {5, 1, 2, 2},
        {5, 2, 2, 2},
        {7, 1, 1, 3},
        {7, 1, 2, 3},
        {7, 2, 1, -233},
    };

    for (int i = 0; i < 16; i++)
    {
        const int k = kdsp[i][0];
        const int d = kdsp[i][1];
        const int s = kdsp[i][2];
        const int p = kdsp[i][3];

        int ret = 0
                  || test_deconvolution1d(9, 1, 1, k, d, s, p, 1, 0, 0)
                  || test_deconvolution1d(9, 4, 13, k, d, s, p, 0, 1, 7)
                  || test_deconvolution1d(9, 13, 4, k, d, s, p, 1, 1, 0)
                  || test_deconvolution1d(9, 4, 8, k, d, s, p, 0, 0, 0)
                  || test_deconvolution1d(9, 8, 4, k, d, s, p, 1, 0, 7)
                  || test_deconvolution1d(9, 8, 13, k, d, s, p, 0, 2, 0)
                  || test_deconvolution1d(9, 13, 8, k, d, s, p, 1, 2, 0)
                  || test_deconvolution1d(9, 16, 16, k, d, s, p, 0, 0, 7);

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_deconvolution1d_dynamic(int w, int h, int outh, int kernel, int dilation, int stride, int pad, int bias, int output_pad_right, int output_w)
{
    ncnn::Mat a = RandomMat(w, h);

    if (output_w > 0 && pad != -233 && pad != -234)
    {
        pad = -233;
    }

    ncnn::ParamDict pd;
    pd.set(0, 0);
    pd.set(1, 0);
    pd.set(2, dilation);
    pd.set(3, stride);
    pd.set(4, pad);
    pd.set(5, bias);
    pd.set(6, 0);
    pd.set(28, 1); // dynamic weight

    int activation_type = RAND() % 5; // 0 1 2 3 4
    ncnn::Mat activation_params(2);
    activation_params[0] = RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);  // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    pd.set(18, output_pad_right);
    pd.set(20, output_w);

    std::vector<ncnn::Mat> as(bias ? 3 : 2);
    as[0] = a;
    as[1] = RandomMat(kernel, outh, h);
    if (bias)
        as[2] = RandomMat(outh);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Deconvolution1D", pd, weights, as);
    if (ret != 0)
    {
        fprintf(stderr, "test_deconvolution1d_dynamic failed w=%d h=%d outh=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d actparams=[%f,%f] output_pad_right=%d output_w=%d\n", w, h, outh, kernel, dilation, stride, pad, bias, activation_type, activation_params[0], activation_params[1], output_pad_right, output_w);
    }

    return ret;
}

static int test_deconvolution1d_1()
{
    static const int kdsp[16][4] = {
        {1, 1, 1, 0},
        {1, 1, 2, 0},
        {2, 1, 1, 1},
        {2, 1, 2, -233},
        {3, 1, 1, 1},
        {3, 1, 2, 1},
        {3, 2, 1, 1},
        {4, 1, 1, -233},
        {4, 1, 2, -234},
        {4, 2, 1, -234},
        {5, 1, 1, 2},
        {5, 1, 2, 2},
        {5, 2, 2, 2},
        {7, 1, 1, 3},
        {7, 1, 2, 3},
        {7, 2, 1, -233},
    };

    for (int i = 0; i < 16; i++)
    {
        const int k = kdsp[i][0];
        const int d = kdsp[i][1];
        const int s = kdsp[i][2];
        const int p = kdsp[i][3];

        int ret = 0
                  || test_deconvolution1d_dynamic(9, 1, 1, k, d, s, p, 1, 0, 0)
                  || test_deconvolution1d_dynamic(9, 4, 13, k, d, s, p, 0, 1, 7)
                  || test_deconvolution1d_dynamic(9, 13, 4, k, d, s, p, 1, 1, 0)
                  || test_deconvolution1d_dynamic(9, 4, 8, k, d, s, p, 0, 0, 0)
                  || test_deconvolution1d_dynamic(9, 8, 4, k, d, s, p, 1, 0, 7)
                  || test_deconvolution1d_dynamic(9, 8, 13, k, d, s, p, 0, 2, 0)
                  || test_deconvolution1d_dynamic(9, 13, 8, k, d, s, p, 1, 2, 0)
                  || test_deconvolution1d_dynamic(9, 16, 16, k, d, s, p, 0, 0, 7);

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_deconvolution1d_load_param_activation(const ncnn::ParamDict& base, int activation_type, const ncnn::Mat& activation_params, int expected_ret)
{
    ncnn::ParamDict pd = base;
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    return test_layer_param(ncnn::LayerType::Deconvolution1D, pd, expected_ret);
}

static int test_deconvolution1d_load_param()
{
    ncnn::ParamDict base;
    base.set(0, 8);
    base.set(1, 3);
    base.set(6, 216);
    if (test_layer_param(ncnn::LayerType::Deconvolution1D, base, 0) != 0)
        return -1;

    ncnn::Mat params(2);
    params[0] = 0.1f;
    params[1] = 0.5f;

    int ret = 0
              || test_deconvolution1d_load_param_activation(base, 0, ncnn::Mat(), 0)
              || test_deconvolution1d_load_param_activation(base, 0, ncnn::Mat(0), 0)
              || test_deconvolution1d_load_param_activation(base, 0, params, 0)
              || test_deconvolution1d_load_param_activation(base, 0, params.range(0, 1), 0)
              || test_deconvolution1d_load_param_activation(base, 1, ncnn::Mat(), 0)
              || test_deconvolution1d_load_param_activation(base, 1, ncnn::Mat(0), 0)
              || test_deconvolution1d_load_param_activation(base, 1, params, 0)
              || test_deconvolution1d_load_param_activation(base, 1, params.range(0, 1), 0)
              || test_deconvolution1d_load_param_activation(base, 2, ncnn::Mat(), -1)
              || test_deconvolution1d_load_param_activation(base, 2, ncnn::Mat(0), -1)
              || test_deconvolution1d_load_param_activation(base, 2, params, 0)
              || test_deconvolution1d_load_param_activation(base, 2, params.range(0, 1), 0)
              || test_deconvolution1d_load_param_activation(base, 3, ncnn::Mat(), -1)
              || test_deconvolution1d_load_param_activation(base, 3, ncnn::Mat(0), -1)
              || test_deconvolution1d_load_param_activation(base, 3, params, 0)
              || test_deconvolution1d_load_param_activation(base, 3, params.range(0, 1), -1)
              || test_deconvolution1d_load_param_activation(base, 4, ncnn::Mat(), 0)
              || test_deconvolution1d_load_param_activation(base, 4, ncnn::Mat(0), 0)
              || test_deconvolution1d_load_param_activation(base, 4, params, 0)
              || test_deconvolution1d_load_param_activation(base, 4, params.range(0, 1), 0)
              || test_deconvolution1d_load_param_activation(base, 5, ncnn::Mat(), 0)
              || test_deconvolution1d_load_param_activation(base, 5, ncnn::Mat(0), 0)
              || test_deconvolution1d_load_param_activation(base, 5, params, 0)
              || test_deconvolution1d_load_param_activation(base, 5, params.range(0, 1), 0)
              || test_deconvolution1d_load_param_activation(base, 6, ncnn::Mat(), -1)
              || test_deconvolution1d_load_param_activation(base, 6, ncnn::Mat(0), -1)
              || test_deconvolution1d_load_param_activation(base, 6, params, 0)
              || test_deconvolution1d_load_param_activation(base, 6, params.range(0, 1), -1)
              || test_deconvolution1d_load_param_activation(base, 7, ncnn::Mat(), -1);
    if (ret != 0)
        return ret;

    if (test_layer_param(ncnn::LayerType::Deconvolution1D, base, 10, 1, -1)
            || test_layer_param(ncnn::LayerType::Deconvolution1D, base, 10, 1.f, -1))
        return -1;

    ncnn::Mat missing_data(0);
    missing_data.w = 1;

    const ncnn::Mat bad[] = {ncnn::Mat(2, (size_t)1u), ncnn::Mat(2, (size_t)2u), ncnn::Mat(2, 2), ncnn::Mat(2, (size_t)16u, 4), missing_data};
    for (int i = 0; i < 5; i++)
    {
        if (test_layer_param(ncnn::LayerType::Deconvolution1D, base, 10, bad[i], -1) != 0)
            return -1;
    }

    ret = 0
          || test_layer_param(ncnn::LayerType::Deconvolution1D, base, 0, 0, -1)
          || test_layer_param(ncnn::LayerType::Deconvolution1D, base, 1, 0, -1)
          || test_layer_param(ncnn::LayerType::Deconvolution1D, base, 2, 0, -1)
          || test_layer_param(ncnn::LayerType::Deconvolution1D, base, 3, 0, -1)
          || test_layer_param(ncnn::LayerType::Deconvolution1D, base, 1, INT_MAX, -1)
          || test_layer_param(ncnn::LayerType::Deconvolution1D, base, 6, 217, -1);
    if (ret != 0)
        return ret;

    return 0;
}

static int test_deconvolution1d_load_param_text()
{
#if NCNN_STRING
    const char* params[] = {"0=1 1=1 6=1 9=3 -23310=2,-1,2", "0=1 1=1 6=1 9=3 -23310=2,-1.0,2.0"};
    for (int i = 0; i < 2; i++)
    {
        TestParamDict pd;
        if (pd.load_param(params[i]) != 0 || pd.type(10) != 5 + i)
            return -1;

        if (test_layer_param(ncnn::LayerType::Deconvolution1D, pd, 0) != 0)
            return -1;

        std::vector<ncnn::Mat> weights(1);
        weights[0].create(1);
        weights[0][0] = 1.f;

        ncnn::Mat a(2, 1);
        a[0] = -3.f;
        a[1] = 3.f;
        ncnn::Mat reference(2, 1);
        reference[0] = -1.f;
        reference[1] = 2.f;
        ncnn::Mat b;
        int ret = test_layer_naive(ncnn::LayerType::Deconvolution1D, pd, weights, a, b, 0);
        if (ret == 0)
            ret = CompareMat(reference, b, 0.f);
        if (ret != 0)
        {
            fprintf(stderr, "test_deconvolution1d_load_param_text failed params=%s ret=%d\n", params[i], ret);
            return ret;
        }

        const ncnn::Mat original = pd.get(10, ncnn::Mat());
        const int* p = original;
        if (i == 0 ? (p[0] != -1 || p[1] != 2) : (original[0] != -1.f || original[1] != 2.f))
        {
            fprintf(stderr, "test_deconvolution1d_load_param_text modified params=%s\n", params[i]);
            return -1;
        }
    }
#endif
    return 0;
}

int main()
{
    SRAND(7767517);

    return test_deconvolution1d_0() || test_deconvolution1d_1() || test_deconvolution1d_load_param() || test_deconvolution1d_load_param_text();
}
