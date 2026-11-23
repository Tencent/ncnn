// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

static int test_convolution3d(int w, int h, int d, int c, int outch, int kernel, int dilation, int stride, int pad, int bias)
{
    ncnn::Mat a = RandomMat(w, h, d, c);

    ncnn::ParamDict pd;
    pd.set(0, outch);    // num_output
    pd.set(1, kernel);   // kernel_w
    pd.set(2, dilation); // dilation_w
    pd.set(3, stride);   // stride_w
    pd.set(4, pad);      // pad_w
    pd.set(5, bias);     // bias_term
    pd.set(6, outch * c * kernel * kernel * kernel);

    int activation_type = RAND() % 6; // 0 1 2 3 4 5
    ncnn::Mat activation_params(2);
    activation_params[0] = RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);  // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outch * c * kernel * kernel * kernel);
    if (bias)
        weights[1] = RandomMat(outch);

    int ret = test_layer("Convolution3D", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolution3d failed w=%d h=%d d=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d actparams=[%f,%f]\n", w, h, d, c, outch, kernel, dilation, stride, pad, bias, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

// full pad control: asymmetric pads and nonzero pad value
static int test_convolution3d_pad(int w, int h, int d, int c, int outch, int kernel, int dilation, int stride, int pad_l, int pad_r, int pad_t, int pad_b, int pad_f, int pad_bh, float pad_value, int bias)
{
    ncnn::Mat a = RandomMat(w, h, d, c);

    ncnn::ParamDict pd;
    pd.set(0, outch);    // num_output
    pd.set(1, kernel);   // kernel_w
    pd.set(2, dilation); // dilation_w
    pd.set(3, stride);   // stride_w
    pd.set(4, pad_l);    // pad_left
    pd.set(15, pad_r);   // pad_right
    pd.set(14, pad_t);   // pad_top
    pd.set(16, pad_b);   // pad_bottom
    pd.set(24, pad_f);   // pad_front
    pd.set(17, pad_bh);  // pad_behind
    pd.set(18, pad_value);
    pd.set(5, bias); // bias_term
    pd.set(6, outch * c * kernel * kernel * kernel);

    int activation_type = RAND() % 6; // 0 1 2 3 4 5
    ncnn::Mat activation_params(2);
    activation_params[0] = RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);  // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outch * c * kernel * kernel * kernel);
    if (bias)
        weights[1] = RandomMat(outch);

    int ret = test_layer("Convolution3D", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolution3d_pad failed w=%d h=%d d=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pads=[%d,%d,%d,%d,%d,%d] pad_value=%f bias=%d act=%d actparams=[%f,%f]\n", w, h, d, c, outch, kernel, dilation, stride, pad_l, pad_r, pad_t, pad_b, pad_f, pad_bh, pad_value, bias, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_convolution3d_0()
{
    static const int kdsp[7][4] = {
        {1, 1, 1, 0},
        {1, 1, 2, 0},
        {2, 1, 1, 1},
        {2, 1, 2, -233},
        {3, 1, 1, 1},
        {3, 1, 2, 1},
        {3, 2, 1, -234},
    };

    for (int i = 0; i < 7; i++)
    {
        const int k = kdsp[i][0];
        const int d = kdsp[i][1];
        const int s = kdsp[i][2];
        const int p = kdsp[i][3];

        int ret = 0
                  || test_convolution3d(11, 10, 9, 1, 1, k, d, s, p, 1)
                  || test_convolution3d(11, 10, 9, 4, 13, k, d, s, p, 0)
                  || test_convolution3d(11, 10, 9, 13, 4, k, d, s, p, 1)
                  || test_convolution3d(11, 10, 9, 12, 12, k, d, s, p, 0)
                  || test_convolution3d(11, 10, 9, 8, 12, k, d, s, p, 1)
                  || test_convolution3d(11, 10, 9, 8, 13, k, d, s, p, 0)
                  || test_convolution3d(11, 10, 9, 13, 8, k, d, s, p, 1)
                  || test_convolution3d(11, 10, 9, 12, 16, k, d, s, p, 0)
                  || test_convolution3d(11, 10, 9, 15, 15, k, d, s, p, 0)
                  || test_convolution3d(11, 10, 9, 16, 16, k, d, s, p, 0);

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_convolution3d_activation_params_text()
{
#if NCNN_STRING
    const char* params[] = {"0=1 1=1 6=1 9=3 -23310=2,-1,2", "0=1 1=1 6=1 9=3 -23310=2,-1.0,2.0"};
    for (int i = 0; i < 2; i++)
    {
        TestParamDict pd;
        if (pd.load_param(params[i]) != 0 || pd.type(10) != 5 + i)
            return -1;

        std::vector<ncnn::Mat> weights(1);
        weights[0].create(1);
        weights[0][0] = 1.f;

        ncnn::Mat a(2, 1, 1, 1);
        a[0] = -3.f;
        a[1] = 3.f;
        ncnn::Mat reference(2, 1, 1, 1);
        reference[0] = -1.f;
        reference[1] = 2.f;
        ncnn::Mat b;
        int ret = test_layer_naive(ncnn::LayerType::Convolution3D, pd, weights, a, b, 0);
        if (ret == 0)
            ret = CompareMat(reference, b, 0.f);
        if (ret != 0)
        {
            fprintf(stderr, "test_convolution3d_activation_params_text failed params=%s ret=%d\n", params[i], ret);
            return ret;
        }

        const ncnn::Mat original = pd.get(10, ncnn::Mat());
        const int* p = original;
        if (i == 0 ? (p[0] != -1 || p[1] != 2) : (original[0] != -1.f || original[1] != 2.f))
        {
            fprintf(stderr, "test_convolution3d_activation_params_text modified params=%s\n", params[i]);
            return -1;
        }
    }
#endif
    return 0;
}

#if NCNN_VALIDATION
static int test_convolution3d_load_param_activation(const ncnn::ParamDict& base, int activation_type, const ncnn::Mat& activation_params, int expected_ret)
{
    ncnn::ParamDict pd = base;
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    return test_layer_param(ncnn::LayerType::Convolution3D, pd, expected_ret);
}

static int test_convolution3d_load_param()
{
    ncnn::ParamDict base;
    base.set(0, 8);
    base.set(1, 3);
    base.set(6, 216);
    if (test_layer_param(ncnn::LayerType::Convolution3D, base, 0) != 0)
        return -1;

    ncnn::Mat params(2);
    params[0] = 0.1f;
    params[1] = 0.5f;

    int ret = 0
              || test_convolution3d_load_param_activation(base, 0, ncnn::Mat(), 0)
              || test_convolution3d_load_param_activation(base, 0, ncnn::Mat(0), 0)
              || test_convolution3d_load_param_activation(base, 0, params, 0)
              || test_convolution3d_load_param_activation(base, 0, params.range(0, 1), 0)
              || test_convolution3d_load_param_activation(base, 1, ncnn::Mat(), 0)
              || test_convolution3d_load_param_activation(base, 1, ncnn::Mat(0), 0)
              || test_convolution3d_load_param_activation(base, 1, params, 0)
              || test_convolution3d_load_param_activation(base, 1, params.range(0, 1), 0)
              || test_convolution3d_load_param_activation(base, 2, ncnn::Mat(), -1)
              || test_convolution3d_load_param_activation(base, 2, ncnn::Mat(0), -1)
              || test_convolution3d_load_param_activation(base, 2, params, 0)
              || test_convolution3d_load_param_activation(base, 2, params.range(0, 1), 0)
              || test_convolution3d_load_param_activation(base, 3, ncnn::Mat(), -1)
              || test_convolution3d_load_param_activation(base, 3, ncnn::Mat(0), -1)
              || test_convolution3d_load_param_activation(base, 3, params, 0)
              || test_convolution3d_load_param_activation(base, 3, params.range(0, 1), -1)
              || test_convolution3d_load_param_activation(base, 4, ncnn::Mat(), 0)
              || test_convolution3d_load_param_activation(base, 4, ncnn::Mat(0), 0)
              || test_convolution3d_load_param_activation(base, 4, params, 0)
              || test_convolution3d_load_param_activation(base, 4, params.range(0, 1), 0)
              || test_convolution3d_load_param_activation(base, 5, ncnn::Mat(), 0)
              || test_convolution3d_load_param_activation(base, 5, ncnn::Mat(0), 0)
              || test_convolution3d_load_param_activation(base, 5, params, 0)
              || test_convolution3d_load_param_activation(base, 5, params.range(0, 1), 0)
              || test_convolution3d_load_param_activation(base, 6, ncnn::Mat(), -1)
              || test_convolution3d_load_param_activation(base, 6, ncnn::Mat(0), -1)
              || test_convolution3d_load_param_activation(base, 6, params, 0)
              || test_convolution3d_load_param_activation(base, 6, params.range(0, 1), -1)
              || test_convolution3d_load_param_activation(base, 7, ncnn::Mat(), -1);
    if (ret != 0)
        return ret;

    if (test_layer_param(ncnn::LayerType::Convolution3D, base, 10, 1, -1)
            || test_layer_param(ncnn::LayerType::Convolution3D, base, 10, 1.f, -1))
        return -1;

    ncnn::Mat missing_data(0);
    missing_data.w = 1;

    const ncnn::Mat bad[] = {ncnn::Mat(2, (size_t)1u), ncnn::Mat(2, (size_t)2u), ncnn::Mat(2, 2), ncnn::Mat(2, (size_t)16u, 4), missing_data};
    for (int i = 0; i < 5; i++)
    {
        if (test_layer_param(ncnn::LayerType::Convolution3D, base, 10, bad[i], -1) != 0)
            return -1;
    }

    ret = 0
          || test_layer_param(ncnn::LayerType::Convolution3D, base, 0, 0, -1)
          || test_layer_param(ncnn::LayerType::Convolution3D, base, 1, 0, -1)
          || test_layer_param(ncnn::LayerType::Convolution3D, base, 2, 0, -1)
          || test_layer_param(ncnn::LayerType::Convolution3D, base, 3, 0, -1)
          || test_layer_param(ncnn::LayerType::Convolution3D, base, 1, INT_MAX, -1)
          || test_layer_param(ncnn::LayerType::Convolution3D, base, 6, 217, -1);
    if (ret != 0)
        return ret;

    return 0;
}

static int test_convolution3d_load_param_text()
{
#if NCNN_STRING
    const char* params[] = {"0=1 1=1 6=1 9=3 -23310=2,-1,2", "0=1 1=1 6=1 9=3 -23310=2,-1.0,2.0"};
    for (int i = 0; i < 2; i++)
    {
        TestParamDict pd;
        if (pd.load_param(params[i]) != 0 || pd.type(10) != 5 + i)
            return -1;

        if (test_layer_param(ncnn::LayerType::Convolution3D, pd, 0) != 0)
            return -1;
    }
#endif
    return 0;
}
#endif // NCNN_VALIDATION

// winograd222 / cooperative matrix targeted cases: 3x3x3 stride 1 dilation 1
static int test_convolution3d_1()
{
    int ret = 0
              || test_convolution3d(7, 6, 5, 16, 16, 3, 1, 1, 1, 1)
              || test_convolution3d(7, 6, 5, 16, 20, 3, 1, 1, 1, 0)
              || test_convolution3d(7, 6, 5, 20, 16, 3, 1, 1, 1, 1)
              || test_convolution3d(7, 6, 5, 17, 16, 3, 1, 1, 1, 0)
              || test_convolution3d(7, 6, 5, 16, 17, 3, 1, 1, 1, 1)
              || test_convolution3d(7, 6, 5, 17, 17, 3, 1, 1, 1, 0)
              || test_convolution3d(7, 6, 5, 32, 32, 3, 1, 1, 1, 0)
              || test_convolution3d(5, 5, 5, 16, 16, 3, 1, 1, 0, 1);

    return ret;
}

// cm gemm / 1x1x1 cm targeted cases
static int test_convolution3d_2()
{
    int ret = 0
              || test_convolution3d(9, 8, 7, 32, 32, 3, 1, 2, 1, 1)
              || test_convolution3d(9, 8, 7, 16, 32, 2, 1, 1, 1, 0)
              || test_convolution3d(9, 8, 7, 32, 16, 1, 1, 1, 0, 1)
              || test_convolution3d(9, 8, 7, 48, 48, 1, 1, 1, 0, 0);

    return ret;
}

// boundary / extreme cases
static int test_convolution3d_boundary()
{
    int ret = 0;

    // tiny volumes, including single-voxel and single-slice dims
    ret |= test_convolution3d(1, 1, 1, 4, 4, 3, 1, 1, 1, 1);
    ret |= test_convolution3d(2, 2, 2, 16, 16, 3, 1, 1, 1, 0);
    ret |= test_convolution3d(3, 3, 3, 16, 16, 3, 1, 1, 1, 1);
    ret |= test_convolution3d(4, 4, 4, 16, 16, 3, 1, 1, 1, 0);
    ret |= test_convolution3d(5, 5, 5, 16, 16, 3, 1, 1, 1, 1);
    ret |= test_convolution3d(6, 7, 8, 16, 16, 3, 1, 1, 1, 0);
    ret |= test_convolution3d(1, 8, 8, 16, 16, 3, 1, 1, 1, 1);
    ret |= test_convolution3d(8, 1, 8, 16, 16, 3, 1, 1, 1, 0);
    ret |= test_convolution3d(8, 8, 1, 16, 16, 3, 1, 1, 1, 1);
    ret |= test_convolution3d(9, 9, 9, 32, 32, 3, 1, 1, 0, 0);

    // channel / outch packing boundaries (non-multiples of 4)
    ret |= test_convolution3d(8, 8, 8, 1, 1, 3, 1, 1, 1, 1);
    ret |= test_convolution3d(8, 8, 8, 2, 3, 3, 1, 1, 1, 0);
    ret |= test_convolution3d(8, 8, 8, 3, 5, 3, 1, 1, 1, 1);
    ret |= test_convolution3d(8, 8, 8, 5, 7, 3, 1, 2, 1, 0);
    ret |= test_convolution3d(8, 8, 8, 6, 9, 3, 1, 1, 1, 1);
    ret |= test_convolution3d(9, 9, 9, 17, 18, 3, 1, 1, 1, 0);
    ret |= test_convolution3d(9, 9, 9, 17, 16, 3, 1, 1, 1, 1);
    ret |= test_convolution3d(9, 9, 9, 16, 17, 3, 1, 1, 1, 0);
    ret |= test_convolution3d(8, 8, 8, 3, 3, 1, 1, 1, 0, 1);
    ret |= test_convolution3d(8, 8, 8, 5, 9, 1, 1, 1, 0, 0);
    ret |= test_convolution3d(8, 8, 8, 1, 13, 3, 1, 1, 1, 1);

    // stride / dilation / kernel extremes
    ret |= test_convolution3d(16, 16, 16, 8, 8, 2, 1, 1, 0, 1);
    ret |= test_convolution3d(16, 16, 16, 8, 8, 2, 1, 2, 1, 0);
    ret |= test_convolution3d(16, 16, 16, 8, 8, 4, 1, 1, 1, 1);
    ret |= test_convolution3d(16, 16, 16, 8, 8, 3, 1, 3, 2, 0);
    ret |= test_convolution3d(16, 16, 16, 8, 8, 3, 2, 1, 2, 1);
    ret |= test_convolution3d(16, 16, 16, 8, 8, 3, 2, 2, 2, 0);
    ret |= test_convolution3d(16, 16, 16, 16, 16, 1, 1, 1, 0, 1);
    ret |= test_convolution3d(17, 15, 13, 8, 8, 3, 1, 2, 1, 1);

    // asymmetric pads and nonzero pad value
    ret |= test_convolution3d_pad(9, 9, 9, 8, 8, 3, 1, 1, 1, 2, 0, 1, 2, 0, 0.f, 1);
    ret |= test_convolution3d_pad(9, 9, 9, 4, 8, 3, 1, 1, 1, 1, 1, 1, 1, 1, 0.5f, 0);
    ret |= test_convolution3d_pad(9, 9, 9, 16, 16, 3, 1, 1, 1, 1, 1, 1, 1, 1, 0.25f, 1);
    ret |= test_convolution3d_pad(9, 9, 9, 8, 8, 3, 1, 1, 2, 2, 2, 2, 2, 2, -1.f, 0);

    // SAME_UPPER / SAME_LOWER on odd sizes
    ret |= test_convolution3d(7, 7, 7, 8, 8, 3, 1, 1, -233, 1);
    ret |= test_convolution3d(8, 9, 10, 8, 8, 3, 1, 2, -233, 0);
    ret |= test_convolution3d(7, 7, 7, 16, 16, 3, 1, 1, -233, 1);
    ret |= test_convolution3d(8, 9, 10, 8, 8, 3, 1, 2, -234, 1);
    ret |= test_convolution3d(9, 9, 9, 8, 8, 3, 1, 1, -234, 0);

    // direct path (outch < 8) with odd outw/outh/outd: the extra 2x2x2 tile lanes
    // must be clamped so border loads stay inside the padded input
    ret |= test_convolution3d(5, 5, 5, 3, 3, 3, 1, 1, 1, 1);
    ret |= test_convolution3d(7, 7, 7, 1, 1, 3, 1, 1, 0, 1);
    ret |= test_convolution3d(7, 3, 5, 2, 5, 3, 1, 1, 1, 0);

    // gemm path (outch >= 8) with outsize % 4 != 0: the partial spatial group
    // lanes must be clamped before the input loads
    ret |= test_convolution3d(5, 5, 5, 8, 8, 3, 1, 1, 1, 1);
    ret |= test_convolution3d(5, 5, 5, 6, 8, 3, 1, 1, 1, 0);
    ret |= test_convolution3d(7, 3, 5, 7, 9, 3, 1, 1, 1, 1);

    // 1x1x1 pack4 path with spatial volume % 4 != 0: the partial spatial lanes
    // must be masked before the input loads
    ret |= test_convolution3d(3, 3, 3, 16, 16, 1, 1, 1, 0, 1);
    ret |= test_convolution3d(5, 5, 5, 16, 20, 1, 1, 1, 0, 0);

    // winograd222 pack4-to-pack1 path with block count % 4 != 0 (5x5x5 -> 27 blocks)
    ret |= test_convolution3d(5, 5, 5, 16, 17, 3, 1, 1, 1, 1);

    // cm gemm elempack==1 with partial spatial volume: the input loads must be
    // bounded by the input cstep, not the output cstep
    ret |= test_convolution3d(5, 5, 5, 13, 8, 3, 1, 1, 1, 1);
    ret |= test_convolution3d(5, 5, 5, 13, 9, 3, 1, 1, 1, 0);

    return ret;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_convolution3d_0()
           || test_convolution3d_1()
           || test_convolution3d_2()
           || test_convolution3d_boundary()
           || test_convolution3d_activation_params_text()
#if NCNN_VALIDATION
           || test_convolution3d_load_param()
           || test_convolution3d_load_param_text()
#endif // NCNN_VALIDATION
           ;
}
