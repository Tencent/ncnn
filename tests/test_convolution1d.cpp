// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

static int test_convolution1d(int w, int h, int outh, int kernel, int dilation, int stride, int pad, int bias, int activation_type = 0)
{
    ncnn::Mat a = RandomMat(w, h);

    ncnn::ParamDict pd;
    pd.set(0, outh);     // num_output
    pd.set(1, kernel);   // kernel_w
    pd.set(2, dilation); // dilation_w
    pd.set(3, stride);   // stride_w
    pd.set(4, pad);      // pad_w
    pd.set(5, bias);     // bias_term
    pd.set(6, outh * h * kernel);

    ncnn::Mat activation_params(2);
    activation_params[0] = activation_type == 2 ? 0.1f : (activation_type == 6 ? 0.2f : -0.5f); // alpha
    activation_params[1] = activation_type == 6 ? 0.5f : 0.25f;                                 // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outh * h * kernel);
    if (bias)
        weights[1] = RandomMat(outh);

    int ret = test_layer("Convolution1D", pd, weights, a, 0.001);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolution1d failed w=%d h=%d outh=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d actparams=[%f,%f]\n", w, h, outh, kernel, dilation, stride, pad, bias, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_convolution1d_0()
{
    // kernel, dilation, stride and padding boundaries use a compact channel group
    static const int kdsp[][4] = {
        {1, 1, 1, 0}, {1, 1, 2, 0}, {2, 1, 1, 1}, {2, 1, 2, -233}, {3, 1, 1, 1}, {3, 1, 2, 1}, {3, 2, 1, 1}, {4, 1, 1, 2}, {4, 1, 2, -233}, {4, 2, 1, -234}, {5, 1, 1, -234}, {5, 1, 2, 2}, {5, 2, 2, 2}, {7, 1, 1, 3}, {7, 1, 2, 3}, {7, 2, 1, -233}
    };
    // scalar and large channel groups cover the kernel geometry independently
    // scalar geometry covers all padding, stride and dilation branches
    for (size_t i = 0; i < sizeof(kdsp) / sizeof(kdsp[0]); i++)
    {
        const int* k = kdsp[i];
        if (test_convolution1d(9, 1, 1, k[0], k[1], k[2], k[3], 0) != 0)
            return -1;
    }

    // large packed inputs retain specialized kernels and generic dilation routes
    static const int packed_kdsp[][4] = {
        {1, 1, 1, 0}, {1, 1, 2, 0}, {2, 1, 1, 1}, {3, 1, 1, 1}, {3, 1, 2, 1}, {3, 2, 1, 1}, {4, 2, 1, -234}, {5, 2, 2, 2}, {7, 1, 1, 3}, {7, 1, 2, 3}
    };
    for (size_t i = 0; i < sizeof(packed_kdsp) / sizeof(packed_kdsp[0]); i++)
    {
        const int* k = packed_kdsp[i];
        if (test_convolution1d(25, 48, 48, k[0], k[1], k[2], k[3], 1) != 0)
            return -1;
    }

    // each kernel retains the scalar channel remainder in its reduction depth
    static const int scalar_kdsp[][4] = {
        {1, 1, 1, 0}, {2, 1, 1, 1}, {3, 1, 1, 1}, {4, 1, 1, 2}, {5, 1, 1, -234}, {7, 1, 1, 3}
    };
    for (size_t i = 0; i < sizeof(scalar_kdsp) / sizeof(scalar_kdsp[0]); i++)
    {
        const int* k = scalar_kdsp[i];
        if (test_convolution1d(25, 31, 31, k[0], k[1], k[2], k[3], 0) != 0)
            return -1;
    }

    // channel tails retain specialized, generic and large kernel representatives
    if (test_convolution1d(9, 3, 3, 1, 1, 1, 0, 1)
            || test_convolution1d(25, 24, 24, 1, 1, 1, 0, 1)
            || test_convolution1d(25, 28, 28, 1, 1, 1, 0, 0)
            || test_convolution1d(9, 3, 3, 3, 1, 1, 1, 1)
            || test_convolution1d(9, 7, 7, 3, 1, 1, 1, 0)
            || test_convolution1d(9, 15, 15, 3, 1, 1, 1, 1)
            || test_convolution1d(25, 24, 24, 3, 1, 1, 1, 1)
            || test_convolution1d(25, 28, 28, 3, 1, 1, 1, 0)
            || test_convolution1d(25, 48, 31, 3, 1, 1, 1, 0)
            || test_convolution1d(9, 7, 7, 7, 1, 1, 3, 0)
            || test_convolution1d(9, 15, 15, 7, 1, 1, 3, 1)
            || test_convolution1d(25, 24, 24, 7, 1, 1, 3, 1)
            || test_convolution1d(25, 48, 31, 7, 1, 1, 3, 0))
        return -1;

    // pointwise kernels cover every mixed packing direction
    static const int pointwise_shapes[][3] = {
        {1, 3, 0}, {3, 1, 1}, {1, 31, 1}, {31, 1, 0}, {28, 31, 0}, {31, 28, 1}, {24, 28, 1}, {28, 24, 0}, {24, 31, 0}, {31, 24, 1}, {24, 48, 0}, {48, 24, 1}, {28, 48, 1}, {48, 28, 0}, {31, 48, 1}
    };
    for (size_t i = 0; i < sizeof(pointwise_shapes) / sizeof(pointwise_shapes[0]); i++)
    {
        const int* a = pointwise_shapes[i];
        if (test_convolution1d(25, a[0], a[1], 1, 1, 1, 0, a[2]) != 0)
            return -1;
    }

    // three-tap kernels retain scalar, partial and full vector packing directions
    static const int generic_shapes[][3] = {
        {1, 3, 0}, {3, 1, 1}, {1, 31, 1}, {31, 1, 0}, {28, 31, 0}, {31, 28, 1}, {24, 48, 0}, {48, 24, 1}
    };
    for (size_t i = 0; i < sizeof(generic_shapes) / sizeof(generic_shapes[0]); i++)
    {
        const int* a = generic_shapes[i];
        if (test_convolution1d(25, a[0], a[1], 3, 1, 1, 1, a[2]) != 0)
            return -1;
    }

    // large kernels cover full and partial cooperative matrix tiles in mixed packing
    if (test_convolution1d(25, 31, 48, 7, 1, 1, 3, 1)
            || test_convolution1d(25, 28, 31, 7, 1, 1, 3, 0)
            || test_convolution1d(25, 24, 48, 7, 1, 1, 3, 0)
            || test_convolution1d(25, 48, 24, 7, 1, 1, 3, 1))
        return -1;

    // each scalar output tier sees every input remainder tier during kernel transformation
    static const int tail_shapes[][2] = {
        {3, 31}, {7, 31}, {15, 31}, {1, 15}, {3, 15}, {7, 15}, {31, 15}, {1, 7}, {3, 7}, {15, 7}, {31, 7}, {7, 3}, {15, 3}, {31, 3}, {7, 1}, {15, 1}
    };
    for (size_t i = 0; i < sizeof(tail_shapes) / sizeof(tail_shapes[0]); i++)
    {
        if (test_convolution1d(9, tail_shapes[i][0], tail_shapes[i][1], 3, 1, 1, 1, 1) != 0)
            return -1;
    }

    // activation types are explicit and cover both packed and scalar output
    for (int activation = 0; activation < 7; activation++)
    {
        if (test_convolution1d(9, 7, 7, 3, 1, 1, 1, 1, activation)
                || test_convolution1d(9, 8, 16, 3, 1, 1, 1, 0, activation))
            return -1;
    }

    // output width and dispatch tile boundaries
    return 0
           || test_convolution1d(7, 1, 4, 3, 1, 1, 1, 1)
           || test_convolution1d(14, 1, 4, 3, 1, 2, 1, 1)
           || test_convolution1d(15, 4, 4, 3, 1, 1, 1, 1)
           || test_convolution1d(15, 8, 8, 3, 1, 1, 1, 1)
           || test_convolution1d(11, 8, 16, 3, 1, 1, 1, 1)
           || test_convolution1d(13, 16, 24, 3, 1, 1, 1, 1)
           || test_convolution1d(8, 16, 24, 3, 1, 1, 1, 0)
           || test_convolution1d(4, 16, 24, 3, 1, 1, 1, 1)
           || test_convolution1d(4, 16, 24, 3, 1, 1, 1, 0)
           || test_convolution1d(6, 64, 64, 3, 1, 2, 0, 1);
}

static int test_convolution1d_dynamic(int w, int h, int outh, int kernel, int dilation, int stride, int pad, int bias)
{
    ncnn::Mat a = RandomMat(w, h);

    ncnn::ParamDict pd;
    pd.set(0, 0);
    pd.set(1, 0);
    pd.set(2, dilation);
    pd.set(3, stride);
    pd.set(4, pad);
    pd.set(5, bias);
    pd.set(6, 0);
    pd.set(19, 1); // dynamic weight

    int activation_type = RAND() % 7; // 0 1 2 3 4 5 6
    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);                                               // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> as(bias ? 3 : 2);
    as[0] = a;
    as[1] = RandomMat(kernel, h, outh);
    if (bias)
        as[2] = RandomMat(outh);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Convolution1D", pd, weights, as);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolution1d_dynamic failed w=%d h=%d outh=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d actparams=[%f,%f]\n", w, h, outh, kernel, dilation, stride, pad, bias, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_convolution1d_1()
{
    // dynamic pointwise and three-tap kernels cover each packing direction
    const int packing_kdsp[][4] = {{1, 1, 1, 0}, {3, 1, 1, 1}};
    const int packing_shapes[][3] = {
        {1, 1, 1}, {4, 13, 0}, {13, 4, 1}, {12, 12, 0}, {8, 12, 1}, {8, 13, 0}, {13, 8, 1}, {12, 16, 0}, {15, 15, 0}, {16, 16, 0}
    };
    for (int g = 0; g < 2; g++)
    {
        for (int p = 0; p < 10; p++)
        {
            const int* k = packing_kdsp[g];
            const int* v = packing_shapes[p];
            if (test_convolution1d_dynamic(11, v[0], v[1], k[0], k[1], k[2], k[3], v[2]) != 0)
                return -1;
        }
    }

    // stride, dilation and automatic padding use scalar, packed and mixed channels
    const int geometry_kdsp[][4] = {
        {1, 1, 2, 0}, {2, 1, 1, 1}, {2, 1, 2, -233}, {3, 1, 2, 1}, {3, 2, 1, -234}
    };
    const int geometry_shapes[][3] = {{1, 1, 1}, {12, 12, 0}, {13, 4, 1}, {8, 13, 0}};
    for (int g = 0; g < 5; g++)
    {
        for (int p = 0; p < 4; p++)
        {
            const int* k = geometry_kdsp[g];
            const int* v = geometry_shapes[p];
            if (test_convolution1d_dynamic(11, v[0], v[1], k[0], k[1], k[2], k[3], v[2]) != 0)
                return -1;
        }
    }

    return 0;
}

static int test_convolution1d_activation_params_text()
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

        ncnn::Mat a(2, 1);
        a[0] = -3.f;
        a[1] = 3.f;
        ncnn::Mat reference(2, 1);
        reference[0] = -1.f;
        reference[1] = 2.f;
        ncnn::Mat b;
        int ret = test_layer_naive(ncnn::LayerType::Convolution1D, pd, weights, a, b, 0);
        if (ret == 0)
            ret = CompareMat(reference, b, 0.f);
        if (ret != 0)
        {
            fprintf(stderr, "test_convolution1d_activation_params_text failed params=%s ret=%d\n", params[i], ret);
            return ret;
        }

        const ncnn::Mat original = pd.get(10, ncnn::Mat());
        const int* p = original;
        if (i == 0 ? (p[0] != -1 || p[1] != 2) : (original[0] != -1.f || original[1] != 2.f))
        {
            fprintf(stderr, "test_convolution1d_activation_params_text modified params=%s\n", params[i]);
            return -1;
        }
    }
#endif
    return 0;
}

#if NCNN_VALIDATION
static int test_convolution1d_load_param_activation(const ncnn::ParamDict& base, int activation_type, const ncnn::Mat& activation_params, int expected_ret)
{
    ncnn::ParamDict pd = base;
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    return test_layer_param(ncnn::LayerType::Convolution1D, pd, expected_ret);
}

static int test_convolution1d_load_param()
{
    ncnn::ParamDict base;
    base.set(0, 8);
    base.set(1, 3);
    base.set(6, 216);
    if (test_layer_param(ncnn::LayerType::Convolution1D, base, 0) != 0)
        return -1;

    ncnn::Mat params(2);
    params[0] = 0.1f;
    params[1] = 0.5f;

    int ret = 0
              || test_convolution1d_load_param_activation(base, 0, ncnn::Mat(), 0)
              || test_convolution1d_load_param_activation(base, 0, ncnn::Mat(0), 0)
              || test_convolution1d_load_param_activation(base, 0, params, 0)
              || test_convolution1d_load_param_activation(base, 0, params.range(0, 1), 0)
              || test_convolution1d_load_param_activation(base, 1, ncnn::Mat(), 0)
              || test_convolution1d_load_param_activation(base, 1, ncnn::Mat(0), 0)
              || test_convolution1d_load_param_activation(base, 1, params, 0)
              || test_convolution1d_load_param_activation(base, 1, params.range(0, 1), 0)
              || test_convolution1d_load_param_activation(base, 2, ncnn::Mat(), -1)
              || test_convolution1d_load_param_activation(base, 2, ncnn::Mat(0), -1)
              || test_convolution1d_load_param_activation(base, 2, params, 0)
              || test_convolution1d_load_param_activation(base, 2, params.range(0, 1), 0)
              || test_convolution1d_load_param_activation(base, 3, ncnn::Mat(), -1)
              || test_convolution1d_load_param_activation(base, 3, ncnn::Mat(0), -1)
              || test_convolution1d_load_param_activation(base, 3, params, 0)
              || test_convolution1d_load_param_activation(base, 3, params.range(0, 1), -1)
              || test_convolution1d_load_param_activation(base, 4, ncnn::Mat(), 0)
              || test_convolution1d_load_param_activation(base, 4, ncnn::Mat(0), 0)
              || test_convolution1d_load_param_activation(base, 4, params, 0)
              || test_convolution1d_load_param_activation(base, 4, params.range(0, 1), 0)
              || test_convolution1d_load_param_activation(base, 5, ncnn::Mat(), 0)
              || test_convolution1d_load_param_activation(base, 5, ncnn::Mat(0), 0)
              || test_convolution1d_load_param_activation(base, 5, params, 0)
              || test_convolution1d_load_param_activation(base, 5, params.range(0, 1), 0)
              || test_convolution1d_load_param_activation(base, 6, ncnn::Mat(), -1)
              || test_convolution1d_load_param_activation(base, 6, ncnn::Mat(0), -1)
              || test_convolution1d_load_param_activation(base, 6, params, 0)
              || test_convolution1d_load_param_activation(base, 6, params.range(0, 1), -1)
              || test_convolution1d_load_param_activation(base, 7, ncnn::Mat(), -1);
    if (ret != 0)
        return ret;

    if (test_layer_param(ncnn::LayerType::Convolution1D, base, 10, 1, -1)
            || test_layer_param(ncnn::LayerType::Convolution1D, base, 10, 1.f, -1))
        return -1;

    ncnn::Mat missing_data(0);
    missing_data.w = 1;

    const ncnn::Mat bad[] = {ncnn::Mat(2, (size_t)1u), ncnn::Mat(2, (size_t)2u), ncnn::Mat(2, 2), ncnn::Mat(2, (size_t)16u, 4), missing_data};
    for (int i = 0; i < 5; i++)
    {
        if (test_layer_param(ncnn::LayerType::Convolution1D, base, 10, bad[i], -1) != 0)
            return -1;
    }

    ret = 0
          || test_layer_param(ncnn::LayerType::Convolution1D, base, 0, 0, -1)
          || test_layer_param(ncnn::LayerType::Convolution1D, base, 1, 0, -1)
          || test_layer_param(ncnn::LayerType::Convolution1D, base, 2, 0, -1)
          || test_layer_param(ncnn::LayerType::Convolution1D, base, 3, 0, -1)
          || test_layer_param(ncnn::LayerType::Convolution1D, base, 1, INT_MAX, -1)
          || test_layer_param(ncnn::LayerType::Convolution1D, base, 6, 217, -1);
    if (ret != 0)
        return ret;

    return 0;
}

static int test_convolution1d_load_param_text()
{
#if NCNN_STRING
    const char* params[] = {"0=1 1=1 6=1 9=3 -23310=2,-1,2", "0=1 1=1 6=1 9=3 -23310=2,-1.0,2.0"};
    for (int i = 0; i < 2; i++)
    {
        TestParamDict pd;
        if (pd.load_param(params[i]) != 0 || pd.type(10) != 5 + i)
            return -1;

        if (test_layer_param(ncnn::LayerType::Convolution1D, pd, 0) != 0)
            return -1;
    }
#endif
    return 0;
}
#endif // NCNN_VALIDATION

static int test_convolution1d_activation_packing_boundaries()
{
    // scalar and packed outputs apply activation after cooperative matrix kernels
    // the mixed packing directions also exercise bias and vector reductions
    for (int activation = 1; activation < 6; activation++)
    {
        if (test_convolution1d(9, 16, 15, 3, 1, 1, 1, 1, activation) != 0)
            return -1;
    }

    return 0
           || test_convolution1d(9, 16, 15, 1, 1, 1, 0, 1, 2)
           || test_convolution1d(9, 16, 16, 1, 1, 1, 0, 0, 2)
           || test_convolution1d(9, 3, 8, 3, 1, 1, 1, 0, 2);
}

static int test_convolution1d_cooperative_scalar_k_tail()
{
    // scalar channels exercise the small cooperative matrix k loop without ping-pong
    return test_convolution1d(9, 15, 15, 1, 1, 1, 0, 1);
}

static int test_convolution1d_activation_saturation()
{
    // identity weights cover hardswish bounds and mish saturation across packing directions
    const int channel_bias_activation[][4] = {{7, 7, 1, 6}, {8, 16, 0, 6}, {8, 7, 1, 6}, {16, 7, 1, 6}, {3, 3, 1, 5}};
    const float values[] = {-4.f, 0.f, 4.f};
    for (int i = 0; i < 5; i++)
    {
        const int input_channels = channel_bias_activation[i][0];
        const int output_channels = channel_bias_activation[i][1];
        const int bias = channel_bias_activation[i][2];
        const int activation = channel_bias_activation[i][3];
        ncnn::Mat a(9, input_channels);
        for (int q = 0; q < input_channels; q++)
        {
            float* ptr = a.row(q);
            for (int x = 0; x < 9; x++)
                ptr[x] = activation == 5 ? values[x / 3] * 4.f : values[x / 3];
        }

        ncnn::ParamDict pd;
        pd.set(0, output_channels);
        pd.set(1, 3);
        pd.set(2, 1);
        pd.set(3, 1);
        pd.set(4, 1);
        pd.set(5, bias);
        pd.set(6, output_channels * input_channels * 3);
        pd.set(9, activation);
        ncnn::Mat params(2);
        params[0] = 0.2f;
        params[1] = 0.5f;
        pd.set(10, params);

        std::vector<ncnn::Mat> weights(bias ? 2 : 1);
        weights[0] = ncnn::Mat(output_channels * input_channels * 3);
        weights[0].fill(0.f);
        for (int p = 0; p < output_channels; p++)
            weights[0][(p * input_channels + p % input_channels) * 3 + 1] = 1.f;
        if (bias)
        {
            weights[1] = ncnn::Mat(output_channels);
            weights[1].fill(0.f);
        }

        if (test_layer("Convolution1D", pd, weights, a, 0.001) != 0)
        {
            fprintf(stderr, "test_convolution1d_activation_saturation failed h=%d outh=%d bias=%d\n", input_channels, output_channels, bias);
            return -1;
        }
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_convolution1d_0()
           || test_convolution1d_1()
           || test_convolution1d_activation_params_text()
#if NCNN_VALIDATION
           || test_convolution1d_load_param()
           || test_convolution1d_load_param_text()
#endif // NCNN_VALIDATION
           || test_convolution1d_activation_packing_boundaries()
           || test_convolution1d_cooperative_scalar_k_tail()
           || test_convolution1d_activation_saturation();
}
