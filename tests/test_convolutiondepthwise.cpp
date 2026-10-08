// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

static int test_convolutiondepthwise(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias, int group, int activation_type)
{
    ncnn::Mat a = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, kernel);
    pd.set(2, dilation);
    pd.set(3, stride);
    pd.set(4, pad);
    pd.set(5, bias);
    pd.set(6, outch / group * c / group * kernel * kernel * group);
    pd.set(7, group);

    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? 0.2f : -0.1f; // alpha
    activation_params[1] = 0.3f;                                  // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(outch / group * c / group * kernel * kernel * group);
    weights[1] = RandomMat(outch);

    int ret = test_layer("ConvolutionDepthWise", pd, weights, a, 0.001);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolutiondepthwise failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d group=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, group, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_convolutiondepthwise_0()
{
    // specialized kernels retain each depthwise and grouped packing transition
    static const int specialized_geometry[][6] = {
        {3, 1, 1, 1, 18, 17},
        {3, 1, 2, 1, 25, 33},
        {5, 1, 1, -234, 18, 17},
        {5, 1, 2, 2, 25, 33},
    };
    static const int specialized_channels[][3] = {
        {7, 7, 7},
        {8, 8, 8},
        {12, 12, 4},
        {8, 8, 2},
        {2, 2, 1},
        {4, 2, 2},
        {16, 8, 2},
        {4, 4, 4},
        {16, 16, 16},
    };
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 9; j++)
        {
            const int* g = specialized_geometry[i];
            const int* ch = specialized_channels[j];
            if (test_convolutiondepthwise(g[4], g[5], ch[0], ch[1], g[0], g[1], g[2], g[3], 0, ch[2], 0) != 0)
                return -1;
        }
    }

    // generic geometry is orthogonal to scalar, pack4/8/16 and grouped implementations
    static const int generic_geometry[][6] = {
        {1, 1, 1, 0, 15, 7},
        {1, 1, 2, 0, 18, 17},
        {2, 1, 1, 1, 25, 33},
        {2, 1, 2, -233, 15, 7},
        {3, 2, 1, 1, 15, 7},
        {4, 1, 1, 2, 18, 17},
        {4, 1, 2, -233, 25, 33},
        {4, 2, 1, -234, 15, 7},
        {5, 2, 2, 2, 15, 7},
        {7, 1, 1, 3, 18, 17},
        {7, 1, 2, 3, 25, 33},
        {7, 2, 1, -233, 15, 7},
    };
    static const int generic_channels[][3] = {
        {7, 7, 7},
        {4, 4, 4},
        {8, 8, 8},
        {16, 16, 16},
        {12, 12, 4},
    };
    for (int i = 0; i < 12; i++)
    {
        const int* g = generic_geometry[i];
        // scalar, pack16 and grouped convolution retain all geometries with intermediate pack anchors
        for (int j = 0; j < 5; j++)
        {
            if (j != 0 && j != 3 && j != 4 && i != 2 && i != 4 && i != 7 && i != 8 && i != 11)
                continue;
            const int* ch = generic_channels[j];
            if (test_convolutiondepthwise(g[4], g[5], ch[0], ch[1], g[0], g[1], g[2], g[3], 0, ch[2], 0) != 0)
                return -1;
        }
    }

    // specialized kernels, generic dilation and group packing conversions
    // input channels, output channels, groups and bias
    static const int packing[][4] = {
        {1, 1, 1, 1},
        {2, 2, 1, 0},
        {2, 2, 2, 1},
        {3, 3, 3, 0},
        {4, 2, 2, 1},
        {4, 4, 4, 0},
        {7, 7, 7, 1},
        {8, 8, 2, 0},
        {8, 8, 8, 1},
        {12, 12, 4, 0},
        {15, 15, 15, 1},
        {16, 8, 2, 0},
        {16, 16, 16, 1},
    };
    static const int kernels[][4] = {
        {3, 1, 1, 1},
        {3, 1, 2, 1},
        {5, 1, 1, 2},
        {5, 1, 2, 2},
        {7, 2, 1, -233},
    };
    for (int i = 0; i < 13; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            // scalar, pack8, pack16 and every grouped route retain activated kernels
            if (i != 0 && i != 1 && i != 4 && i != 7 && i != 8 && i != 9 && i != 11 && i != 12 && j != i % 5)
                continue;
            const int* ch = packing[i];
            const int* k = kernels[j];
            if (test_convolutiondepthwise(25, 33, ch[0], ch[1], k[0], k[1], k[2], k[3], ch[3], ch[2], 1) != 0)
                return -1;
        }
    }

    // grouped 1x1 output tiles with four, two and one element remainders
    static const int tile_shapes[][2] = {{7, 2}, {7, 1}, {15, 7}};
    for (int i = 0; i < 3; i++)
    {
        for (int bias = 0; bias < 2; bias++)
        {
            if (test_convolutiondepthwise(tile_shapes[i][0], tile_shapes[i][1], 16, 8, 1, 1, 1, 0, bias, 2, 0) != 0)
                return -1;
        }
    }

    // vector width remainders and both bias paths in specialized kernels
    static const int tail_shapes[][4] = {{15, 7, 0, 0}, {18, 17, 1, 1}};
    static const int tail_channels[] = {4, 8, 16};
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 4; k++)
            {
                const int c = tail_channels[j];
                const int* shape = tail_shapes[i];
                const int* geom = kernels[k];
                if (test_convolutiondepthwise(shape[0], shape[1], c, c, geom[0], geom[1], geom[2], geom[3], shape[2], c, shape[3]) != 0)
                    return -1;
            }
        }
    }

    // explicit activations and bias for scalar, packed and grouped paths
    static const int activation_channels[][3] = {
        {3, 3, 3},
        {4, 4, 4},
        {8, 8, 8},
        {16, 16, 16},
        {4, 2, 2},
        {12, 12, 4},
        {8, 8, 2},
        {16, 8, 2},
    };
    for (int i = 0; i < 8; i++)
    {
        for (int bias = 0; bias < 2; bias++)
        {
            for (int act = 0; act < 7; act++)
            {
                const int* ch = activation_channels[i];
                if (test_convolutiondepthwise(9, 7, ch[0], ch[1], 2, 1, 1, 1, bias, ch[2], act) != 0)
                    return -1;
            }
        }
    }

    // kernel larger than the input and partial workgroups on large inputs
    return 0
           || test_convolutiondepthwise(18, 17, 2, 2, 2, 1, 2, -233, 0, 1, 0)
           || test_convolutiondepthwise(15, 7, 1, 1, 5, 2, 2, 2, 1, 1, 0)
           || test_convolutiondepthwise(15, 7, 3, 3, 5, 2, 2, 2, 0, 3, 1)
           || test_convolutiondepthwise(65, 33, 7, 7, 3, 1, 1, 1, 1, 7, 2)
           || test_convolutiondepthwise(65, 33, 16, 16, 3, 1, 1, 1, 0, 16, 3);
}

static int test_convolutiondepthwise_activation_boundaries()
{
    // identity kernels exercise all hard-swish regions without random saturation
    static const int channels[][2] = {
        {3, 3},
        {4, 4},
        {8, 8},
        {16, 16},
        {2, 1},
        {12, 4},
        {8, 2},
        {16, 2},
    };
    const float input[] = {-4.f, -1.f, 0.f, 1.f, 4.f, 0.f};
    const float output[] = {0.f, -0.1f, 0.f, 0.5f, 4.f};
    for (int i = 0; i < 8; i++)
    {
        const int c = channels[i][0];
        const int group = channels[i][1];
        ncnn::Mat a(6, 2, c);
        a.fill(0.f);
        ncnn::Mat reference(5, 1, c);
        for (int q = 0; q < c; q++)
        {
            float* aptr = a.channel(q);
            float* rptr = reference.channel(q);
            for (int x = 0; x < 6; x++)
                aptr[x] = input[x];
            for (int x = 0; x < 5; x++)
                rptr[x] = output[x];
        }

        ncnn::ParamDict pd;
        pd.set(0, c);
        pd.set(1, 2);
        pd.set(2, 1);
        pd.set(3, 1);
        pd.set(4, 0);
        pd.set(5, 0);
        pd.set(6, c * c / group * 4);
        pd.set(7, group);
        pd.set(9, 6);
        ncnn::Mat params(2);
        params[0] = 0.2f;
        params[1] = 0.3f;
        pd.set(10, params);
        std::vector<ncnn::Mat> weights(1);
        weights[0].create(c * c / group * 4);
        weights[0].fill(0.f);
        for (int q = 0; q < c; q++)
            weights[0][q * c / group * 4] = 1.f;

        ncnn::Mat b;
        int ret = test_layer_naive(ncnn::LayerType::ConvolutionDepthWise, pd, weights, a, b, 0);
        if (ret == 0)
            ret = CompareMat(reference, b, 0.000001f);
        if (ret == 0)
            ret = test_layer("ConvolutionDepthWise", pd, weights, a);
        if (ret != 0)
        {
            fprintf(stderr, "test_convolutiondepthwise_activation_boundaries failed c=%d group=%d\n", c, group);
            return ret;
        }
    }

    return 0;
}

static int test_convolutiondepthwise_packed_activation_tails()
{
    // odd output rows and columns cover both packed vector blocks and tails
    for (int bias = 0; bias < 2; bias++)
    {
        for (int act = 1; act <= 6; act++)
        {
            if (test_convolutiondepthwise(8, 6, 8, 8, 2, 1, 1, 1, bias, 8, act) != 0)
                return -1;
        }
    }

    return 0;
}

static int test_convolutiondepthwise_activation_params_text()
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

        ncnn::Mat a(2, 1, 1);
        a[0] = -3.f;
        a[1] = 3.f;
        ncnn::Mat reference(2, 1, 1);
        reference[0] = -1.f;
        reference[1] = 2.f;
        ncnn::Mat b;
        int ret = test_layer_naive(ncnn::LayerType::ConvolutionDepthWise, pd, weights, a, b, 0);
        if (ret == 0)
            ret = CompareMat(reference, b, 0.f);
        if (ret != 0)
        {
            fprintf(stderr, "test_convolutiondepthwise_activation_params_text failed params=%s ret=%d\n", params[i], ret);
            return ret;
        }

        const ncnn::Mat original = pd.get(10, ncnn::Mat());
        const int* p = original;
        if (i == 0 ? (p[0] != -1 || p[1] != 2) : (original[0] != -1.f || original[1] != 2.f))
        {
            fprintf(stderr, "test_convolutiondepthwise_activation_params_text modified params=%s\n", params[i]);
            return -1;
        }
    }
#endif
    return 0;
}

#if NCNN_VALIDATION
static int test_convolutiondepthwise_load_param_activation(const ncnn::ParamDict& base, int activation_type, const ncnn::Mat& activation_params, int expected_ret)
{
    ncnn::ParamDict pd = base;
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    return test_layer_param(ncnn::LayerType::ConvolutionDepthWise, pd, expected_ret);
}

static int test_convolutiondepthwise_load_param()
{
    ncnn::ParamDict base;
    base.set(0, 8);
    base.set(1, 3);
    base.set(6, 216);
    base.set(7, 2);
    if (test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, 0) != 0)
        return -1;

    ncnn::Mat params(2);
    params[0] = 0.1f;
    params[1] = 0.5f;

    int ret = 0
              || test_convolutiondepthwise_load_param_activation(base, 0, ncnn::Mat(), 0)
              || test_convolutiondepthwise_load_param_activation(base, 0, ncnn::Mat(0), 0)
              || test_convolutiondepthwise_load_param_activation(base, 0, params, 0)
              || test_convolutiondepthwise_load_param_activation(base, 0, params.range(0, 1), 0)
              || test_convolutiondepthwise_load_param_activation(base, 1, ncnn::Mat(), 0)
              || test_convolutiondepthwise_load_param_activation(base, 1, ncnn::Mat(0), 0)
              || test_convolutiondepthwise_load_param_activation(base, 1, params, 0)
              || test_convolutiondepthwise_load_param_activation(base, 1, params.range(0, 1), 0)
              || test_convolutiondepthwise_load_param_activation(base, 2, ncnn::Mat(), -1)
              || test_convolutiondepthwise_load_param_activation(base, 2, ncnn::Mat(0), -1)
              || test_convolutiondepthwise_load_param_activation(base, 2, params, 0)
              || test_convolutiondepthwise_load_param_activation(base, 2, params.range(0, 1), 0)
              || test_convolutiondepthwise_load_param_activation(base, 3, ncnn::Mat(), -1)
              || test_convolutiondepthwise_load_param_activation(base, 3, ncnn::Mat(0), -1)
              || test_convolutiondepthwise_load_param_activation(base, 3, params, 0)
              || test_convolutiondepthwise_load_param_activation(base, 3, params.range(0, 1), -1)
              || test_convolutiondepthwise_load_param_activation(base, 4, ncnn::Mat(), 0)
              || test_convolutiondepthwise_load_param_activation(base, 4, ncnn::Mat(0), 0)
              || test_convolutiondepthwise_load_param_activation(base, 4, params, 0)
              || test_convolutiondepthwise_load_param_activation(base, 4, params.range(0, 1), 0)
              || test_convolutiondepthwise_load_param_activation(base, 5, ncnn::Mat(), 0)
              || test_convolutiondepthwise_load_param_activation(base, 5, ncnn::Mat(0), 0)
              || test_convolutiondepthwise_load_param_activation(base, 5, params, 0)
              || test_convolutiondepthwise_load_param_activation(base, 5, params.range(0, 1), 0)
              || test_convolutiondepthwise_load_param_activation(base, 6, ncnn::Mat(), -1)
              || test_convolutiondepthwise_load_param_activation(base, 6, ncnn::Mat(0), -1)
              || test_convolutiondepthwise_load_param_activation(base, 6, params, 0)
              || test_convolutiondepthwise_load_param_activation(base, 6, params.range(0, 1), -1)
              || test_convolutiondepthwise_load_param_activation(base, 7, ncnn::Mat(), -1);
    if (ret != 0)
        return ret;

    if (test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, 10, 1, -1)
            || test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, 10, 1.f, -1))
        return -1;

    ncnn::Mat missing_data(0);
    missing_data.w = 1;

    const ncnn::Mat bad[] = {ncnn::Mat(2, (size_t)1u), ncnn::Mat(2, (size_t)2u), ncnn::Mat(2, 2), ncnn::Mat(2, (size_t)16u, 4), missing_data};
    for (int i = 0; i < 5; i++)
    {
        if (test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, 10, bad[i], -1) != 0)
            return -1;
    }

    ret = 0
          || test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, 0, 0, -1)
          || test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, 1, 0, -1)
          || test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, 2, 0, -1)
          || test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, 3, 0, -1)
          || test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, 1, INT_MAX, -1)
          || test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, 6, 217, -1);
    if (ret != 0)
        return ret;

    const int groups[] = {0, -1, -8, INT_MIN, 3};
    for (int i = 0; i < 5; i++)
    {
        if (test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, 7, groups[i], -1) != 0)
            return -1;
    }

    {
        ncnn::ParamDict pd = base;
        pd.set(0, INT_MIN);
        pd.set(7, -1); // avoid evaluating INT_MIN % -1
        if (test_layer_param(ncnn::LayerType::ConvolutionDepthWise, pd, -1) != 0)
            return -1;
    }

    return 0;
}

static int test_convolutiondepthwise_load_param_int8()
{
    ncnn::ParamDict base;
    base.set(0, 1);
    base.set(1, 1);
    base.set(6, 1);
    if (test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, 0) != 0)
        return -1;

    const int valid[] = {1, 2, 101, 102};
    for (int i = 0; i < 4; i++)
    {
#if NCNN_INT8
        if (test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, 8, valid[i], 0) != 0)
#else
        if (test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, 8, valid[i], -1) != 0)
#endif
            return -1;
    }

    const int invalid[] = {-1, 3, 100, 103, INT_MIN, INT_MAX};
    for (int i = 0; i < 6; i++)
    {
        if (test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, 8, invalid[i], -1) != 0)
            return -1;
    }

    return 0;
}

static int test_convolutiondepthwise_load_param_dynamic()
{
    ncnn::ParamDict base;
    base.set(0, 5);
    base.set(1, 1);
    base.set(6, 10);
    base.set(7, 2);
    if (test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, -1) != 0)
        return -1;

    ncnn::ParamDict dynamic = base;
    dynamic.set(19, 1);

    const int num_outputs[] = {0, -1, 5, INT_MIN, INT_MAX};
    for (int i = 0; i < 5; i++)
    {
        if (test_layer_param(ncnn::LayerType::ConvolutionDepthWise, dynamic, 0, num_outputs[i], 0) != 0)
            return -1;
    }

    return 0
           || test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, 0, 2, 0)
           || test_layer_param(ncnn::LayerType::ConvolutionDepthWise, dynamic, 7, 0, -1)
           || test_layer_param(ncnn::LayerType::ConvolutionDepthWise, dynamic, 7, -1, -1)
           || test_layer_param(ncnn::LayerType::ConvolutionDepthWise, dynamic, 7, INT_MIN, -1);
}

static int test_convolutiondepthwise_load_param_text()
{
#if NCNN_STRING
    const char* params[] = {"0=1 1=1 6=1 9=3 -23310=2,-1,2", "0=1 1=1 6=1 9=3 -23310=2,-1.0,2.0"};
    for (int i = 0; i < 2; i++)
    {
        TestParamDict pd;
        if (pd.load_param(params[i]) != 0 || pd.type(10) != 5 + i)
            return -1;

        if (test_layer_param(ncnn::LayerType::ConvolutionDepthWise, pd, 0) != 0)
            return -1;
    }
#endif
    return 0;
}
#endif // NCNN_VALIDATION

static int test_convolutiondepthwise_activation_extremes()
{
    // identity inputs cover sigmoid exponent and mish saturation boundaries
    static const float values[] = {-20.f, -2.f, 0.f, 2.f, 20.f};
    for (int act = 4; act <= 5; act++)
    {
        ncnn::Mat a(5, 1, 1);
        for (int i = 0; i < 5; i++)
            a[i] = values[i];
        ncnn::ParamDict pd;
        pd.set(0, 1);
        pd.set(1, 1);
        pd.set(5, 0);
        pd.set(6, 1);
        pd.set(7, 1);
        pd.set(9, act);
        std::vector<ncnn::Mat> weights(1);
        weights[0].create(1);
        weights[0][0] = 1.f;
        if (test_layer("ConvolutionDepthWise", pd, weights, a, 0.001) != 0)
        {
            fprintf(stderr, "test_convolutiondepthwise_activation_extremes failed act=%d\n", act);
            return -1;
        }
    }
    return 0;
}

static int test_rectangular_kernels()
{
    // channel dimensions and the two independent strip-kernel orientations
    static const int cases[][4] = {{8, 8, 1, 3}, {8, 8, 3, 1}};
    static const float values[] = {-4.f, -2.f, -1.f, 0.f, 1.f, 2.f, 4.f};
    for (int i = 0; i < 2; i++)
    {
        const int* c = cases[i];
        ncnn::Mat a(7, 5, c[0]);
        for (int q = 0; q < c[0]; q++)
        {
            float* ptr = a.channel(q);
            for (int y = 0; y < 5; y++)
            {
                for (int x = 0; x < 7; x++)
                    ptr[y * 7 + x] = values[(x + y + q) % 7];
            }
        }

        ncnn::ParamDict pd;
        pd.set(0, c[1]);
        pd.set(1, c[2]);
        pd.set(11, c[3]);
        pd.set(2, 1);
        pd.set(3, 1);
        pd.set(4, c[2] == 3 ? 1 : 0);
        pd.set(15, c[2] == 3 ? 1 : 0);
        pd.set(14, c[3] == 3 ? 1 : 0);
        pd.set(16, c[3] == 3 ? 1 : 0);
        pd.set(5, 0);
        pd.set(7, c[0]);
        const int weight_count = c[0] * c[2] * c[3];
        pd.set(6, weight_count);
        std::vector<ncnn::Mat> weights(1);
        weights[0].create(weight_count);
        for (int k = 0; k < weight_count; k++)
            weights[0][k] = (float)(k % 3 - 1);
        if (test_layer("ConvolutionDepthWise", pd, weights, a, 0.001) != 0)
        {
            fprintf(stderr, "test_rectangular_kernels failed c=%d outch=%d kernel=%dx%d\n", c[0], c[1], c[2], c[3]);
            return -1;
        }
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_convolutiondepthwise_0()
           || test_convolutiondepthwise_activation_boundaries()
           || test_convolutiondepthwise_packed_activation_tails()
           || test_convolutiondepthwise_activation_params_text()
#if NCNN_VALIDATION
           || test_convolutiondepthwise_load_param()
           || test_convolutiondepthwise_load_param_text()
           || test_convolutiondepthwise_load_param_dynamic()
           || test_convolutiondepthwise_load_param_int8()
#endif // NCNN_VALIDATION
           || test_rectangular_kernels()
           || test_convolutiondepthwise_activation_extremes();
}
