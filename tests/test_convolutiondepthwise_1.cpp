// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

static int test_convolutiondepthwise_dynamic(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias, int group, int activation_type)
{
    ncnn::Mat a = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, 0);
    pd.set(1, 0);
    pd.set(2, dilation);
    pd.set(3, stride);
    pd.set(4, pad);
    pd.set(5, bias);
    pd.set(6, 0);
    pd.set(7, group);
    pd.set(19, 1); // dynamic weight

    ncnn::Mat activation_params(2);
    activation_params[0] = activation_type == 6 ? 0.5f : (activation_type == 2 ? 0.1f : -0.5f); // alpha
    activation_params[1] = activation_type == 6 ? 0.5f : 0.3f;                                  // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> as(bias ? 3 : 2);
    as[0] = a;
    as[1] = RandomMat(kernel, kernel, c / group, outch);
    if (bias)
        as[2] = RandomMat(outch);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("ConvolutionDepthWise", pd, weights, as);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolutiondepthwise_dynamic failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d group=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, group, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_convolutiondepthwise_2()
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

    // each channel route traverses all activation types independently of input random values
    // zero activation retains depthwise stride2 and grouped stride1 postprocessing boundaries
    static const int activations[][13] = {
        {2, 3, 2, 2, 3, 2, 2, 3, 2, 3, 2, 3, 2},
        {3, 4, 3, 3, 4, 3, 3, 4, 3, 4, 3, 4, 3},
        {4, 5, 4, 4, 5, 4, 4, 5, 4, 5, 4, 5, 4},
        {5, 6, 5, 5, 6, 5, 5, 6, 5, 6, 5, 6, 5},
        {6, 0, 6, 6, 0, 6, 6, 0, 6, 0, 6, 0, 6},
        {0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0},
        {1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1},
    };

    for (int i = 0; i < 7; i++)
    {
        const int k = kdsp[i][0];
        const int d = kdsp[i][1];
        const int s = kdsp[i][2];
        const int p = kdsp[i][3];

        int ret = 0
                  || test_convolutiondepthwise_dynamic(11, 10, 1, 1, k, d, s, p, 1, 1, activations[i][0])
                  || test_convolutiondepthwise_dynamic(11, 10, 2, 2, k, d, s, p, 0, 1, activations[i][1])
                  || test_convolutiondepthwise_dynamic(11, 10, 2, 2, k, d, s, p, 1, 2, activations[i][2])
                  || test_convolutiondepthwise_dynamic(11, 10, 3, 3, k, d, s, p, 0, 3, activations[i][3])
                  || test_convolutiondepthwise_dynamic(11, 10, 4, 2, k, d, s, p, 1, 2, activations[i][4])
                  || test_convolutiondepthwise_dynamic(11, 10, 4, 4, k, d, s, p, 0, 4, activations[i][5])
                  || test_convolutiondepthwise_dynamic(11, 10, 7, 7, k, d, s, p, 1, 7, activations[i][6])
                  || test_convolutiondepthwise_dynamic(11, 10, 8, 8, k, d, s, p, 0, 2, activations[i][7])
                  || test_convolutiondepthwise_dynamic(11, 10, 8, 8, k, d, s, p, 1, 8, activations[i][8])
                  || test_convolutiondepthwise_dynamic(11, 10, 12, 12, k, d, s, p, 0, 4, activations[i][9])
                  || test_convolutiondepthwise_dynamic(11, 10, 15, 15, k, d, s, p, 1, 15, activations[i][10])
                  || test_convolutiondepthwise_dynamic(11, 10, 16, 8, k, d, s, p, 0, 2, activations[i][11])
                  || test_convolutiondepthwise_dynamic(11, 10, 16, 16, k, d, s, p, 1, 16, activations[i][12]);

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_convolutiondepthwise_dynamic_activation_tails()
{
    // generic scalar and packed channel routes retain every activation in full and partial tiles
    static const int channels[][4] = {{16, 16, 1, 16}, {12, 12, 0, 4}};
    for (int i = 0; i < 2; i++)
    {
        const int* c = channels[i];
        for (int activation = 0; activation < 7; activation++)
        {
            if (test_convolutiondepthwise_dynamic(112, 2, c[0], c[1], 2, 1, 1, 0, c[2], c[3], activation) != 0)
                return -1;
        }
    }
    return 0;
}

#if NCNN_INT8
static int test_convolutiondepthwise_int8(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias, int group, bool requant = false, int int8_scale_term = 0, bool input_int8 = false, int activation_type = 0)
{
    ncnn::Mat a = RandomMat(w, h, c);

    if (int8_scale_term == 0)
        int8_scale_term = requant ? 101 : 1;
    const bool use_requant = int8_scale_term > 100;

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, kernel);
    pd.set(2, dilation);
    pd.set(3, stride);
    pd.set(4, pad);
    pd.set(5, bias);
    pd.set(6, outch / group * c / group * kernel * kernel * group);
    pd.set(7, group);
    pd.set(8, int8_scale_term); // int8_scale_term

    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? 0.2f : -0.1f; // alpha
    activation_params[1] = 0.3f;                                  // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 5 : 4);
    weights[0] = RandomMat(outch / group * c / group * kernel * kernel * group);
    ncnn::Mat weight_scales;
    if (int8_scale_term == 2 || int8_scale_term == 102)
        weight_scales = scales_mat(weights[0], 1, weights[0].w, weights[0].w);
    else
        weight_scales = scales_mat(weights[0], group, c * kernel * kernel / group, c * kernel * kernel / group);
    ncnn::Mat input_scales = scales_mat(a, 1, w * h * c, a.cstep);
    ncnn::Mat top_scales = use_requant ? scales_mat(a, 1, w * h * c, a.cstep) : ncnn::Mat();

    ncnn::Mat a_int8 = a;
    if (input_int8)
    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = false;
        ncnn::quantize_to_int8(a, a_int8, input_scales, opt);
    }

    if (bias)
    {
        weights[1] = RandomMat(outch);
        weights[2] = weight_scales;
        weights[3] = input_scales;
        weights[4] = top_scales;
    }
    else
    {
        weights[1] = weight_scales;
        weights[2] = input_scales;
        weights[3] = top_scales;
    }

    int flag = input_int8 ? TEST_LAYER_DISABLE_AUTO_INPUT_CASTING : 0;
    int ret = 0;
    if (input_int8)
    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = true;
        opt.use_fp16_packed = false;
        opt.use_fp16_storage = false;
        opt.use_fp16_arithmetic = false;
        opt.use_bf16_packed = false;
        opt.use_bf16_storage = false;

        ret = test_layer_opt("ConvolutionDepthWise", pd, weights, opt, a_int8, use_requant ? 1.0f : 0.001f, flag);
    }
    else
    {
        ret = test_layer("ConvolutionDepthWise", pd, weights, a_int8, use_requant ? 1.0f : 0.001f, flag);
    }
    if (ret != 0)
    {
        fprintf(stderr, "test_convolutiondepthwise_int8 failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d group=%d int8_scale_term=%d input_int8=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, group, int8_scale_term, input_int8, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_convolutiondepthwise_1()
{
    static const int kdsp[16][4] = {
        {1, 1, 1, 0},
        {1, 1, 2, 0},
        {2, 1, 1, 1},
        {2, 1, 2, -233},
        {3, 1, 1, 1},
        {3, 1, 2, 1},
        {3, 2, 1, 1},
        {4, 1, 1, 2},
        {4, 1, 2, -233},
        {4, 2, 1, -234},
        {5, 1, 1, -234},
        {5, 1, 2, 2},
        {5, 2, 2, 2},
        {7, 1, 1, 3},
        {7, 1, 2, 3},
        {7, 2, 1, -233},
    };

    // scalar depthwise channels retain every geometry and both output types
    for (int requant = 0; requant < 2; requant++)
    {
        for (int i = 0; i < 16; i++)
        {
            const int* k = kdsp[i];
            if (test_convolutiondepthwise_int8(requant ? 9 : 15, 7, 7, 7, k[0], k[1], k[2], k[3], 0, 7, requant != 0, 0, false, 0) != 0)
                return -1;
        }
    }

    // packed geometry covers dedicated 3x3 kernels and full or partial dot-product blocks
    static const int packed_kdsp[][4] = {
        {1, 1, 1, 0},
        {2, 1, 1, 1},
        {3, 1, 1, 1},
        {3, 1, 2, 1},
        {3, 2, 1, 1},
        {4, 2, 1, -234},
        {5, 2, 2, 2},
        {7, 2, 1, -233},
    };
    for (int requant = 0; requant < 2; requant++)
    {
        for (size_t i = 0; i < sizeof(packed_kdsp) / sizeof(packed_kdsp[0]); i++)
        {
            const int* k = packed_kdsp[i];
            if (test_convolutiondepthwise_int8(requant ? 9 : 15, 7, 8, 8, k[0], k[1], k[2], k[3], 0, 8, requant != 0, 0, false, 0) != 0)
                return -1;
        }
    }

    // group packing is independent of the full depthwise geometry matrix
    static const int grouped_channels[][3] = {{12, 12, 4}, {8, 8, 2}};
    static const int grouped_kdsp[][4] = {
        {1, 1, 1, 0}, {3, 1, 1, 1}, {3, 1, 2, 1}, {5, 1, 2, 2}, {7, 2, 1, -233}
    };
    for (int requant = 0; requant < 2; requant++)
    {
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                const int* ch = grouped_channels[j];
                const int* k = grouped_kdsp[i];
                if (test_convolutiondepthwise_int8(requant ? 9 : 15, 7, ch[0], ch[1], k[0], k[1], k[2], k[3], 0, ch[2], requant != 0, 0, false, 0) != 0)
                    return -1;
            }
        }
    }

    // specialized int8 kernels and group packing conversions
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
    for (int requant = 0; requant < 2; requant++)
    {
        for (int i = 0; i < 13; i++)
        {
            for (int stride = 1; stride <= 2; stride++)
            {
                const int* ch = packing[i];
                if (test_convolutiondepthwise_int8(25, 33, ch[0], ch[1], 3, 1, stride, 1, ch[3], ch[2], requant != 0, 0, false, 0) != 0)
                    return -1;
            }
        }
    }

    // both post-kernel activation branches in scalar int8 specializations
    for (int requant = 0; requant < 2; requant++)
    {
        int ret = 0
                  || test_convolutiondepthwise_int8(15, 7, 3, 3, 3, 1, 1, 1, 1, 3, requant != 0, 0, false, 1)
                  || test_convolutiondepthwise_int8(15, 7, 3, 3, 3, 1, 2, 1, 0, 3, requant != 0, 0, false, 0)
                  || test_convolutiondepthwise_int8(15, 7, 3, 3, 3, 1, 2, 1, 0, 3, requant != 0, 0, false, 1);
        if (ret != 0)
            return -1;
    }

    // activations outside relu use the generic scalar int8 kernel
    for (int requant = 0; requant < 2; requant++)
    {
        int ret = 0
                  || test_convolutiondepthwise_int8(15, 7, 3, 3, 3, 1, 1, 1, 1, 3, requant != 0, 0, false, 2)
                  || test_convolutiondepthwise_int8(15, 7, 3, 3, 3, 1, 2, 1, 0, 3, requant != 0, 0, false, 2);
        if (ret != 0)
            return -1;
    }

    // grouped pack8 inputs retain the 1x1 and generic im2col spatial tails
    for (int requant = 0; requant < 2; requant++)
    {
        int ret = 0
                  || test_convolutiondepthwise_int8(9, 7, 16, 8, 1, 1, 1, 0, 0, 2, requant != 0, 0, false, 0)
                  || test_convolutiondepthwise_int8(9, 7, 16, 8, 3, 1, 1, 1, 1, 2, requant != 0, 0, false, 0)
                  || test_convolutiondepthwise_int8(13, 9, 16, 8, 3, 1, 2, 1, 0, 2, requant != 0, 0, false, 0);
        if (ret != 0)
            return -1;
    }

    // every packing route keeps both bias states independently of activation types
    static const int activation_channels[][3] = {
        {3, 3, 3},
        {8, 8, 8},
        {12, 12, 4},
        {8, 8, 2},
        {2, 2, 1},
        {4, 2, 2},
        {16, 8, 2},
    };
    // bias representatives for activation types one through six
    static const int activation_bias[][6] = {
        {0, 1, 0, 1, 0, 1},
        {1, 0, 1, 0, 1, 0},
        {0, 1, 0, 1, 0, 1},
        {1, 0, 1, 0, 1, 0},
        {0, 1, 0, 1, 0, 1},
        {1, 0, 1, 0, 1, 0},
        {0, 1, 0, 1, 0, 1},
    };
    for (int requant = 0; requant < 2; requant++)
    {
        for (int i = 0; i < 7; i++)
        {
            const int* ch = activation_channels[i];
            if (test_convolutiondepthwise_int8(10, 8, ch[0], ch[1], 2, 1, 1, 0, 0, ch[2], requant != 0, 0, false, 0)
                    || test_convolutiondepthwise_int8(10, 8, ch[0], ch[1], 2, 1, 1, 0, 1, ch[2], requant != 0, 0, false, 0))
                return -1;
            for (int act = 1; act < 7; act++)
            {
                if (test_convolutiondepthwise_int8(10, 8, ch[0], ch[1], 2, 1, 1, 0, activation_bias[i][act - 1], ch[2], requant != 0, 0, false, act) != 0)
                    return -1;
            }
        }
    }

    // depthwise scale layouts retain both small and large reduction depths
    static const int scale_kernels[][4] = {{1, 1, 1, 0}, {7, 2, 1, -233}};
    static const int scale_packing_kernels[][4] = {{1, 1, 1, 0}, {3, 1, 2, 1}};
    for (int requant = 0; requant < 2; requant++)
    {
        for (int i = 0; i < 2; i++)
        {
            const int* k = scale_kernels[i];
            const int* p = scale_packing_kernels[i];
            if (test_convolutiondepthwise_int8(15, 7, 8, 8, k[0], k[1], k[2], k[3], 1, 8, requant != 0, requant ? 102 : 2, false, 0)
                    || test_convolutiondepthwise_int8(15, 7, 2, 2, p[0], p[1], p[2], p[3], 0, 1, requant != 0, 0, false, 0)
                    || test_convolutiondepthwise_int8(15, 7, 4, 2, p[0], p[1], p[2], p[3], 1, 2, requant != 0, 0, false, 0))
                return -1;
        }

        // grouped uniform scales independently cover dequantized and requantized output
        if (test_convolutiondepthwise_int8(15, 7, 8, 8, 1, 1, 1, 0, 1, 2, requant != 0, requant ? 102 : 2, false, 0) != 0)
            return -1;
    }

    return 0;
}

static int test_convolutiondepthwise_int8_specialized_activations()
{
    // pack8 3x3 kernels have separate dequantized and requantized relu paths
    static const int cases[][4] = {
        {1, 0, 0, 1},
        {1, 1, 0, 1},
        {1, 0, 1, 1},
        {1, 1, 1, 1},
        {2, 0, 0, 1},
        {2, 1, 0, 1},
        {2, 0, 1, 1},
        {2, 1, 1, 1},
        {2, 0, 0, 2},
    };
    for (int i = 0; i < 9; i++)
    {
        const int* c = cases[i];
        if (test_convolutiondepthwise_int8(15, 7, 8, 8, 3, 1, c[0], 1, c[1], 8, c[2] != 0, 0, false, c[3]) != 0)
            return -1;
    }

    return 0;
}

static int test_convolutiondepthwise_int8_activation_boundaries()
{
    // scaled identity kernels cover all hard-swish regions in int8 arithmetic
    static const int channels[][3] = {
        {3, 3, 3},
        {4, 4, 4},
        {8, 8, 8},
        {16, 16, 16},
        {2, 2, 1},
        {12, 12, 4},
        {8, 8, 2},
        {16, 8, 2},
    };
    const float input[] = {-4.f, -1.f, 0.f, 1.f, 4.f, 0.f};
    for (int requant = 0; requant < 2; requant++)
    {
        for (int i = 0; i < 8; i++)
        {
            const int c = channels[i][0];
            const int outch = channels[i][1];
            const int group = channels[i][2];
            ncnn::Mat a(6, 2, c);
            a.fill(0.f);
            for (int q = 0; q < c; q++)
            {
                float* aptr = a.channel(q);
                for (int x = 0; x < 6; x++)
                    aptr[x] = input[x];
            }

            ncnn::ParamDict pd;
            pd.set(0, outch);
            pd.set(1, 2);
            pd.set(2, 1);
            pd.set(3, 1);
            pd.set(4, 0);
            pd.set(5, 0);
            pd.set(6, outch * c / group * 4);
            pd.set(7, group);
            pd.set(8, requant ? 101 : 1);
            pd.set(9, 6);
            ncnn::Mat params(2);
            params[0] = 0.2f;
            params[1] = 0.3f;
            pd.set(10, params);
            std::vector<ncnn::Mat> weights(4);
            weights[0].create(outch * c / group * 4);
            weights[0].fill(0.f);
            for (int q = 0; q < outch; q++)
                weights[0][q * c / group * 4] = 1.f;
            weights[1].create(group);
            weights[1].fill(127.f);
            weights[2].create(1);
            weights[2][0] = 31.75f;
            if (requant)
            {
                weights[3].create(1);
                weights[3][0] = 31.75f;
            }

            ncnn::Mat b;
            int ret = test_layer_naive(ncnn::LayerType::ConvolutionDepthWise, pd, weights, a, b, 0);
            if (ret == 0)
            {
                for (int q = 0; q < outch; q++)
                {
                    if (requant)
                    {
                        const signed char* ptr = b.channel(q);
                        if (ptr[0] != 0 || ptr[2] != 0 || ptr[4] != 127 || ptr[1] >= 0 || ptr[3] <= 0 || ptr[3] >= 127)
                            ret = -1;
                    }
                    else
                    {
                        const float* ptr = b.channel(q);
                        if (fabs(ptr[0]) > 0.000001f || fabs(ptr[2]) > 0.000001f || fabs(ptr[4] - 4.f) > 0.000001f || ptr[1] >= 0.f || ptr[3] <= 0.f || ptr[3] >= 4.f)
                            ret = -1;
                    }
                }
            }
            if (ret == 0)
                ret = test_layer("ConvolutionDepthWise", pd, weights, a, requant ? 1.0f : 0.001f);
            if (ret != 0)
            {
                fprintf(stderr, "test_convolutiondepthwise_int8_activation_boundaries failed c=%d outch=%d group=%d requant=%d\n", c, outch, group, requant);
                return ret;
            }
        }
    }

    return 0;
}

static int test_convolutiondepthwise_1_int8_input()
{
    return 0
           || test_convolutiondepthwise_int8(9, 7, 1, 1, 3, 1, 1, 1, 1, 1, false, 1, true)
           || test_convolutiondepthwise_int8(9, 7, 4, 4, 3, 1, 1, 1, 1, 4, false, 1, true)
           || test_convolutiondepthwise_int8(9, 7, 8, 8, 3, 1, 1, 1, 1, 2, false, 1, true)
           || test_convolutiondepthwise_int8(9, 7, 8, 8, 3, 1, 1, 1, 1, 2, false, 2, true)
           || test_convolutiondepthwise_int8(9, 7, 8, 8, 3, 1, 1, 1, 1, 8, true, 101, true)
           || test_convolutiondepthwise_int8(9, 7, 8, 8, 3, 1, 1, 1, 1, 2, true, 102, true);
}
static int test_convolutiondepthwise_int8_input_tile_boundaries()
{
    // the three geometries select distinct scalar input tile templates
    static const int kernels[][4] = {{1, 1, 2, 0}, {5, 1, 1, -234}, {7, 1, 2, 3}};
    for (int requant = 0; requant < 2; requant++)
    {
        for (int i = 0; i < 3; i++)
        {
            const int* k = kernels[i];
            if (test_convolutiondepthwise_int8(requant ? 9 : 15, 7, 12, 12, k[0], k[1], k[2], k[3], 0, 4, requant != 0, 0, false, 0) != 0)
                return -1;
        }
        // n66 and K27 retain two-column tiles and odd reduction tails together
        if (test_convolutiondepthwise_int8(11, 6, 12, 12, 3, 1, 1, 1, 0, 4, requant != 0, 0, false, 0) != 0)
            return -1;
    }
    return 0;
}

#endif // NCNN_INT8

static int test_convolutiondepthwise_dynamic_postprocessing_boundaries()
{
    // non-null activation retains specialized scalar and pack4/8/16 postprocessing
    static const int depthwise_channels[] = {3, 4, 8, 16};
    for (int i = 0; i < 4; i++)
    {
        const int c = depthwise_channels[i];
        if (test_convolutiondepthwise_dynamic(11, 10, c, c, 3, 1, 2, 1, 1, c, 1) != 0)
            return -1;
    }

    // grouped stride1 retains non-null winograd and gemm postprocessing
    static const int groups[][4] = {{2, 2, 0, 1}, {8, 8, 0, 2}, {12, 12, 0, 4}, {16, 8, 0, 2}};
    for (int i = 0; i < 4; i++)
    {
        const int* c = groups[i];
        if (test_convolutiondepthwise_dynamic(11, 10, c[0], c[1], 3, 1, 1, 1, c[2], c[3], 1) != 0)
            return -1;
    }

    // generic gemm and dilation fallback retain the no-postprocessing side
    return test_convolutiondepthwise_dynamic(11, 10, 16, 8, 2, 1, 1, 0, 0, 2, 0)
           || test_convolutiondepthwise_dynamic(11, 10, 2, 2, 3, 2, 1, -234, 0, 1, 0);
}

static int test_convolutiondepthwise_dynamic_scalar_pack4_activation_tails()
{
    // wide output retains vector bodies and tails for scalar2 and pack4 activations
    static const int channels[][4] = {{4, 4, 0, 4}, {2, 2, 0, 1}};
    for (int i = 0; i < 2; i++)
    {
        const int* c = channels[i];
        for (int activation = 0; activation < 7; activation++)
        {
            if (test_convolutiondepthwise_dynamic(112, 2, c[0], c[1], 2, 1, 1, 0, c[2], c[3], activation) != 0)
                return -1;
        }
    }
    return 0;
}

#if NCNN_INT8
static int test_convolutiondepthwise_int8_input_tile_tails()
{
    // w, h, channels, output channels, kernel, dilation, stride, padding, requant
    static const int cases[][9] = {
        // contiguous, strided, and cross-row spatial tails with odd reduction tails
        {19, 14, 12, 12, 5, 2, 1, 0, 0},
        {19, 14, 12, 12, 5, 2, 1, 0, 1},
        {29, 19, 12, 12, 5, 2, 2, 0, 0},
        {29, 19, 12, 12, 5, 2, 2, 0, 1},
        {9, 76, 12, 12, 5, 2, 1, 0, 0},
        {9, 76, 12, 12, 5, 2, 1, 0, 1},
        {9, 76, 28, 12, 5, 2, 1, 0, 0},
        {9, 76, 28, 12, 5, 2, 1, 0, 1},
        // grouped kernels cover dilation and stride dispatch boundaries
        {15, 7, 12, 12, 3, 2, 1, 2, 0},
        {15, 7, 12, 12, 7, 1, 1, 3, 0},
        {9, 7, 12, 12, 3, 2, 1, 2, 1},
        {9, 7, 12, 12, 7, 1, 1, 3, 1},
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
    {
        const int* c = cases[i];
        if (test_convolutiondepthwise_int8(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], 0, 4, c[8] != 0, 0, false, 0) != 0)
            return -1;
    }

    return 0;
}
static int test_convolutiondepthwise_int8_zero_weight_scales()
{
    // zero weight scales disable a group contribution while preserving its bias
    // channels, output channels and groups cover depthwise and grouped descales
    static const int cases[][3] = {{8, 8, 8}, {4, 4, 2}};
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
    {
        const int c = cases[i][0];
        const int outch = cases[i][1];
        const int group = cases[i][2];
        ncnn::Mat a(1, 1, c);
        a.fill(1.f);

        ncnn::ParamDict pd;
        pd.set(0, outch);
        pd.set(1, 1);
        pd.set(2, 1);
        pd.set(3, 1);
        pd.set(4, 0);
        pd.set(5, 1);
        pd.set(6, outch * c / group);
        pd.set(7, group);
        pd.set(8, 1);
        pd.set(9, 0);

        std::vector<ncnn::Mat> weights(5);
        weights[0].create(outch * c / group);
        weights[0].fill(3.f);
        weights[1].create(outch);
        for (int q = 0; q < outch; q++)
            weights[1][q] = q % 2 == 0 ? 1.f : -1.f;
        weights[2].create(group);
        for (int g = 0; g < group; g++)
            weights[2][g] = g % 2 == 0 ? 0.f : 1.f;
        weights[3].create(1);
        weights[3][0] = 1.f;

        ncnn::Mat b;
        int ret = test_layer_naive(ncnn::LayerType::ConvolutionDepthWise, pd, weights, a, b, 0);
        if (ret == 0 && (b.dims != 3 || b.w != 1 || b.h != 1 || b.c != outch || b.elempack != 1 || b.elemsize != 4u))
            ret = -1;
        if (ret == 0)
        {
            for (int q = 0; q < outch; q++)
            {
                const int g = q / (outch / group);
                const float expected = (g % 2 == 0 ? 0.f : 3.f * (c / group)) + weights[1][q];
                const float actual = ((const float*)b.channel(q))[0];
                if (actual != expected)
                {
                    fprintf(stderr, "test_convolutiondepthwise_int8_zero_weight_scales oracle failed c=%d outch=%d group=%d q=%d expected=%f actual=%f\n", c, outch, group, q, expected, actual);
                    ret = -1;
                    break;
                }
            }
        }
        if (ret == 0)
            ret = test_layer("ConvolutionDepthWise", pd, weights, a, 0.001f);
        if (ret != 0)
        {
            fprintf(stderr, "test_convolutiondepthwise_int8_zero_weight_scales failed c=%d outch=%d group=%d\n", c, outch, group);
            return -1;
        }
    }
    return 0;
}

#endif // NCNN_INT8

int main()
{
    SRAND(7767517);

#if NCNN_INT8
    return test_convolutiondepthwise_1()
           || test_convolutiondepthwise_1_int8_input()
           || test_convolutiondepthwise_int8_activation_boundaries()
           || test_convolutiondepthwise_2()
           || test_convolutiondepthwise_int8_specialized_activations()
           || test_convolutiondepthwise_int8_input_tile_boundaries()
           || test_convolutiondepthwise_dynamic_activation_tails()
           || test_convolutiondepthwise_dynamic_postprocessing_boundaries()
           || test_convolutiondepthwise_dynamic_scalar_pack4_activation_tails()
           || test_convolutiondepthwise_int8_input_tile_tails()
           || test_convolutiondepthwise_int8_zero_weight_scales();
#else
    return test_convolutiondepthwise_2()
           || test_convolutiondepthwise_dynamic_activation_tails()
           || test_convolutiondepthwise_dynamic_postprocessing_boundaries()
           || test_convolutiondepthwise_dynamic_scalar_pack4_activation_tails();
#endif
}
