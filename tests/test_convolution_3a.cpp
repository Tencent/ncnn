// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#if NCNN_INT8
static int test_convolution_int8(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias, bool requant = false, int int8_scale_term = 0, bool sgemm = false, bool input_int8 = false, bool test_winograd43 = false)
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
    pd.set(6, outch * c * kernel * kernel);
    pd.set(8, int8_scale_term); // int8_scale_term

    int activation_type = RAND() % 7; // 0 1 2 3 4 5 6
    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);                                               // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 5 : 4);
    weights[0] = RandomMat(outch * c * kernel * kernel);

    ncnn::Mat weight_scales = scales_mat(weights[0], outch, c * kernel * kernel, c * kernel * kernel);
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

    if (kernel == 3 && dilation == 1 && stride == 1)
    {
        // test for 6bit quant
        for (int i = 0; i < weight_scales.w; i++)
            weight_scales[i] = weight_scales[i] / 4.f;
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
        opt.use_sgemm_convolution = sgemm;
        opt.use_winograd_convolution = false;

        ret = test_layer_opt("Convolution", pd, weights, opt, a_int8, use_requant ? 1.0f : 0.001f, flag);
    }
    else
    {
        ret = test_layer("Convolution", pd, weights, a_int8, use_requant ? 1.0f : 0.001f, flag);
    }
    if (ret != 0)
    {
        fprintf(stderr, "test_convolution_int8 failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d int8_scale_term=%d sgemm=%d input_int8=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, int8_scale_term, sgemm, input_int8, activation_type, activation_params[0], activation_params[1]);
        return ret;
    }

    if (input_int8)
        return ret;

    if (kernel == 3 && dilation == 1 && stride == 1)
    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = true;
        opt.use_fp16_packed = false;
        opt.use_fp16_storage = false;
        opt.use_fp16_arithmetic = false;
        opt.use_bf16_packed = false;
        opt.use_bf16_storage = false;
        opt.use_sgemm_convolution = false;
        opt.use_winograd_convolution = true;
        for (int i = 0; i < (test_winograd43 ? 2 : 1); i++)
        {
            opt.use_winograd23_convolution = i == 0;
            opt.use_winograd43_convolution = i == 1;

            ret = test_layer_opt("Convolution", pd, weights, opt, a, use_requant ? 1.0f : 0.001f, flag);
            if (ret != 0)
            {
                fprintf(stderr, "test_convolution_int8 failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d int8_scale_term=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, int8_scale_term, activation_type, activation_params[0], activation_params[1]);
                return ret;
            }
        }
    }

    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = false;
        opt.use_fp16_packed = false;
        opt.use_fp16_storage = false;
        opt.use_fp16_arithmetic = false;
        opt.use_bf16_packed = false;
        opt.use_bf16_storage = false;
        opt.use_sgemm_convolution = false;
        opt.use_winograd_convolution = false;

        ret = test_layer_opt("Convolution", pd, weights, opt, a, use_requant ? 1.0f : 0.001f, flag);
        if (ret != 0)
        {
            fprintf(stderr, "test_convolution_int8 failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d int8_scale_term=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, int8_scale_term, activation_type, activation_params[0], activation_params[1]);
            return ret;
        }
    }

    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = true;
        opt.use_fp16_packed = false;
        opt.use_fp16_storage = false;
        opt.use_fp16_arithmetic = false;
        opt.use_bf16_packed = false;
        opt.use_bf16_storage = false;
        opt.use_sgemm_convolution = false;
        opt.use_winograd_convolution = false;

        ret = test_layer_opt("Convolution", pd, weights, opt, a, use_requant ? 1.0f : 0.001f, flag);
        if (ret != 0)
        {
            fprintf(stderr, "test_convolution_int8 failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d int8_scale_term=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, int8_scale_term, activation_type, activation_params[0], activation_params[1]);
            return ret;
        }
    }

    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = true;
        opt.use_fp16_packed = true;
        opt.use_fp16_storage = true;
        opt.use_fp16_arithmetic = true;
        opt.use_bf16_packed = true;
        opt.use_bf16_storage = true;
        opt.use_sgemm_convolution = false;
        opt.use_winograd_convolution = false;

        ret = test_layer_opt("Convolution", pd, weights, opt, a, use_requant ? 1.0f : 0.001f, flag);
        if (ret != 0)
        {
            fprintf(stderr, "test_convolution_int8 failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d int8_scale_term=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, int8_scale_term, activation_type, activation_params[0], activation_params[1]);
            return ret;
        }
    }

    if (sgemm)
    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = true;
        opt.use_fp16_packed = false;
        opt.use_fp16_storage = false;
        opt.use_fp16_arithmetic = false;
        opt.use_bf16_packed = false;
        opt.use_bf16_storage = false;
        opt.use_sgemm_convolution = true;
        opt.use_winograd_convolution = false;

        ret = test_layer_opt("Convolution", pd, weights, opt, a, use_requant ? 1.0f : 0.001f, flag);
        if (ret != 0)
        {
            fprintf(stderr, "test_convolution_int8 failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d int8_scale_term=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, int8_scale_term, activation_type, activation_params[0], activation_params[1]);
            return ret;
        }
    }

    return ret;
}

static int test_convolution_1()
{
    // exercise each packing direction on the pointwise and 3x3 paths
    static const int kdsp_packing[2][4] = {
        {1, 1, 1, 0},
        {3, 1, 1, 1},
    };

    static const int channels_packing[10][2] = {
        {1, 1},
        {2, 2},
        {3, 3},
        {4, 4},
        {7, 7},
        {8, 8},
        {15, 15},
        {16, 15},
        {15, 16},
        {16, 16},
    };

    for (int q = 0; q < 2; q++)
    {
        for (int i = 0; i < 2; i++)
        {
            const int k = kdsp_packing[i][0];
            const int d = kdsp_packing[i][1];
            const int s = kdsp_packing[i][2];
            const int p = kdsp_packing[i][3];

            for (int j = 0; j < 10; j++)
            {
                const int c = channels_packing[j][0];
                const int outch = channels_packing[j][1];
                if (test_convolution_int8(9, 7, c, outch, k, d, s, p, 1, q != 0) != 0)
                    return -1;
            }
        }
    }

    // vary kernel geometry with packed, unpacked and mixed channel directions
    static const int kdsp_geometry[5][4] = {
        {2, 1, 1, 1},
        {3, 2, 1, 1},
        {4, 1, 1, 2},
        {4, 2, 1, -234},
        {5, 1, 1, -234},
    };

    static const int channels_geometry[6][2] = {
        {1, 1},
        {4, 4},
        {8, 8},
        {16, 15},
        {15, 16},
        {16, 16},
    };

    for (int q = 0; q < 2; q++)
    {
        for (int i = 0; i < 5; i++)
        {
            const int k = kdsp_geometry[i][0];
            const int d = kdsp_geometry[i][1];
            const int s = kdsp_geometry[i][2];
            const int p = kdsp_geometry[i][3];

            for (int j = 0; j < 6; j++)
            {
                const int c = channels_geometry[j][0];
                const int outch = channels_geometry[j][1];
                if (test_convolution_int8(9, 7, c, outch, k, d, s, p, 1, q != 0) != 0)
                    return -1;
            }
        }
    }

    // vary large kernels separately from channel packing transitions
    static const int kdsp_7x7[2][4] = {
        {7, 1, 1, 3},
        {7, 2, 1, -233},
    };

    static const int channels_7x7[2][2] = {
        {1, 1},
        {8, 8},
    };

    for (int q = 0; q < 2; q++)
    {
        for (int i = 0; i < 2; i++)
        {
            const int k = kdsp_7x7[i][0];
            const int d = kdsp_7x7[i][1];
            const int s = kdsp_7x7[i][2];
            const int p = kdsp_7x7[i][3];

            for (int j = 0; j < 2; j++)
            {
                const int c = channels_7x7[j][0];
                const int outch = channels_7x7[j][1];
                if (test_convolution_int8(9, 7, c, outch, k, d, s, p, 1, q != 0) != 0)
                    return -1;
            }
        }
    }

    return 0
           || test_convolution_int8(11, 11, 8, 16, 3, 1, 1, 1, 1)
           || test_convolution_int8(13, 16, 16, 24, 3, 1, 1, 1, 1)
           || test_convolution_int8(8, 8, 16, 24, 3, 1, 1, 1, 0)
           || test_convolution_int8(4, 8, 16, 24, 3, 1, 1, 1, 1)
           || test_convolution_int8(4, 20, 16, 24, 3, 1, 1, 1, 0)
           || test_convolution_int8(25, 33, 16, 15, 3, 1, 1, 1, 0)
           || test_convolution_int8(25, 33, 31, 31, 3, 1, 1, 1, 0)
           || test_convolution_int8(9, 7, 32, 16, 3, 1, 1, 1, 1, false, 0, false, false, true)
           || test_convolution_int8(17, 13, 64, 64, 3, 1, 1, 1, 1, false, 0, false, false, true)
           || test_convolution_int8(13, 11, 65, 97, 3, 1, 1, 1, 1, false, 0, false, false, true)
           || test_convolution_int8(13, 11, 65, 97, 3, 1, 1, 1, 1, true, 0, false, false, true)
           || test_convolution_int8(7, 7, 15, 12, 3, 1, 1, 1, 0)
           || test_convolution_int8(5, 6, 31, 9, 5, 1, 1, 0, 1)
           || test_convolution_int8(5, 10, 5, 32, 3, 2, 1, 0, 1)
           || test_convolution_int8(3, 9, 16, 13, 2, 2, 1, 0, 0)
           || test_convolution_int8(23, 11, 33, 28, 5, 1, 1, 0, 1)
           || test_convolution_int8(7, 5, 4, 8, 1, 1, 1, 0, 1, false, 2)
           || test_convolution_int8(7, 5, 4, 8, 1, 1, 1, 0, 1, true, 102);
}

static int test_convolution_1_int8_input()
{
    return 0
           || test_convolution_int8(7, 5, 1, 1, 3, 1, 1, 1, 1, false, 1, false, true)
           || test_convolution_int8(7, 5, 4, 4, 3, 1, 1, 1, 1, false, 1, false, true)
           || test_convolution_int8(8, 6, 4, 8, 1, 1, 1, 0, 1, false, 1, false, true)
           || test_convolution_int8(8, 6, 4, 8, 1, 1, 1, 0, 1, false, 2, false, true)
           || test_convolution_int8(8, 6, 4, 8, 1, 1, 1, 0, 1, true, 102, false, true)
           || test_convolution_int8(9, 7, 8, 8, 2, 1, 1, 1, 1, false, 1, true, true);
}

#endif // NCNN_INT8

int main()
{
    SRAND(7767517);

#if NCNN_INT8
    return 0
           || test_convolution_1()
           || test_convolution_1_int8_input();
#endif

    return 0;
}
