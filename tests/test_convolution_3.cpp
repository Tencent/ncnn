// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_convolution_vec(int w, int outch, int kernel, int dilation, int stride, int pad, int bias)
{
    ncnn::Mat a = RandomMat(w);

    ncnn::ParamDict pd;
    pd.set(0, outch);    // num_output
    pd.set(1, kernel);   // kernel_w
    pd.set(2, dilation); // dilation_w
    pd.set(3, stride);   // stride_w
    pd.set(4, pad);      // pad_w
    pd.set(5, bias);     // bias_term
    pd.set(6, outch * w * kernel * kernel);

    int activation_type = RAND() % 7; // 0 1 2 3 4 5 6
    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);                                               // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outch * w * kernel * kernel);
    if (bias)
        weights[1] = RandomMat(outch);

    int ret = test_layer("Convolution", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolution_vec failed w=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d actparams=[%f,%f]\n", w, outch, kernel, dilation, stride, pad, bias, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_convolution_2()
{
    return 0
           || test_convolution_vec(1, 1, 1, 1, 1, 0, 1)
           || test_convolution_vec(11, 12, 1, 1, 1, 0, 0)
           || test_convolution_vec(20, 15, 1, 1, 1, 0, 1)
           || test_convolution_vec(12, 20, 1, 1, 1, 0, 0)
           || test_convolution_vec(3, 24, 1, 1, 1, 0, 1)
           || test_convolution_vec(24, 5, 1, 1, 1, 0, 0)
           || test_convolution_vec(32, 24, 1, 1, 1, 0, 1)
           || test_convolution_vec(12, 32, 1, 1, 1, 0, 0)
           || test_convolution_vec(64, 20, 1, 1, 1, 0, 1)
           || test_convolution_vec(64, 128, 1, 1, 1, 0, 0);
}

static int test_convolution_dynamic(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias)
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
    pd.set(19, 1); // dynamic weight

    int activation_type = RAND() % 7; // 0 1 2 3 4 5 6
    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);                                               // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> as(bias ? 3 : 2);
    as[0] = a;
    as[1] = RandomMat(kernel, kernel, c, outch);
    if (bias)
        as[2] = RandomMat(outch);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Convolution", pd, weights, as);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolution_dynamic failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_convolution_3()
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
                  || test_convolution_dynamic(11, 10, 1, 1, k, d, s, p, 1)
                  || test_convolution_dynamic(11, 10, 4, 13, k, d, s, p, 0)
                  || test_convolution_dynamic(11, 10, 13, 4, k, d, s, p, 1)
                  || test_convolution_dynamic(11, 10, 12, 12, k, d, s, p, 0)
                  || test_convolution_dynamic(11, 10, 8, 12, k, d, s, p, 1)
                  || test_convolution_dynamic(11, 10, 8, 13, k, d, s, p, 0)
                  || test_convolution_dynamic(11, 10, 13, 8, k, d, s, p, 1)
                  || test_convolution_dynamic(11, 10, 12, 16, k, d, s, p, 0)
                  || test_convolution_dynamic(11, 10, 15, 15, k, d, s, p, 0)
                  || test_convolution_dynamic(11, 10, 16, 16, k, d, s, p, 0);

        if (ret != 0)
            return -1;
    }

    return 0;
}

#if NCNN_INT8
static int test_convolution_int8(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias, bool requant = false)
{
    ncnn::Mat a = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, kernel);
    pd.set(2, dilation);
    pd.set(3, stride);
    pd.set(4, pad);
    pd.set(5, bias);
    pd.set(6, outch * c * kernel * kernel);
    pd.set(8, requant ? 101 : 1); // int8_scale_term

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
    ncnn::Mat top_scales = requant ? scales_mat(a, 1, w * h * c, a.cstep) : ncnn::Mat();

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

    int flag = TEST_LAYER_DISABLE_GPU_TESTING;
    int ret = test_layer("Convolution", pd, weights, a, requant ? 1.0f : 0.001f, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolution_int8 failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d requant=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, requant, activation_type, activation_params[0], activation_params[1]);
        return ret;
    }

    if (kernel == 3 && dilation == 1 && stride == 1)
    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = true;
        opt.use_fp16_packed = false;
        opt.use_fp16_storage = false;
        opt.use_fp16_arithmetic = false;
        opt.use_bf16_storage = false;
        opt.use_sgemm_convolution = false;
        opt.use_winograd_convolution = true;
        opt.use_winograd23_convolution = true;
        opt.use_winograd43_convolution = false;

        ret = test_layer_opt("Convolution", pd, weights, opt, a, requant ? 1.0f : 0.001f, flag);
        if (ret != 0)
        {
            fprintf(stderr, "test_convolution_int8 failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d requant=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, requant, activation_type, activation_params[0], activation_params[1]);
            return ret;
        }
    }

    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = false;
        opt.use_fp16_packed = false;
        opt.use_fp16_storage = false;
        opt.use_fp16_arithmetic = false;
        opt.use_bf16_storage = false;
        opt.use_sgemm_convolution = false;
        opt.use_winograd_convolution = false;

        ret = test_layer_opt("Convolution", pd, weights, opt, a, requant ? 1.0f : 0.001f, flag);
        if (ret != 0)
        {
            fprintf(stderr, "test_convolution_int8 failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d requant=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, requant, activation_type, activation_params[0], activation_params[1]);
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
        opt.use_bf16_storage = false;
        opt.use_sgemm_convolution = false;
        opt.use_winograd_convolution = false;

        ret = test_layer_opt("Convolution", pd, weights, opt, a, requant ? 1.0f : 0.001f, flag);
        if (ret != 0)
        {
            fprintf(stderr, "test_convolution_int8 failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d requant=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, requant, activation_type, activation_params[0], activation_params[1]);
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
        opt.use_bf16_storage = true;
        opt.use_sgemm_convolution = false;
        opt.use_winograd_convolution = false;

        ret = test_layer_opt("Convolution", pd, weights, opt, a, requant ? 1.0f : 0.001f, flag);
        if (ret != 0)
        {
            fprintf(stderr, "test_convolution_int8 failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d requant=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, requant, activation_type, activation_params[0], activation_params[1]);
            return ret;
        }
    }

    return ret;
}

static int test_convolution_1()
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

    for (int i = 0; i < 16; i++)
    {
        const int k = kdsp[i][0];
        const int d = kdsp[i][1];
        const int s = kdsp[i][2];
        const int p = kdsp[i][3];

        int ret = 0
                  || test_convolution_int8(9, 7, 1, 1, k, d, s, p, 1)
                  || test_convolution_int8(9, 7, 2, 2, k, d, s, p, 1)
                  || test_convolution_int8(9, 7, 3, 3, k, d, s, p, 1)
                  || test_convolution_int8(9, 7, 4, 4, k, d, s, p, 1)
                  || test_convolution_int8(9, 7, 7, 7, k, d, s, p, 1)
                  || test_convolution_int8(9, 7, 8, 8, k, d, s, p, 1)
                  || test_convolution_int8(9, 7, 15, 15, k, d, s, p, 1)
                  || test_convolution_int8(9, 7, 16, 15, k, d, s, p, 1)
                  || test_convolution_int8(9, 7, 15, 16, k, d, s, p, 1)
                  || test_convolution_int8(9, 7, 16, 16, k, d, s, p, 1);

        if (ret != 0)
            return -1;
    }
    for (int i = 0; i < 16; i++)
    {
        const int k = kdsp[i][0];
        const int d = kdsp[i][1];
        const int s = kdsp[i][2];
        const int p = kdsp[i][3];

        int ret = 0
                  || test_convolution_int8(9, 7, 1, 1, k, d, s, p, 1, true)
                  || test_convolution_int8(9, 7, 1, 1, k, d, s, p, 1, true)
                  || test_convolution_int8(9, 7, 2, 2, k, d, s, p, 1, true)
                  || test_convolution_int8(9, 7, 3, 3, k, d, s, p, 1, true)
                  || test_convolution_int8(9, 7, 4, 4, k, d, s, p, 1, true)
                  || test_convolution_int8(9, 7, 7, 7, k, d, s, p, 1, true)
                  || test_convolution_int8(9, 7, 8, 8, k, d, s, p, 1, true)
                  || test_convolution_int8(9, 7, 15, 15, k, d, s, p, 1, true)
                  || test_convolution_int8(9, 7, 16, 15, k, d, s, p, 1, true)
                  || test_convolution_int8(9, 7, 15, 16, k, d, s, p, 1, true)
                  || test_convolution_int8(9, 7, 16, 16, k, d, s, p, 1, true);

        if (ret != 0)
            return -1;
    }

    return 0
           || test_convolution_int8(11, 11, 8, 16, 3, 1, 1, 1, 1)
           || test_convolution_int8(13, 16, 16, 24, 3, 1, 1, 1, 1)
           || test_convolution_int8(8, 8, 16, 24, 3, 1, 1, 1, 0)
           || test_convolution_int8(4, 8, 16, 24, 3, 1, 1, 1, 1)
           || test_convolution_int8(4, 20, 16, 24, 3, 1, 1, 1, 0)
           || test_convolution_int8(6, 7, 64, 64, 3, 1, 2, 0, 1)
           || test_convolution_int8(25, 33, 16, 15, 3, 1, 1, 1, 0)
           || test_convolution_int8(25, 33, 31, 31, 3, 1, 1, 1, 0)
           || test_convolution_int8(7, 7, 15, 12, 3, 1, 1, 1, 0)
           || test_convolution_int8(5, 6, 31, 9, 5, 1, 1, 0, 1)
           || test_convolution_int8(5, 7, 32, 8, 5, 1, 2, 0, 1)
           || test_convolution_int8(16, 10, 31, 32, 2, 1, 3, 0, 0)
           || test_convolution_int8(5, 10, 5, 32, 3, 2, 1, 0, 1)
           || test_convolution_int8(3, 9, 16, 13, 2, 2, 1, 0, 0)
           || test_convolution_int8(33, 5, 15, 5, 2, 1, 3, 0, 1)
           || test_convolution_int8(23, 11, 33, 28, 5, 1, 1, 0, 1)
           || test_convolution_int8(3, 63, 2, 28, 2, 1, 2, 0, 0);
}

static int test_convolution_1_2()
{
    return 0
           || test_convolution_int8(19, 17, 1, 1, 3, 2, 2, 0, 1)
           || test_convolution_int8(19, 17, 2, 1, 3, 2, 2, 0, 0)
           || test_convolution_int8(19, 17, 7, 1, 3, 2, 2, 0, 1)
           || test_convolution_int8(19, 17, 8, 1, 3, 2, 2, 0, 0)
           || test_convolution_int8(19, 17, 15, 1, 3, 2, 2, 0, 1)
           || test_convolution_int8(19, 17, 16, 1, 3, 2, 2, 0, 0)
           || test_convolution_int8(19, 17, 31, 1, 3, 2, 2, 0, 1)
           || test_convolution_int8(19, 17, 32, 1, 3, 2, 2, 0, 0)

           || test_convolution_int8(19, 17, 1, 2, 5, 2, 2, 0, 1)
           || test_convolution_int8(19, 17, 2, 2, 5, 2, 3, 0, 0)
           || test_convolution_int8(19, 17, 7, 2, 5, 2, 2, 0, 1)
           || test_convolution_int8(19, 17, 8, 2, 5, 2, 3, 0, 0)
           || test_convolution_int8(19, 17, 15, 2, 5, 2, 2, 0, 1)
           || test_convolution_int8(19, 17, 16, 2, 5, 2, 3, 0, 0)
           || test_convolution_int8(19, 17, 31, 2, 5, 2, 2, 0, 1)
           || test_convolution_int8(19, 17, 32, 2, 5, 2, 3, 0, 0)

           || test_convolution_int8(19, 17, 1, 7, 5, 2, 3, 0, 1)
           || test_convolution_int8(19, 17, 2, 7, 5, 2, 2, 0, 0)
           || test_convolution_int8(19, 17, 7, 7, 5, 2, 3, 0, 1)
           || test_convolution_int8(19, 17, 8, 7, 5, 2, 2, 0, 0)
           || test_convolution_int8(19, 17, 15, 7, 5, 2, 3, 0, 1)
           || test_convolution_int8(19, 17, 16, 7, 5, 2, 2, 0, 0)
           || test_convolution_int8(19, 17, 31, 7, 5, 2, 3, 0, 1)
           || test_convolution_int8(19, 17, 32, 7, 5, 2, 2, 0, 0)

           || test_convolution_int8(19, 17, 1, 8, 5, 2, 2, 0, 1)
           || test_convolution_int8(19, 17, 2, 8, 5, 2, 3, 0, 0)
           || test_convolution_int8(19, 17, 7, 8, 5, 2, 2, 0, 1)
           || test_convolution_int8(19, 17, 8, 8, 5, 2, 3, 0, 0)
           || test_convolution_int8(19, 17, 15, 8, 5, 2, 2, 0, 1)
           || test_convolution_int8(19, 17, 16, 8, 5, 2, 3, 0, 0)
           || test_convolution_int8(19, 17, 31, 8, 5, 2, 2, 0, 1)
           || test_convolution_int8(19, 17, 32, 8, 5, 2, 3, 0, 0)

           || test_convolution_int8(19, 17, 1, 15, 5, 2, 3, 0, 1)
           || test_convolution_int8(19, 17, 2, 15, 5, 2, 2, 0, 0)
           || test_convolution_int8(19, 17, 7, 15, 5, 2, 3, 0, 1)
           || test_convolution_int8(19, 17, 8, 15, 5, 2, 2, 0, 0)
           || test_convolution_int8(19, 17, 15, 15, 5, 2, 3, 0, 1)
           || test_convolution_int8(19, 17, 16, 15, 5, 2, 2, 0, 0)
           || test_convolution_int8(19, 17, 31, 15, 5, 2, 3, 0, 1)
           || test_convolution_int8(19, 17, 32, 15, 5, 2, 2, 0, 0)

           || test_convolution_int8(19, 17, 1, 16, 5, 2, 3, 0, 1)
           || test_convolution_int8(19, 17, 2, 16, 5, 2, 3, 0, 0)
           || test_convolution_int8(19, 17, 7, 16, 5, 2, 2, 0, 1)
           || test_convolution_int8(19, 17, 8, 16, 5, 2, 2, 0, 0)
           || test_convolution_int8(19, 17, 15, 16, 5, 2, 2, 0, 1)
           || test_convolution_int8(19, 17, 16, 16, 5, 2, 2, 0, 0)
           || test_convolution_int8(19, 17, 31, 16, 5, 2, 3, 0, 1)
           || test_convolution_int8(19, 17, 32, 16, 5, 2, 3, 0, 0)

           || test_convolution_int8(19, 17, 1, 31, 5, 2, 2, 0, 1)
           || test_convolution_int8(19, 17, 2, 31, 5, 2, 3, 0, 0)
           || test_convolution_int8(19, 17, 7, 31, 5, 2, 3, 0, 1)
           || test_convolution_int8(19, 17, 8, 31, 5, 2, 2, 0, 0)
           || test_convolution_int8(19, 17, 15, 31, 5, 2, 2, 0, 1)
           || test_convolution_int8(19, 17, 16, 31, 5, 2, 3, 0, 0)
           || test_convolution_int8(19, 17, 31, 31, 5, 2, 3, 0, 1)
           || test_convolution_int8(19, 17, 32, 31, 5, 2, 2, 0, 0)

           || test_convolution_int8(19, 17, 1, 32, 5, 2, 2, 0, 1)
           || test_convolution_int8(19, 17, 2, 32, 5, 2, 2, 0, 0)
           || test_convolution_int8(19, 17, 7, 32, 5, 2, 3, 0, 1)
           || test_convolution_int8(19, 17, 8, 32, 5, 2, 3, 0, 0)
           || test_convolution_int8(19, 17, 15, 32, 5, 2, 3, 0, 1)
           || test_convolution_int8(19, 17, 16, 32, 5, 2, 3, 0, 0)
           || test_convolution_int8(19, 17, 31, 32, 5, 2, 2, 0, 1)
           || test_convolution_int8(19, 17, 32, 32, 5, 2, 2, 0, 0);
}
#endif // NCNN_INT8

int main()
{
    SRAND(7767517);

#if NCNN_INT8
    return 0
           || test_convolution_1()
           || test_convolution_1_2()
           || test_convolution_2()
           || test_convolution_3();
#else
    return 0
           || test_convolution_2()
           || test_convolution_3();
#endif
}
