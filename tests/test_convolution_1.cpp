// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_convolution_impl(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias, int activation_type, const ncnn::Option* opt, int flag)
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

    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? 0.2f : -0.1f; // alpha
    activation_params[1] = 0.3f;                                  // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outch * c * kernel * kernel);
    if (bias)
        weights[1] = RandomMat(outch);

    float epsilon = 0.001;

    int ret = opt ? test_layer_opt("Convolution", pd, weights, *opt, a, epsilon, flag) : test_layer("Convolution", pd, weights, a, epsilon, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolution failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, activation_type, activation_params[0], activation_params[1]);
        return ret;
    }

    return ret;
}

static int test_convolution(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias, int activation_type = 0)
{
    return test_convolution_impl(w, h, c, outch, kernel, dilation, stride, pad, bias, activation_type, 0, 0);
}

static int test_convolution_direct(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias, int activation_type, bool fp16)
{
    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = true;
    opt.use_fp16_packed = fp16;
    opt.use_fp16_storage = fp16;
    opt.use_fp16_arithmetic = fp16;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;
    opt.use_sgemm_convolution = false;
    opt.use_winograd_convolution = false;

    return test_convolution_impl(w, h, c, outch, kernel, dilation, stride, pad, bias, activation_type, &opt, 0);
}

static int test_convolution_packing(int w, int h, int k, int d, int s, int p)
{
    // scalar, pack4, pack8 and pack16 input/output transitions
    static const int channels[][3] = {
        {1, 1, 1}, {4, 13, 0}, {13, 4, 1}, {12, 12, 0}, {8, 12, 1}, {8, 13, 0}, {13, 8, 1}, {12, 16, 0}, {15, 15, 0}, {16, 16, 0}
    };
    for (int i = 0; i < 10; i++)
    {
        if (test_convolution(w, h, channels[i][0], channels[i][1], k, d, s, p, channels[i][2], 0) != 0)
            return -1;
    }

    return 0;
}

static int test_convolution_spatial(int k, int d, int s, int p)
{
    // cover odd and even spatial boundaries independently of channel transitions
    return 0
           || test_convolution(18, 17, 1, 1, k, d, s, p, 1)
           || test_convolution(18, 17, 12, 12, k, d, s, p, 0)
           || test_convolution(18, 17, 16, 16, k, d, s, p, 0)
           || test_convolution(25, 33, 1, 1, k, d, s, p, 1)
           || test_convolution(25, 33, 12, 12, k, d, s, p, 0)
           || test_convolution(25, 33, 16, 16, k, d, s, p, 0);
}

static int test_convolution_direct_geometry()
{
    // direct implementations independently cover geometry and packing transitions
    static const int geometry[][4] = {
        {5, 2, 2, 2},
        {7, 1, 1, 3},
        {7, 1, 2, 3},
        {7, 2, 1, -233},
    };
    static const int channels[][3] = {
        {13, 4, 1},
        {8, 13, 0},
        {12, 12, 0},
        {16, 16, 0},
        {13, 24, 1},
        {8, 8, 1},
        {4, 13, 0},
        {15, 15, 0},
    };
    for (int fp16 = 0; fp16 < 2; fp16++)
    {
        for (int i = 0; i < (int)(sizeof(geometry) / sizeof(geometry[0])); i++)
        {
            for (int j = 0; j < 8; j++)
            {
                const int* g = geometry[i];
                if (test_convolution_direct(9, 7, channels[j][0], channels[j][1], g[0], g[1], g[2], g[3], channels[j][2], 0, fp16 != 0) != 0)
                    return -1;
            }
        }
    }

    return 0;
}

#if __aarch64__
static int test_convolution_cpu_tuning()
{
    // a53/a55 tuning selects cpu kernels independently of vulkan algorithms
    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_a53_a55_optimized_kernel = true;
    static const int channels[][3] = {
        {1, 1, 1},
        {4, 13, 0},
        {13, 4, 1},
        {16, 16, 0},
        {13, 24, 1},
        {8, 12, 0},
        {15, 15, 1},
    };
    static const int geometry[][4] = {
        {5, 2, 2, 2},
        {7, 1, 1, 3},
        {7, 1, 2, 3},
        {7, 2, 1, -233},
    };
    static const int shapes[][2] = {{9, 7}, {18, 17}, {25, 33}, {18, 17}};
    for (int i = 0; i < 7; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            const int* ch = channels[i];
            const int* g = geometry[j];
            const int* shape = shapes[j];
            if (test_convolution_impl(shape[0], shape[1], ch[0], ch[1], g[0], g[1], g[2], g[3], ch[2], 0, &opt, TEST_LAYER_DISABLE_GPU_TESTING) != 0)
                return -1;
        }
    }

    return 0;
}
#endif // __aarch64__

static int test_convolution_generic_geometry(int k, int d, int s, int p)
{
    // generic geometry retains scalar and pack4/8/16 channels with large spatial tails
    return test_convolution(9, 7, 1, 1, k, d, s, p, 1)
           || test_convolution(9, 7, 12, 12, k, d, s, p, 0)
           || test_convolution(9, 7, 16, 16, k, d, s, p, 0)
           // generic packing retains scalar output transpose and pack16 kernel reduction tails
           || test_convolution(9, 7, 4, 13, k, d, s, p, 0)
           || test_convolution(9, 7, 12, 16, k, d, s, p, 0)
           // generic sgemm keeps scalar input/output and both mixed packing directions
           || test_convolution(9, 7, 15, 15, k, d, s, p, 0)
           || test_convolution(9, 7, 13, 8, k, d, s, p, 1)
           || test_convolution(9, 7, 8, 13, k, d, s, p, 0)
           || test_convolution(18, 17, 1, 1, k, d, s, p, 1)
           // packed outputs retain large n tiles and their spatial remainders
           || test_convolution(18, 17, 12, 12, k, d, s, p, 0)
           || test_convolution(18, 17, 16, 16, k, d, s, p, 0)
           || test_convolution(25, 33, 1, 1, k, d, s, p, 1)
           || test_convolution(25, 33, 12, 12, k, d, s, p, 0)
           || test_convolution(25, 33, 16, 16, k, d, s, p, 0);
}

static int test_convolution_0()
{
    // both specialized 7x7 strides retain all packing and spatial directions
    return test_convolution_generic_geometry(5, 2, 2, 2)
           || test_convolution_packing(9, 7, 7, 1, 1, 3)
           || test_convolution_spatial(7, 1, 1, 3)
           || test_convolution_packing(9, 7, 7, 1, 2, 3)
           || test_convolution_spatial(7, 1, 2, 3)
           || test_convolution_generic_geometry(7, 2, 1, -233)
           // the arm 7x7s2 scalar-to-packed implementation has its own spatial boundary
           || test_convolution(18, 17, 13, 8, 7, 1, 2, 3, 1);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_convolution_0()
#if __aarch64__
           || test_convolution_cpu_tuning()
#endif // __aarch64__
           || test_convolution_direct_geometry();
}
