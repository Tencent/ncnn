// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

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
        {1, 1, 1}, {4, 13, 0}, {13, 4, 1}, {12, 12, 0}, {8, 12, 1}, {8, 13, 0}, {13, 24, 1}, {12, 16, 0}, {15, 15, 0}, {16, 16, 0}
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
    // scalar outputs retain every generic geometry at both spatial boundaries
    return 0
           || test_convolution(18, 17, 1, 1, k, d, s, p, 1)
           || test_convolution(25, 33, 1, 1, k, d, s, p, 1);
}

static int test_convolution_packed_spatial()
{
    // packed spatial blocks and tails use contiguous, dilated and strided kernels
    static const int geometry[][4] = {
        {2, 1, 1, 1},
        {4, 2, 1, -234},
        {5, 1, 2, 2},
    };
    for (int i = 0; i < 3; i++)
    {
        const int* g = geometry[i];
        if (test_convolution(18, 17, 12, 12, g[0], g[1], g[2], g[3], 0)
                || test_convolution(18, 17, 16, 16, g[0], g[1], g[2], g[3], 0)
                || test_convolution(25, 33, 12, 12, g[0], g[1], g[2], g[3], 0)
                || test_convolution(25, 33, 16, 16, g[0], g[1], g[2], g[3], 0))
            return -1;
    }

    // dilated pack16 inputs produce winograd subimages at the size threshold
    return test_convolution(25, 33, 16, 16, 3, 2, 1, 1, 0);
}

static int test_convolution_direct_geometry()
{
    // dedicated kernels and the generic contiguous kernel retain every packing transition
    static const int geometry[][4] = {
        {1, 1, 1, 0},
        {2, 1, 1, 1},
        {3, 1, 1, 1},
        {3, 1, 2, 1},
        {5, 1, 1, -234},
        {5, 1, 2, 2},
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
        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                const int* g = geometry[i];
                if (test_convolution_direct(9, 7, channels[j][0], channels[j][1], g[0], g[1], g[2], g[3], channels[j][2], 0, fp16 != 0) != 0)
                    return -1;
            }
        }
    }

    // generic stride and dilation retain scalar, pack4 and pack16 output bodies and tails
    static const int generic_geometry[][4] = {
        {2, 1, 2, -233},
        {3, 2, 1, 1},
        {4, 2, 1, -234},
    };
    static const int generic_channels[][3] = {{15, 15, 0}, {12, 12, 0}, {16, 16, 0}};
    for (int fp16 = 0; fp16 < 2; fp16++)
    {
        for (int i = 0; i < 3; i++)
        {
            const int* g = generic_geometry[i];
            for (int j = 0; j < 3; j++)
            {
                const int* ch = generic_channels[j];
                if (test_convolution_direct(9, 7, ch[0], ch[1], g[0], g[1], g[2], g[3], ch[2], 0, fp16 != 0) != 0)
                    return -1;
            }
        }
    }

    // stride and dilation distinguish dedicated pack8 kernels from the generic fallback
    // input channels, output channels, bias, kernel, dilation, stride and padding
    static const int dispatch_boundaries[][7] = {
        {8, 8, 1, 2, 1, 2, -233},
        {8, 8, 1, 3, 2, 1, 1},
        {13, 24, 1, 3, 2, 1, 1},
    };
    for (int fp16 = 0; fp16 < 2; fp16++)
    {
        for (int i = 0; i < 3; i++)
        {
            const int* c = dispatch_boundaries[i];
            if (test_convolution_direct(9, 7, c[0], c[1], c[3], c[4], c[5], c[6], c[2], 0, fp16 != 0) != 0)
                return -1;
        }
    }

    return 0;
}

static int test_convolution_hardswish_boundaries()
{
    // outputs include values below, within and above the linear interval
    ncnn::Mat a(10, 2, 1);
    static const float values[] = {-2.f, 0.f, 4.f};
    for (int i = 0; i < 20; i++)
        a[i] = values[i % 3];

    ncnn::ParamDict pd;
    pd.set(0, 17);
    pd.set(1, 2);
    pd.set(2, 1);
    pd.set(3, 1);
    pd.set(4, 0);
    pd.set(5, 0);
    pd.set(6, 68);
    pd.set(9, 6);
    ncnn::Mat activation_params(2);
    activation_params[0] = 0.2f;
    activation_params[1] = 0.3f;
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(1);
    weights[0] = ncnn::Mat(68);
    weights[0].fill(0.f);
    for (int i = 0; i < 17; i++)
        weights[0][i * 4] = 1.f;

    if (test_layer("Convolution", pd, weights, a) != 0)
        return -1;

    for (int fp16 = 0; fp16 < 2; fp16++)
    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = true;
        opt.use_fp16_packed = fp16 != 0;
        opt.use_fp16_storage = fp16 != 0;
        opt.use_fp16_arithmetic = fp16 != 0;
        opt.use_bf16_packed = false;
        opt.use_bf16_storage = false;
        opt.use_sgemm_convolution = false;
        opt.use_winograd_convolution = false;
        if (test_layer_opt("Convolution", pd, weights, opt, a) != 0)
            return -1;
    }

    return 0;
}

static int test_convolution_bf16_winograd_hardswish_boundaries()
{
    // zero weights and constant bias isolate the activation at every output tile
    ncnn::Mat a(9, 7, 16);
    a.fill(0.f);
    ncnn::ParamDict pd;
    pd.set(0, 31);
    pd.set(1, 3);
    pd.set(2, 1);
    pd.set(3, 1);
    pd.set(4, 1);
    pd.set(5, 1);
    pd.set(6, 31 * 16 * 9);
    pd.set(9, 6);
    ncnn::Mat activation_params(2);
    activation_params[0] = 0.2f;
    activation_params[1] = 0.3f;
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = ncnn::Mat(31 * 16 * 9);
    weights[0].fill(0.f);
    weights[1] = ncnn::Mat(31);
    static const float values[] = {-2.f, 0.f, 4.f};
    for (int variant = 0; variant < 3; variant++)
    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = true;
        opt.use_fp16_packed = false;
        opt.use_fp16_storage = false;
        opt.use_fp16_arithmetic = false;
        opt.use_bf16_storage = true;
        opt.use_winograd23_convolution = variant == 0;
        opt.use_winograd43_convolution = variant == 1;
        opt.use_winograd63_convolution = variant == 2;
        for (int i = 0; i < 3; i++)
        {
            weights[1].fill(values[i]);
            if (test_layer_opt("Convolution", pd, weights, opt, a) != 0)
                return -1;
        }
    }

    return 0;
}

static const int activation_bias_cases[][2] = {
    {0, 0},
    {1, 0},
    {0, 1},
    {1, 2},
    {0, 3},
    {1, 4},
    {0, 5},
    {1, 6},
};

static int test_convolution_activations()
{
    // cover each activation and both bias states for scalar and pack4/8/16 transitions
    static const int channels[][2] = {
        {1, 1},
        {13, 4},
        {4, 13},
        {12, 12},
        {13, 24},
        {8, 13},
        {8, 12},
        {16, 16},
        {15, 15},
        {31, 31},
    };
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < (int)(sizeof(activation_bias_cases) / sizeof(activation_bias_cases[0])); j++)
        {
            const int* activation = activation_bias_cases[j];
            if (test_convolution(8, 6, channels[i][0], channels[i][1], 2, 1, 1, 1, activation[0], activation[1]) != 0)
                return -1;
        }
    }

    // fused activations on direct fp32/fp16 packed and scalar output paths
    static const int direct_channels[][2] = {
        {13, 4},
        {8, 13},
        {12, 12},
        {16, 16},
        {13, 24},
        {8, 8},
        {4, 13},
        {15, 15},
    };
    for (int fp16 = 0; fp16 < 2; fp16++)
    {
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < (int)(sizeof(activation_bias_cases) / sizeof(activation_bias_cases[0])); j++)
            {
                const int* activation = activation_bias_cases[j];
                if (test_convolution_direct(8, 6, direct_channels[i][0], direct_channels[i][1], 2, 1, 1, 1, activation[0], activation[1], fp16 != 0) != 0)
                    return -1;
            }
        }
    }

    // 1x1 cooperative matrix epilogues apply activation to scalar and packed output
    return test_convolution(9, 7, 16, 15, 1, 1, 1, 0, 1, 2)
           || test_convolution(9, 7, 16, 16, 1, 1, 1, 0, 0, 2);
}

static int test_convolution_winograd_activations()
{
    // each winograd output tile includes vector blocks and scalar remainders
    for (int variant = 0; variant < 3; variant++)
    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = true;
        opt.use_fp16_packed = false;
        opt.use_fp16_storage = false;
        opt.use_fp16_arithmetic = false;
        opt.use_bf16_storage = true;
        opt.use_winograd23_convolution = variant == 0;
        opt.use_winograd43_convolution = variant == 1;
        opt.use_winograd63_convolution = variant == 2;
        for (int j = 0; j < (int)(sizeof(activation_bias_cases) / sizeof(activation_bias_cases[0])); j++)
        {
            const int* activation = activation_bias_cases[j];
            if (test_convolution_impl(9, 7, 16, 31, 3, 1, 1, 1, activation[0], activation[1], &opt, 0) != 0)
                return -1;
        }
    }

    return 0;
}

static int test_convolution_post_activation()
{
    // specialized direct kernels apply the activation as a separate layer
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
        for (int i = 0; i < 8; i++)
        {
            for (int stride = 1; stride <= 2; stride++)
            {
                const int* ch = channels[i];
                if (test_convolution_direct(18, 17, ch[0], ch[1], 3, 1, stride, 1, ch[2], 1, fp16 != 0) != 0)
                    return -1;
            }
        }
    }

    return 0
           || test_convolution(18, 17, 16, 16, 3, 1, 1, 1, 1, 1)
           || test_convolution(18, 17, 1, 1, 3, 2, 1, 1, 1, 1);
}

static int test_convolution_direct_spatial()
{
    // large scalar-to-packed widths exercise both output packs and vector tails
    return 0
           || test_convolution_direct(15, 7, 13, 12, 3, 1, 1, 1, 1, 0, false)
           || test_convolution_direct(15, 7, 13, 24, 3, 1, 1, 1, 0, 0, false)
           || test_convolution_direct(18, 17, 13, 24, 3, 1, 1, 1, 1, 0, false)
           || test_convolution_direct(18, 17, 8, 8, 3, 1, 1, 1, 0, 0, false)
           || test_convolution_direct(25, 33, 13, 12, 3, 1, 2, 1, 0, 0, false)
           || test_convolution_direct(25, 33, 13, 24, 3, 1, 2, 1, 1, 1, false)
           || test_convolution_direct(25, 33, 13, 12, 3, 1, 2, 1, 1, 1, true)
           || test_convolution_direct(25, 33, 13, 24, 3, 1, 2, 1, 0, 0, true)
           || test_convolution_direct(18, 17, 8, 13, 3, 1, 1, 1, 0, 0, false)
           || test_convolution_direct(18, 17, 13, 4, 3, 1, 1, 1, 0, 0, false);
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
        {1, 1, 1, 0},
        {2, 1, 1, 1},
        {3, 1, 1, 1},
        {3, 1, 2, 1},
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

static int test_convolution_0()
{
    // specialized im2col templates retain every channel packing direction
    static const int specialized_geometry[][4] = {
        {2, 1, 1, 1},
        {3, 1, 2, 1},
        {5, 1, 1, -234},
        {5, 1, 2, 2},
    };
    for (size_t i = 0; i < sizeof(specialized_geometry) / sizeof(specialized_geometry[0]); i++)
    {
        const int* g = specialized_geometry[i];
        if (test_convolution_packing(9, 7, g[0], g[1], g[2], g[3])
                || test_convolution_spatial(g[0], g[1], g[2], g[3]))
            return -1;
    }

    // generic geometry and large spatial boundaries use scalar and pack16 anchors
    static const int generic_geometry[][4] = {
        {2, 1, 2, -233},
        {3, 2, 1, 1},
        {4, 1, 1, 2},
        {4, 1, 2, -233},
        {4, 2, 1, -234},
    };
    for (size_t i = 0; i < sizeof(generic_geometry) / sizeof(generic_geometry[0]); i++)
    {
        const int* g = generic_geometry[i];
        if (test_convolution(9, 7, 1, 1, g[0], g[1], g[2], g[3], 1)
                || test_convolution(9, 7, 16, 16, g[0], g[1], g[2], g[3], 0)
                || test_convolution_spatial(g[0], g[1], g[2], g[3]))
            return -1;
    }

    // strided and dilated mixed packing retains generic im2col dispatch boundaries
    if (test_convolution(9, 7, 13, 24, 3, 2, 1, 1, 1)
            || test_convolution(9, 7, 13, 24, 4, 2, 1, -234, 1)
            || test_convolution(9, 7, 13, 4, 3, 2, 1, 1, 1)
            || test_convolution(9, 7, 8, 13, 3, 2, 1, 1, 0)
            || test_convolution(9, 7, 13, 24, 2, 1, 2, -233, 1))
        return -1;

    // 1x1 and winograd implementations have distinct spatial tile boundaries
    return 0
           || test_convolution_packed_spatial()
           || test_convolution_packing(9, 7, 1, 1, 1, 0)
           || test_convolution_packing(9, 7, 1, 1, 2, 0)
           || test_convolution_packing(9, 7, 3, 1, 1, 1)
           || test_convolution(18, 17, 1, 1, 1, 1, 1, 0, 1)
           || test_convolution(18, 17, 12, 12, 1, 1, 1, 0, 0)
           || test_convolution(18, 17, 16, 16, 1, 1, 1, 0, 0)
           || test_convolution(18, 17, 13, 24, 1, 1, 1, 0, 1)
           // scalar reduction depth includes the two-channel tail in 1x1 input tiles
           || test_convolution(18, 17, 15, 15, 1, 1, 1, 0, 0)
           // strided 1x1 spatial tiles retain scalar, pack4/16 and mixed channel anchors
           || test_convolution(18, 17, 1, 1, 1, 1, 2, 0, 1)
           || test_convolution(18, 17, 12, 12, 1, 1, 2, 0, 0)
           || test_convolution(18, 17, 16, 16, 1, 1, 2, 0, 0)
           || test_convolution(18, 17, 13, 24, 1, 1, 2, 0, 1)
           // odd scalar channels retain two-element reduction tails in spatial tiles
           || test_convolution(18, 17, 15, 15, 1, 1, 2, 0, 0)
           || test_convolution(18, 17, 1, 1, 3, 1, 1, 1, 1)
           || test_convolution(18, 17, 12, 12, 3, 1, 1, 1, 0)
           || test_convolution(18, 17, 16, 16, 3, 1, 1, 1, 0)
           || test_convolution(18, 17, 13, 24, 3, 1, 1, 1, 1)
           || test_convolution(18, 17, 15, 15, 3, 1, 1, 1, 0)
           // rvv fp16 pack16 output retains the width two-element remainder
           || test_convolution(18, 17, 12, 16, 3, 1, 1, 1, 0)
           || test_convolution_packing(25, 33, 1, 1, 1, 0)
           || test_convolution_packing(25, 33, 1, 1, 2, 0)
           || test_convolution_packing(25, 33, 3, 1, 1, 1);
}

static int test_convolution_specialized_activations()
{
    // specialized large kernels apply relu after the packed convolution
    static const int cases[][6] = {
        {13, 4, 7, 2, 1, 1},
        {13, 13, 7, 1, 3, 0},
        {13, 13, 7, 2, 3, 1},
        {8, 8, 5, 1, 2, 0},
        {8, 8, 5, 2, 2, 1},
        {13, 8, 7, 2, 3, 0},
    };
    for (int i = 0; i < 6; i++)
    {
        const int* c = cases[i];
        if (test_convolution(15, 13, c[0], c[1], c[2], 1, c[3], c[4], c[5], 1) != 0)
            return -1;
    }

    return 0;
}

static int test_convolution_im2col_boundaries()
{
    // sixteen sgemm input positions fit one output row before the row tail
    return test_convolution(25, 33, 13, 24, 2, 1, 1, 1, 0, 0);
}

static int test_convolution_scalar_winograd_activations()
{
    // scalar bf16 winograd runs on architectures without packing support
    for (int variant = 0; variant < 3; variant++)
    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = false;
        opt.use_fp16_packed = false;
        opt.use_fp16_storage = false;
        opt.use_fp16_arithmetic = false;
        opt.use_bf16_storage = true;
        opt.use_winograd23_convolution = variant == 0;
        opt.use_winograd43_convolution = variant == 1;
        opt.use_winograd63_convolution = variant == 2;
        for (int bias = 0; bias < 2; bias++)
        {
            if (test_convolution_impl(9, 7, 16, 31, 3, 1, 1, 1, bias, 2, &opt, 0) != 0)
                return -1;
        }
    }

    return 0;
}

static int test_convolution_bf16_sgemm_activation_boundaries()
{
    // diagonal weights cover numeric regions at every scalar sgemm tail
    const int activations[] = {2, 3, 6};
    const float inputs[] = {-4.f, 0.f, 4.f};
    const float outputs[][3] = {
        {0.4f, 0.f, 4.f},
        {-0.1f, 0.f, 0.3f},
        {0.f, 0.f, 4.f},
    };
    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = true;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = true;
    opt.use_winograd_convolution = false;

    for (int i = 0; i < 3; i++)
    {
        ncnn::ParamDict pd;
        pd.set(0, 31);
        pd.set(1, 1);
        pd.set(6, 31 * 31);
        pd.set(9, activations[i]);
        ncnn::Mat params(2);
        params[0] = activations[i] == 6 ? 0.2f : -0.1f;
        params[1] = 0.3f;
        pd.set(10, params);

        std::vector<ncnn::Mat> weights(1);
        weights[0].create(31 * 31);
        weights[0].fill(0.f);
        for (int q = 0; q < 31; q++)
            weights[0][q * 31 + q] = 1.f;

        for (int j = 0; j < 3; j++)
        {
            ncnn::Mat a(9, 7, 31);
            a.fill(inputs[j]);
            ncnn::Mat reference(9, 7, 31);
            reference.fill(outputs[i][j]);
            ncnn::Mat b;
            int ret = test_layer_naive(ncnn::LayerType::Convolution, pd, weights, a, b, 0);
            if (ret == 0)
                ret = CompareMat(reference, b, 0.000001f);
            if (ret == 0)
                ret = test_layer_opt("Convolution", pd, weights, opt, a, 0.001f);
            if (ret != 0)
            {
                fprintf(stderr, "test_convolution_bf16_sgemm_activation_boundaries failed act=%d input=%f output=%f\n", activations[i], inputs[j], outputs[i][j]);
                return ret;
            }
        }
    }

    return 0;
}

static int test_convolution_im2col_odd_k()
{
    // stride two keeps odd reduction depth on the generic sgemm route
    return test_convolution(51, 9, 13, 24, 1, 1, 2, 0, 0, 0);
}

static int test_convolution_vector_packing_boundaries()
{
    // pack1 input covers eight-column blocks for wider fp16 output packs
    return 0
           || test_convolution(25, 17, 3, 16, 3, 1, 2, 1, 0, 1)
           || test_convolution(25, 17, 3, 16, 7, 1, 2, 1, 0, 1)
           // direct kernel packing retains scalar input and output channel tails
           || test_convolution_direct(8, 6, 3, 3, 2, 1, 1, 1, 0, 0, false);
}

static int test_convolution_fp16_generic_packing()
{
    // direct fp16 packing retains generic kernel, dilation and stride paths
    static const int geometry[][3] = {
        {2, 1, 1},
        {3, 2, 1},
        {3, 1, 3},
        {7, 2, 1},
        {7, 1, 3},
    };
    for (int i = 0; i < 5; i++)
    {
        if (test_convolution_direct(17, 15, 3, 16, geometry[i][0], geometry[i][1], geometry[i][2], 1, 0, 0, true) != 0)
            return -1;
    }

    return 0;
}

static int test_convolution_activation_params_text()
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
        int ret = test_layer_naive(ncnn::LayerType::Convolution, pd, weights, a, b, 0);
        if (ret == 0)
            ret = CompareMat(reference, b, 0.f);
        if (ret != 0)
        {
            fprintf(stderr, "test_convolution_activation_params_text failed params=%s ret=%d\n", params[i], ret);
            return ret;
        }

        const ncnn::Mat original = pd.get(10, ncnn::Mat());
        const int* p = original;
        if (i == 0 ? (p[0] != -1 || p[1] != 2) : (original[0] != -1.f || original[1] != 2.f))
        {
            fprintf(stderr, "test_convolution_activation_params_text modified params=%s\n", params[i]);
            return -1;
        }
    }
#endif
    return 0;
}

#if NCNN_VALIDATION
static int test_convolution_load_param_activation(const ncnn::ParamDict& base, int activation_type, const ncnn::Mat& activation_params, int expected_ret)
{
    ncnn::ParamDict pd = base;
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    return test_layer_param(ncnn::LayerType::Convolution, pd, expected_ret);
}

static int test_convolution_load_param()
{
    ncnn::ParamDict base;
    base.set(0, 8);
    base.set(1, 3);
    base.set(6, 216);
    if (test_layer_param(ncnn::LayerType::Convolution, base, 0) != 0)
        return -1;

    ncnn::Mat params(2);
    params[0] = 0.1f;
    params[1] = 0.5f;

    int ret = 0
              || test_convolution_load_param_activation(base, 0, ncnn::Mat(), 0)
              || test_convolution_load_param_activation(base, 0, ncnn::Mat(0), 0)
              || test_convolution_load_param_activation(base, 0, params, 0)
              || test_convolution_load_param_activation(base, 0, params.range(0, 1), 0)
              || test_convolution_load_param_activation(base, 1, ncnn::Mat(), 0)
              || test_convolution_load_param_activation(base, 1, ncnn::Mat(0), 0)
              || test_convolution_load_param_activation(base, 1, params, 0)
              || test_convolution_load_param_activation(base, 1, params.range(0, 1), 0)
              || test_convolution_load_param_activation(base, 2, ncnn::Mat(), -1)
              || test_convolution_load_param_activation(base, 2, ncnn::Mat(0), -1)
              || test_convolution_load_param_activation(base, 2, params, 0)
              || test_convolution_load_param_activation(base, 2, params.range(0, 1), 0)
              || test_convolution_load_param_activation(base, 3, ncnn::Mat(), -1)
              || test_convolution_load_param_activation(base, 3, ncnn::Mat(0), -1)
              || test_convolution_load_param_activation(base, 3, params, 0)
              || test_convolution_load_param_activation(base, 3, params.range(0, 1), -1)
              || test_convolution_load_param_activation(base, 4, ncnn::Mat(), 0)
              || test_convolution_load_param_activation(base, 4, ncnn::Mat(0), 0)
              || test_convolution_load_param_activation(base, 4, params, 0)
              || test_convolution_load_param_activation(base, 4, params.range(0, 1), 0)
              || test_convolution_load_param_activation(base, 5, ncnn::Mat(), 0)
              || test_convolution_load_param_activation(base, 5, ncnn::Mat(0), 0)
              || test_convolution_load_param_activation(base, 5, params, 0)
              || test_convolution_load_param_activation(base, 5, params.range(0, 1), 0)
              || test_convolution_load_param_activation(base, 6, ncnn::Mat(), -1)
              || test_convolution_load_param_activation(base, 6, ncnn::Mat(0), -1)
              || test_convolution_load_param_activation(base, 6, params, 0)
              || test_convolution_load_param_activation(base, 6, params.range(0, 1), -1)
              || test_convolution_load_param_activation(base, 7, ncnn::Mat(), -1);
    if (ret != 0)
        return ret;

    if (test_layer_param(ncnn::LayerType::Convolution, base, 10, 1, -1)
            || test_layer_param(ncnn::LayerType::Convolution, base, 10, 1.f, -1))
        return -1;

    ncnn::Mat missing_data(0);
    missing_data.w = 1;

    const ncnn::Mat bad[] = {ncnn::Mat(2, (size_t)1u), ncnn::Mat(2, (size_t)2u), ncnn::Mat(2, 2), ncnn::Mat(2, (size_t)16u, 4), missing_data};
    for (int i = 0; i < 5; i++)
    {
        if (test_layer_param(ncnn::LayerType::Convolution, base, 10, bad[i], -1) != 0)
            return -1;
    }

    ret = 0
          || test_layer_param(ncnn::LayerType::Convolution, base, 0, 0, -1)
          || test_layer_param(ncnn::LayerType::Convolution, base, 1, 0, -1)
          || test_layer_param(ncnn::LayerType::Convolution, base, 2, 0, -1)
          || test_layer_param(ncnn::LayerType::Convolution, base, 3, 0, -1)
          || test_layer_param(ncnn::LayerType::Convolution, base, 1, INT_MAX, -1)
          || test_layer_param(ncnn::LayerType::Convolution, base, 6, 217, -1);
    if (ret != 0)
        return ret;

    ncnn::ParamDict dynamic;
    dynamic.set(19, 1); // runtime weight dimensions are unspecified at load time
    if (test_layer_param(ncnn::LayerType::Convolution, dynamic, 0) != 0)
        return -1;

    return test_layer_param(ncnn::LayerType::Convolution, dynamic, 3, 0, -1);
}

static int test_convolution_load_param_text()
{
#if NCNN_STRING
    const char* params[] = {"0=1 1=1 6=1 9=3 -23310=2,-1,2", "0=1 1=1 6=1 9=3 -23310=2,-1.0,2.0"};
    for (int i = 0; i < 2; i++)
    {
        TestParamDict pd;
        if (pd.load_param(params[i]) != 0 || pd.type(10) != 5 + i)
            return -1;

        if (test_layer_param(ncnn::LayerType::Convolution, pd, 0) != 0)
            return -1;
    }
#endif
    return 0;
}
#endif // NCNN_VALIDATION

static int test_convolution_scalar_activation_boundaries()
{
    // odd output channels and spatial size exercise both scalar sgemm tails
    ncnn::Mat a(8, 6, 16);
    a.fill(0.f);
    ncnn::ParamDict pd;
    pd.set(0, 31);
    pd.set(1, 2);
    pd.set(2, 1);
    pd.set(3, 1);
    pd.set(4, 1);
    pd.set(5, 1);
    pd.set(6, 31 * 16 * 4);
    ncnn::Mat params(2);
    params[1] = 0.3f;
    pd.set(10, params);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = ncnn::Mat(31 * 16 * 4);
    weights[0].fill(0.f);
    weights[1] = ncnn::Mat(31);

    const int activations[] = {3, 6};
    const float values[] = {-2.f, 0.f, 4.f};
    for (int i = 0; i < 2; i++)
    {
        pd.set(9, activations[i]);
        params[0] = activations[i] == 6 ? 0.2f : -0.1f;
        for (int j = 0; j < 3; j++)
        {
            weights[1].fill(values[j]);
            if (test_layer("Convolution", pd, weights, a) != 0)
                return -1;
        }
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_convolution_0()
           || test_convolution_direct_geometry()
           || test_convolution_activations()
           || test_convolution_winograd_activations()
           || test_convolution_direct_spatial()
           || test_convolution_post_activation()
#if __aarch64__
           || test_convolution_cpu_tuning()
#endif // __aarch64__
           || test_convolution_specialized_activations()
           || test_convolution_im2col_boundaries()
           || test_convolution_scalar_winograd_activations()
           || test_convolution_bf16_sgemm_activation_boundaries()
           || test_convolution_im2col_odd_k()
           || test_convolution_vector_packing_boundaries()
           || test_convolution_fp16_generic_packing()
           || test_convolution_activation_params_text()
           || test_convolution_hardswish_boundaries()
           || test_convolution_bf16_winograd_hardswish_boundaries()
           || test_convolution_scalar_activation_boundaries()
#if NCNN_VALIDATION
           || test_convolution_load_param()
           || test_convolution_load_param_text()
#endif // NCNN_VALIDATION
           ;
}
