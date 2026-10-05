// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

static int test_pooling(int w, int h, int c, int pooling_type, int kernel, int stride, int pad, int global_pooling, int pad_mode, int avgpool_count_include_pad, int adaptive_pooling, int out_w, int out_h = 0)
{
    ncnn::Mat a = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, pooling_type);                // pooling_type
    pd.set(1, kernel);                      // kernel_w
    pd.set(2, stride);                      // stride_w
    pd.set(3, pad);                         // pad_w
    pd.set(4, global_pooling);              // global_pooling
    pd.set(5, pad_mode);                    // pad_mode
    pd.set(6, avgpool_count_include_pad);   // avgpool_count_include_pad
    pd.set(7, adaptive_pooling);            // adaptive_pooling
    pd.set(8, out_w);                       // out_w
    pd.set(18, out_h == 0 ? out_w : out_h); // out_h

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Pooling", pd, weights, a, 0.001);
    if (ret != 0)
    {
        fprintf(stderr, "test_pooling failed w=%d h=%d c=%d pooling_type=%d kernel=%d stride=%d pad=%d global_pooling=%d pad_mode=%d avgpool_count_include_pad=%d adaptive_pooling=%d out_w=%d out_h=%d\n", w, h, c, pooling_type, kernel, stride, pad, global_pooling, pad_mode, avgpool_count_include_pad, adaptive_pooling, out_w, out_h == 0 ? out_w : out_h);
    }

    return ret;
}

static int test_pooling_0()
{
    // specialized max kernels retain scalar channel tails and each packing class
    static const int channels_pad[][2] = {
        {1, 0},
        {2, 1},
        {3, 2},
        {4, 3},
        {7, 0},
        {8, 1},
        {15, 2},
        {16, 3},
    };
    static const int specialized[][3] = {{2, 2, 0}, {3, 2, 1}};
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            const int* k = specialized[i];
            const int* ch = channels_pad[j];
            if (test_pooling(9, 7, ch[0], 0, k[0], k[1], k[2], 0, ch[1], 0, 0, 0) != 0)
                return -1;
        }
    }

    // packed stride1 contrasts the specialized stride2 dispatch guards
    static const int generic_packing[][2] = {{4, 3}, {8, 1}, {16, 3}};
    for (int kernel = 2; kernel <= 3; kernel++)
    {
        for (int i = 0; i < 3; i++)
        {
            const int* ch = generic_packing[i];
            if (test_pooling(9, 7, ch[0], 0, kernel, 1, 0, 0, ch[1], 0, 0, 0) != 0)
                return -1;
        }
    }

    // generic kernel geometry is independent of padding and channel packing
    static const int geometry[][3] = {
        {2, 1, 0},
        {3, 1, 0},
        {4, 1, 0},
        {5, 1, 0},
        {5, 2, 2},
        {7, 1, 0},
        {7, 2, 1},
        {7, 3, 2},
    };
    for (int i = 0; i < 8; i++)
    {
        const int* k = geometry[i];
        if (test_pooling(9, 7, 1, 0, k[0], k[1], k[2], 0, 0, 0, 0, 0) != 0)
            return -1;
    }

    // 4x4 stride2 gives independent right and bottom padding tails
    static const int packing[] = {1, 4, 8, 16};
    for (int i = 0; i < 4; i++)
    {
        for (int mode = 0; mode < 4; mode++)
        {
            if (test_pooling(9, 7, packing[i], 0, 4, 2, 1, 0, mode, 0, 0, 0) != 0)
                return -1;
        }
    }

    return 0;
}

static int test_pooling_1()
{
    // averaging geometry does not select the specialized max kernels
    static const int geometry[][3] = {
        {2, 1, 0},
        {2, 2, 0},
        {3, 1, 0},
        {3, 2, 1},
        {4, 1, 0},
        {5, 1, 0},
        {5, 2, 2},
        {7, 1, 0},
        {7, 2, 1},
        {7, 3, 2},
    };
    for (int i = 0; i < 10; i++)
    {
        const int* k = geometry[i];
        if (test_pooling(9, 7, 1, 1, k[0], k[1], k[2], 0, 0, 0, 0, 0) != 0)
            return -1;
    }

    // padding divisors cover each pack and the full-padding spatial tails
    // c64 also retains multiple channel packs on avx512
    static const int packing[] = {1, 4, 8, 64};
    for (int i = 0; i < 4; i++)
    {
        for (int mode = 0; mode < 4; mode++)
        {
            for (int include_pad = 0; include_pad < 2; include_pad++)
            {
                if (test_pooling(9, 7, packing[i], 1, 4, 2, 1, 0, mode, include_pad, 0, 0) != 0)
                    return -1;
            }
        }
    }

    static const int tails[][3] = {{2, 1, 0}, {3, 0, 1}, {7, 0, 0}, {15, 0, 0}};
    for (int i = 0; i < 4; i++)
    {
        const int* ch = tails[i];
        if (test_pooling(9, 7, ch[0], 1, 3, 2, 1, 0, ch[1], ch[2], 0, 0) != 0)
            return -1;
    }

    return 0;
}

static int test_pooling_2()
{
    return 0
           || test_pooling(2, 5, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling(5, 2, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling(3, 6, 3, 0, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling(6, 3, 3, 1, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling(4, 4, 4, 0, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling(6, 4, 4, 1, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling(8, 7, 8, 0, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling(7, 8, 8, 1, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling(11, 13, 16, 0, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling(13, 11, 16, 1, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling(110, 103, 106, 0, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling(130, 101, 106, 1, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling(80, 93, 128, 0, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling(80, 91, 128, 1, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling(48, 48, 4, 0, 2, 2, 0, 0, 0, 0, 0, 0)
           || test_pooling(48, 48, 15, 0, 2, 2, 1, 0, 0, 0, 0, 0);
}

// adaptive avg pool
static int test_pooling_adaptive(int pooling_type)
{
    // outputs below, equal to and above either input axis for each packing class
    static const int shapes[][3] = {
        {2, 5, 1},
        {5, 2, 1},
        {3, 6, 3},
        {6, 3, 3},
        {4, 4, 4},
        {6, 4, 4},
        {8, 7, 8},
        {7, 8, 8},
    };
    static const int outputs[][4] = {
        {1, 2, 5, 6},
        {1, 2, 5, 6},
        {1, 3, 6, 7},
        {1, 3, 6, 7},
        {1, 3, 4, 5},
        {1, 4, 6, 7},
        {1, 7, 8, 9},
        {1, 7, 8, 9},
    };
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            if (test_pooling(shapes[i][0], shapes[i][1], shapes[i][2], pooling_type, 1, 1, 0, 0, 0, 0, 1, outputs[i][j]) != 0)
                return -1;
        }
    }

    // independent output width and height with pack16 and reduction tails
    static const int outputs16[][2] = {{-233, 1}, {2, -233}, {-233, -233}, {13, 11}};
    for (int i = 0; i < 4; i++)
    {
        if (test_pooling(11, 13, 16, pooling_type, 1, 1, 0, 0, 0, 0, 1, outputs16[i][0], outputs16[i][1]) != 0)
            return -1;
    }

    return 0
           || test_pooling(11, 13, 16, pooling_type, 1, 1, 0, 0, 0, 1, 0, 1)
           || test_pooling(13, 11, 16, pooling_type, 1, 1, 0, 0, 0, 1, 0, 2);
}

static int test_pooling_3()
{
    return test_pooling_adaptive(1);
}

static int test_pooling_4()
{
    return test_pooling_adaptive(0);
}

#if NCNN_VALIDATION
static int test_pooling_load_param()
{
    ncnn::ParamDict base;
    base.set(1, 3);
    if (test_layer_param(ncnn::LayerType::Pooling, base, 0) != 0)
        return -1;

    // global pooling does not use the local stride
    ncnn::ParamDict global = base;
    global.set(4, 1);

    return 0
           || test_layer_param(ncnn::LayerType::Pooling, base, 2, 0, -1)
           || test_layer_param(ncnn::LayerType::Pooling, global, 2, 0, 0);
}

static int test_pooling_load_param_type()
{
    ncnn::ParamDict base;
    base.set(1, 3);
    if (test_layer_param(ncnn::LayerType::Pooling, base, 0) != 0)
        return -1;

    for (int i = 0; i <= 1; i++)
    {
        if (test_layer_param(ncnn::LayerType::Pooling, base, 0, i, 0) != 0)
            return -1;
    }

    const int invalid[] = {-1, 2, INT_MIN, INT_MAX};
    for (int i = 0; i < 4; i++)
    {
        if (test_layer_param(ncnn::LayerType::Pooling, base, 0, invalid[i], -1) != 0)
            return -1;
    }

    for (int i = 0; i <= 3; i++)
    {
        if (test_layer_param(ncnn::LayerType::Pooling, base, 5, i, 0) != 0)
            return -1;
    }

    // global and adaptive pooling do not use pad_mode
    ncnn::ParamDict global = base;
    global.set(4, 1);
    ncnn::ParamDict adaptive = base;
    adaptive.set(4, 0);
    adaptive.set(8, 1);
    adaptive.set(7, 1);

    const int invalid_pad[] = {-1, 4, INT_MIN, INT_MAX};
    for (int i = 0; i < 4; i++)
    {
        int ret = 0
                  || test_layer_param(ncnn::LayerType::Pooling, base, 5, invalid_pad[i], -1)
                  || test_layer_param(ncnn::LayerType::Pooling, global, 5, invalid_pad[i], 0)
                  || test_layer_param(ncnn::LayerType::Pooling, adaptive, 5, invalid_pad[i], 0);
        if (ret != 0)
            return ret;
    }

    return 0;
}

static int test_pooling_load_param_adaptive()
{
    ncnn::ParamDict base;
    base.set(1, 3);
    base.set(7, 1);
    if (test_layer_param(ncnn::LayerType::Pooling, base, -1) != 0)
        return -1;

    base.set(8, 2);
    if (test_layer_param(ncnn::LayerType::Pooling, base, 0) != 0)
        return -1;

    base.set(18, 2);

    // global and ordinary pooling ignore adaptive output sizes
    ncnn::ParamDict global = base;
    global.set(4, 1);
    ncnn::ParamDict ordinary = base;
    ordinary.set(4, 0);
    ordinary.set(7, 0);

    const int ids[] = {8, 18};
    const int invalid[] = {0, -1, -234, INT_MIN};
    for (int i = 0; i < 2; i++)
    {
        if (test_layer_param(ncnn::LayerType::Pooling, base, ids[i], 1, 0) != 0)
            return -1;

        if (test_layer_param(ncnn::LayerType::Pooling, base, ids[i], -233, 0) != 0)
            return -1;

        for (int j = 0; j < 4; j++)
        {
            int ret = 0
                      || test_layer_param(ncnn::LayerType::Pooling, base, ids[i], invalid[j], -1)
                      || test_layer_param(ncnn::LayerType::Pooling, global, ids[i], invalid[j], 0)
                      || test_layer_param(ncnn::LayerType::Pooling, ordinary, ids[i], invalid[j], 0);
            if (ret != 0)
                return ret;
        }
    }

    return 0;
}
#endif // NCNN_VALIDATION

static int test_pooling_rectangular()
{
    // channels, type, kernel width/height, stride width/height and include-pad
    static const int cases[][7] = {
        {3, 0, 3, 2, 2, 2, 0},
        {3, 1, 2, 2, 2, 1, 0},
        {4, 0, 2, 3, 2, 2, 0},
        {8, 0, 3, 2, 2, 2, 0},
        {16, 0, 2, 3, 2, 1, 0},
        {8, 1, 3, 2, 1, 2, 0},
    };
    static const float values[] = {-4.f, -2.f, -1.f, 0.f, 1.f, 2.f, 4.f};
    for (int i = 0; i < 6; i++)
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
        pd.set(2, c[4]);
        pd.set(12, c[5]);
        pd.set(3, 1);
        pd.set(14, 0);
        pd.set(13, 0);
        pd.set(15, 1);
        pd.set(5, 0);
        pd.set(6, c[6]);
        std::vector<ncnn::Mat> weights;
        if (test_layer("Pooling", pd, weights, a, 0.001) != 0)
        {
            fprintf(stderr, "test_pooling_rectangular failed c=%d type=%d kernel=%dx%d stride=%dx%d include_pad=%d\n", c[0], c[1], c[2], c[3], c[4], c[5], c[6]);
            return -1;
        }
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_pooling_0()
           || test_pooling_1()
           || test_pooling_2()
           || test_pooling_3()
           || test_pooling_4()
#if NCNN_VALIDATION
           || test_pooling_load_param()
           || test_pooling_load_param_type()
           || test_pooling_load_param_adaptive()
#endif // NCNN_VALIDATION
           || test_pooling_rectangular()
           // height-one output retains the scalar channel tail dispatch
           || test_pooling(5, 3, 3, 0, 3, 2, 0, 0, 1, 0, 0, 0);
}
