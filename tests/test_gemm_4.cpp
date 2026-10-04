// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#if NCNN_INT8
static void RandomizeA(ncnn::Mat& m, int transA, float absmax)
{
    if (transA == 0)
    {
        const int h = m.dims == 3 ? m.c : m.h;
        for (int i = 0; i < h; i++)
        {
            float* p = m.dims == 3 ? m.channel(i) : m.row(i);
            float randabsmax = RandomFloat(absmax * 0.5f, absmax);
            randabsmax = ncnn::float16_to_float32(ncnn::float32_to_float16(randabsmax));
            randabsmax = ncnn::bfloat16_to_float32(ncnn::float32_to_bfloat16(randabsmax));

            for (int j = 0; j < m.w; j++)
            {
                p[j] = RandomFloat(-randabsmax, randabsmax);
            }

            // set random a and b
            p[RandomInt(0, m.w - 1)] = -randabsmax;
            p[RandomInt(0, m.w - 1)] = randabsmax;

            // drop 0.45 ~ 0.55
            for (int j = 0; j < m.w; j++)
            {
                float v = p[j] * (127.f / randabsmax);
                float vv = fabs(v - (int)v);

                float hp = ncnn::float16_to_float32(ncnn::float32_to_float16(p[j]));
                float hv = hp * (127.f / randabsmax);
                float hvv = fabs(hv - (int)hv);

                float bp = ncnn::bfloat16_to_float32(ncnn::float32_to_bfloat16(p[j]));
                float bv = bp * (127.f / randabsmax);
                float bvv = fabs(bv - (int)bv);

                while ((vv > 0.45f && vv < 0.55f) || (hvv > 0.45f && hvv < 0.55f) || (bvv > 0.45f && bvv < 0.55f))
                {
                    p[j] = RandomFloat(-randabsmax, randabsmax);
                    v = p[j] * (127.f / randabsmax);
                    vv = fabs(v - (int)v);

                    hp = ncnn::float16_to_float32(ncnn::float32_to_float16(p[j]));
                    hv = hp * (127.f / randabsmax);
                    hvv = fabs(hv - (int)hv);

                    bp = ncnn::bfloat16_to_float32(ncnn::float32_to_bfloat16(p[j]));
                    bv = bp * (127.f / randabsmax);
                    bvv = fabs(bv - (int)bv);
                }
            }
        }
    }
    else // if (transA == 1)
    {
        std::vector<float> randabsmaxes(m.w);
        for (int j = 0; j < m.w; j++)
        {
            float randabsmax = RandomFloat(absmax * 0.5f, absmax);
            randabsmax = ncnn::float16_to_float32(ncnn::float32_to_float16(randabsmax));
            randabsmax = ncnn::bfloat16_to_float32(ncnn::float32_to_bfloat16(randabsmax));
            randabsmaxes[j] = randabsmax;
        }

        const int h = m.dims == 3 ? m.c : m.h;
        for (int i = 0; i < h; i++)
        {
            float* p = m.dims == 3 ? m.channel(i) : m.row(i);
            for (int j = 0; j < m.w; j++)
            {
                const float randabsmax = randabsmaxes[j];
                p[j] = RandomFloat(-randabsmax, randabsmax);
            }

            // drop 0.45 ~ 0.55
            for (int j = 0; j < m.w; j++)
            {
                const float randabsmax = randabsmaxes[j];
                float v = p[j] * (127.f / randabsmax);
                float vv = fabs(v - (int)v);

                float hp = ncnn::float16_to_float32(ncnn::float32_to_float16(p[j]));
                float hv = hp * (127.f / randabsmax);
                float hvv = fabs(hv - (int)hv);

                float bp = ncnn::bfloat16_to_float32(ncnn::float32_to_bfloat16(p[j]));
                float bv = bp * (127.f / randabsmax);
                float bvv = fabs(bv - (int)bv);

                while ((vv > 0.45f && vv < 0.55f) || (hvv > 0.45f && hvv < 0.55f) || (bvv > 0.45f && bvv < 0.55f))
                {
                    p[j] = RandomFloat(-randabsmax, randabsmax);
                    v = p[j] * (127.f / randabsmax);
                    vv = fabs(v - (int)v);

                    hp = ncnn::float16_to_float32(ncnn::float32_to_float16(p[j]));
                    hv = hp * (127.f / randabsmax);
                    hvv = fabs(hv - (int)hv);

                    bp = ncnn::bfloat16_to_float32(ncnn::float32_to_bfloat16(p[j]));
                    bv = bp * (127.f / randabsmax);
                    bvv = fabs(bv - (int)bv);
                }
            }
        }

        for (int j = 0; j < m.w; j++)
        {
            const int randi0 = RandomInt(0, h - 1);
            const int randi1 = RandomInt(0, h - 1);
            float* p0 = m.dims == 3 ? m.channel(randi0) : m.row(randi0);
            float* p1 = m.dims == 3 ? m.channel(randi1) : m.row(randi1);

            const float randabsmax = randabsmaxes[j];

            // set random a and b
            p0[j] = -randabsmax;
            p1[j] = randabsmax;
        }
    }
}

static void RandomizeB(ncnn::Mat& m, float absmax)
{
    absmax = ncnn::float16_to_float32(ncnn::float32_to_float16(absmax));
    absmax = ncnn::bfloat16_to_float32(ncnn::float32_to_bfloat16(absmax));

    const int h = m.dims == 3 ? m.c : m.h;
    float* p = m;
    for (int i = 0; i < h; i++)
    {
        float* p = m.dims == 3 ? m.channel(i) : m.row(i);
        for (int j = 0; j < m.w; j++)
        {
            p[j] = RandomFloat(-absmax, absmax);

            // drop 0.45 ~ 0.55
            float v = p[j] * (127.f / absmax);
            float vv = fabs(v - (int)v);

            float hp = ncnn::float16_to_float32(ncnn::float32_to_float16(p[j]));
            float hv = hp * (127.f / absmax);
            float hvv = fabs(hv - (int)hv);

            float bp = ncnn::bfloat16_to_float32(ncnn::float32_to_bfloat16(p[j]));
            float bv = bp * (127.f / absmax);
            float bvv = fabs(bv - (int)bv);

            while ((vv > 0.45f && vv < 0.55f) || (hvv > 0.45f && hvv < 0.55f) || (bvv > 0.45f && bvv < 0.55f))
            {
                p[j] = RandomFloat(-absmax, absmax);
                v = p[j] * (127.f / absmax);
                vv = fabs(v - (int)v);

                hp = ncnn::float16_to_float32(ncnn::float32_to_float16(p[j]));
                hv = hp * (127.f / absmax);
                hvv = fabs(hv - (int)hv);

                bp = ncnn::bfloat16_to_float32(ncnn::float32_to_bfloat16(p[j]));
                bv = bp * (127.f / absmax);
                bvv = fabs(bv - (int)bv);
            }
        }
    }

    // set random a and b
    if (m.dims == 3)
    {
        m.channel(RandomInt(0, h - 1))[RandomInt(0, m.w - 1)] = -absmax;
        m.channel(RandomInt(0, h - 1))[RandomInt(0, m.w - 1)] = absmax;
    }
    else
    {
        m.row(RandomInt(0, h - 1))[RandomInt(0, m.w - 1)] = -absmax;
        m.row(RandomInt(0, h - 1))[RandomInt(0, m.w - 1)] = absmax;
    }
}

static int test_gemm_int8(int M, int N, int K, int TILE_M, int TILE_N, int TILE_K, float alpha, int transA, int transB, int output_transpose, int flag = 0)
{
    ncnn::ParamDict pd;
    pd.set(0, alpha);
    pd.set(1, 1.f); // beta
    pd.set(2, transA);
    pd.set(3, transB);
    pd.set(14, output_transpose);
    pd.set(18, 2); // int8_scale_term

    pd.set(20, TILE_M);
    pd.set(21, TILE_N);
    pd.set(22, TILE_K);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> a(2);
    a[0] = transA ? ncnn::Mat(M, K) : ncnn::Mat(K, M);
    a[1] = transB ? ncnn::Mat(K, N) : ncnn::Mat(N, K);

    RandomizeA(a[0], transA, 10.f);
    RandomizeB(a[1], 10.f);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = true;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;

    int ret = test_layer_opt("Gemm", pd, weights, opt, a, 1, 0.001, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_gemm_int8 failed M=%d N=%d K=%d TILE_M=%d TILE_N=%d TILE_K=%d alpha=%f transA=%d transB=%d output_transpose=%d\n", M, N, K, TILE_M, TILE_N, TILE_K, alpha, transA, transB, output_transpose);
    }

    return ret;
}

static int test_gemm_0(int M, int N, int K, int TILE_M, int TILE_N, int TILE_K, int flag = 0)
{
    return 0
           || test_gemm_int8(M, N, K, TILE_M, TILE_N, TILE_K, 2.1f, 0, 0, 0, flag)
           || test_gemm_int8(M, N, K, TILE_M, TILE_N, TILE_K, 3.1f, 0, 1, 0, flag)
           || test_gemm_int8(M, N, K, TILE_M, TILE_N, TILE_K, 4.1f, 1, 0, 0, flag)
           || test_gemm_int8(M, N, K, TILE_M, TILE_N, TILE_K, 5.1f, 1, 1, 0, flag)
           || test_gemm_int8(M, N, K, TILE_M, TILE_N, TILE_K, 2.1f, 0, 0, 1, flag)
           || test_gemm_int8(M, N, K, TILE_M, TILE_N, TILE_K, 3.1f, 0, 1, 1, flag)
           || test_gemm_int8(M, N, K, TILE_M, TILE_N, TILE_K, 4.1f, 1, 0, 1, flag)
           || test_gemm_int8(M, N, K, TILE_M, TILE_N, TILE_K, 5.1f, 1, 1, 1, flag);
}

// non-unit scaling is independent of the dynamic input transpose layouts
static int test_gemm_layouts(int M, int N, int K, int TILE_M, int TILE_N, int TILE_K)
{
    return 0
           || test_gemm_int8(M, N, K, TILE_M, TILE_N, TILE_K, 2.1f, 0, 0, 0)
           || test_gemm_int8(M, N, K, TILE_M, TILE_N, TILE_K, 2.1f, 0, 1, 0)
           || test_gemm_int8(M, N, K, TILE_M, TILE_N, TILE_K, 2.1f, 1, 0, 0)
           || test_gemm_int8(M, N, K, TILE_M, TILE_N, TILE_K, 2.1f, 1, 1, 0)
           || test_gemm_int8(M, N, K, TILE_M, TILE_N, TILE_K, 2.1f, 0, 0, 1)
           || test_gemm_int8(M, N, K, TILE_M, TILE_N, TILE_K, 2.1f, 0, 1, 1)
           || test_gemm_int8(M, N, K, TILE_M, TILE_N, TILE_K, 2.1f, 1, 0, 1)
           || test_gemm_int8(M, N, K, TILE_M, TILE_N, TILE_K, 2.1f, 1, 1, 1);
}
#endif // NCNN_INT8

int main()
{
    SRAND(7767517);

#if NCNN_INT8
    // transpose and output layouts cover short, aligned and remainder dimensions
    static const int layouts[][3] = {
        {1, 1, 1},
        {31, 31, 31},
        {31, 32, 31},
        {32, 31, 32},
        {32, 32, 32},
        {20, 32, 20},
        {47, 47, 47},
        {63, 64, 63},
        {64, 63, 64},
        {64, 64, 64},
    };
    for (int i = 0; i < (int)(sizeof(layouts) / sizeof(layouts[0])); i++)
    {
        const int* t = layouts[i];
        if (test_gemm_layouts(t[0], t[1], t[2], 100, 100, 100) != 0)
            return -1;
    }

    // independent dimensions retain small and intermediate matrix sizes
    static const int dimensions[][3] = {
        {2, 2, 2},
        {3, 3, 3},
        {4, 4, 4},
        {5, 5, 5},
        {6, 6, 6},
        {7, 7, 7},
        {8, 8, 8},
        {15, 15, 15},
        {16, 16, 16},
        {24, 24, 24},
        {40, 40, 40},
        {48, 48, 48},
        {52, 52, 52},
    };
    for (int i = 0; i < (int)(sizeof(dimensions) / sizeof(dimensions[0])); i++)
    {
        const int* t = dimensions[i];
        if (test_gemm_int8(t[0], t[1], t[2], 100, 100, 100, 2.1f, 0, 0, 0) != 0)
            return -1;
    }

    // aligned and remainder shapes exercise every tile and transpose layout
    static const int boundary[][3] = {
        {31, 31, 31},
        {47, 47, 47},
        {64, 64, 64},
    };
    static const int tiles[] = {1, 2, 4, 8, 12, 16, 20, 24, 28};
    const int boundary_count = sizeof(boundary) / sizeof(boundary[0]);
    for (int i = 0; i < boundary_count; i++)
    {
        for (int j = 0; j < 9; j++)
        {
            // tile overrides affect cpu kernels
            if (test_gemm_0(boundary[i][0], boundary[i][1], boundary[i][2], tiles[j], tiles[j], tiles[j], TEST_LAYER_DISABLE_GPU_TESTING) != 0)
                return -1;
        }
    }

    // short matrices and asymmetric dimensions cover partial tiles independently
    static const int partial[][6] = {
        {2, 2, 2, 1, 1, 1}, {3, 3, 3, 2, 2, 2},
        {5, 5, 5, 4, 4, 4}, {9, 9, 9, 8, 8, 8},
        {13, 13, 13, 12, 12, 12}, {17, 17, 17, 16, 16, 16},
        {21, 21, 21, 20, 20, 20}, {25, 25, 25, 24, 24, 24},
        {29, 29, 29, 28, 28, 28},
        {31, 32, 31, 1, 1, 1}, {32, 31, 32, 4, 8, 4}
    };
    const int partial_count = sizeof(partial) / sizeof(partial[0]);
    for (int i = 0; i < partial_count; i++)
    {
        const int* t = partial[i];
        if (test_gemm_0(t[0], t[1], t[2], t[3], t[4], t[5], TEST_LAYER_DISABLE_GPU_TESTING) != 0)
            return -1;
    }

#else
    // test nothing for non-int8 build
#endif

    return 0;
}
