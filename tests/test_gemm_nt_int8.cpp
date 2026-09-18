// Copyright 2026 Tencent
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

static int test_gemm_int8_nt(int M, int N, int K, float alpha, int transA, int transB, int output_transpose, int constantA, int constantB)
{
    ncnn::ParamDict pd;
    pd.set(0, alpha);
    pd.set(1, 1.f); // beta
    pd.set(2, transA);
    pd.set(3, transB);
    pd.set(4, constantA);
    pd.set(5, constantB);
    pd.set(6, 1);
    pd.set(7, M);
    pd.set(8, N);
    pd.set(9, K);
    pd.set(10, -1);
    pd.set(13, 0); // output_elemtype = fp32
    pd.set(14, output_transpose);
    pd.set(18, 2); // int8_scale_term

    std::vector<ncnn::Mat> weights;
    if (constantA) weights.push_back(transA ? RandomS8Mat(M, K) : RandomS8Mat(K, M));
    if (constantB) weights.push_back(transB ? RandomS8Mat(K, N) : RandomS8Mat(N, K));
    if (constantA) weights.push_back(RandomMat(M, 10.f, 20.f));
    if (constantB) weights.push_back(RandomMat(1, 10.f, 20.f));

    std::vector<ncnn::Mat> a;
    if (!constantA)
    {
        a.push_back(transA ? ncnn::Mat(M, K) : ncnn::Mat(K, M));
        RandomizeA(a[a.size() - 1], transA, 10.f);
    }
    if (!constantB)
    {
        a.push_back(transB ? ncnn::Mat(K, N) : ncnn::Mat(N, K));
        RandomizeB(a[a.size() - 1], 10.f);
    }

    float epsilon = 0.001;

    int ret = test_layer("Gemm", pd, weights, a, 1, epsilon, TEST_LAYER_ENABLE_THREADING);
    if (ret != 0)
    {
        fprintf(stderr, "test_gemm_int8_nt failed M=%d N=%d K=%d alpha=%f transA=%d transB=%d output_transpose=%d constantA=%d constantB=%d\n", M, N, K, alpha, transA, transB, output_transpose, constantA, constantB);
    }

    return ret;
}

static int test_gemm_int8_bias_nt(int M, int N, int K, const ncnn::Mat& C, float alpha, float beta, int transA, int transB, int output_transpose, int constantA, int constantB, int constantC)
{
    int broadcast_type_C = 0;
    if (C.dims == 1 && C.w == 1)
    {
        // scalar
        broadcast_type_C = 0;
    }
    if (C.dims == 1 && C.w == M)
    {
        // M
        // auto broadcast from h to w is the ncnn-style convention
        broadcast_type_C = 1;
    }
    if (C.dims == 1 && C.w == N)
    {
        // N
        broadcast_type_C = 4;
    }
    if (C.dims == 2 && C.w == 1 && C.h == M)
    {
        // Mx1
        broadcast_type_C = 2;
    }
    if (C.dims == 2 && C.w == N && C.h == M)
    {
        // MxN
        broadcast_type_C = 3;
    }
    if (C.dims == 2 && C.w == N && C.h == 1)
    {
        // 1xN
        broadcast_type_C = 4;
    }

    ncnn::ParamDict pd;
    pd.set(0, alpha);
    pd.set(1, beta);
    pd.set(2, transA);
    pd.set(3, transB);
    pd.set(4, constantA);
    pd.set(5, constantB);
    pd.set(6, constantC);
    pd.set(7, M);
    pd.set(8, N);
    pd.set(9, K);
    pd.set(10, broadcast_type_C);
    pd.set(13, 0); // output_elemtype = fp32
    pd.set(14, output_transpose);
    pd.set(18, 2); // int8_scale_term

    std::vector<ncnn::Mat> weights;
    if (constantA) weights.push_back(transA ? RandomS8Mat(M, K) : RandomS8Mat(K, M));
    if (constantB) weights.push_back(transB ? RandomS8Mat(K, N) : RandomS8Mat(N, K));
    if (constantC) weights.push_back(C);
    if (constantA) weights.push_back(RandomMat(M, 10.f, 20.f));
    if (constantB) weights.push_back(RandomMat(1, 10.f, 20.f));

    std::vector<ncnn::Mat> a;
    if (!constantA)
    {
        a.push_back(transA ? ncnn::Mat(M, K) : ncnn::Mat(K, M));
        RandomizeA(a[a.size() - 1], transA, 10.f);
    }
    if (!constantB)
    {
        a.push_back(transB ? ncnn::Mat(K, N) : ncnn::Mat(N, K));
        RandomizeB(a[a.size() - 1], 10.f);
    }
    if (!constantC) a.push_back(C);

    float epsilon = 0.001;

    int ret = test_layer("Gemm", pd, weights, a, 1, epsilon, TEST_LAYER_ENABLE_THREADING);
    if (ret != 0)
    {
        fprintf(stderr, "test_gemm_int8_bias_nt failed M=%d N=%d K=%d C.dims=%d C=(%d %d %d) alpha=%f beta=%f transA=%d transB=%d output_transpose=%d constantA=%d constantB=%d constantC=%d\n", M, N, K, C.dims, C.w, C.h, C.c, alpha, beta, transA, transB, output_transpose, constantA, constantB, constantC);
    }

    return ret;
}

static int test_gemm_0(int M, int N, int K)
{
    return 0
           || test_gemm_int8_nt(M, N, K, 2.1f, 0, 0, 0, 0, 0)
           || test_gemm_int8_nt(M, N, K, 3.1f, 0, 1, 0, 0, 0)
           || test_gemm_int8_nt(M, N, K, 4.1f, 1, 0, 1, 0, 0)
           || test_gemm_int8_nt(M, N, K, 5.1f, 1, 1, 1, 0, 0)

           || test_gemm_int8_nt(M, N, K, 0.2f, 0, 0, 1, 1, 0)
           || test_gemm_int8_nt(M, N, K, 0.3f, 0, 1, 1, 1, 0)
           || test_gemm_int8_nt(M, N, K, 0.4f, 1, 0, 0, 1, 0)
           || test_gemm_int8_nt(M, N, K, 0.5f, 1, 1, 0, 1, 0)

           || test_gemm_int8_nt(M, N, K, 1.2f, 0, 0, 0, 0, 1)
           || test_gemm_int8_nt(M, N, K, 1.3f, 0, 1, 1, 0, 1)
           || test_gemm_int8_nt(M, N, K, 1.4f, 1, 0, 0, 0, 1)
           || test_gemm_int8_nt(M, N, K, 1.5f, 1, 1, 1, 0, 1)

           || test_gemm_int8_nt(M, N, K, -1.2f, 0, 0, 1, 1, 1)
           || test_gemm_int8_nt(M, N, K, -1.3f, 0, 1, 0, 1, 1)
           || test_gemm_int8_nt(M, N, K, -1.4f, 1, 0, 1, 1, 1)
           || test_gemm_int8_nt(M, N, K, -1.5f, 1, 1, 0, 1, 1);
}

static int test_gemm_1(int M, int N, int K)
{
    return 0
           || test_gemm_int8_bias_nt(M, N, K, RandomMat(1), 2.1f, 0.5f, 0, 0, 0, 0, 0, 0)
           || test_gemm_int8_bias_nt(M, N, K, RandomMat(M), 3.1f, 0.6f, 0, 1, 0, 0, 0, 0)
           || test_gemm_int8_bias_nt(M, N, K, RandomMat(1, M), 4.1f, 0.7f, 1, 0, 1, 0, 0, 0)
           || test_gemm_int8_bias_nt(M, N, K, RandomMat(N, M), 5.1f, 0.8f, 1, 1, 1, 0, 0, 0)
           || test_gemm_int8_bias_nt(M, N, K, RandomMat(N, 1), 2.1f, 0.5f, 0, 0, 0, 0, 0, 0)
           || test_gemm_int8_bias_nt(M, N, K, RandomMat(N), 3.1f, 0.6f, 0, 1, 0, 0, 0, 0)

           || test_gemm_int8_bias_nt(M, N, K, RandomMat(1), -2.1f, 0.5f, 1, 1, 0, 1, 1, 1)
           || test_gemm_int8_bias_nt(M, N, K, RandomMat(M), -3.1f, 0.6f, 0, 0, 0, 1, 1, 1)
           || test_gemm_int8_bias_nt(M, N, K, RandomMat(1, M), -4.1f, 0.7f, 0, 1, 1, 1, 1, 1)
           || test_gemm_int8_bias_nt(M, N, K, RandomMat(N, M), -5.1f, 0.8f, 1, 0, 1, 1, 1, 1)
           || test_gemm_int8_bias_nt(M, N, K, RandomMat(N, 1), -2.1f, 0.5f, 1, 1, 0, 1, 1, 1)
           || test_gemm_int8_bias_nt(M, N, K, RandomMat(N), -3.1f, 0.6f, 0, 0, 0, 1, 1, 1);
}
#endif // NCNN_INT8

int main()
{
    SRAND(7767517);

#if NCNN_INT8
    int mnk[][3] = {
        {1, 20, 40},
        {20, 2, 39},
        {3, 30, 13},
        {33, 1, 19}
    };

    int mnk_count = sizeof(mnk) / sizeof(int) / 3;

    for (int i = 0; i < mnk_count; i++)
    {
        int M = mnk[i][0];
        int N = mnk[i][1];
        int K = mnk[i][2];

        int ret = test_gemm_0(M, N, K) || test_gemm_1(M, N, K);

        if (ret != 0)
            return ret;
    }
#else
    // test nothing for non-int8 build
#endif

    return 0;
}
