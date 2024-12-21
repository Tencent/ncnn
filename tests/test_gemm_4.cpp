// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

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

static int test_gemm_int8(int M, int N, int K, int TILE_M, int TILE_N, int TILE_K, float alpha, int transA, int transB, int output_transpose)
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

    int ret = test_layer("Gemm", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_gemm_int8 failed M=%d N=%d K=%d TILE_M=%d TILE_N=%d TILE_K=%d alpha=%f transA=%d transB=%d output_transpose=%d\n", M, N, K, TILE_M, TILE_N, TILE_K, alpha, transA, transB, output_transpose);
    }

    return ret;
}

static int test_gemm_0(int M, int N, int K, int TILE_M, int TILE_N, int TILE_K)
{
    return 0
           || test_gemm_int8(M, N, K, TILE_M, TILE_N, TILE_K, 2.1f, 0, 0, 0)
           || test_gemm_int8(M, N, K, TILE_M, TILE_N, TILE_K, 3.1f, 0, 1, 0)
           || test_gemm_int8(M, N, K, TILE_M, TILE_N, TILE_K, 4.1f, 1, 0, 0)
           || test_gemm_int8(M, N, K, TILE_M, TILE_N, TILE_K, 5.1f, 1, 1, 0)
           || test_gemm_int8(M, N, K, TILE_M, TILE_N, TILE_K, 2.1f, 0, 0, 1)
           || test_gemm_int8(M, N, K, TILE_M, TILE_N, TILE_K, 3.1f, 0, 1, 1)
           || test_gemm_int8(M, N, K, TILE_M, TILE_N, TILE_K, 4.1f, 1, 0, 1)
           || test_gemm_int8(M, N, K, TILE_M, TILE_N, TILE_K, 5.1f, 1, 1, 1);
}
#endif // NCNN_INT8

int main()
{
    SRAND(7767517);

#if NCNN_INT8
    int mnk[][3] = {
        {1, 1, 1},
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
        {31, 31, 31},
        {31, 32, 31},
        {32, 31, 32},
        {32, 32, 32},
        {20, 32, 20},
        {40, 40, 40},
        {47, 47, 47},
        {48, 48, 48},
        {52, 52, 52},
        {63, 64, 63},
        {64, 63, 64},
        {64, 64, 64}
    };

    int tile_mnk[][3] = {
        {1, 1, 1},
        {2, 2, 2},
        {4, 4, 4},
        {8, 8, 8},
        {12, 12, 12},
        {16, 16, 16},
        {20, 20, 20},
        {24, 24, 24},
        {28, 28, 28}
    };

    int mnk_count = sizeof(mnk) / sizeof(int) / 3;
    int tile_mnk_count = sizeof(tile_mnk) / sizeof(int) / 3;

    for (int i = 0; i < mnk_count; i++)
    {
        int M = mnk[i][0];
        int N = mnk[i][1];
        int K = mnk[i][2];

        for (int j = 0; j < tile_mnk_count; j++)
        {
            int TILE_M = tile_mnk[j][0];
            int TILE_N = tile_mnk[j][1];
            int TILE_K = tile_mnk[j][2];

            if (TILE_M >= M && TILE_N >= N && TILE_K >= K)
                continue;

            int ret = test_gemm_0(M, N, K, TILE_M, TILE_N, TILE_K);
            if (ret != 0)
                return ret;
        }

        // test no tiling
        int ret = test_gemm_0(M, N, K, 100, 100, 100);
        if (ret != 0)
            return ret;
    }
#else
    // test nothing for non-int8 build
#endif

    return 0;
}
