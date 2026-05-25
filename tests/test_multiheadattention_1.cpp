// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#if NCNN_INT8
static void RandomizeDynamicQuantMat(ncnn::Mat& m, float absmax)
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

        p[RandomInt(0, m.w - 1)] = -randabsmax;
        p[RandomInt(0, m.w - 1)] = randabsmax;

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

static ncnn::Mat RandomDynamicQuantMat(int w, int h)
{
    ncnn::Mat m(w, h);
    RandomizeDynamicQuantMat(m, 1.2f);
    return m;
}

static int test_multiheadattention_int8(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v, int embed_dim, int num_heads, int attn_mask)
{
    const int qdim = q.w;
    const int kdim = k.w;
    const int vdim = v.w;

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * qdim);
    pd.set(3, kdim);
    pd.set(4, vdim);
    pd.set(5, attn_mask);
    pd.set(6, 1.f / sqrtf(embed_dim / num_heads));
    pd.set(18, 2); // int8_scale_term

    std::vector<ncnn::Mat> weights(12);
    weights[0] = RandomS8Mat(embed_dim * qdim);
    weights[1] = RandomMat(embed_dim);
    weights[2] = RandomS8Mat(embed_dim * kdim);
    weights[3] = RandomMat(embed_dim);
    weights[4] = RandomS8Mat(embed_dim * vdim);
    weights[5] = RandomMat(embed_dim);
    weights[6] = RandomS8Mat(qdim * embed_dim);
    weights[7] = RandomMat(qdim);
    weights[8] = RandomMat(embed_dim, 160.f, 200.f);
    weights[9] = RandomMat(embed_dim, 160.f, 200.f);
    weights[10] = RandomMat(embed_dim, 160.f, 200.f);
    weights[11] = RandomMat(1, 160.f, 200.f);

    std::vector<ncnn::Mat> as(3);
    as[0] = q;
    as[1] = k;
    as[2] = v;

    if (attn_mask)
    {
        as.push_back(RandomMat(k.h, q.h));
    }

    float epsilon = 0.1;

    int ret = test_layer("MultiHeadAttention", pd, weights, as, 1, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_int8 failed q=(%d %d) k=(%d %d) v=(%d %d) embed_dim=%d num_heads=%d kdim=%d vdim=%d attn_mask=%d\n", q.w, q.h, k.w, k.h, v.w, v.h, embed_dim, num_heads, kdim, vdim, attn_mask);
    }

    return ret;
}

static int test_multiheadattention_int8_samekv(const ncnn::Mat& q, const ncnn::Mat& kv, int embed_dim, int num_heads)
{
    const int qdim = q.w;
    const int kvdim = kv.w;

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * qdim);
    pd.set(3, kvdim);
    pd.set(4, kvdim);
    pd.set(6, 1.f / sqrtf(embed_dim / num_heads));
    pd.set(18, 2); // int8_scale_term

    std::vector<ncnn::Mat> weights(12);
    weights[0] = RandomS8Mat(embed_dim * qdim);
    weights[1] = RandomMat(embed_dim);
    weights[2] = RandomS8Mat(embed_dim * kvdim);
    weights[3] = RandomMat(embed_dim);
    weights[4] = RandomS8Mat(embed_dim * kvdim);
    weights[5] = RandomMat(embed_dim);
    weights[6] = RandomS8Mat(qdim * embed_dim);
    weights[7] = RandomMat(qdim);
    weights[8] = RandomMat(embed_dim, 160.f, 200.f);
    weights[9] = RandomMat(embed_dim, 160.f, 200.f);
    weights[10] = RandomMat(embed_dim, 160.f, 200.f);
    weights[11] = RandomMat(1, 160.f, 200.f);

    std::vector<ncnn::Mat> as(2);
    as[0] = q;
    as[1] = kv;

    float epsilon = 0.1;

    int ret = test_layer("MultiHeadAttention", pd, weights, as, 1, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_int8_samekv failed q=(%d %d) kv=(%d %d) embed_dim=%d num_heads=%d kvdim=%d\n", q.w, q.h, kv.w, kv.h, embed_dim, num_heads, kvdim);
    }

    return ret;
}

static int test_multiheadattention_int8_sameqkv(const ncnn::Mat& a, int embed_dim, int num_heads)
{
    const int qdim = a.w;

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * qdim);
    pd.set(3, qdim);
    pd.set(4, qdim);
    pd.set(6, 1.f / sqrtf(embed_dim / num_heads));
    pd.set(18, 2); // int8_scale_term

    std::vector<ncnn::Mat> weights(12);
    weights[0] = RandomS8Mat(embed_dim * qdim);
    weights[1] = RandomMat(embed_dim);
    weights[2] = RandomS8Mat(embed_dim * qdim);
    weights[3] = RandomMat(embed_dim);
    weights[4] = RandomS8Mat(embed_dim * qdim);
    weights[5] = RandomMat(embed_dim);
    weights[6] = RandomS8Mat(qdim * embed_dim);
    weights[7] = RandomMat(qdim);
    weights[8] = RandomMat(embed_dim, 160.f, 200.f);
    weights[9] = RandomMat(embed_dim, 160.f, 200.f);
    weights[10] = RandomMat(embed_dim, 160.f, 200.f);
    weights[11] = RandomMat(1, 160.f, 200.f);

    std::vector<ncnn::Mat> as(1);
    as[0] = a;

    float epsilon = 0.1;

    int ret = test_layer("MultiHeadAttention", pd, weights, as, 1, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_int8_sameqkv failed a=(%d %d) embed_dim=%d num_heads=%d\n", a.w, a.h, embed_dim, num_heads);
    }

    return ret;
}

static int test_multiheadattention_0()
{
    return 0
           || test_multiheadattention_int8(RandomDynamicQuantMat(62, 66), RandomDynamicQuantMat(32, 66), RandomDynamicQuantMat(20, 66), 62, 2, 0)
           || test_multiheadattention_int8(RandomDynamicQuantMat(26, 64), RandomDynamicQuantMat(32, 64), RandomDynamicQuantMat(18, 64), 26, 2, 1)
           || test_multiheadattention_int8(RandomDynamicQuantMat(64, 128), RandomDynamicQuantMat(64, 128), RandomDynamicQuantMat(64, 128), 64, 4, 0)
           || test_multiheadattention_int8(RandomDynamicQuantMat(48, 127), RandomDynamicQuantMat(64, 127), RandomDynamicQuantMat(64, 127), 64, 16, 1)
           || test_multiheadattention_int8(RandomDynamicQuantMat(16, 128), RandomDynamicQuantMat(44, 128), RandomDynamicQuantMat(55, 128), 16, 2, 0)
           || test_multiheadattention_int8(RandomDynamicQuantMat(12, 128), RandomDynamicQuantMat(44, 127), RandomDynamicQuantMat(55, 127), 16, 4, 1)
           || test_multiheadattention_int8(RandomDynamicQuantMat(12, 17), RandomDynamicQuantMat(28, 127), RandomDynamicQuantMat(32, 127), 12, 3, 0)
           || test_multiheadattention_int8(RandomDynamicQuantMat(12, 17), RandomDynamicQuantMat(28, 32), RandomDynamicQuantMat(11, 32), 12, 3, 1);
}

static int test_multiheadattention_1()
{
    return 0
           || test_multiheadattention_int8_samekv(RandomDynamicQuantMat(64, 128), RandomDynamicQuantMat(64, 128), 64, 4)
           || test_multiheadattention_int8_samekv(RandomDynamicQuantMat(48, 127), RandomDynamicQuantMat(64, 127), 64, 16)
           || test_multiheadattention_int8_samekv(RandomDynamicQuantMat(16, 128), RandomDynamicQuantMat(44, 128), 16, 2)
           || test_multiheadattention_int8_samekv(RandomDynamicQuantMat(12, 128), RandomDynamicQuantMat(22, 127), 16, 4)
           || test_multiheadattention_int8_samekv(RandomDynamicQuantMat(12, 17), RandomDynamicQuantMat(28, 127), 12, 3)
           || test_multiheadattention_int8_samekv(RandomDynamicQuantMat(12, 17), RandomDynamicQuantMat(11, 32), 12, 3);
}

static int test_multiheadattention_2()
{
    return 0
           || test_multiheadattention_int8_sameqkv(RandomDynamicQuantMat(64, 128), 64, 4)
           || test_multiheadattention_int8_sameqkv(RandomDynamicQuantMat(48, 127), 64, 8);
}
#endif

int main()
{
    SRAND(7767517);

#if NCNN_INT8
    return 0
           || test_multiheadattention_0()
           || test_multiheadattention_1()
           || test_multiheadattention_2();
#else
    // test nothing
    return 0;
#endif
}
