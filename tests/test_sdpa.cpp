// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_sdpa(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v, int mask_type, float scale = 0.f, int flag = TEST_LAYER_DISABLE_AUTO_INPUT_PACKING)
{
    const int src_seqlen = q.h;
    const int dst_seqlen = k.h;

    ncnn::ParamDict pd;
    pd.set(5, mask_type != 0);
    pd.set(6, scale);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(3);
    as[0] = q;
    as[1] = k;
    as[2] = v;

    if (mask_type)
    {
        ncnn::Mat mask;
        if (mask_type == 2)
            mask = RandomMat(dst_seqlen, src_seqlen, 1);
        else if (mask_type == 3)
            mask = RandomMat(dst_seqlen, src_seqlen, q.c);
        else
            mask = RandomMat(dst_seqlen, src_seqlen);

        if (mask_type == 4)
        {
            const int masked_seqlen = std::min(dst_seqlen, 256);
            for (int i = 0; i < src_seqlen; i++)
            {
                float* mptr = mask.row(i);
                for (int j = 0; j < masked_seqlen; j++)
                    mptr[j] = -10000.f;
            }
        }

        as.push_back(mask);
    }

    float epsilon = 0.001;

    int ret = test_layer("SDPA", pd, weights, as, 1, epsilon, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_sdpa failed q=(%d %d %d) k=(%d %d %d) v=(%d %d %d) mask_type=%d scale=%f\n", q.w, q.h, q.c, k.w, k.h, k.c, v.w, v.h, v.c, mask_type, scale);
    }

    return ret;
}

static int test_sdpa_0()
{
    return 0
           || test_sdpa(RandomMat(64, 1, 16), RandomMat(64, 513, 1), RandomMat(48, 513, 1), 0)
           || test_sdpa(RandomMat(64, 1, 16), RandomMat(64, 521, 1), RandomMat(48, 521, 1), 3)
           || test_sdpa(RandomMat(64, 1, 12), RandomMat(64, 521, 1), RandomMat(40, 521, 1), 1)
           || test_sdpa(RandomMat(63, 1, 8), RandomMat(63, 513, 1), RandomMat(37, 513, 1), 0)
           || test_sdpa(RandomMat(47, 1, 2), RandomMat(47, 513, 1), RandomMat(29, 513, 1), 0)
           || test_sdpa(RandomMat(55, 1, 3), RandomMat(55, 521, 1), RandomMat(31, 521, 1), 1)
           || test_sdpa(RandomMat(65, 1, 4), RandomMat(65, 509, 4), RandomMat(33, 509, 4), 3)
           || test_sdpa(RandomMat(80, 1, 8), RandomMat(80, 521, 2), RandomMat(96, 521, 2), 1, -0.4f)
           || test_sdpa(RandomMat(96, 1, 8), RandomMat(96, 521, 2), RandomMat(80, 521, 2), 3)
           || test_sdpa(RandomMat(27, 1, 30), RandomMat(27, 37, 1), RandomMat(23, 37, 1), 3)
           || test_sdpa(RandomMat(17, 1, 1), RandomMat(17, 37, 1), RandomMat(65, 37, 1), 0)
           || test_sdpa(RandomMat(27, 4, 1), RandomMat(27, 513, 1), RandomMat(23, 513, 1), 3, 0.f, TEST_LAYER_DISABLE_AUTO_INPUT_PACKING | TEST_LAYER_ENABLE_THREADING)
           || test_sdpa(RandomMat(128, 17, 4), RandomMat(128, 513, 4), RandomMat(192, 513, 4), 4)
           || test_sdpa(RandomMat(32, 66, 8), RandomMat(32, 66, 8), RandomMat(20, 66, 8), 0)
           || test_sdpa(RandomMat(26, 64, 8), RandomMat(26, 61, 8), RandomMat(18, 61, 8), 1)
           || test_sdpa(RandomMat(192, 9, 8), RandomMat(192, 17, 2), RandomMat(128, 17, 2), 2, 0.2f)
           || test_sdpa(RandomMat(256, 5, 4), RandomMat(256, 13, 4), RandomMat(96, 13, 4), 3)
           || test_sdpa(RandomMat(64, 17, 1), RandomMat(64, 13, 1), RandomMat(28, 13, 1), 1)
           || test_sdpa(RandomMat(64, 12, 4), RandomMat(64, 16, 1), RandomMat(32, 16, 1), 1)
           || test_sdpa(RandomMat(64, 8, 1), RandomMat(64, 13, 1), RandomMat(28, 13, 1), 0)
           || test_sdpa(RandomMat(64, 4, 1), RandomMat(64, 13, 1), RandomMat(20, 13, 1), 0)
           || test_sdpa(RandomMat(40, 62, 7), RandomMat(40, 61, 7), RandomMat(24, 61, 7), 0)
           || test_sdpa(RandomMat(24, 22, 6), RandomMat(24, 19, 6), RandomMat(16, 19, 6), 1)
           || test_sdpa(RandomMat(64, 128, 12), RandomMat(64, 128, 2), RandomMat(64, 128, 2), 0)
           || test_sdpa(RandomMat(64, 122, 12), RandomMat(64, 127, 2), RandomMat(48, 127, 2), 1)
           || test_sdpa(RandomMat(44, 128, 4), RandomMat(44, 123, 4), RandomMat(55, 123, 4), 0, 1.f)
           || test_sdpa(RandomMat(12, 127, 4), RandomMat(12, 127, 4), RandomMat(55, 127, 4), 1, 1.f)
           || test_sdpa(RandomMat(28, 17, 15), RandomMat(28, 127, 5), RandomMat(32, 127, 5), 0, 0.1f)
           || test_sdpa(RandomMat(28, 17, 15), RandomMat(28, 32, 5), RandomMat(11, 32, 5), 1, -0.4f);
}

#if NCNN_INT8
static int test_sdpa_int8(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v, int attn_mask, float scale = 0.f)
{
    const int src_seqlen = q.h;
    const int dst_seqlen = k.h;

    ncnn::ParamDict pd;
    pd.set(5, attn_mask);
    pd.set(6, scale);
    pd.set(18, 2); // int8_scale_term

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(3);
    as[0] = q;
    as[1] = k;
    as[2] = v;

    if (attn_mask)
    {
        as.push_back(RandomMat(dst_seqlen, src_seqlen));
    }

    float epsilon = 0.01;

    int ret = test_layer("SDPA", pd, weights, as, 1, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_sdpa_int8 failed q=(%d %d %d) k=(%d %d %d) v=(%d %d %d) attn_mask=%d scale=%f\n", q.w, q.h, q.c, k.w, k.h, k.c, v.w, v.h, v.c, attn_mask, scale);
    }

    return ret;
}

static int test_sdpa_1()
{
    return 0
           || test_sdpa_int8(RandomMat(32, 66, 8), RandomMat(32, 66, 8), RandomMat(20, 66, 8), 0)
           || test_sdpa_int8(RandomMat(26, 64, 8), RandomMat(26, 61, 8), RandomMat(18, 61, 8), 1)
           || test_sdpa_int8(RandomMat(40, 62, 7), RandomMat(40, 61, 7), RandomMat(24, 61, 7), 0)
           || test_sdpa_int8(RandomMat(24, 22, 6), RandomMat(24, 19, 6), RandomMat(16, 19, 6), 1)
           || test_sdpa_int8(RandomMat(64, 128, 12), RandomMat(64, 128, 2), RandomMat(64, 128, 2), 0)
           || test_sdpa_int8(RandomMat(48, 122, 12), RandomMat(64, 127, 2), RandomMat(64, 127, 2), 1)
           || test_sdpa_int8(RandomMat(44, 128, 4), RandomMat(44, 123, 4), RandomMat(55, 123, 4), 0, 1.f)
           || test_sdpa_int8(RandomMat(12, 127, 4), RandomMat(12, 127, 4), RandomMat(55, 127, 4), 1, 1.f)
           || test_sdpa_int8(RandomMat(28, 17, 15), RandomMat(28, 127, 5), RandomMat(32, 127, 5), 0, 0.1f)
           || test_sdpa_int8(RandomMat(28, 17, 15), RandomMat(28, 32, 5), RandomMat(11, 32, 5), 1, -0.4f);
}
#endif

int main()
{
    SRAND(7767517);

#if NCNN_INT8
    return test_sdpa_0() || test_sdpa_1();
#else
    return test_sdpa_0();
#endif
}
