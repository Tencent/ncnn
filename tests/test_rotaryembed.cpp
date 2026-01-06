// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_rotaryembed(const ncnn::Mat& a, int interleaved)
{
    const int embed_dim = a.w;
    const int seqlen = a.h;
    const int num_heads = a.c;

    ncnn::Mat cos_cache = RandomMat(embed_dim / 2, seqlen);
    ncnn::Mat sin_cache = RandomMat(embed_dim / 2, seqlen);

    ncnn::ParamDict pd;
    pd.set(0, interleaved);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(3);
    as[0] = a;
    as[1] = cos_cache;
    as[2] = sin_cache;

    int ret = test_layer("RotaryEmbed", pd, weights, as, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_rotaryembed failed a=(%d %d %d) interleaved=%d\n", a.w, a.h, a.c, interleaved);
    }

    return ret;
}

static int test_rotaryembed_0()
{
    return 0
           || test_rotaryembed(RandomMat(32, 66, 8), 0)
           || test_rotaryembed(RandomMat(26, 64, 8), 1)
           || test_rotaryembed(RandomMat(64, 28, 12), 0)
           || test_rotaryembed(RandomMat(48, 22, 12), 1)
           || test_rotaryembed(RandomMat(44, 28, 64), 0)
           || test_rotaryembed(RandomMat(12, 27, 64), 1)
           || test_rotaryembed(RandomMat(28, 17, 15), 0)
           || test_rotaryembed(RandomMat(28, 17, 15), 1);
}

int main()
{
    SRAND(7767517);

    return test_rotaryembed_0();
}
