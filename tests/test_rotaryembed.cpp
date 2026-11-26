// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_rotaryembed(const ncnn::Mat& a, const ncnn::Mat& cos_cache, int interleaved)
{
    ncnn::ParamDict pd;
    pd.set(0, interleaved);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(3);
    as[0] = a;
    as[1] = cos_cache;
    as[2].create_like(cos_cache);
    Randomize(as[2]);

    int ret = test_layer("RotaryEmbed", pd, weights, as, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_rotaryembed failed a.dims=%d a=(%d %d %d) cache.dims=%d cache=(%d %d) interleaved=%d\n", a.dims, a.w, a.h, a.c, cos_cache.dims, cos_cache.w, cos_cache.h, interleaved);
    }

    return ret;
}

// full-width caches exercise independent frequencies for the two halves and the interleaved row stride
static int test_rotaryembed_cache_widths(const ncnn::Mat& a, int interleaved, int cache_h = 0)
{
    const int h = cache_h > 0 ? cache_h : a.h;
    return test_rotaryembed(a, RandomMat(a.w / 2, h), interleaved)
           || test_rotaryembed(a, RandomMat(a.w, h), interleaved);
}

static int test_rotaryembed_3d()
{
    return 0
           || test_rotaryembed_cache_widths(RandomMat(2, 3, 2), 0)
           || test_rotaryembed_cache_widths(RandomMat(2, 3, 2), 1)
           || test_rotaryembed_cache_widths(RandomMat(62, 3, 2), 0)
           || test_rotaryembed_cache_widths(RandomMat(62, 3, 2), 1)
           || test_rotaryembed_cache_widths(RandomMat(66, 3, 2), 0)
           || test_rotaryembed_cache_widths(RandomMat(66, 3, 2), 1)
           || test_rotaryembed_cache_widths(RandomMat(32, 66, 8), 0)
           || test_rotaryembed(RandomMat(32, 66, 8), RandomMat(32, 66), 1)
           || test_rotaryembed_cache_widths(RandomMat(26, 64, 8), 1)
           || test_rotaryembed_cache_widths(RandomMat(64, 28, 12), 0)
           || test_rotaryembed_cache_widths(RandomMat(48, 22, 12), 1)
           || test_rotaryembed_cache_widths(RandomMat(44, 28, 64), 0)
           || test_rotaryembed_cache_widths(RandomMat(12, 27, 64), 1)
           || test_rotaryembed_cache_widths(RandomMat(28, 17, 15), 0)
           || test_rotaryembed_cache_widths(RandomMat(28, 17, 15), 1)
           || test_rotaryembed_cache_widths(RandomMat(128, 1, 16), 0)
           || test_rotaryembed_cache_widths(RandomMat(128, 1, 16), 1)
           || test_rotaryembed_cache_widths(RandomMat(196, 5, 3), 0)
           || test_rotaryembed_cache_widths(RandomMat(196, 5, 3), 1)
           || test_rotaryembed_cache_widths(RandomMat(62, 8, 3), 0)
           || test_rotaryembed_cache_widths(RandomMat(62, 8, 3), 1);
}

// vulkan packs 2d input along seqlen, with a different cos/sin row for each lane
static int test_rotaryembed_2d()
{
    return 0
           || test_rotaryembed(RandomMat(2, 3), RandomMat(1, 3), 0)
           || test_rotaryembed(RandomMat(2, 4), RandomMat(1, 4), 1)
           || test_rotaryembed(RandomMat(2, 4), RandomMat(2, 4), 0)
           || test_rotaryembed(RandomMat(64, 1), RandomMat(32, 1), 0)
           || test_rotaryembed(RandomMat(64, 1), RandomMat(32, 1), 1)
           || test_rotaryembed_cache_widths(RandomMat(64, 4), 0)
           || test_rotaryembed_cache_widths(RandomMat(64, 4), 1)
           || test_rotaryembed_cache_widths(RandomMat(62, 8), 0)
           || test_rotaryembed_cache_widths(RandomMat(62, 8), 1)
           || test_rotaryembed_cache_widths(RandomMat(128, 16), 0)
           || test_rotaryembed_cache_widths(RandomMat(128, 16), 1)
           || test_rotaryembed_cache_widths(RandomMat(48, 13), 0)
           || test_rotaryembed_cache_widths(RandomMat(48, 13), 1);
}

static int test_rotaryembed_1d()
{
    return 0
           || test_rotaryembed(RandomMat(4), RandomMat(2, 1), 0)
           || test_rotaryembed(RandomMat(60), RandomMat(60, 1), 0)
           || test_rotaryembed_cache_widths(RandomMat(62), 0)
           || test_rotaryembed_cache_widths(RandomMat(62), 1)
           || test_rotaryembed_cache_widths(RandomMat(64), 0)
           || test_rotaryembed_cache_widths(RandomMat(64), 1)
           || test_rotaryembed_cache_widths(RandomMat(62), 0, 4)
           || test_rotaryembed_cache_widths(RandomMat(62), 1, 4)
           || test_rotaryembed_cache_widths(RandomMat(64), 0, 4)
           || test_rotaryembed_cache_widths(RandomMat(64), 1, 4);
}

static int test_rotaryembed_1d_cache()
{
    // a half-width 1d cache can have a different packing from the input
    const int widths[] = {60, 62, 64};
    for (size_t i = 0; i < sizeof(widths) / sizeof(widths[0]); i++)
    {
        const ncnn::Mat a = RandomMat(widths[i]);
        for (int interleaved = 0; interleaved < 2; interleaved++)
        {
            if (test_rotaryembed(a, RandomMat(a.w / 2), interleaved)
                || test_rotaryembed(a, RandomMat(a.w), interleaved))
                return -1;
        }
    }

    return 0;
}

// cache can be longer than the input, so its packing is independent of input packing
static int test_rotaryembed_longer_cache()
{
    return 0
           || test_rotaryembed(RandomMat(2, 4), RandomMat(1, 5), 0)
           || test_rotaryembed(RandomMat(62, 4), RandomMat(31, 5), 1)
           || test_rotaryembed_cache_widths(RandomMat(62, 8), 0, 9)
           || test_rotaryembed_cache_widths(RandomMat(62, 8), 1, 9)
           || test_rotaryembed_cache_widths(RandomMat(62, 5), 0, 8)
           || test_rotaryembed_cache_widths(RandomMat(62, 5), 1, 8)
           || test_rotaryembed(RandomMat(64, 5, 4), RandomMat(64, 8), 0);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_rotaryembed_3d()
           || test_rotaryembed_2d()
           || test_rotaryembed_1d()
           || test_rotaryembed_1d_cache()
           || test_rotaryembed_longer_cache();
}
