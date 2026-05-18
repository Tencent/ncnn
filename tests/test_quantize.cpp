// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static bool near_round(float v)
{
    float vv = fabs(v - (int)v);

    return vv > 0.45f && vv < 0.55f;
}

static float RandomQuantizeValue(float scale)
{
    float hscale = ncnn::float16_to_float32(ncnn::float32_to_float16(scale));
    float bscale = ncnn::bfloat16_to_float32(ncnn::float32_to_bfloat16(scale));

    float v = RandomFloat();

    float hv = ncnn::float16_to_float32(ncnn::float32_to_float16(v));
    float bv = ncnn::bfloat16_to_float32(ncnn::float32_to_bfloat16(v));

    float v0 = v * scale;
    float hv0 = hv * hscale;
    float hhv0 = ncnn::float16_to_float32(ncnn::float32_to_float16(hv0));
    float bv0 = bv * bscale;

    while (near_round(v0) || near_round(hv0) || near_round(hhv0) || near_round(bv0))
    {
        v = RandomFloat();

        hv = ncnn::float16_to_float32(ncnn::float32_to_float16(v));
        bv = ncnn::bfloat16_to_float32(ncnn::float32_to_bfloat16(v));

        v0 = v * scale;
        hv0 = hv * hscale;
        hhv0 = ncnn::float16_to_float32(ncnn::float32_to_float16(hv0));
        bv0 = bv * bscale;
    }

    return v;
}

static void RandomizeQuantize(ncnn::Mat& m, const ncnn::Mat& scale_data)
{
    if (m.dims == 1)
    {
        float* p = m;
        const float scale = scale_data[0];

        for (int i = 0; i < m.w; i++)
        {
            p[i] = RandomQuantizeValue(scale);
        }
    }

    if (m.dims == 2)
    {
        for (int i = 0; i < m.h; i++)
        {
            float* p = m.row(i);
            const float scale = scale_data.w == 1 ? scale_data[0] : scale_data[i];

            for (int j = 0; j < m.w; j++)
            {
                p[j] = RandomQuantizeValue(scale);
            }
        }
    }

    if (m.dims == 3)
    {
        for (int q = 0; q < m.c; q++)
        {
            float* p = m.channel(q);
            const float scale = scale_data.w == 1 ? scale_data[0] : scale_data[q];

            for (int i = 0; i < m.w * m.h; i++)
            {
                p[i] = RandomQuantizeValue(scale);
            }
        }
    }
}

static int test_quantize(ncnn::Mat a, float scale_low, float scale_high)
{
    ncnn::Mat scale_data;
    if (scale_low == scale_high)
    {
        scale_data.create(1);
        scale_data[0] = scale_low;
    }
    else
    {
        if (a.dims == 1) scale_data.create(1);
        if (a.dims == 2) scale_data.create(a.h);
        if (a.dims == 3) scale_data.create(a.c);
        Randomize(scale_data, scale_low, scale_high);
    }

    RandomizeQuantize(a, scale_data);

    ncnn::ParamDict pd;
    pd.set(0, scale_data.w);

    std::vector<ncnn::Mat> weights(1);
    weights[0] = scale_data;

    int ret = test_layer("Quantize", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_quantize failed a.dims=%d a=(%d %d %d) scale_low=%f scale_high=%f\n", a.dims, a.w, a.h, a.c, scale_low, scale_high);
    }

    return ret;
}

static int test_quantize_0()
{
    return 0
           || test_quantize(RandomMat(5, 7, 24), 100.f, 100.f)
           || test_quantize(RandomMat(5, 7, 24), 120.f, 140.f)
           || test_quantize(RandomMat(7, 9, 12), 100.f, 100.f)
           || test_quantize(RandomMat(7, 9, 12), 120.f, 140.f)
           || test_quantize(RandomMat(3, 5, 13), 100.f, 100.f)
           || test_quantize(RandomMat(3, 5, 13), 120.f, 140.f)
           || test_quantize(RandomMat(3, 3, 256), 100.f, 100.f)
           || test_quantize(RandomMat(3, 3, 256), 120.f, 140.f)
           || test_quantize(RandomMat(3, 3, 255), 100.f, 100.f)
           || test_quantize(RandomMat(3, 3, 255), 120.f, 140.f);
}

static int test_quantize_1()
{
    return 0
           || test_quantize(RandomMat(15, 24), 100.f, 100.f)
           || test_quantize(RandomMat(15, 24), 120.f, 140.f)
           || test_quantize(RandomMat(17, 12), 100.f, 100.f)
           || test_quantize(RandomMat(17, 12), 120.f, 140.f)
           || test_quantize(RandomMat(19, 15), 100.f, 100.f)
           || test_quantize(RandomMat(19, 15), 120.f, 140.f)
           || test_quantize(RandomMat(15, 256), 100.f, 100.f)
           || test_quantize(RandomMat(15, 256), 120.f, 140.f)
           || test_quantize(RandomMat(15, 255), 100.f, 100.f)
           || test_quantize(RandomMat(15, 255), 120.f, 140.f);
}

static int test_quantize_2()
{
    return 0
           || test_quantize(RandomMat(128), 100.f, 100.f)
           || test_quantize(RandomMat(128), 120.f, 140.f)
           || test_quantize(RandomMat(124), 100.f, 100.f)
           || test_quantize(RandomMat(124), 120.f, 140.f)
           || test_quantize(RandomMat(127), 100.f, 100.f)
           || test_quantize(RandomMat(127), 120.f, 140.f)
           || test_quantize(RandomMat(256), 100.f, 100.f)
           || test_quantize(RandomMat(256), 120.f, 140.f)
           || test_quantize(RandomMat(255), 100.f, 100.f)
           || test_quantize(RandomMat(255), 120.f, 140.f);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_quantize_0()
           || test_quantize_1()
           || test_quantize_2();
}
