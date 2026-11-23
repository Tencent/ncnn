// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static float cubic_weight(float x)
{
    const float A = -0.75f;
    x = fabs(x);
    if (x <= 1.f)
        return ((A + 2.f) * x - (A + 3.f)) * x * x + 1.f;
    if (x < 2.f)
        return ((A * x - 5.f * A) * x + 8.f * A) * x - 4.f * A;
    return 0.f;
}

static int test_interp_scale(const ncnn::Mat& a, int resize_type, float height_scale, float width_scale, int target = 0, int align_corner = 0)
{
    const int h = a.dims == 2 ? 1 : a.h;
    const int channels = a.dims == 2 ? a.h : a.c;
    const int outw = (int)(a.w * width_scale);
    const int outh = (int)(h * height_scale);

    ncnn::ParamDict pd;
    pd.set(0, resize_type);
    pd.set(6, align_corner);
    if (target == 0)
    {
        pd.set(1, height_scale);
        pd.set(2, width_scale);
    }
    if (target == 1)
    {
        pd.set(3, outh);
        pd.set(4, outw);
    }
    if (target == 2)
        pd.set(5, 1);
    if (target == 3)
    {
        char expr[256];
        if (a.dims == 2)
            snprintf(expr, sizeof(expr), "floor(*(0w,%.9e))", width_scale);
        else
            snprintf(expr, sizeof(expr), "floor(*(0w,%.9e)),floor(*(0h,%.9e))", width_scale, height_scale);
        pd.set(9, expr);
    }

    ncnn::Mat expected;
    if (a.dims == 2)
        expected.create(outw, channels);
    else
        expected.create(outw, outh, channels);

    double ws = target == 0 ? 1.0 / width_scale : (double)a.w / outw;
    double hs = target == 0 ? 1.0 / height_scale : (double)h / outh;
    if (resize_type == 2 && a.w == outw)
        ws = 1.0;
    if (resize_type == 2 && h == outh)
        hs = 1.0;
    if (align_corner)
    {
        ws = outw > 1 ? (double)(a.w - 1) / (outw - 1) : 0.0;
        hs = outh > 1 ? (double)(h - 1) / (outh - 1) : 0.0;
    }

    // independent clamped-tap reference, not the naive layer's coefficient tables
    for (int q = 0; q < channels; q++)
    {
        const ncnn::Mat src = a.dims == 2 ? a.row_range(q, 1) : a.channel(q);
        ncnn::Mat dst = a.dims == 2 ? expected.row_range(q, 1) : expected.channel(q);
        for (int y = 0; y < outh; y++)
        {
            float* ptr = dst.row(y);
            for (int x = 0; x < outw; x++)
            {
                if (resize_type == 1)
                {
                    const int ix = std::min((int)(x * (float)ws), a.w - 1);
                    const int iy = a.dims == 2 ? 0 : std::min((int)(y * (float)hs), h - 1);
                    ptr[x] = src.row(iy)[ix];
                    continue;
                }

                const float fx = (float)(align_corner ? x * ws : (x + 0.5) * ws - 0.5);
                const float fy = a.dims == 2 ? 0.f : (float)(align_corner ? y * hs : (y + 0.5) * hs - 0.5);
                const int sx = (int)floor(fx);
                const int sy = (int)floor(fy);
                const int begin = resize_type == 2 ? 0 : -1;
                const int end = resize_type == 2 ? 1 : 2;
                float v = 0.f;
                for (int i = begin; i <= end; i++)
                {
                    const int iy = std::max(0, std::min(sy + i, h - 1));
                    const float b = resize_type == 2 ? (i == 0 ? 1.f - (fy - sy) : fy - sy) : cubic_weight(fy - (sy + i));
                    float row = 0.f;
                    for (int j = begin; j <= end; j++)
                    {
                        const int ix = std::max(0, std::min(sx + j, a.w - 1));
                        const float alpha = resize_type == 2 ? (j == 0 ? 1.f - (fx - sx) : fx - sx) : cubic_weight(fx - (sx + j));
                        row += src.row(iy)[ix] * alpha;
                    }
                    v += row * b;
                }
                ptr[x] = v;
            }
        }
    }

    std::vector<ncnn::Mat> as(2);
    as[0] = a;
    as[1] = ncnn::Mat(outw, outh, 1);
    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = false;
    opt.use_fp16_storage = false;
    opt.use_bf16_storage = false;
    std::vector<ncnn::Mat> bs(1);
    ncnn::Layer* op = ncnn::create_layer_naive("Interp");
    int ret = op->load_param(pd);
    if (ret == 0)
        ret = op->create_pipeline(opt);
    if (ret == 0)
    {
        if (target == 2)
            ret = op->forward(as, bs, opt);
        else
            ret = op->forward(a, bs[0], opt);
    }
    op->destroy_pipeline(opt);
    delete op;

    if (ret == 0)
        ret = CompareMat(expected, bs[0], 0.0001f);
    if (ret != 0)
    {
        fprintf(stderr, "test_interp_scale reference failed a=(%d %d %d) dims=%d type=%d scale=(%f %f) target=%d align_corner=%d\n", a.w, a.h, a.c, a.dims, resize_type, height_scale, width_scale, target, align_corner);
        return ret;
    }

    std::vector<ncnn::Mat> weights(0);
    if (target == 2)
        ret = test_layer("Interp", pd, weights, as);
    else
        ret = test_layer("Interp", pd, weights, a);
    if (ret != 0)
        fprintf(stderr, "test_interp_scale backend failed a=(%d %d %d) dims=%d type=%d scale=(%f %f) target=%d align_corner=%d\n", a.w, a.h, a.c, a.dims, resize_type, height_scale, width_scale, target, align_corner);
    return ret;
}

static int test_interp_scale_0()
{
    ncnn::Mat a = RandomMat(7, 7, 3);
    ncnn::Mat b = RandomMat(7, 7, 8);
    ncnn::Mat c = RandomMat(7, 8);
    for (int type = 1; type <= 3; type++)
    {
        for (int target = 0; target < 4; target++)
        {
            if (test_interp_scale(a, type, 1.5f, 1.5f, target)
                    || test_interp_scale(b, type, 0.8f, 0.8f, target)
                    || test_interp_scale(c, type, 1.f, 1.5f, target))
                return -1;
        }
    }
    return 0;
}

static int test_interp_scale_1()
{
    ncnn::Mat a = RandomMat(7, 7, 8);
    ncnn::Mat b = RandomMat(7, 8);
    for (int type = 2; type <= 3; type++)
    {
        if (test_interp_scale(a, type, 1.1f, 1.1f)
                || test_interp_scale(a, type, 1.1f, 1.5f)
                || test_interp_scale(a, type, 1.5f, 1.1f)
                || test_interp_scale(b, type, 1.f, 1.1f)
                || test_interp_scale(a, type, 1.5f, 1.5f, 0, 1)
                || test_interp_scale(a, type, 0.2f, 0.2f, 0, 1)
                || test_interp_scale(a, type, 1.f, 1.f))
            return -1;
    }
    return 0;
}

static int test_interp_scale_2()
{
    ncnn::Mat a = RandomMat(1, 7, 8);
    ncnn::Mat b = RandomMat(7, 1, 8);
    ncnn::Mat c = RandomMat(3, 2, 8);
    ncnn::Mat d = RandomMat(3, 8);
    ncnn::Mat e = RandomMat(1, 7, 3);
    ncnn::Mat f = RandomMat(7, 1, 3);
    ncnn::Mat g = RandomMat(3, 2, 3);
    ncnn::Mat h = RandomMat(3, 3);
    for (int type = 2; type <= 3; type++)
    {
        for (int target = 0; target < 4; target++)
        {
            if (test_interp_scale(a, type, 1.5f, 2.f, target)
                    || test_interp_scale(b, type, 2.f, 1.5f, target)
                    || test_interp_scale(c, type, 1.5f, 1.5f, target)
                    || test_interp_scale(d, type, 1.f, 1.5f, target)
                    || test_interp_scale(a, type, 0.2f, 2.f, target, 1)
                    || test_interp_scale(b, type, 2.f, 0.2f, target, 1)
                    || test_interp_scale(c, type, 0.75f, 1.5f, target, 1)
                    || test_interp_scale(d, type, 1.f, 0.5f, target, 1))
                return -1;
        }
        if (test_interp_scale(c, type, 1.1f, 1.1f)
                || test_interp_scale(d, type, 1.f, 1.1f))
            return -1;
        // pack1 vulkan paths for short spatial axes
        if (test_interp_scale(e, type, 1.5f, 2.f)
                || test_interp_scale(f, type, 2.f, 1.5f)
                || test_interp_scale(g, type, 1.5f, 1.5f)
                || test_interp_scale(h, type, 1.f, 1.5f))
            return -1;
    }
    return 0;
}

static int test_interp_scale_3()
{
    ncnn::Mat a = RandomMat(7, 1, 8);
    ncnn::Mat b = RandomMat(3, 2, 8);
    ncnn::Mat c = RandomMat(1, 8);
    ncnn::Mat d = RandomMat(3, 8);
    for (int target = 0; target < 4; target++)
    {
        if (test_interp_scale(a, 2, 1.f, 1.f, target)
                || test_interp_scale(b, 3, 1.f, 1.f, target)
                || test_interp_scale(c, 2, 1.f, 1.f, target)
                || test_interp_scale(d, 3, 1.f, 1.f, target))
            return -1;
    }

    // equal shapes still require resampling for bicubic with a non-unit scale
    return 0
           || test_interp_scale(a, 2, 1.1f, 1.1f)
           || test_interp_scale(b, 3, 1.1f, 1.1f)
           || test_interp_scale(c, 2, 1.f, 1.1f)
           || test_interp_scale(d, 3, 1.f, 1.1f);
}

static int test_interp_scale_identity()
{
    ncnn::Mat a = RandomMat(7, 1, 8);
    ncnn::Mat b = RandomMat(3, 2, 8);
    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = false;
    opt.use_fp16_storage = false;
    opt.use_bf16_storage = false;

    for (int i = 0; i < 2; i++)
    {
        const ncnn::Mat& input = i == 0 ? a : b;
        ncnn::ParamDict pd;
        pd.set(0, i == 0 ? 2 : 3);
        pd.set(1, 1.f);
        pd.set(2, 1.f);

        for (int backend = 0; backend < 2; backend++)
        {
            ncnn::Layer* op = backend == 0 ? ncnn::create_layer_naive("Interp") : ncnn::create_layer_cpu("Interp");
            int ret = op->load_param(pd);
            if (ret == 0)
                ret = op->create_pipeline(opt);
            ncnn::Mat output;
            if (ret == 0)
                ret = op->forward(input, output, opt);
            op->destroy_pipeline(opt);
            delete op;
            if (ret != 0 || output.data != input.data)
            {
                fprintf(stderr, "test_interp_scale_identity failed type=%d backend=%d\n", i == 0 ? 2 : 3, backend);
                return -1;
            }
        }
    }
    return 0;
}

int main()
{
    SRAND(7767517);
    return 0
           || test_interp_scale_0()
           || test_interp_scale_1()
           || test_interp_scale_2()
           || test_interp_scale_3()
           || test_interp_scale_identity();
}
