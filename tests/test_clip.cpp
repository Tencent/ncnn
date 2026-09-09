// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"
#include "datareader.h"

static int test_clip(const ncnn::Mat& a, float min, float max)
{
    ncnn::ParamDict pd;
    pd.set(0, min);
    pd.set(1, max);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Clip", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_clip failed a.dims=%d a=(%d %d %d %d) min=%f max=%f\n", a.dims, a.w, a.h, a.d, a.c, min, max);
    }

    return ret;
}

static int test_clip_0()
{
    return 0
           || test_clip(RandomMat(3, 3, 3, 48), -1.f, 1.f)
           || test_clip(RandomMat(5, 6, 7, 24), -1.f, 1.f)
           || test_clip(RandomMat(7, 8, 9, 12), -1.f, 1.f)
           || test_clip(RandomMat(3, 4, 5, 13), -1.f, 1.f);
}

static int test_clip_1()
{
    return 0
           || test_clip(RandomMat(3, 3, 48), -1.f, 1.f)
           || test_clip(RandomMat(5, 7, 24), -1.f, 1.f)
           || test_clip(RandomMat(7, 9, 12), -1.f, 1.f)
           || test_clip(RandomMat(3, 5, 13), -1.f, 1.f);
}

static int test_clip_2()
{
    return 0
           || test_clip(RandomMat(19, 48), -1.f, 1.f)
           || test_clip(RandomMat(15, 24), -1.f, 1.f)
           || test_clip(RandomMat(17, 12), -1.f, 1.f)
           || test_clip(RandomMat(19, 15), -1.f, 1.f);
}

static int test_clip_3()
{
    return 0
           || test_clip(RandomMat(128), -1.f, 1.f)
           || test_clip(RandomMat(124), -1.f, 1.f)
           || test_clip(RandomMat(127), -1.f, 1.f);
}

class ClipParamDict : public ncnn::ParamDict
{
public:
    using ncnn::ParamDict::load_param;
};

static int test_clip_zero_type()
{
    for (int mode = 0; mode < 4; mode++)
    {
        const bool floating = mode % 2 == 1;
        ClipParamDict pd;
        if (mode < 2)
        {
            if (floating)
                pd.set(0, 0.f);
            else
                pd.set(0, 0);
            pd.set(1, 6.f);
        }
        else
        {
#if NCNN_STRING
            const unsigned char* text = (const unsigned char*)(floating ? "0=0.0 1=6.000000e+00" : "0=0 1=6.000000e+00");
            ncnn::DataReaderFromMemory reader(text);
            if (pd.load_param(reader))
                return -1;
#else
            continue;
#endif
        }

        ncnn::Option opt;
        opt.num_threads = 1;
        ncnn::Layer* op = ncnn::create_layer_cpu("Clip");
        if (!op)
            return -1;
        int ret = op->load_param(pd);
        if (ret == 0)
            ret = op->create_pipeline(opt);

        ncnn::Mat values(3);
        values[0] = -1.f;
        values[1] = 0.f;
        values[2] = 7.f;
        if (ret == 0)
            ret = op->forward_inplace(values, opt);
        op->destroy_pipeline(opt);
        delete op;

        // integer zero selects the nonzero default lower bound of -FLT_MAX
        const float lower = floating ? 0.f : -1.f;
        if (ret != 0 || values[0] != lower || values[1] != 0.f || values[2] != 6.f)
        {
            fprintf(stderr, "test_clip_zero_type failed mode=%d ret=%d\n", mode, ret);
            return -1;
        }
    }
    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_clip_0()
           || test_clip_1()
           || test_clip_2()
           || test_clip_3()
           || test_clip_zero_type();
}
