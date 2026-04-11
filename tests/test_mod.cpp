// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "layer/mod.h"
#include "testutil.h"

#include <gtest/gtest.h>

static int test_mod_cpu(int fmode, int w, int h, int c)
{
    ncnn::Mat a = RandomMat(w, h, c);
    ncnn::Mat b = RandomMat(w, h, c);

    // Ensure b is not zero to avoid division by zero
    for (int i = 0; i < (int)b.total(); i++)
    {
        float val = ((float*)b)[i];
        if (val == 0.0f)
            ((float*)b)[i] = 1.0f;
    }

    ncnn::Option opt;
    opt.num_threads = 1;

    ncnn::Layer* op = ncnn::create_layer("Mod");
    op->vkdev = ncnn::get_gpu_device();

    ncnn::ParamDict pd;
    pd.set(0, fmode);
    op->load_param(pd);

    std::vector<ncnn::Mat> bottom_blobs(2);
    bottom_blobs[0] = a;
    bottom_blobs[1] = b;

    std::vector<ncnn::Mat> top_blobs(1);
    int ret = op->forward(bottom_blobs, top_blobs, opt);

    delete op;

    if (ret != 0)
        return -1;

    // Check output shape
    const ncnn::Mat& out = top_blobs[0];
    if (out.w != w || out.h != h || out.c != c)
    {
        fprintf(stderr, "Output shape mismatch\n");
        return -1;
    }

    // Verify correctness
    const float* pa = a;
    const float* pb = b;
    const float* pout = out;
    
    for (int i = 0; i < (int)out.total(); i++)
    {
        float expected;
        if (fmode == 0)
        {
            // Python-style modulo
            expected = std::fmod(pa[i], pb[i]);
            if ((expected != 0.0f) && ((pb[i] < 0.0f) != (expected < 0.0f)))
            {
                expected += pb[i];
            }
        }
        else
        {
            // C-style fmod
            expected = std::fmod(pa[i], pb[i]);
        }
        
        if (std::abs(pout[i] - expected) > 0.001f)
        {
            fprintf(stderr, "Value mismatch at index %d: expected %f, got %f\n", 
                    i, expected, pout[i]);
            return -1;
        }
    }

    return 0;
}

TEST(Mod, test_fmod_python_style)
{
    EXPECT_EQ(0, test_mod_cpu(0, 10, 1, 1));
}

TEST(Mod, test_fmod_c_style)
{
    EXPECT_EQ(0, test_mod_cpu(1, 10, 1, 1));
}

TEST(Mod, test_2d)
{
    EXPECT_EQ(0, test_mod_cpu(0, 8, 6, 1));
}

TEST(Mod, test_3d)
{
    EXPECT_EQ(0, test_mod_cpu(0, 4, 6, 8));
}

TEST(Mod, test_negative_values)
{
    ncnn::Mat a(10);
    ncnn::Mat b(10);
    
    for (int i = 0; i < 10; i++)
    {
        ((float*)a)[i] = -10.0f + i * 2.0f;
        ((float*)b)[i] = 3.0f;
    }

    ncnn::Option opt;
    opt.num_threads = 1;

    ncnn::Layer* op = ncnn::create_layer("Mod");
    
    ncnn::ParamDict pd;
    pd.set(0, 0); // Python-style
    op->load_param(pd);

    std::vector<ncnn::Mat> bottom_blobs(2);
    bottom_blobs[0] = a;
    bottom_blobs[1] = b;

    std::vector<ncnn::Mat> top_blobs(1);
    int ret = op->forward(bottom_blobs, top_blobs, opt);

    delete op;

    EXPECT_EQ(0, ret);
}
