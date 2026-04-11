// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "layer/expand.h"
#include "testutil.h"

#include <gtest/gtest.h>

static int test_expand_cpu(int in_w, int in_h, int in_c, int out_w, int out_h, int out_c)
{
    ncnn::Mat input(in_w, in_h, in_c);
    Randomize(input);

    // Create shape tensor
    ncnn::Mat shape_tensor(3);
    ((int*)shape_tensor)[0] = out_w;
    ((int*)shape_tensor)[1] = out_h;
    ((int*)shape_tensor)[2] = out_c;

    ncnn::Option opt;
    opt.num_threads = 1;

    ncnn::Layer* op = ncnn::create_layer("Expand");
    op->vkdev = ncnn::get_gpu_device();

    ncnn::ParamDict pd;
    op->load_param(pd);

    std::vector<ncnn::Mat> bottom_blobs(2);
    bottom_blobs[0] = input;
    bottom_blobs[1] = shape_tensor;

    std::vector<ncnn::Mat> top_blobs(1);
    int ret = op->forward(bottom_blobs, top_blobs, opt);

    delete op;

    if (ret != 0)
        return -1;

    // Check output shape
    const ncnn::Mat& out = top_blobs[0];
    if (out.w != out_w || out.h != out_h || out.c != out_c)
    {
        fprintf(stderr, "Output shape mismatch: expected (%d,%d,%d), got (%d,%d,%d)\n",
                out_w, out_h, out_c, out.w, out.h, out.c);
        return -1;
    }

    return 0;
}

TEST(Expand, test_1d_to_1d)
{
    EXPECT_EQ(0, test_expand_cpu(1, 1, 1, 10, 1, 1));
}

TEST(Expand, test_1d_to_2d)
{
    EXPECT_EQ(0, test_expand_cpu(5, 1, 1, 5, 3, 1));
}

TEST(Expand, test_2d_broadcast)
{
    EXPECT_EQ(0, test_expand_cpu(1, 5, 1, 4, 5, 1));
}

TEST(Expand, test_3d_expand)
{
    EXPECT_EQ(0, test_expand_cpu(2, 3, 1, 2, 3, 5));
}

TEST(Expand, test_full_broadcast)
{
    EXPECT_EQ(0, test_expand_cpu(1, 1, 1, 4, 6, 8));
}
