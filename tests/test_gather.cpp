// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "layer/gather.h"
#include "testutil.h"

#include <gtest/gtest.h>

static int test_gather_cpu(int dims, int axis, const std::vector<int>& data_shape, const std::vector<int>& index_shape)
{
    ncnn::Mat data;
    if (dims == 1)
        data = RandomMat(data_shape[0]);
    else if (dims == 2)
        data = RandomMat(data_shape[0], data_shape[1]);
    else
        data = RandomMat(data_shape[0], data_shape[1], data_shape[2]);

    ncnn::Mat indices;
    if (dims == 1)
        indices = RandomMat(index_shape[0]);
    else if (dims == 2)
        indices = RandomMat(index_shape[0], index_shape[1]);
    else
        indices = RandomMat(index_shape[0], index_shape[1], index_shape[2]);

    // Convert to int32 indices clamped to valid range
    int axis_size = (dims == 1) ? data_shape[0] : (axis == 0) ? data_shape[0] : (axis == 1) ? data_shape[1] : data_shape[2];
    ncnn::Mat indices_int(indices.w, indices.h, indices.c, 4u);
    for (int i = 0; i < (int)indices.total(); i++)
    {
        int idx = (int)(((float*)indices)[i] * axis_size);
        if (idx < 0) idx = 0;
        if (idx >= axis_size) idx = axis_size - 1;
        ((int*)indices_int)[i] = idx;
    }

    ncnn::Option opt;
    opt.num_threads = 1;

    ncnn::Layer* op = ncnn::create_layer("Gather");
    op->vkdev = ncnn::get_gpu_device();

    ncnn::ParamDict pd;
    pd.set(0, axis);
    op->load_param(pd);

    std::vector<ncnn::Mat> bottom_blobs(2);
    bottom_blobs[0] = data;
    bottom_blobs[1] = indices_int;

    std::vector<ncnn::Mat> top_blobs(1);
    int ret = op->forward(bottom_blobs, top_blobs, opt);

    delete op;

    if (ret != 0)
        return -1;

    // Output rank must match index blob
    const ncnn::Mat& out = top_blobs[0];
    if (out.dims != indices_int.dims || out.w != indices_int.w || out.h != indices_int.h || out.c != indices_int.c)
    {
        fprintf(stderr, "Output shape mismatch: got %dx%dx%d (dims=%d), expected %dx%dx%d (dims=%d)\n",
                out.w, out.h, out.c, out.dims,
                indices_int.w, indices_int.h, indices_int.c, indices_int.dims);
        return -1;
    }

    return 0;
}

TEST(Gather, test_1d_axis0)
{
    EXPECT_EQ(0, test_gather_cpu(1, 0, {10}, {5}));
}

TEST(Gather, test_2d_axis0)
{
    EXPECT_EQ(0, test_gather_cpu(2, 0, {5, 8}, {3, 8}));
}

TEST(Gather, test_2d_axis1)
{
    EXPECT_EQ(0, test_gather_cpu(2, 1, {5, 8}, {5, 4}));
}

TEST(Gather, test_3d_axis0)
{
    EXPECT_EQ(0, test_gather_cpu(3, 0, {4, 6, 8}, {2, 6, 8}));
}

TEST(Gather, test_3d_axis1)
{
    EXPECT_EQ(0, test_gather_cpu(3, 1, {4, 6, 8}, {4, 3, 8}));
}

TEST(Gather, test_3d_axis2)
{
    EXPECT_EQ(0, test_gather_cpu(3, 2, {4, 6, 8}, {4, 6, 5}));
}

TEST(Gather, test_negative_axis)
{
    EXPECT_EQ(0, test_gather_cpu(3, -1, {4, 6, 8}, {4, 6, 5}));
}

TEST(Gather, test_1d_index_from_3d_data)
{
    // index rank may differ from data rank (Gather spec allows this)
    EXPECT_EQ(0, test_gather_cpu(1, 0, {10}, {7}));
}
