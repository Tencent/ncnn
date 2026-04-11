// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "layer/gatherelements.h"
#include "testutil.h"

#include <gtest/gtest.h>

static int test_gatherelements_cpu(int dims, int axis, const std::vector<int>& data_shape, const std::vector<int>& index_shape)
{
    ncnn::Mat data;
    if (dims == 1)
    {
        data = RandomMat(data_shape[0]);
    }
    else if (dims == 2)
    {
        data = RandomMat(data_shape[0], data_shape[1]);
    }
    else if (dims == 3)
    {
        data = RandomMat(data_shape[0], data_shape[1], data_shape[2]);
    }

    ncnn::Mat indices;
    if (dims == 1)
    {
        indices = RandomMat(index_shape[0]);
    }
    else if (dims == 2)
    {
        indices = RandomMat(index_shape[0], index_shape[1]);
    }
    else if (dims == 3)
    {
        indices = RandomMat(index_shape[0], index_shape[1], index_shape[2]);
    }

    // Convert indices to int32
    ncnn::Mat indices_int(indices.w, indices.h, indices.c, 4u);
    for (int i = 0; i < (int)indices.total(); i++)
    {
        ((int*)indices_int)[i] = (int)((float*)indices)[i];
    }

    ncnn::Option opt;
    opt.num_threads = 1;

    ncnn::Layer* op = ncnn::create_layer("GatherElements");
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

    // Check output shape matches indices shape
    const ncnn::Mat& out = top_blobs[0];
    if (out.w != indices.w || out.h != indices.h || out.c != indices.c)
    {
        fprintf(stderr, "Output shape mismatch\n");
        return -1;
    }

    return 0;
}

TEST(GatherElements, test_1d)
{
    std::vector<int> data_shape = {10};
    std::vector<int> index_shape = {5};
    EXPECT_EQ(0, test_gatherelements_cpu(1, 0, data_shape, index_shape));
}

TEST(GatherElements, test_2d_axis0)
{
    std::vector<int> data_shape = {5, 8};
    std::vector<int> index_shape = {3, 8};
    EXPECT_EQ(0, test_gatherelements_cpu(2, 0, data_shape, index_shape));
}

TEST(GatherElements, test_2d_axis1)
{
    std::vector<int> data_shape = {5, 8};
    std::vector<int> index_shape = {5, 4};
    EXPECT_EQ(0, test_gatherelements_cpu(2, 1, data_shape, index_shape));
}

TEST(GatherElements, test_3d_axis0)
{
    std::vector<int> data_shape = {4, 6, 8};
    std::vector<int> index_shape = {2, 6, 8};
    EXPECT_EQ(0, test_gatherelements_cpu(3, 0, data_shape, index_shape));
}

TEST(GatherElements, test_3d_axis1)
{
    std::vector<int> data_shape = {4, 6, 8};
    std::vector<int> index_shape = {4, 3, 8};
    EXPECT_EQ(0, test_gatherelements_cpu(3, 1, data_shape, index_shape));
}

TEST(GatherElements, test_3d_axis2)
{
    std::vector<int> data_shape = {4, 6, 8};
    std::vector<int> index_shape = {4, 6, 5};
    EXPECT_EQ(0, test_gatherelements_cpu(3, 2, data_shape, index_shape));
}

TEST(GatherElements, test_negative_axis)
{
    std::vector<int> data_shape = {4, 6, 8};
    std::vector<int> index_shape = {4, 6, 5};
    EXPECT_EQ(0, test_gatherelements_cpu(3, -1, data_shape, index_shape));
}
