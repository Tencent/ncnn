// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "net.h"
#include "testutil.h"

static int test_gridsample_nearest_even()
{
    ncnn::Mat image(5, 5, 1);
    float* image_ptr = image.channel(0);
    for (int y = 0; y < 5; y++)
        for (int x = 0; x < 5; x++)
            image_ptr[y * 5 + x] = (float)x;

    const float coordinates[4] = {-0.75f, -0.25f, 0.25f, 0.75f};
    const float expected[4] = {0.f, 2.f, 2.f, 4.f};

    for (int permute_fusion = 0; permute_fusion < 2; permute_fusion++)
    {
        ncnn::Mat grid = permute_fusion ? ncnn::Mat(16, 1, 2) : ncnn::Mat(2, 16, 1);
        for (int i = 0; i < 16; i++)
        {
            if (permute_fusion)
            {
                ((float*)grid.channel(0))[i] = coordinates[i % 4];
                ((float*)grid.channel(1))[i] = 0.f;
            }
            else
            {
                ((float*)grid.channel(0))[i * 2] = coordinates[i % 4];
                ((float*)grid.channel(0))[i * 2 + 1] = 0.f;
            }
        }

        ncnn::Layer* layer = ncnn::create_layer("GridSample");
        ncnn::ParamDict pd;
        pd.set(0, 2); // nearest
        pd.set(1, 1); // zeros padding
        pd.set(2, 1); // align corners
        pd.set(3, permute_fusion);
        int ret = layer->load_param(pd);
        std::vector<ncnn::Mat> inputs(2);
        inputs[0] = image;
        inputs[1] = grid;
        std::vector<ncnn::Mat> outputs(1);
        if (ret == 0)
            ret = layer->forward(inputs, outputs, ncnn::Option());
        delete layer;
        if (ret != 0)
            return ret;

        const float* output_ptr = outputs[0].channel(0);
        for (int i = 0; i < 16; i++)
        {
            if (output_ptr[i] != expected[i % 4])
            {
                fprintf(stderr, "test_gridsample_nearest_even failed permute_fusion=%d index=%d value=%f expected=%f\n", permute_fusion, i, output_ptr[i], expected[i % 4]);
                return -1;
            }
        }
    }

    return 0;
}

static int test_gridsample_3d_nearest_even()
{
    ncnn::Mat image(5, 5, 5, 1);
    float* image_ptr = image.channel(0);
    for (int z = 0; z < 5; z++)
        for (int y = 0; y < 5; y++)
            for (int x = 0; x < 5; x++)
                image_ptr[z * 25 + y * 5 + x] = (float)(100 * z + 10 * y + x);

    const float coordinates[4] = {-0.75f, -0.25f, 0.25f, 0.75f};
    const float expected[4] = {0.f, 222.f, 222.f, 444.f};

    for (int permute_fusion = 0; permute_fusion < 2; permute_fusion++)
    {
        ncnn::Mat grid = permute_fusion ? ncnn::Mat(16, 1, 1, 3) : ncnn::Mat(3, 16, 1, 1);
        for (int i = 0; i < 16; i++)
        {
            for (int axis = 0; axis < 3; axis++)
            {
                if (permute_fusion)
                    ((float*)grid.channel(axis))[i] = coordinates[i % 4];
                else
                    ((float*)grid.channel(0))[i * 3 + axis] = coordinates[i % 4];
            }
        }

        ncnn::Layer* layer = ncnn::create_layer("GridSample");
        ncnn::ParamDict pd;
        pd.set(0, 2); // nearest
        pd.set(1, 1); // zeros padding
        pd.set(2, 1); // align corners
        pd.set(3, permute_fusion);
        int ret = layer->load_param(pd);
        std::vector<ncnn::Mat> inputs(2);
        inputs[0] = image;
        inputs[1] = grid;
        std::vector<ncnn::Mat> outputs(1);
        if (ret == 0)
            ret = layer->forward(inputs, outputs, ncnn::Option());
        delete layer;
        if (ret != 0)
            return ret;

        const float* output_ptr = outputs[0].channel(0);
        for (int i = 0; i < 16; i++)
        {
            if (output_ptr[i] != expected[i % 4])
            {
                fprintf(stderr, "test_gridsample_3d_nearest_even failed permute_fusion=%d index=%d value=%f expected=%f\n", permute_fusion, i, output_ptr[i], expected[i % 4]);
                return -1;
            }
        }
    }

    return 0;
}

static int test_gridsample(const ncnn::Mat& a, const ncnn::Mat& grid, int sample_type, int padding_mode, int align_corner, int permute_fusion)
{
    ncnn::ParamDict pd;
    pd.set(0, sample_type);
    pd.set(1, padding_mode);
    pd.set(2, align_corner);
    pd.set(3, permute_fusion);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(2);
    as[0] = a;
    as[1] = grid;

    int ret = test_layer("GridSample", pd, weights, as);
    if (ret != 0)
    {
        fprintf(stderr, "test_gridsample failed a.dims=%d a=(%d %d %d %d) grid.dims=%d grid=(%d %d %d %d) sample_type=%d padding_mode=%d align_corner=%d permute_fusion=%d",
                a.dims, a.w, a.h, a.d, a.c, grid.dims, grid.w, grid.h, grid.d, grid.c,
                sample_type, padding_mode, align_corner, permute_fusion);
    }

    return ret;
}

static int test_gridsample_0()
{
    return 0
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 1, 1, 0, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 1, 1, 1, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 1, 2, 0, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 1, 2, 1, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 1, 3, 0, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 1, 3, 1, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 2, 1, 0, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 2, 1, 1, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 2, 2, 0, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 2, 2, 1, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 2, 3, 0, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 2, 3, 1, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 3, 1, 0, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 3, 1, 1, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 3, 2, 0, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 3, 2, 1, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 3, 3, 0, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(2, 11, 13), 3, 3, 1, 0)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 1, 1, 0, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 1, 1, 1, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 1, 2, 0, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 1, 2, 1, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 1, 3, 0, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 1, 3, 1, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 2, 1, 0, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 2, 1, 1, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 2, 2, 0, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 2, 2, 1, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 2, 3, 0, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 2, 3, 1, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 3, 1, 0, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 3, 1, 1, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 3, 2, 0, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 3, 2, 1, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 3, 3, 0, 1)
           || test_gridsample(RandomMat(3, 7, 1), RandomMat(11, 13, 2), 3, 3, 1, 1);
}

static int test_gridsample_1()
{
    return 0
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 1, 1, 0, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 1, 1, 1, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 1, 2, 0, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 1, 2, 1, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 1, 3, 0, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 1, 3, 1, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 2, 1, 0, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 2, 1, 1, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 2, 2, 0, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 2, 2, 1, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 2, 3, 0, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 2, 3, 1, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 3, 1, 0, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 3, 1, 1, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 3, 2, 0, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 3, 2, 1, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 3, 3, 0, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(2, 24, 16), 3, 3, 1, 0)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 1, 1, 0, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 1, 1, 1, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 1, 2, 0, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 1, 2, 1, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 1, 3, 0, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 1, 3, 1, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 2, 1, 0, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 2, 1, 1, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 2, 2, 0, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 2, 2, 1, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 2, 3, 0, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 2, 3, 1, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 3, 1, 0, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 3, 1, 1, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 3, 2, 0, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 3, 2, 1, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 3, 3, 0, 1)
           || test_gridsample(RandomMat(8, 12, 16), RandomMat(24, 16, 2), 3, 3, 1, 1);
}

static int test_gridsample_2()
{
    return 0
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(3, 17, 11, 13), 1, 1, 0, 0)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(3, 17, 11, 13), 1, 1, 1, 0)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(3, 17, 11, 13), 1, 2, 0, 0)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(3, 17, 11, 13), 1, 2, 1, 0)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(3, 17, 11, 13), 1, 3, 0, 0)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(3, 17, 11, 13), 1, 3, 1, 0)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(3, 17, 11, 13), 2, 1, 0, 0)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(3, 17, 11, 13), 2, 1, 1, 0)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(3, 17, 11, 13), 2, 2, 0, 0)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(3, 17, 11, 13), 2, 2, 1, 0)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(3, 17, 11, 13), 2, 3, 0, 0)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(3, 17, 11, 13), 2, 3, 1, 0)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(17, 11, 13, 3), 1, 1, 0, 1)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(17, 11, 13, 3), 1, 1, 1, 1)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(17, 11, 13, 3), 1, 2, 0, 1)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(17, 11, 13, 3), 1, 2, 1, 1)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(17, 11, 13, 3), 1, 3, 0, 1)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(17, 11, 13, 3), 1, 3, 1, 1)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(17, 11, 13, 3), 2, 1, 0, 1)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(17, 11, 13, 3), 2, 1, 1, 1)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(17, 11, 13, 3), 2, 2, 0, 1)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(17, 11, 13, 3), 2, 2, 1, 1)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(17, 11, 13, 3), 2, 3, 0, 1)
           || test_gridsample(RandomMat(5, 7, 11, 13), RandomMat(17, 11, 13, 3), 2, 3, 1, 1);
}

static int test_gridsample_3()
{
    return 0
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(3, 11, 12, 16), 1, 1, 0, 0)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(3, 11, 12, 16), 1, 1, 1, 0)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(3, 11, 12, 16), 1, 2, 0, 0)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(3, 11, 12, 16), 1, 2, 1, 0)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(3, 11, 12, 16), 1, 3, 0, 0)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(3, 11, 12, 16), 1, 3, 1, 0)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(3, 11, 12, 16), 2, 1, 0, 0)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(3, 11, 12, 16), 2, 1, 1, 0)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(3, 11, 12, 16), 2, 2, 0, 0)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(3, 11, 12, 16), 2, 2, 1, 0)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(3, 11, 12, 16), 2, 3, 0, 0)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(3, 11, 12, 16), 2, 3, 1, 0)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(11, 12, 16, 3), 1, 1, 0, 1)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(11, 12, 16, 3), 1, 1, 1, 1)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(11, 12, 16, 3), 1, 2, 0, 1)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(11, 12, 16, 3), 1, 2, 1, 1)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(11, 12, 16, 3), 1, 3, 0, 1)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(11, 12, 16, 3), 1, 3, 1, 1)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(11, 12, 16, 3), 2, 1, 0, 1)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(11, 12, 16, 3), 2, 1, 1, 1)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(11, 12, 16, 3), 2, 2, 0, 1)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(11, 12, 16, 3), 2, 2, 1, 1)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(11, 12, 16, 3), 2, 3, 0, 1)
           || test_gridsample(RandomMat(16, 12, 11, 16), RandomMat(11, 12, 16, 3), 2, 3, 1, 1);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_gridsample_nearest_even()
           || test_gridsample_3d_nearest_even()
           || test_gridsample_0()
           || test_gridsample_1()
           || test_gridsample_2()
           || test_gridsample_3();
}
