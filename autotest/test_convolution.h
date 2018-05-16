#pragma once
#include "gtest/gtest.h"
#include "layer/convolution.h"
using namespace ncnn;

/*
forward - pass:
    [0,1,2,3,4,
    1,2,3,4,5,          [1,1,1,           [ 9.5,    18.5,
    2,3,4,5,6,  *  0.5*  1,1,1,  + 0.5 =
    3,4,5,6,7,           1,1,1]            18.5,    27.5]
    4,5,6,7,8]
*/

TEST(convolution, forward)
{
    // layer params
    Convolution convolution_layer;
    convolution_layer.num_output = 1;
    convolution_layer.kernel_size = 3;
    convolution_layer.dilation = 1;
    convolution_layer.stride = 2;
    convolution_layer.pad = 0;
    convolution_layer.bias_term = 1;
    convolution_layer.weight_data_size = 9;

    // input & output
    float_t in[] = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
        4.0f, 5.0f, 6.0f, 7.0f, 8.0f
    };

    float_t expected_out[] = {
        9.5f, 18.5f,
        18.5f, 27.5f
    };


    // weights & bias
    float_t w[] = {
        0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f
    };

    float_t b[] = {
        0.5f
    };

    // forward
    Mat mat_in(5, 5, 1, in);
    Mat mat_out;

    convolution_layer.bias_data.data = b;
    convolution_layer.weight_data.data = w;
    convolution_layer.forward(mat_in, mat_out);

    // check expect
    EXPECT_EQ(mat_out.w, 2);
    EXPECT_EQ(mat_out.h, 2);
    EXPECT_EQ(mat_out.c, 1);
    for (int i = 0; i < _countof(expected_out); ++i)
    {
        EXPECT_NEAR(mat_out[i], expected_out[i], 1E-5);
    }

}
