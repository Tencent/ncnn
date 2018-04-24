#pragma once
#include "gtest/gtest.h"
#include "layer/innerproduct.h"

/*
forward - pass:
[0,1,2,3] * [1,1,1,1  + [0.5, = [6.5,
             1,1,1,1]    0.5]    6.5]
*/

TEST(innerproduct, forward)
{
    // layer params
    InnerProduct inner_product_layer;
    inner_product_layer.num_output = 2; // W
    inner_product_layer.bias_term = 1;  // bias
    inner_product_layer.weight_data_size = 3; // W + bias


    // input & output
    float_t in[] = {
        0.0f, 1.0f, 2.0f, 3.0f
    };

    float_t expected_out[] = {
        6.5, 6.5  /// 0+1+2+3+0.5
    };


    // weights & bias
    float_t w[] = {
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f
    };

    float_t b[] = {
        0.5f, 0.5f
    };

    // forward
    Mat mat_in(4, in);
    Mat mat_out;

    inner_product_layer.bias_data.data = b;
    inner_product_layer.weight_data.data = w;
    inner_product_layer.forward(mat_in, mat_out);

    // check expect
    EXPECT_EQ(mat_out.c, 2);
    for (int i = 0; i < _countof(expected_out); ++i)
    {
        float output_value = *(mat_out.data + mat_out.cstep * i);
        EXPECT_NEAR(output_value, expected_out[i], 1E-5);
    }
}
