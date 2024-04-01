// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "testutil.h"

static int test_gru(const ncnn::Mat& a, int outch, int direction)
{
    int input_size = a.w;
    int num_directions = direction == 2 ? 2 : 1;

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, outch * input_size * 3 * num_directions);
    pd.set(2, direction);

    std::vector<ncnn::Mat> weights(3);
    weights[0] = RandomMat(outch * input_size * 3 * num_directions);
    weights[1] = RandomMat(outch * 4 * num_directions);
    weights[2] = RandomMat(outch * outch * 3 * num_directions);

    int ret = test_layer("GRU", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_gru failed a.dims=%d a=(%d %d %d) outch=%d, direction = %d \n", a.dims, a.w, a.h, a.c, outch, direction);
    }

    return ret;
}

int test_gru_layer_with_hidden(const ncnn::Mat& a, int outch, int direction)
{
    int input_size = a.w;
    int num_directions = direction == 2 ? 2 : 1;

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, outch * input_size * 3 * num_directions);
    pd.set(2, direction);

    std::vector<ncnn::Mat> weights(3);
    weights[0] = RandomMat(outch * input_size * 3 * num_directions);
    weights[1] = RandomMat(outch * 4 * num_directions);
    weights[2] = RandomMat(outch * outch * 3 * num_directions);

    // initial hidden state
    ncnn::Mat hidden = RandomMat(outch, num_directions);

    std::vector<ncnn::Mat> as(2);
    as[0] = a;
    as[1] = hidden;

    int ret = test_layer("GRU", pd, weights, as, 2);
    if (ret != 0)
    {
        fprintf(stderr, "test_gru_layer_with_hidden failed a.dims=%d a=(%d %d %d) outch=%d, direction = %d \n", a.dims, a.w, a.h, a.c, outch, direction);
    }

    return ret;
}

int test_gru_layer_with_hidden_input(const ncnn::Mat& a, int outch, int direction)
{
    int input_size = a.w;
    int num_directions = direction == 2 ? 2 : 1;

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, outch * input_size * 3 * num_directions);
    pd.set(2, direction);

    std::vector<ncnn::Mat> weights(3);
    weights[0] = RandomMat(outch * input_size * 3 * num_directions);
    weights[1] = RandomMat(outch * 4 * num_directions);
    weights[2] = RandomMat(outch * outch * 3 * num_directions);

    // initial hidden state
    ncnn::Mat hidden = RandomMat(outch, num_directions);

    std::vector<ncnn::Mat> as(2);
    as[0] = a;
    as[1] = hidden;

    int ret = test_layer("GRU", pd, weights, as, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_gru_layer_with_hidden_input failed a.dims=%d a=(%d %d %d) outch=%d, direction = %d \n", a.dims, a.w, a.h, a.c, outch, direction);
    }

    return ret;
}

int test_gru_layer_with_hidden_output(const ncnn::Mat& a, int outch, int direction)
{
    int input_size = a.w;
    int num_directions = direction == 2 ? 2 : 1;

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, outch * input_size * 3 * num_directions);
    pd.set(2, direction);

    std::vector<ncnn::Mat> weights(3);
    weights[0] = RandomMat(outch * input_size * 3 * num_directions);
    weights[1] = RandomMat(outch * 4 * num_directions);
    weights[2] = RandomMat(outch * outch * 3 * num_directions);

    std::vector<ncnn::Mat> as(1);
    as[0] = a;

    int ret = test_layer("GRU", pd, weights, as, 2);
    if (ret != 0)
    {
        fprintf(stderr, "test_gru_layer_with_hidden_output failed a.dims=%d a=(%d %d %d) outch=%d, direction = %d \n", a.dims, a.w, a.h, a.c, outch, direction);
    }

    return ret;
}

static int test_gru_0()
{
    return 0
           || test_gru(RandomMat(4, 1), 2, 2)
           || test_gru(RandomMat(8, 2), 2, 2)
           || test_gru(RandomMat(16, 8), 7, 2)
           || test_gru(RandomMat(17, 8), 8, 2)
           || test_gru(RandomMat(19, 15), 8, 2)
           || test_gru(RandomMat(5, 16), 16, 2)
           || test_gru(RandomMat(3, 16), 8, 2)
           || test_gru(RandomMat(8, 16), 16, 2)
           || test_gru(RandomMat(2, 5), 17, 2);
}

static int test_gru_1()
{
    return 0
           || test_gru_layer_with_hidden(RandomMat(4, 4), 1, 2)
           || test_gru_layer_with_hidden(RandomMat(8, 2), 2, 2)
           || test_gru_layer_with_hidden(RandomMat(16, 8), 7, 2)
           || test_gru_layer_with_hidden(RandomMat(17, 8), 8, 2)
           || test_gru_layer_with_hidden(RandomMat(19, 15), 8, 2)
           || test_gru_layer_with_hidden(RandomMat(5, 16), 16, 2)
           || test_gru_layer_with_hidden(RandomMat(3, 16), 8, 2)
           || test_gru_layer_with_hidden(RandomMat(2, 5), 99, 2)
           || test_gru_layer_with_hidden(RandomMat(4, 4), 1, 1)
           || test_gru_layer_with_hidden(RandomMat(8, 2), 2, 1)
           || test_gru_layer_with_hidden(RandomMat(16, 8), 7, 1)
           || test_gru_layer_with_hidden(RandomMat(17, 8), 8, 1)
           || test_gru_layer_with_hidden(RandomMat(19, 15), 8, 1)
           || test_gru_layer_with_hidden(RandomMat(5, 16), 16, 1)
           || test_gru_layer_with_hidden(RandomMat(3, 16), 8, 1)
           || test_gru_layer_with_hidden(RandomMat(2, 5), 99, 1)
           || test_gru_layer_with_hidden(RandomMat(4, 2), 1, 0)
           || test_gru_layer_with_hidden(RandomMat(8, 2), 2, 0)
           || test_gru_layer_with_hidden(RandomMat(16, 8), 7, 0)
           || test_gru_layer_with_hidden(RandomMat(17, 8), 8, 0)
           || test_gru_layer_with_hidden(RandomMat(19, 15), 8, 0)
           || test_gru_layer_with_hidden(RandomMat(5, 16), 16, 0)
           || test_gru_layer_with_hidden(RandomMat(3, 16), 8, 0)
           || test_gru_layer_with_hidden(RandomMat(2, 5), 17, 0)

           || test_gru_layer_with_hidden_input(RandomMat(4, 4), 1, 2)
           || test_gru_layer_with_hidden_input(RandomMat(8, 2), 2, 2)
           || test_gru_layer_with_hidden_input(RandomMat(16, 8), 7, 2)
           || test_gru_layer_with_hidden_input(RandomMat(17, 8), 8, 2)
           || test_gru_layer_with_hidden_input(RandomMat(19, 15), 8, 2)
           || test_gru_layer_with_hidden_input(RandomMat(5, 16), 16, 2)
           || test_gru_layer_with_hidden_input(RandomMat(3, 16), 8, 2)
           || test_gru_layer_with_hidden_input(RandomMat(2, 5), 99, 2)
           || test_gru_layer_with_hidden_input(RandomMat(4, 4), 1, 1)
           || test_gru_layer_with_hidden_input(RandomMat(8, 2), 2, 1)
           || test_gru_layer_with_hidden_input(RandomMat(16, 8), 7, 1)
           || test_gru_layer_with_hidden_input(RandomMat(17, 8), 8, 1)
           || test_gru_layer_with_hidden_input(RandomMat(19, 15), 8, 1)
           || test_gru_layer_with_hidden_input(RandomMat(5, 16), 16, 1)
           || test_gru_layer_with_hidden_input(RandomMat(3, 16), 8, 1)
           || test_gru_layer_with_hidden_input(RandomMat(2, 5), 99, 1)
           || test_gru_layer_with_hidden_input(RandomMat(4, 2), 1, 0)
           || test_gru_layer_with_hidden_input(RandomMat(8, 2), 2, 0)
           || test_gru_layer_with_hidden_input(RandomMat(16, 8), 7, 0)
           || test_gru_layer_with_hidden_input(RandomMat(17, 8), 8, 0)
           || test_gru_layer_with_hidden_input(RandomMat(19, 15), 8, 0)
           || test_gru_layer_with_hidden_input(RandomMat(5, 16), 16, 0)
           || test_gru_layer_with_hidden_input(RandomMat(3, 16), 8, 0)
           || test_gru_layer_with_hidden_input(RandomMat(2, 5), 17, 0)

           || test_gru_layer_with_hidden_output(RandomMat(4, 4), 1, 2)
           || test_gru_layer_with_hidden_output(RandomMat(8, 2), 2, 2)
           || test_gru_layer_with_hidden_output(RandomMat(16, 8), 7, 2)
           || test_gru_layer_with_hidden_output(RandomMat(17, 8), 8, 2)
           || test_gru_layer_with_hidden_output(RandomMat(19, 15), 8, 2)
           || test_gru_layer_with_hidden_output(RandomMat(5, 16), 16, 2)
           || test_gru_layer_with_hidden_output(RandomMat(3, 16), 8, 2)
           || test_gru_layer_with_hidden_output(RandomMat(2, 5), 99, 2)
           || test_gru_layer_with_hidden_output(RandomMat(4, 4), 1, 1)
           || test_gru_layer_with_hidden_output(RandomMat(8, 2), 2, 1)
           || test_gru_layer_with_hidden_output(RandomMat(16, 8), 7, 1)
           || test_gru_layer_with_hidden_output(RandomMat(17, 8), 8, 1)
           || test_gru_layer_with_hidden_output(RandomMat(19, 15), 8, 1)
           || test_gru_layer_with_hidden_output(RandomMat(5, 16), 16, 1)
           || test_gru_layer_with_hidden_output(RandomMat(3, 16), 8, 1)
           || test_gru_layer_with_hidden_output(RandomMat(2, 5), 99, 1)
           || test_gru_layer_with_hidden_output(RandomMat(4, 2), 1, 0)
           || test_gru_layer_with_hidden_output(RandomMat(8, 2), 2, 0)
           || test_gru_layer_with_hidden_output(RandomMat(16, 8), 7, 0)
           || test_gru_layer_with_hidden_output(RandomMat(17, 8), 8, 0)
           || test_gru_layer_with_hidden_output(RandomMat(19, 15), 8, 0)
           || test_gru_layer_with_hidden_output(RandomMat(5, 16), 16, 0)
           || test_gru_layer_with_hidden_output(RandomMat(3, 16), 8, 0)
           || test_gru_layer_with_hidden_output(RandomMat(2, 5), 17, 0);
}

static int test_gru_2()
{
    return 0
           || test_gru(RandomMat(4, 1), 1, 0)
           || test_gru(RandomMat(8, 2), 2, 0)
           || test_gru(RandomMat(16, 8), 7, 0)
           || test_gru(RandomMat(17, 8), 8, 0)
           || test_gru(RandomMat(19, 15), 8, 0)
           || test_gru(RandomMat(5, 16), 16, 0)
           || test_gru(RandomMat(3, 16), 8, 0)
           || test_gru(RandomMat(8, 16), 16, 0)
           || test_gru(RandomMat(2, 5), 17, 0);
}

static int test_gru_3()
{
    return 0
           || test_gru(RandomMat(4, 1), 1, 1)
           || test_gru(RandomMat(8, 2), 2, 1)
           || test_gru(RandomMat(16, 8), 7, 1)
           || test_gru(RandomMat(17, 8), 8, 1)
           || test_gru(RandomMat(19, 15), 8, 1)
           || test_gru(RandomMat(5, 16), 16, 1)
           || test_gru(RandomMat(3, 16), 8, 1)
           || test_gru(RandomMat(8, 16), 16, 1)
           || test_gru(RandomMat(2, 5), 17, 1);
}

int main()
{
    SRAND(7767517);
    return test_gru_0() || test_gru_1() || test_gru_2() || test_gru_3();
}
