// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

static int test_lstm(const ncnn::Mat& a, int outch, int direction, int hidden_size = 0)
{
    int input_size = a.w;
    int num_directions = direction == 2 ? 2 : 1;
    if (hidden_size == 0)
        hidden_size = outch;

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, hidden_size * input_size * 4 * num_directions);
    pd.set(2, direction);
    pd.set(3, hidden_size);

    std::vector<ncnn::Mat> weights(hidden_size == 0 ? 3 : 4);
    weights[0] = RandomMat(hidden_size * input_size * 4 * num_directions);
    weights[1] = RandomMat(hidden_size * 4 * num_directions);
    weights[2] = RandomMat(outch * hidden_size * 4 * num_directions);
    if (hidden_size)
    {
        weights[3] = RandomMat(hidden_size * outch * num_directions);
    }

    int ret = test_layer("LSTM", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_lstm failed a.dims=%d a=(%d %d %d) outch=%d direction=%d hidden_size=%d\n", a.dims, a.w, a.h, a.c, outch, direction, hidden_size);
    }

    return ret;
}

int test_lstm_layer_with_hidden(const ncnn::Mat& a, int outch, int direction, int hidden_size = 0)
{
    int input_size = a.w;
    int num_directions = direction == 2 ? 2 : 1;
    if (hidden_size == 0)
        hidden_size = outch;

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, hidden_size * input_size * 4 * num_directions);
    pd.set(2, direction);
    pd.set(3, hidden_size);

    std::vector<ncnn::Mat> weights(hidden_size == 0 ? 3 : 4);
    weights[0] = RandomMat(hidden_size * input_size * 4 * num_directions);
    weights[1] = RandomMat(hidden_size * 4 * num_directions);
    weights[2] = RandomMat(outch * hidden_size * 4 * num_directions);
    if (hidden_size)
    {
        weights[3] = RandomMat(hidden_size * outch * num_directions);
    }

    // initial hidden state
    ncnn::Mat hidden = RandomMat(outch, num_directions);

    // initial cell state
    ncnn::Mat cell = RandomMat(hidden_size, num_directions);

    std::vector<ncnn::Mat> as(3);
    as[0] = a;
    as[1] = hidden;
    as[2] = cell;

    int ret = test_layer("LSTM", pd, weights, as, 3);
    if (ret != 0)
    {
        fprintf(stderr, "test_lstm_layer_with_hidden failed a.dims=%d a=(%d %d %d) outch=%d direction=%d hidden_size=%d\n", a.dims, a.w, a.h, a.c, outch, direction, hidden_size);
    }

    return ret;
}

int test_lstm_layer_with_hidden_input(const ncnn::Mat& a, int outch, int direction, int hidden_size = 0)
{
    int input_size = a.w;
    int num_directions = direction == 2 ? 2 : 1;
    if (hidden_size == 0)
        hidden_size = outch;

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, hidden_size * input_size * 4 * num_directions);
    pd.set(2, direction);
    pd.set(3, hidden_size);

    std::vector<ncnn::Mat> weights(hidden_size == 0 ? 3 : 4);
    weights[0] = RandomMat(hidden_size * input_size * 4 * num_directions);
    weights[1] = RandomMat(hidden_size * 4 * num_directions);
    weights[2] = RandomMat(outch * hidden_size * 4 * num_directions);
    if (hidden_size)
    {
        weights[3] = RandomMat(hidden_size * outch * num_directions);
    }

    // initial hidden state
    ncnn::Mat hidden = RandomMat(outch, num_directions);

    // initial cell state
    ncnn::Mat cell = RandomMat(hidden_size, num_directions);

    std::vector<ncnn::Mat> as(3);
    as[0] = a;
    as[1] = hidden;
    as[2] = cell;

    int ret = test_layer("LSTM", pd, weights, as, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_lstm_layer_with_hidden_input failed a.dims=%d a=(%d %d %d) outch=%d direction=%d hidden_size=%d\n", a.dims, a.w, a.h, a.c, outch, direction, hidden_size);
    }

    return ret;
}

int test_lstm_layer_with_hidden_output(const ncnn::Mat& a, int outch, int direction, int hidden_size = 0)
{
    int input_size = a.w;
    int num_directions = direction == 2 ? 2 : 1;
    if (hidden_size == 0)
        hidden_size = outch;

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, hidden_size * input_size * 4 * num_directions);
    pd.set(2, direction);
    pd.set(3, hidden_size);

    std::vector<ncnn::Mat> weights(hidden_size == 0 ? 3 : 4);
    weights[0] = RandomMat(hidden_size * input_size * 4 * num_directions);
    weights[1] = RandomMat(hidden_size * 4 * num_directions);
    weights[2] = RandomMat(outch * hidden_size * 4 * num_directions);
    if (hidden_size)
    {
        weights[3] = RandomMat(hidden_size * outch * num_directions);
    }

    std::vector<ncnn::Mat> as(1);
    as[0] = a;

    int ret = test_layer("LSTM", pd, weights, as, 3);
    if (ret != 0)
    {
        fprintf(stderr, "test_lstm_layer_with_hidden_output failed a.dims=%d a=(%d %d %d) outch=%d direction=%d hidden_size=%d\n", a.dims, a.w, a.h, a.c, outch, direction, hidden_size);
    }

    return ret;
}

static int test_lstm_0()
{
    return 0
           || test_lstm(RandomMat(4, 1), 2, 2)
           || test_lstm(RandomMat(8, 2), 2, 2)
           || test_lstm(RandomMat(16, 8), 7, 2)
           || test_lstm(RandomMat(17, 8), 8, 2)
           || test_lstm(RandomMat(19, 15), 8, 2)
           || test_lstm(RandomMat(5, 16), 16, 2)
           || test_lstm(RandomMat(3, 16), 8, 2)
           || test_lstm(RandomMat(8, 16), 16, 2)
           || test_lstm(RandomMat(2, 5), 17, 2, 15);
}

static int test_lstm_1()
{
    return 0
           || test_lstm_layer_with_hidden(RandomMat(4, 4), 1, 2)
           || test_lstm_layer_with_hidden(RandomMat(8, 2), 2, 2)
           || test_lstm_layer_with_hidden(RandomMat(16, 8), 7, 2)
           || test_lstm_layer_with_hidden(RandomMat(17, 8), 8, 2)
           || test_lstm_layer_with_hidden(RandomMat(19, 15), 8, 2)
           || test_lstm_layer_with_hidden(RandomMat(5, 16), 16, 2)
           || test_lstm_layer_with_hidden(RandomMat(3, 16), 8, 2)
           || test_lstm_layer_with_hidden(RandomMat(2, 5), 99, 2, 33)
           || test_lstm_layer_with_hidden(RandomMat(4, 4), 1, 1)
           || test_lstm_layer_with_hidden(RandomMat(8, 2), 2, 1)
           || test_lstm_layer_with_hidden(RandomMat(16, 8), 7, 1)
           || test_lstm_layer_with_hidden(RandomMat(17, 8), 8, 1)
           || test_lstm_layer_with_hidden(RandomMat(19, 15), 8, 1)
           || test_lstm_layer_with_hidden(RandomMat(5, 16), 16, 1)
           || test_lstm_layer_with_hidden(RandomMat(3, 16), 8, 1)
           || test_lstm_layer_with_hidden(RandomMat(2, 5), 99, 1, 33)
           || test_lstm_layer_with_hidden(RandomMat(4, 2), 1, 0)
           || test_lstm_layer_with_hidden(RandomMat(8, 2), 2, 0)
           || test_lstm_layer_with_hidden(RandomMat(16, 8), 7, 0)
           || test_lstm_layer_with_hidden(RandomMat(17, 8), 8, 0)
           || test_lstm_layer_with_hidden(RandomMat(19, 15), 8, 0)
           || test_lstm_layer_with_hidden(RandomMat(5, 16), 16, 0)
           || test_lstm_layer_with_hidden(RandomMat(3, 16), 8, 0)
           || test_lstm_layer_with_hidden(RandomMat(2, 5), 17, 0, 15)

           || test_lstm_layer_with_hidden_input(RandomMat(4, 4), 1, 2)
           || test_lstm_layer_with_hidden_input(RandomMat(8, 2), 2, 2)
           || test_lstm_layer_with_hidden_input(RandomMat(16, 8), 7, 2)
           || test_lstm_layer_with_hidden_input(RandomMat(17, 8), 8, 2)
           || test_lstm_layer_with_hidden_input(RandomMat(19, 15), 8, 2)
           || test_lstm_layer_with_hidden_input(RandomMat(5, 16), 16, 2)
           || test_lstm_layer_with_hidden_input(RandomMat(3, 16), 8, 2)
           || test_lstm_layer_with_hidden_input(RandomMat(2, 5), 99, 2, 33)
           || test_lstm_layer_with_hidden_input(RandomMat(4, 4), 1, 1)
           || test_lstm_layer_with_hidden_input(RandomMat(8, 2), 2, 1)
           || test_lstm_layer_with_hidden_input(RandomMat(16, 8), 7, 1)
           || test_lstm_layer_with_hidden_input(RandomMat(17, 8), 8, 1)
           || test_lstm_layer_with_hidden_input(RandomMat(19, 15), 8, 1)
           || test_lstm_layer_with_hidden_input(RandomMat(5, 16), 16, 1)
           || test_lstm_layer_with_hidden_input(RandomMat(3, 16), 8, 1)
           || test_lstm_layer_with_hidden_input(RandomMat(2, 5), 99, 1, 33)
           || test_lstm_layer_with_hidden_input(RandomMat(4, 2), 1, 0)
           || test_lstm_layer_with_hidden_input(RandomMat(8, 2), 2, 0)
           || test_lstm_layer_with_hidden_input(RandomMat(16, 8), 7, 0)
           || test_lstm_layer_with_hidden_input(RandomMat(17, 8), 8, 0)
           || test_lstm_layer_with_hidden_input(RandomMat(19, 15), 8, 0)
           || test_lstm_layer_with_hidden_input(RandomMat(5, 16), 16, 0)
           || test_lstm_layer_with_hidden_input(RandomMat(3, 16), 8, 0)
           || test_lstm_layer_with_hidden_input(RandomMat(2, 5), 17, 0, 15)

           || test_lstm_layer_with_hidden_output(RandomMat(4, 4), 1, 2)
           || test_lstm_layer_with_hidden_output(RandomMat(8, 2), 2, 2)
           || test_lstm_layer_with_hidden_output(RandomMat(16, 8), 7, 2)
           || test_lstm_layer_with_hidden_output(RandomMat(17, 8), 8, 2)
           || test_lstm_layer_with_hidden_output(RandomMat(19, 15), 8, 2)
           || test_lstm_layer_with_hidden_output(RandomMat(5, 16), 16, 2)
           || test_lstm_layer_with_hidden_output(RandomMat(3, 16), 8, 2)
           || test_lstm_layer_with_hidden_output(RandomMat(2, 5), 99, 2, 33)
           || test_lstm_layer_with_hidden_output(RandomMat(4, 4), 1, 1)
           || test_lstm_layer_with_hidden_output(RandomMat(8, 2), 2, 1)
           || test_lstm_layer_with_hidden_output(RandomMat(16, 8), 7, 1)
           || test_lstm_layer_with_hidden_output(RandomMat(17, 8), 8, 1)
           || test_lstm_layer_with_hidden_output(RandomMat(19, 15), 8, 1)
           || test_lstm_layer_with_hidden_output(RandomMat(5, 16), 16, 1)
           || test_lstm_layer_with_hidden_output(RandomMat(3, 16), 8, 1)
           || test_lstm_layer_with_hidden_output(RandomMat(2, 5), 99, 1, 33)
           || test_lstm_layer_with_hidden_output(RandomMat(4, 2), 1, 0)
           || test_lstm_layer_with_hidden_output(RandomMat(8, 2), 2, 0)
           || test_lstm_layer_with_hidden_output(RandomMat(16, 8), 7, 0)
           || test_lstm_layer_with_hidden_output(RandomMat(17, 8), 8, 0)
           || test_lstm_layer_with_hidden_output(RandomMat(19, 15), 8, 0)
           || test_lstm_layer_with_hidden_output(RandomMat(5, 16), 16, 0)
           || test_lstm_layer_with_hidden_output(RandomMat(3, 16), 8, 0)
           || test_lstm_layer_with_hidden_output(RandomMat(2, 5), 17, 0, 15);
}

static int test_lstm_2()
{
    return 0
           || test_lstm(RandomMat(4, 1), 1, 0)
           || test_lstm(RandomMat(8, 2), 2, 0)
           || test_lstm(RandomMat(16, 8), 7, 0)
           || test_lstm(RandomMat(17, 8), 8, 0)
           || test_lstm(RandomMat(19, 15), 8, 0)
           || test_lstm(RandomMat(5, 16), 16, 0)
           || test_lstm(RandomMat(3, 16), 8, 0)
           || test_lstm(RandomMat(8, 16), 16, 0)
           || test_lstm(RandomMat(2, 5), 17, 0, 15);
}
static int test_lstm_3()
{
    return 0
           || test_lstm(RandomMat(4, 1), 1, 1)
           || test_lstm(RandomMat(8, 2), 2, 1)
           || test_lstm(RandomMat(16, 8), 7, 1)
           || test_lstm(RandomMat(17, 8), 8, 1)
           || test_lstm(RandomMat(19, 15), 8, 1)
           || test_lstm(RandomMat(5, 16), 16, 1)
           || test_lstm(RandomMat(3, 16), 8, 1)
           || test_lstm(RandomMat(8, 16), 16, 1)
           || test_lstm(RandomMat(2, 5), 17, 1, 15);
}

int main()
{
    SRAND(7767517);
    return 0 || test_lstm_0() || test_lstm_1() || test_lstm_2() || test_lstm_3();
}
