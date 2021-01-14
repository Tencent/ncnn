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

#include "layer/lstm.h"
#include "testutil.h"

static int test_lstm(const ncnn::Mat& a, int outch, int direction)
{
    int input_size = a.w;
    int num_directions = direction == 2 ? 2 : 1;

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, outch * input_size * 4 * num_directions);
    pd.set(2, direction);

    std::vector<ncnn::Mat> weights(3);
    weights[0] = RandomMat(outch * input_size * 4 * num_directions);
    weights[1] = RandomMat(outch * 4 * num_directions);
    weights[2] = RandomMat(outch * outch * 4 * num_directions);

    int ret = test_layer<ncnn::LSTM>("LSTM", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_lstm failed a.dims=%d a=(%d %d %d) outch=%d, direction = %d \n", a.dims, a.w, a.h, a.c, outch, direction);
    }

    return ret;
}

int test_lstm_layer_with_hidden(const ncnn::Mat& a, int outch, int direction)
{
    int input_size = a.w;

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, outch * input_size * 4);
    pd.set(2, direction);

    std::vector<ncnn::Mat> weights(3);
    weights[0] = RandomMat(outch * input_size * 4);
    weights[1] = RandomMat(outch * 4);
    weights[2] = RandomMat(outch * outch * 4);

    // initial hidden state
    ncnn::Mat hidden = RandomMat(outch);

    // initial cell state
    ncnn::Mat cell = RandomMat(outch);

    std::vector<ncnn::Mat> as(3);
    as[0] = a;
    as[1] = hidden;
    as[2] = cell;

    int ret = test_layer<ncnn::LSTM>("LSTM", pd, weights, as, 3);
    if (ret != 0)
    {
        fprintf(stderr, "test_lstm_layer_with_hidden failed a.dims=%d a=(%d %d %d) outch=%d, direction = %d \n", a.dims, a.w, a.h, a.c, outch, direction);
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
           || test_lstm(RandomMat(2, 5), 17, 2);
}

static int test_lstm_1()
{
    return 0
           || test_lstm_layer_with_hidden(RandomMat(4, 4), 1, 1)
           || test_lstm_layer_with_hidden(RandomMat(8, 2), 2, 1)
           || test_lstm_layer_with_hidden(RandomMat(16, 8), 7, 1)
           || test_lstm_layer_with_hidden(RandomMat(17, 8), 8, 1)
           || test_lstm_layer_with_hidden(RandomMat(19, 15), 8, 1)
           || test_lstm_layer_with_hidden(RandomMat(5, 16), 16, 1)
           || test_lstm_layer_with_hidden(RandomMat(3, 16), 8, 1)
           || test_lstm_layer_with_hidden(RandomMat(2, 5), 99, 1)
           || test_lstm_layer_with_hidden(RandomMat(4, 2), 1, 0)
           || test_lstm_layer_with_hidden(RandomMat(8, 2), 2, 0)
           || test_lstm_layer_with_hidden(RandomMat(16, 8), 7, 0)
           || test_lstm_layer_with_hidden(RandomMat(17, 8), 8, 0)
           || test_lstm_layer_with_hidden(RandomMat(19, 15), 8, 0)
           || test_lstm_layer_with_hidden(RandomMat(5, 16), 16, 0)
           || test_lstm_layer_with_hidden(RandomMat(3, 16), 8, 0)
           || test_lstm_layer_with_hidden(RandomMat(2, 5), 17, 0);
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
           || test_lstm(RandomMat(2, 5), 17, 0);
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
           || test_lstm(RandomMat(2, 5), 17, 1);
}

int main()
{
    SRAND(7767517);
    return 0 || test_lstm_0() || test_lstm_1() || test_lstm_2() || test_lstm_3();
}
