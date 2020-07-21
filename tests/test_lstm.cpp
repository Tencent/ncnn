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
    int input_size = a.w * a.h * a.c;
    int num_directions = direction == 2 ? 2 : 1;

    ncnn::ParamDict pd;
    pd.set(0, outch); // num_output
    pd.set(1, outch * input_size * 4 * num_directions);
    pd.set(2, direction); // bias_term

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

int test_lstm_layer(const ncnn::Mat& a, int outch, int direction, float epsilon = 0.01)
{
    int input_size = a.w * a.h * a.c;
    ncnn::ParamDict pd;
    pd.set(0, outch); // num_output
    pd.set(1, outch * input_size * 4);
    pd.set(2, direction); // bias_term
    int num_directions = direction == 2 ? 2 : 1;

    std::vector<ncnn::Mat> weights(3);
    weights[0] = RandomMat(outch * input_size * 4 * num_directions);
    weights[1] = RandomMat(outch * 4 * num_directions);
    weights[2] = RandomMat(outch * outch * 4 * num_directions);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_int8_inference = false;

    ncnn::LSTM* op = (ncnn::LSTM*)ncnn::create_layer(ncnn::layer_to_index("LSTM"));

    if (!op->support_vulkan) opt.use_vulkan_compute = false;
    if (!op->support_packing) opt.use_packing_layout = false;
    if (!op->support_bf16_storage) opt.use_bf16_storage = false;
    if (!op->support_image_storage) opt.use_image_storage = false;

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    op->create_pipeline(opt);

    ncnn::Mat b;
    op->LSTM::forward(a, b, opt);

    std::vector<ncnn::Mat> _c1(3);
    std::vector<ncnn::Mat> _c2(3);
    std::vector<ncnn::Mat> a1(3);
    std::vector<ncnn::Mat> a2(3);
    if (direction == 0)
    {
        a1[0] = a.row_range(0, a.h / 2).clone();
        a2[0] = a.row_range(a.h / 2, a.h - a.h / 2).clone();
    }
    else
    {
        a2[0] = a.row_range(0, a.h / 2).clone();
        a1[0] = a.row_range(a.h / 2, a.h - a.h / 2).clone();
    }

    // initial hidden state
    ncnn::Mat hidden(outch);
    if (hidden.empty())
        return -100;
    hidden.fill(0.f);

    ncnn::Mat cell(outch);
    if (cell.empty())
        return -100;
    cell.fill(0.f);

    a1[1] = hidden;
    a1[2] = cell;
    op->forward(a1, _c1, opt);
    a2[1] = _c1[1];
    a2[2] = _c1[2];
    op->forward(a2, _c2, opt);

    ncnn::Mat c1 = _c1[0];
    ncnn::Mat c2 = _c2[0];

    if (direction == 1)
    {
        c2 = _c1[0];
        c1 = _c2[0];
    }

    // total height
    ncnn::Mat c;
    c.create(b.w, b.h, b.elemsize, opt.blob_allocator);
    if (c.empty())
        return -100;

    unsigned char* outptr = c;
    int c1_size = c1.w * c1.h;
    const unsigned char* c1ptr = c1;
    memcpy(outptr, c1ptr, c1_size * c1.elemsize);
    outptr += c1_size * c1.elemsize;
    int c2_size = c2.w * c2.h;
    const unsigned char* c2ptr = c2;
    memcpy(outptr, c2ptr, c2_size * c2.elemsize);

    op->destroy_pipeline(opt);

    delete op;

    if (CompareMat(b, c, epsilon) != 0)
    {
        fprintf(stderr, "test_lstm two step failed a.dims=%d a=(%d %d %d) outch=%d, direction = %d \n", a.dims, a.w, a.h, a.c, outch, direction);
        return -1;
    }

    return 0;
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
           || test_lstm_layer(RandomMat(4, 4), 1, 1)
           || test_lstm_layer(RandomMat(8, 2), 2, 1)
           || test_lstm_layer(RandomMat(16, 8), 7, 1)
           || test_lstm_layer(RandomMat(17, 8), 8, 1)
           || test_lstm_layer(RandomMat(19, 15), 8, 1)
           || test_lstm_layer(RandomMat(5, 16), 16, 1)
           || test_lstm_layer(RandomMat(3, 16), 8, 1)
           || test_lstm_layer(RandomMat(2, 5), 99, 1)
           || test_lstm_layer(RandomMat(4, 2), 1, 0)
           || test_lstm_layer(RandomMat(8, 2), 2, 0)
           || test_lstm_layer(RandomMat(16, 8), 7, 0)
           || test_lstm_layer(RandomMat(17, 8), 8, 0)
           || test_lstm_layer(RandomMat(19, 15), 8, 0)
           || test_lstm_layer(RandomMat(5, 16), 16, 0)
           || test_lstm_layer(RandomMat(3, 16), 8, 0)
           || test_lstm_layer(RandomMat(2, 5), 17, 0);
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
