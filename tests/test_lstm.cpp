// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_lstm(int size, int T, int outch, int direction, int hidden_size = 0)
{
    ncnn::Mat a = RandomMat(size, T);
    int num_directions = direction == 2 ? 2 : 1;
    if (hidden_size == 0)
        hidden_size = outch;

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, hidden_size * size * 4 * num_directions);
    pd.set(2, direction);
    pd.set(3, hidden_size);

    std::vector<ncnn::Mat> weights(hidden_size == outch ? 3 : 4);
    weights[0] = RandomMat(hidden_size * size * 4 * num_directions);
    weights[1] = RandomMat(hidden_size * 4 * num_directions);
    weights[2] = RandomMat(outch * hidden_size * 4 * num_directions);
    if (hidden_size != outch)
    {
        weights[3] = RandomMat(hidden_size * outch * num_directions);
    }

    int ret = test_layer("LSTM", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_lstm failed size=%d T=%d outch=%d direction=%d hidden_size=%d\n", size, T, outch, direction, hidden_size);
    }

    return ret;
}

static int test_lstm_with_hidden(int size, int T, int outch, int direction, int hidden_size = 0)
{
    ncnn::Mat a = RandomMat(size, T);
    int num_directions = direction == 2 ? 2 : 1;
    if (hidden_size == 0)
        hidden_size = outch;

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, hidden_size * size * 4 * num_directions);
    pd.set(2, direction);
    pd.set(3, hidden_size);

    std::vector<ncnn::Mat> weights(hidden_size == outch ? 3 : 4);
    weights[0] = RandomMat(hidden_size * size * 4 * num_directions);
    weights[1] = RandomMat(hidden_size * 4 * num_directions);
    weights[2] = RandomMat(outch * hidden_size * 4 * num_directions);
    if (hidden_size != outch)
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
        fprintf(stderr, "test_lstm_with_hidden failed size=%d T=%d outch=%d direction=%d hidden_size=%d\n", size, T, outch, direction, hidden_size);
    }

    return ret;
}

static int test_lstm_with_hidden_input(int size, int T, int outch, int direction, int hidden_size = 0)
{
    ncnn::Mat a = RandomMat(size, T);
    int num_directions = direction == 2 ? 2 : 1;
    if (hidden_size == 0)
        hidden_size = outch;

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, hidden_size * size * 4 * num_directions);
    pd.set(2, direction);
    pd.set(3, hidden_size);

    std::vector<ncnn::Mat> weights(hidden_size == outch ? 3 : 4);
    weights[0] = RandomMat(hidden_size * size * 4 * num_directions);
    weights[1] = RandomMat(hidden_size * 4 * num_directions);
    weights[2] = RandomMat(outch * hidden_size * 4 * num_directions);
    if (hidden_size != outch)
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
        fprintf(stderr, "test_lstm_with_hidden_input failed size=%d T=%d outch=%d direction=%d hidden_size=%d\n", size, T, outch, direction, hidden_size);
    }

    return ret;
}

static int test_lstm_with_hidden_output(int size, int T, int outch, int direction, int hidden_size = 0)
{
    ncnn::Mat a = RandomMat(size, T);
    int num_directions = direction == 2 ? 2 : 1;
    if (hidden_size == 0)
        hidden_size = outch;

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, hidden_size * size * 4 * num_directions);
    pd.set(2, direction);
    pd.set(3, hidden_size);

    std::vector<ncnn::Mat> weights(hidden_size == outch ? 3 : 4);
    weights[0] = RandomMat(hidden_size * size * 4 * num_directions);
    weights[1] = RandomMat(hidden_size * 4 * num_directions);
    weights[2] = RandomMat(outch * hidden_size * 4 * num_directions);
    if (hidden_size != outch)
    {
        weights[3] = RandomMat(hidden_size * outch * num_directions);
    }

    std::vector<ncnn::Mat> as(1);
    as[0] = a;

    int ret = test_layer("LSTM", pd, weights, as, 3);
    if (ret != 0)
    {
        fprintf(stderr, "test_lstm_with_hidden_output failed size=%d T=%d outch=%d direction=%d hidden_size=%d\n", size, T, outch, direction, hidden_size);
    }

    return ret;
}

static int test_lstm_0()
{
    return 0
           || test_lstm(4, 1, 2, 2)
           || test_lstm(8, 2, 2, 2)
           || test_lstm(16, 8, 7, 2)
           || test_lstm(17, 8, 8, 2)
           || test_lstm(19, 15, 8, 2)
           || test_lstm(5, 16, 16, 2)
           || test_lstm(3, 16, 8, 2)
           || test_lstm(8, 16, 16, 2)
           || test_lstm(31, 3, 31, 2)
           || test_lstm(2, 5, 17, 2, 15);
}

static int test_lstm_1()
{
    return 0
           || test_lstm_with_hidden(4, 4, 1, 2)
           || test_lstm_with_hidden(8, 2, 2, 2)
           || test_lstm_with_hidden(16, 8, 7, 2)
           || test_lstm_with_hidden(17, 8, 8, 2)
           || test_lstm_with_hidden(19, 15, 8, 2)
           || test_lstm_with_hidden(5, 16, 16, 2)
           || test_lstm_with_hidden(3, 16, 8, 2)
           || test_lstm_with_hidden(2, 5, 79, 2, 33)
           || test_lstm_with_hidden(4, 4, 1, 1)
           || test_lstm_with_hidden(8, 2, 2, 1)
           || test_lstm_with_hidden(16, 8, 7, 1)
           || test_lstm_with_hidden(17, 8, 8, 1)
           || test_lstm_with_hidden(19, 15, 8, 1)
           || test_lstm_with_hidden(5, 16, 16, 1)
           || test_lstm_with_hidden(3, 16, 8, 1)
           || test_lstm_with_hidden(2, 5, 79, 1, 33)
           || test_lstm_with_hidden(4, 2, 1, 0)
           || test_lstm_with_hidden(8, 2, 2, 0)
           || test_lstm_with_hidden(16, 8, 7, 0)
           || test_lstm_with_hidden(17, 8, 8, 0)
           || test_lstm_with_hidden(19, 15, 8, 0)
           || test_lstm_with_hidden(5, 16, 16, 0)
           || test_lstm_with_hidden(3, 16, 8, 0)
           || test_lstm_with_hidden(2, 5, 17, 0, 15)

           || test_lstm_with_hidden_input(4, 4, 1, 2)
           || test_lstm_with_hidden_input(8, 2, 2, 2)
           || test_lstm_with_hidden_input(16, 8, 7, 2)
           || test_lstm_with_hidden_input(17, 8, 8, 2)
           || test_lstm_with_hidden_input(19, 15, 8, 2)
           || test_lstm_with_hidden_input(5, 16, 16, 2)
           || test_lstm_with_hidden_input(3, 16, 8, 2)
           || test_lstm_with_hidden_input(2, 5, 79, 2, 33)
           || test_lstm_with_hidden_input(4, 4, 1, 1)
           || test_lstm_with_hidden_input(8, 2, 2, 1)
           || test_lstm_with_hidden_input(16, 8, 7, 1)
           || test_lstm_with_hidden_input(17, 8, 8, 1)
           || test_lstm_with_hidden_input(19, 15, 8, 1)
           || test_lstm_with_hidden_input(5, 16, 16, 1)
           || test_lstm_with_hidden_input(3, 16, 8, 1)
           || test_lstm_with_hidden_input(2, 5, 79, 1, 33)
           || test_lstm_with_hidden_input(4, 2, 1, 0)
           || test_lstm_with_hidden_input(8, 2, 2, 0)
           || test_lstm_with_hidden_input(16, 8, 7, 0)
           || test_lstm_with_hidden_input(17, 8, 8, 0)
           || test_lstm_with_hidden_input(19, 15, 8, 0)
           || test_lstm_with_hidden_input(5, 16, 16, 0)
           || test_lstm_with_hidden_input(3, 16, 8, 0)
           || test_lstm_with_hidden_input(2, 5, 17, 0, 15)

           || test_lstm_with_hidden_output(4, 4, 1, 2)
           || test_lstm_with_hidden_output(8, 2, 2, 2)
           || test_lstm_with_hidden_output(16, 8, 7, 2)
           || test_lstm_with_hidden_output(17, 8, 8, 2)
           || test_lstm_with_hidden_output(19, 15, 8, 2)
           || test_lstm_with_hidden_output(5, 16, 16, 2)
           || test_lstm_with_hidden_output(3, 16, 8, 2)
           || test_lstm_with_hidden_output(2, 5, 79, 2, 33)
           || test_lstm_with_hidden_output(4, 4, 1, 1)
           || test_lstm_with_hidden_output(8, 2, 2, 1)
           || test_lstm_with_hidden_output(16, 8, 7, 1)
           || test_lstm_with_hidden_output(17, 8, 8, 1)
           || test_lstm_with_hidden_output(19, 15, 8, 1)
           || test_lstm_with_hidden_output(5, 16, 16, 1)
           || test_lstm_with_hidden_output(3, 16, 8, 1)
           || test_lstm_with_hidden_output(2, 5, 79, 1, 33)
           || test_lstm_with_hidden_output(4, 2, 1, 0)
           || test_lstm_with_hidden_output(8, 2, 2, 0)
           || test_lstm_with_hidden_output(16, 8, 7, 0)
           || test_lstm_with_hidden_output(17, 8, 8, 0)
           || test_lstm_with_hidden_output(19, 15, 8, 0)
           || test_lstm_with_hidden_output(5, 16, 16, 0)
           || test_lstm_with_hidden_output(3, 16, 8, 0)
           || test_lstm_with_hidden_output(2, 5, 17, 0, 15);
}

static int test_lstm_2()
{
    return 0
           || test_lstm(4, 1, 1, 0)
           || test_lstm(8, 2, 2, 0)
           || test_lstm(16, 8, 7, 0)
           || test_lstm(17, 8, 8, 0)
           || test_lstm(19, 15, 8, 0)
           || test_lstm(5, 16, 16, 0)
           || test_lstm(3, 16, 8, 0)
           || test_lstm(8, 16, 16, 0)
           || test_lstm(2, 5, 17, 0, 15);
}

static int test_lstm_3()
{
    return 0
           || test_lstm(4, 1, 1, 1)
           || test_lstm(8, 2, 2, 1)
           || test_lstm(16, 8, 7, 1)
           || test_lstm(17, 8, 8, 1)
           || test_lstm(19, 15, 8, 1)
           || test_lstm(5, 16, 16, 1)
           || test_lstm(3, 16, 8, 1)
           || test_lstm(8, 16, 16, 1)
           || test_lstm(2, 5, 17, 1, 15);
}

#if NCNN_INT8
static void RandomizeA(ncnn::Mat& m, float absmax)
{
    absmax = ncnn::float16_to_float32(ncnn::float32_to_float16(absmax));
    absmax = ncnn::bfloat16_to_float32(ncnn::float32_to_bfloat16(absmax));

    const int h = m.h;
    float* p = m;
    for (int i = 0; i < h; i++)
    {
        float* p = m.row(i);
        for (int j = 0; j < m.w; j++)
        {
            p[j] = RandomFloat(-absmax, absmax);

            // drop 0.45 ~ 0.55
            float v = p[j] * (127.f / absmax);
            float vv = fabs(v - (int)v);

            float hp = ncnn::float16_to_float32(ncnn::float32_to_float16(p[j]));
            float hv = hp * (127.f / absmax);
            float hvv = fabs(hv - (int)hv);

            float bp = ncnn::bfloat16_to_float32(ncnn::float32_to_bfloat16(p[j]));
            float bv = bp * (127.f / absmax);
            float bvv = fabs(bv - (int)bv);

            while ((vv > 0.45f && vv < 0.55f) || (hvv > 0.45f && hvv < 0.55f) || (bvv > 0.45f && bvv < 0.55f))
            {
                p[j] = RandomFloat(-absmax, absmax);
                v = p[j] * (127.f / absmax);
                vv = fabs(v - (int)v);

                hp = ncnn::float16_to_float32(ncnn::float32_to_float16(p[j]));
                hv = hp * (127.f / absmax);
                hvv = fabs(hv - (int)hv);

                bp = ncnn::bfloat16_to_float32(ncnn::float32_to_bfloat16(p[j]));
                bv = bp * (127.f / absmax);
                bvv = fabs(bv - (int)bv);
            }
        }
    }

    // set random a and b
    m.row(RandomInt(0, h - 1))[RandomInt(0, m.w - 1)] = -absmax;
    m.row(RandomInt(0, h - 1))[RandomInt(0, m.w - 1)] = absmax;
}

static int test_lstm_int8(int size, int T, int outch, int direction, int hidden_size = 0)
{
    int num_directions = direction == 2 ? 2 : 1;
    if (hidden_size == 0)
        hidden_size = outch;

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, hidden_size * size * 4 * num_directions);
    pd.set(2, direction);
    pd.set(3, hidden_size);
    pd.set(8, 2); // int8_scale_term

    std::vector<ncnn::Mat> weights(hidden_size == outch ? 5 : 6);
    weights[0] = RandomS8Mat(hidden_size * size * 4 * num_directions);
    weights[1] = RandomMat(hidden_size * 4 * num_directions);
    weights[2] = RandomS8Mat(outch * hidden_size * 4 * num_directions);
    if (hidden_size != outch)
    {
        weights[3] = RandomMat(hidden_size * outch * num_directions);
        weights[4] = RandomMat(hidden_size * 4 * num_directions, 100.f, 200.f);
        weights[5] = RandomMat(hidden_size * 4 * num_directions, 100.f, 200.f);
    }
    else
    {
        weights[3] = RandomMat(hidden_size * 4 * num_directions, 100.f, 200.f);
        weights[4] = RandomMat(hidden_size * 4 * num_directions, 100.f, 200.f);
    }

    ncnn::Mat a(size, T);
    RandomizeA(a, 10.f);

    int ret = test_layer("LSTM", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_lstm_int8 failed size=%d T=%d outch=%d direction=%d hidden_size=%d\n", size, T, outch, direction, hidden_size);
    }

    return ret;
}

static int test_lstm_int8_with_hidden(int size, int T, int outch, int direction, int hidden_size = 0)
{
    int num_directions = direction == 2 ? 2 : 1;
    if (hidden_size == 0)
        hidden_size = outch;

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, hidden_size * size * 4 * num_directions);
    pd.set(2, direction);
    pd.set(3, hidden_size);
    pd.set(8, 2); // int8_scale_term

    std::vector<ncnn::Mat> weights(hidden_size == outch ? 5 : 6);
    weights[0] = RandomS8Mat(hidden_size * size * 4 * num_directions);
    weights[1] = RandomMat(hidden_size * 4 * num_directions);
    weights[2] = RandomS8Mat(outch * hidden_size * 4 * num_directions);
    if (hidden_size != outch)
    {
        weights[3] = RandomMat(hidden_size * outch * num_directions);
        weights[4] = RandomMat(hidden_size * 4 * num_directions, 100.f, 200.f);
        weights[5] = RandomMat(hidden_size * 4 * num_directions, 100.f, 200.f);
    }
    else
    {
        weights[3] = RandomMat(hidden_size * 4 * num_directions, 100.f, 200.f);
        weights[4] = RandomMat(hidden_size * 4 * num_directions, 100.f, 200.f);
    }

    ncnn::Mat a(size, T);
    RandomizeA(a, 10.f);

    // initial hidden state
    ncnn::Mat hidden(outch, num_directions);
    RandomizeA(hidden, 10.f);

    // initial cell state
    ncnn::Mat cell(hidden_size, num_directions);
    RandomizeA(cell, 10.f);

    std::vector<ncnn::Mat> as(3);
    as[0] = a;
    as[1] = hidden;
    as[2] = cell;

    int ret = test_layer("LSTM", pd, weights, as, 3);
    if (ret != 0)
    {
        fprintf(stderr, "test_lstm_int8_with_hidden failed size=%d T=%d outch=%d direction=%d hidden_size=%d\n", size, T, outch, direction, hidden_size);
    }

    return ret;
}

static int test_lstm_int8_with_hidden_input(int size, int T, int outch, int direction, int hidden_size = 0)
{
    int num_directions = direction == 2 ? 2 : 1;
    if (hidden_size == 0)
        hidden_size = outch;

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, hidden_size * size * 4 * num_directions);
    pd.set(2, direction);
    pd.set(3, hidden_size);
    pd.set(8, 2); // int8_scale_term

    std::vector<ncnn::Mat> weights(hidden_size == outch ? 5 : 6);
    weights[0] = RandomS8Mat(hidden_size * size * 4 * num_directions);
    weights[1] = RandomMat(hidden_size * 4 * num_directions);
    weights[2] = RandomS8Mat(outch * hidden_size * 4 * num_directions);
    if (hidden_size != outch)
    {
        weights[3] = RandomMat(hidden_size * outch * num_directions);
        weights[4] = RandomMat(hidden_size * 4 * num_directions, 100.f, 200.f);
        weights[5] = RandomMat(hidden_size * 4 * num_directions, 100.f, 200.f);
    }
    else
    {
        weights[3] = RandomMat(hidden_size * 4 * num_directions, 100.f, 200.f);
        weights[4] = RandomMat(hidden_size * 4 * num_directions, 100.f, 200.f);
    }

    ncnn::Mat a(size, T);
    RandomizeA(a, 10.f);

    // initial hidden state
    ncnn::Mat hidden(outch, num_directions);
    RandomizeA(hidden, 10.f);

    // initial cell state
    ncnn::Mat cell(hidden_size, num_directions);
    RandomizeA(cell, 10.f);

    std::vector<ncnn::Mat> as(3);
    as[0] = a;
    as[1] = hidden;
    as[2] = cell;

    int ret = test_layer("LSTM", pd, weights, as, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_lstm_int8_with_hidden_input failed size=%d T=%d outch=%d direction=%d hidden_size=%d\n", size, T, outch, direction, hidden_size);
    }

    return ret;
}

static int test_lstm_int8_with_hidden_output(int size, int T, int outch, int direction, int hidden_size = 0)
{
    int num_directions = direction == 2 ? 2 : 1;
    if (hidden_size == 0)
        hidden_size = outch;

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, hidden_size * size * 4 * num_directions);
    pd.set(2, direction);
    pd.set(3, hidden_size);
    pd.set(8, 2); // int8_scale_term

    std::vector<ncnn::Mat> weights(hidden_size == outch ? 5 : 6);
    weights[0] = RandomS8Mat(hidden_size * size * 4 * num_directions);
    weights[1] = RandomMat(hidden_size * 4 * num_directions);
    weights[2] = RandomS8Mat(outch * hidden_size * 4 * num_directions);
    if (hidden_size != outch)
    {
        weights[3] = RandomMat(hidden_size * outch * num_directions);
        weights[4] = RandomMat(hidden_size * 4 * num_directions, 100.f, 200.f);
        weights[5] = RandomMat(hidden_size * 4 * num_directions, 100.f, 200.f);
    }
    else
    {
        weights[3] = RandomMat(hidden_size * 4 * num_directions, 100.f, 200.f);
        weights[4] = RandomMat(hidden_size * 4 * num_directions, 100.f, 200.f);
    }

    ncnn::Mat a(size, T);
    RandomizeA(a, 10.f);

    std::vector<ncnn::Mat> as(1);
    as[0] = a;

    int ret = test_layer("LSTM", pd, weights, as, 3);
    if (ret != 0)
    {
        fprintf(stderr, "test_lstm_int8_with_hidden_output failed size=%d T=%d outch=%d direction=%d hidden_size=%d\n", size, T, outch, direction, hidden_size);
    }

    return ret;
}

static int test_lstm_4()
{
    return 0
           || test_lstm_int8(4, 1, 2, 2)
           || test_lstm_int8(8, 2, 2, 2)
           || test_lstm_int8(16, 8, 7, 2)
           || test_lstm_int8(17, 8, 8, 2)
           || test_lstm_int8(19, 15, 8, 2)
           || test_lstm_int8(5, 16, 16, 2)
           || test_lstm_int8(3, 16, 8, 2)
           || test_lstm_int8(8, 16, 16, 2)
           || test_lstm_int8(31, 3, 31, 2)
           || test_lstm_int8(2, 5, 17, 2, 15);
}

static int test_lstm_5()
{
    return 0
           || test_lstm_int8_with_hidden(4, 4, 1, 2)
           || test_lstm_int8_with_hidden(8, 2, 2, 2)
           || test_lstm_int8_with_hidden(16, 8, 7, 2)
           || test_lstm_int8_with_hidden(17, 8, 8, 2)
           || test_lstm_int8_with_hidden(19, 15, 8, 2)
           || test_lstm_int8_with_hidden(5, 16, 16, 2)
           || test_lstm_int8_with_hidden(3, 16, 8, 2)
           || test_lstm_int8_with_hidden(2, 5, 79, 2, 33)
           || test_lstm_int8_with_hidden(4, 4, 1, 1)
           || test_lstm_int8_with_hidden(8, 2, 2, 1)
           || test_lstm_int8_with_hidden(16, 8, 7, 1)
           || test_lstm_int8_with_hidden(17, 8, 8, 1)
           || test_lstm_int8_with_hidden(19, 15, 8, 1)
           || test_lstm_int8_with_hidden(5, 16, 16, 1)
           || test_lstm_int8_with_hidden(3, 16, 8, 1)
           || test_lstm_int8_with_hidden(2, 5, 79, 1, 33)
           || test_lstm_int8_with_hidden(4, 2, 1, 0)
           || test_lstm_int8_with_hidden(8, 2, 2, 0)
           || test_lstm_int8_with_hidden(16, 8, 7, 0)
           || test_lstm_int8_with_hidden(17, 8, 8, 0)
           || test_lstm_int8_with_hidden(19, 15, 8, 0)
           || test_lstm_int8_with_hidden(5, 16, 16, 0)
           || test_lstm_int8_with_hidden(3, 16, 8, 0)
           || test_lstm_int8_with_hidden(2, 5, 17, 0, 15)

           || test_lstm_int8_with_hidden_input(4, 4, 1, 2)
           || test_lstm_int8_with_hidden_input(8, 2, 2, 2)
           || test_lstm_int8_with_hidden_input(16, 8, 7, 2)
           || test_lstm_int8_with_hidden_input(17, 8, 8, 2)
           || test_lstm_int8_with_hidden_input(19, 15, 8, 2)
           || test_lstm_int8_with_hidden_input(5, 16, 16, 2)
           || test_lstm_int8_with_hidden_input(3, 16, 8, 2)
           || test_lstm_int8_with_hidden_input(2, 5, 79, 2, 33)
           || test_lstm_int8_with_hidden_input(4, 4, 1, 1)
           || test_lstm_int8_with_hidden_input(8, 2, 2, 1)
           || test_lstm_int8_with_hidden_input(16, 8, 7, 1)
           || test_lstm_int8_with_hidden_input(17, 8, 8, 1)
           || test_lstm_int8_with_hidden_input(19, 15, 8, 1)
           || test_lstm_int8_with_hidden_input(5, 16, 16, 1)
           || test_lstm_int8_with_hidden_input(3, 16, 8, 1)
           || test_lstm_int8_with_hidden_input(2, 5, 79, 1, 33)
           || test_lstm_int8_with_hidden_input(4, 2, 1, 0)
           || test_lstm_int8_with_hidden_input(8, 2, 2, 0)
           || test_lstm_int8_with_hidden_input(16, 8, 7, 0)
           || test_lstm_int8_with_hidden_input(17, 8, 8, 0)
           || test_lstm_int8_with_hidden_input(19, 15, 8, 0)
           || test_lstm_int8_with_hidden_input(5, 16, 16, 0)
           || test_lstm_int8_with_hidden_input(3, 16, 8, 0)
           || test_lstm_int8_with_hidden_input(2, 5, 17, 0, 15)

           || test_lstm_int8_with_hidden_output(4, 4, 1, 2)
           || test_lstm_int8_with_hidden_output(8, 2, 2, 2)
           || test_lstm_int8_with_hidden_output(16, 8, 7, 2)
           || test_lstm_int8_with_hidden_output(17, 8, 8, 2)
           || test_lstm_int8_with_hidden_output(19, 15, 8, 2)
           || test_lstm_int8_with_hidden_output(5, 16, 16, 2)
           || test_lstm_int8_with_hidden_output(3, 16, 8, 2)
           || test_lstm_int8_with_hidden_output(2, 5, 79, 2, 33)
           || test_lstm_int8_with_hidden_output(4, 4, 1, 1)
           || test_lstm_int8_with_hidden_output(8, 2, 2, 1)
           || test_lstm_int8_with_hidden_output(16, 8, 7, 1)
           || test_lstm_int8_with_hidden_output(17, 8, 8, 1)
           || test_lstm_int8_with_hidden_output(19, 15, 8, 1)
           || test_lstm_int8_with_hidden_output(5, 16, 16, 1)
           || test_lstm_int8_with_hidden_output(3, 16, 8, 1)
           || test_lstm_int8_with_hidden_output(2, 5, 79, 1, 33)
           || test_lstm_int8_with_hidden_output(4, 2, 1, 0)
           || test_lstm_int8_with_hidden_output(8, 2, 2, 0)
           || test_lstm_int8_with_hidden_output(16, 8, 7, 0)
           || test_lstm_int8_with_hidden_output(17, 8, 8, 0)
           || test_lstm_int8_with_hidden_output(19, 15, 8, 0)
           || test_lstm_int8_with_hidden_output(5, 16, 16, 0)
           || test_lstm_int8_with_hidden_output(3, 16, 8, 0)
           || test_lstm_int8_with_hidden_output(2, 5, 17, 0, 15);
}

static int test_lstm_6()
{
    return 0
           || test_lstm_int8(4, 1, 1, 0)
           || test_lstm_int8(8, 2, 2, 0)
           || test_lstm_int8(16, 8, 7, 0)
           || test_lstm_int8(17, 8, 8, 0)
           || test_lstm_int8(19, 15, 8, 0)
           || test_lstm_int8(5, 16, 16, 0)
           || test_lstm_int8(3, 16, 8, 0)
           || test_lstm_int8(8, 16, 16, 0)
           || test_lstm_int8(2, 5, 17, 0, 15);
}

static int test_lstm_7()
{
    return 0
           || test_lstm_int8(4, 1, 1, 1)
           || test_lstm_int8(8, 2, 2, 1)
           || test_lstm_int8(16, 8, 7, 1)
           || test_lstm_int8(17, 8, 8, 1)
           || test_lstm_int8(19, 15, 8, 1)
           || test_lstm_int8(5, 16, 16, 1)
           || test_lstm_int8(3, 16, 8, 1)
           || test_lstm_int8(8, 16, 16, 1)
           || test_lstm_int8(2, 5, 17, 1, 15);
}
#endif

int main()
{
    SRAND(7767517);

#if NCNN_INT8
    return 0
           || test_lstm_0()
           || test_lstm_1()
           || test_lstm_2()
           || test_lstm_3()
           || test_lstm_4()
           || test_lstm_5()
           || test_lstm_6()
           || test_lstm_7();
#else
    return 0
           || test_lstm_0()
           || test_lstm_1()
           || test_lstm_2()
           || test_lstm_3();
#endif
}
