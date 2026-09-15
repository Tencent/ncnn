// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

static int test_inversespectrogram(int frames, int freqs, int n_fft, int returns, int hoplen, int winlen, int window_type, int center, int normalized)
{
    ncnn::Mat a = RandomMat(2, frames, freqs);

    ncnn::ParamDict pd;
    pd.set(0, n_fft);
    pd.set(1, returns);
    pd.set(2, hoplen);
    pd.set(3, winlen);
    pd.set(4, window_type);
    pd.set(5, center);
    pd.set(7, normalized);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("InverseSpectrogram", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_inversespectrogram failed frames=%d freqs=%d n_fft=%d returns=%d hoplen=%d winlen=%d window_type=%d center=%d normalized=%d\n", frames, freqs, n_fft, returns, hoplen, winlen, window_type, center, normalized);
    }

    return ret;
}

static int test_inversespectrogram_0()
{
    return 0
           || test_inversespectrogram(17, 1, 1, 0, 1, 1, 0, 1, 0)
           || test_inversespectrogram(39, 9, 17, 0, 7, 15, 0, 0, 1)
           || test_inversespectrogram(128, 6, 10, 0, 2, 7, 1, 1, 1)
           || test_inversespectrogram(255, 17, 17, 1, 14, 17, 2, 0, 0)
           || test_inversespectrogram(124, 28, 55, 2, 12, 55, 1, 1, 2);
}

static int test_inversespectrogram_load_param_case(const ncnn::ParamDict& pd, bool valid)
{
    ncnn::Layer* layer = ncnn::create_layer_naive(ncnn::LayerType::InverseSpectrogram);
    if (!layer)
        return -1;

    int ret = layer->load_param(pd);
    delete layer;

    if ((ret == 0) != valid)
    {
        fprintf(stderr, "InverseSpectrogram load_param returned %d, expected %s\n", ret, valid ? "success" : "failure");
        return -1;
    }

    return 0;
}

static int test_inversespectrogram_load_param()
{
    ncnn::ParamDict base;
    base.set(0, 16);
    if (test_inversespectrogram_load_param_case(base, true) != 0)
        return -1;

    const int ids[] = {0, 0, 3, 3, 3, 2, 4, 7};
    const int values[] = {0, -1, 0, -1, 17, 0, 3, 3};
    for (int i = 0; i < 8; i++)
    {
        ncnn::ParamDict pd = base;
        pd.set(ids[i], values[i]);
        if (test_inversespectrogram_load_param_case(pd, false) != 0)
            return -1;
    }

    ncnn::ParamDict pd = base;
    pd.set(0, INT_MAX);
    pd.set(7, 2);
    if (test_inversespectrogram_load_param_case(pd, false) != 0)
        return -1;

    pd.set(0, 1);
    pd.set(2, 1); // a one-sample FFT is valid with an explicit hop length
    if (test_inversespectrogram_load_param_case(pd, true) != 0)
        return -1;
    return 0;
}

int main()
{
    SRAND(7767517);

    return test_inversespectrogram_0() || test_inversespectrogram_load_param();
}
