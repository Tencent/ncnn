// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

static int test_spectrogram(int size, int n_fft, int power, int hoplen, int winlen, int window_type, int center, int pad_type, int normalized, int onesided)
{
    ncnn::Mat a = RandomMat(size);

    ncnn::ParamDict pd;
    pd.set(0, n_fft);
    pd.set(1, power);
    pd.set(2, hoplen);
    pd.set(3, winlen);
    pd.set(4, window_type);
    pd.set(5, center);
    pd.set(6, pad_type);
    pd.set(7, normalized);
    pd.set(8, onesided);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Spectrogram", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_spectrogram failed size=%d n_fft=%d power=%d hoplen=%d winlen=%d window_type=%d center=%d pad_type=%d normalized=%d onesided=%d\n", size, n_fft, power, hoplen, winlen, window_type, center, pad_type, normalized, onesided);
    }

    return ret;
}

static int test_spectrogram_0()
{
    return 0
           || test_spectrogram(17, 1, 0, 1, 1, 0, 1, 0, 0, 0)
           || test_spectrogram(39, 17, 0, 7, 15, 0, 0, 0, 1, 0)
           || test_spectrogram(128, 10, 0, 2, 7, 1, 1, 1, 1, 1)
           || test_spectrogram(255, 17, 1, 14, 17, 2, 0, 0, 0, 1)
           || test_spectrogram(124, 55, 2, 12, 55, 1, 1, 2, 2, 0);
}

static int test_spectrogram_load_param_case(const ncnn::ParamDict& pd, bool valid)
{
    for (int backend = 0; backend < 2; backend++)
    {
        ncnn::Layer* layer = backend == 0 ? ncnn::create_layer_naive(ncnn::LayerType::Spectrogram) : ncnn::create_layer_cpu(ncnn::LayerType::Spectrogram);
        if (!layer)
            return -1;

        int ret = layer->load_param(pd);
        delete layer;

        if ((ret == 0) != valid)
        {
            fprintf(stderr, "Spectrogram load_param backend=%d returned %d, expected %s\n", backend, ret, valid ? "success" : "failure");
            return -1;
        }
    }

    return 0;
}

static int test_spectrogram_load_param()
{
    ncnn::ParamDict base;
    base.set(0, 16);
    if (test_spectrogram_load_param_case(base, true) != 0)
        return -1;

    const int ids[] = {0, 0, 3, 3, 3, 2, 4, 7};
    const int values[] = {0, -1, 0, -1, 17, 0, 3, 3};
    for (int i = 0; i < 8; i++)
    {
        ncnn::ParamDict pd = base;
        pd.set(ids[i], values[i]);
        if (test_spectrogram_load_param_case(pd, false) != 0)
            return -1;
    }

    ncnn::ParamDict pd = base;
    pd.set(0, INT_MAX);
    pd.set(7, 2);
    if (test_spectrogram_load_param_case(pd, false) != 0)
        return -1;

    pd.set(0, 1);
    pd.set(2, 1); // a one-sample FFT is valid with an explicit hop length
    if (test_spectrogram_load_param_case(pd, true) != 0)
        return -1;
    return 0;
}

int main()
{
    SRAND(7767517);

    return test_spectrogram_0() || test_spectrogram_load_param();
}
