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

static int test_spectrogram_load_param()
{
    ncnn::ParamDict base;
    base.set(0, 16);
    if (test_layer_param(ncnn::LayerType::Spectrogram, base, 0) != 0)
        return -1;

    int ret = 0
              || test_layer_param(ncnn::LayerType::Spectrogram, base, 0, 0, -1)
              || test_layer_param(ncnn::LayerType::Spectrogram, base, 0, -1, -1)
              || test_layer_param(ncnn::LayerType::Spectrogram, base, 3, 0, -1)
              || test_layer_param(ncnn::LayerType::Spectrogram, base, 3, -1, -1)
              || test_layer_param(ncnn::LayerType::Spectrogram, base, 3, 17, -1)
              || test_layer_param(ncnn::LayerType::Spectrogram, base, 2, 0, -1)
              || test_layer_param(ncnn::LayerType::Spectrogram, base, 4, 3, -1)
              || test_layer_param(ncnn::LayerType::Spectrogram, base, 7, 3, -1);
    if (ret != 0)
        return ret;

    {
        ncnn::ParamDict pd = base;
        pd.set(0, INT_MAX);
        pd.set(7, 2);
        if (test_layer_param(ncnn::LayerType::Spectrogram, pd, -1) != 0)
            return -1;
    }

    {
        ncnn::ParamDict pd = base;
        pd.set(0, 1);
        pd.set(7, 2);
        pd.set(2, 1); // a one-sample FFT is valid with an explicit hop length
        if (test_layer_param(ncnn::LayerType::Spectrogram, pd, 0) != 0)
            return -1;
    }

    if (sizeof(size_t) == 4)
    {
        // window allocation overflows on 32-bit platforms
        const int n_ffts[] = {0x40000001, 0x40000000, 0x3fffffff, INT_MAX};
        for (int i = 0; i < 4; i++)
        {
            for (int normalized = 0; normalized <= 2; normalized++)
            {
                ncnn::ParamDict pd = base;
                pd.set(0, n_ffts[i]);
                pd.set(7, normalized);
                if (test_layer_param(ncnn::LayerType::Spectrogram, pd, -1) != 0)
                    return -1;
            }
        }
    }

    return 0;
}

static int test_spectrogram_load_param_type()
{
    ncnn::ParamDict base;
    base.set(0, 16);
    if (test_layer_param(ncnn::LayerType::Spectrogram, base, 0) != 0)
        return -1;

    for (int i = 0; i <= 2; i++)
    {
        if (test_layer_param(ncnn::LayerType::Spectrogram, base, 1, i, 0) != 0)
            return -1;
    }

    const int invalid[] = {-1, 3, INT_MIN, INT_MAX};
    for (int i = 0; i < 4; i++)
    {
        if (test_layer_param(ncnn::LayerType::Spectrogram, base, 1, invalid[i], -1) != 0)
            return -1;
    }

    for (int i = 0; i <= 2; i++)
    {
        if (test_layer_param(ncnn::LayerType::Spectrogram, base, 6, i, 0) != 0)
            return -1;
    }

    // padding is unused without centering
    ncnn::ParamDict uncentered = base;
    uncentered.set(5, 0);

    const int invalid_pad[] = {-1, 3, INT_MIN, INT_MAX};
    for (int i = 0; i < 4; i++)
    {
        int ret = 0
                  || test_layer_param(ncnn::LayerType::Spectrogram, base, 6, invalid_pad[i], -1)
                  || test_layer_param(ncnn::LayerType::Spectrogram, uncentered, 6, invalid_pad[i], 0);
        if (ret != 0)
            return ret;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return test_spectrogram_0() || test_spectrogram_load_param() || test_spectrogram_load_param_type();
}
