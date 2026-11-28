// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_INVERSESPECTROGRAM_H
#define LAYER_INVERSESPECTROGRAM_H

#include "layer.h"

namespace ncnn {

class InverseSpectrogram : public Layer
{
public:
    InverseSpectrogram();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    int n_fft;
    int returns; // 0=complex 1=real 2=imag
    int hoplen;
    int winlen;
    int window_type; // 0=ones 1=hann 2=hamming
    int center;
    int normalized; // 0=disabled 1=sqrt(n_fft) 2=window-l2-energy

    Mat window_data;

    // power-of-two fast path tables (fft_is_fast(n_fft)), see spectrogram_fft.h
    Mat fft_rev;     // n_fft/2-point bit-reversal permutation
    Mat fft_tw;      // n_fft/2-point twiddles, n_fft/4 (re, im) pairs
    Mat fft_tw_real; // n-point twiddles, n_fft/2+1 (re, im) pairs
};

} // namespace ncnn

#endif // LAYER_INVERSESPECTROGRAM_H
