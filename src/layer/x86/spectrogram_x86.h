// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SPECTROGRAM_X86_H
#define LAYER_SPECTROGRAM_X86_H

#include "spectrogram.h"

namespace ncnn {

class Spectrogram_x86 : virtual public Spectrogram
{
public:
    Spectrogram_x86();

    virtual int load_param(const ParamDict& pd);

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    // stft as conv1d: kernel=n_fft stride=hoplen
    // out channel [0, freqs)         = real part = window[k] * cos(2*pi*i*k/n_fft)
    // out channel [freqs, 2*freqs)   = imag part = -window[k] * sin(2*pi*i*k/n_fft)
    Layer* conv1d;
    Mat dft_weight;
};

} // namespace ncnn

#endif // LAYER_SPECTROGRAM_X86_H
