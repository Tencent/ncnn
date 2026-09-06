// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_INVERSESPECTROGRAM_X86_H
#define LAYER_INVERSESPECTROGRAM_X86_H

#include "inversespectrogram.h"

namespace ncnn {

class InverseSpectrogram_x86 : virtual public InverseSpectrogram
{
public:
    InverseSpectrogram_x86();

    virtual int load_param(const ParamDict& pd);

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    // istft per-frame idft as conv1d 1x1 (sgemm): out[t][m] = sum_q sp[q][t] * W[q][m]
    // num_input = 2*n_fft (re bins then im bins), num_output = 2*n_fft (re taps then im taps)
    // overlap-add over the hoplen-shifted taps gives the waveform
    Layer* conv1d;
    Mat idft_weight;
};

} // namespace ncnn

#endif // LAYER_INVERSESPECTROGRAM_X86_H
