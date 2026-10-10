// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SPECTROGRAM_H
#define LAYER_SPECTROGRAM_H

#include "layer.h"

namespace ncnn {

class Spectrogram : public Layer
{
public:
    Spectrogram();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    int n_fft;
    int power;
    int hoplen;
    int winlen;
    int window_type; // 0=ones 1=hann 2=hamming
    int center;
    int pad_type;   // 0=CONSTANT 1=REPLICATE 2=REFLECT
    int normalized; // 0=disabled 1=sqrt(n_fft) 2=window-l2-energy
    int onesided;

    Mat window_data;
};

} // namespace ncnn

#endif // LAYER_SPECTROGRAM_H
