// Copyright 2026 pchar.cn
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SPECTROGRAM_X86_H
#define LAYER_SPECTROGRAM_X86_H

#include "spectrogram.h"

namespace ncnn {

class Spectrogram_x86 : public Spectrogram
{
public:
    Spectrogram_x86();

    virtual int create_pipeline(const Option& opt);

    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    int n_freq;

    Mat basis_data;

    Layer* unfold;
    Layer* gemm;
};

} // namespace ncnn

#endif // LAYER_SPECTROGRAM_X86_H