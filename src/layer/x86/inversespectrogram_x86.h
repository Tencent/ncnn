// Copyright 2026 pchar.cn
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_INVERSESPECTROGRAM_X86_H
#define LAYER_INVERSESPECTROGRAM_X86_H

#include "inversespectrogram.h"

namespace ncnn {

class InverseSpectrogram_x86 : public InverseSpectrogram
{
public:
    InverseSpectrogram_x86();

    virtual int create_pipeline(const Option& opt);

    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    Mat basis_data_real;
    Mat basis_data_imag;

    Layer* gemm_real;
    Layer* gemm_imag;
};

} // namespace ncnn

#endif // LAYER_INVERSESPECTROGRAM_X86_H