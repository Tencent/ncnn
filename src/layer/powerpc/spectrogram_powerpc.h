// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SPECTROGRAM_POWERPC_H
#define LAYER_SPECTROGRAM_POWERPC_H

#include "spectrogram.h"

namespace ncnn {

class Spectrogram_powerpc : public Spectrogram
{
public:
    Spectrogram_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SPECTROGRAM_POWERPC_H
