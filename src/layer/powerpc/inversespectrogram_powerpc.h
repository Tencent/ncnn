// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_INVERSESPECTROGRAM_POWERPC_H
#define LAYER_INVERSESPECTROGRAM_POWERPC_H

#include "inversespectrogram.h"

namespace ncnn {

class InverseSpectrogram_powerpc : public InverseSpectrogram
{
public:
    InverseSpectrogram_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_INVERSESPECTROGRAM_POWERPC_H
