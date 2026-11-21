// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_QUANTIZE_POWERPC_H
#define LAYER_QUANTIZE_POWERPC_H

#include "quantize.h"

namespace ncnn {

class Quantize_powerpc : public Quantize
{
public:
    Quantize_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_QUANTIZE_POWERPC_H
