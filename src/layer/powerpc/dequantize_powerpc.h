// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DEQUANTIZE_POWERPC_H
#define LAYER_DEQUANTIZE_POWERPC_H

#include "dequantize.h"

namespace ncnn {

class Dequantize_powerpc : public Dequantize
{
public:
    Dequantize_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_DEQUANTIZE_POWERPC_H
