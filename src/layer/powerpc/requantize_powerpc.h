// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_REQUANTIZE_POWERPC_H
#define LAYER_REQUANTIZE_POWERPC_H

#include "requantize.h"

namespace ncnn {

class Requantize_powerpc : public Requantize
{
public:
    Requantize_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_REQUANTIZE_POWERPC_H
