// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_PADDING_POWERPC_H
#define LAYER_PADDING_POWERPC_H

#include "padding.h"

namespace ncnn {

class Padding_powerpc : public Padding
{
public:
    Padding_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_PADDING_POWERPC_H
