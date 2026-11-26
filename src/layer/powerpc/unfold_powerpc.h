// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_UNFOLD_POWERPC_H
#define LAYER_UNFOLD_POWERPC_H

#include "unfold.h"

namespace ncnn {

class Unfold_powerpc : public Unfold
{
public:
    Unfold_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_UNFOLD_POWERPC_H
