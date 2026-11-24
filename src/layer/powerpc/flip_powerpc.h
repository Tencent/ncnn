// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_FLIP_POWERPC_H
#define LAYER_FLIP_POWERPC_H

#include "flip.h"

namespace ncnn {

class Flip_powerpc : public Flip
{
public:
    Flip_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_FLIP_POWERPC_H
