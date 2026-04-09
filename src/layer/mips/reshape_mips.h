// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_RESHAPE_MIPS_H
#define LAYER_RESHAPE_MIPS_H

#include "reshape.h"

namespace ncnn {

class Reshape_mips : public Reshape
{
public:
    Reshape_mips();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_RESHAPE_MIPS_H
