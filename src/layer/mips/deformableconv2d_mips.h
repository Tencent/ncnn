// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DEFORMABLECONV2D_MIPS_H
#define LAYER_DEFORMABLECONV2D_MIPS_H

#include "deformableconv2d.h"

namespace ncnn {

class DeformableConv2D_mips : public DeformableConv2D
{
public:
    DeformableConv2D_mips();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_DEFORMABLECONV2D_MIPS_H
