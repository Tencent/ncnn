// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DEFORMABLECONV2D_POWERPC_H
#define LAYER_DEFORMABLECONV2D_POWERPC_H

#include "deformableconv2d.h"

namespace ncnn {

class DeformableConv2D_powerpc : public DeformableConv2D
{
public:
    DeformableConv2D_powerpc();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_DEFORMABLECONV2D_POWERPC_H
