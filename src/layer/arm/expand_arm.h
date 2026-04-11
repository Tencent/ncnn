// ARM NEON header for Expand
// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_EXPAND_ARM_H
#define LAYER_EXPAND_ARM_H

#include "expand.h"

namespace ncnn {

class Expand_arm : public virtual Expand
{
public:
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_EXPAND_ARM_H
