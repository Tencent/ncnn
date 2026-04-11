// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_MOD_ARM_H
#define LAYER_MOD_ARM_H

#include "mod.h"

namespace ncnn {

class Mod_arm : public Mod
{
public:
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_MOD_ARM_H
