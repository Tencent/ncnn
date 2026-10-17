// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ROTARYEMBED_RISCV_H
#define LAYER_ROTARYEMBED_RISCV_H

#include "rotaryembed.h"

namespace ncnn {

class RotaryEmbed_riscv : public RotaryEmbed
{
public:
    RotaryEmbed_riscv();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_ROTARYEMBED_RISCV_H
