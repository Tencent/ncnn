// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ROTARYEMBED_ARM_H
#define LAYER_ROTARYEMBED_ARM_H

#include "rotaryembed.h"

namespace ncnn {

class RotaryEmbed_arm : public RotaryEmbed
{
public:
    RotaryEmbed_arm();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
#if NCNN_ARM82
    int forward_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
    int forward_fp16sa(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_ROTARYEMBED_ARM_H
