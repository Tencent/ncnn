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
    /* fp16 arithmetic version for the fused layout (input_hc_swapped):
     * delegating to the generic implementation would upgrade to fp32 compute,
     * changing the requested arithmetic mode and output bits; this version
     * keeps fp16sa semantics. */
    int forward_fp16sa_swapped(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
#endif
#if NCNN_BF16
    int forward_bf16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_ROTARYEMBED_ARM_H
