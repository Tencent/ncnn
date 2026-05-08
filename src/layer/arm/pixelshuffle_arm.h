// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_PIXELSHUFFLE_ARM_H
#define LAYER_PIXELSHUFFLE_ARM_H

#include "pixelshuffle.h"

namespace ncnn {

class PixelShuffle_arm : public PixelShuffle
{
public:
    PixelShuffle_arm();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

protected:
    int forward_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_PIXELSHUFFLE_ARM_H
