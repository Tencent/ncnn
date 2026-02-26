// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CLIP_ARM_H
#define LAYER_CLIP_ARM_H

#include "clip.h"

namespace ncnn {

class Clip_arm : public Clip
{
public:
    Clip_arm();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_ARM82
    int forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
#if NCNN_BF16
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_CLIP_ARM_H
