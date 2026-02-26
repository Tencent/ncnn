// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CLIP_RISCV_H
#define LAYER_CLIP_RISCV_H

#include "clip.h"

namespace ncnn {

class Clip_riscv : public Clip
{
public:
    Clip_riscv();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_ZFH
    int forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_CLIP_RISCV_H
