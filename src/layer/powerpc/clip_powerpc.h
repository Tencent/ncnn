// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CLIP_POWERPC_H
#define LAYER_CLIP_POWERPC_H

#include "clip.h"

namespace ncnn {

class Clip_powerpc : public Clip
{
public:
    Clip_powerpc();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_BF16
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_CLIP_POWERPC_H
