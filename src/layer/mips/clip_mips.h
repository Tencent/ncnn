// Copyright 2019 Leo <leo@nullptr.com.cn>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CLIP_MIPS_H
#define LAYER_CLIP_MIPS_H

#include "clip.h"

namespace ncnn {

class Clip_mips : public Clip
{
public:
    Clip_mips();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_CLIP_MIPS_H
