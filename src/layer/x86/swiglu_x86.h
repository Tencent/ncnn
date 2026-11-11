// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SWIGLU_X86_H
#define LAYER_SWIGLU_X86_H

#include "swiglu.h"

namespace ncnn {

class SwiGLU_x86 : public SwiGLU
{
public:
    SwiGLU_x86();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SWIGLU_X86_H
