// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GLU_X86_H
#define LAYER_GLU_X86_H

#include "glu.h"

namespace ncnn {

class GLU_x86 : public GLU
{
public:
    GLU_x86();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_GLU_X86_H
