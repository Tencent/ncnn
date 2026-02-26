// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CAST_X86_H
#define LAYER_CAST_X86_H

#include "cast.h"

namespace ncnn {

class Cast_x86 : public Cast
{
public:
    Cast_x86();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_CAST_X86_H
