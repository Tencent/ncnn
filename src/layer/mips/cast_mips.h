// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CAST_MIPS_H
#define LAYER_CAST_MIPS_H

#include "cast.h"

namespace ncnn {

class Cast_mips : public Cast
{
public:
    Cast_mips();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_CAST_MIPS_H
