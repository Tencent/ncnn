// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CAST_ARM_H
#define LAYER_CAST_ARM_H

#include "cast.h"

namespace ncnn {

class Cast_arm : public Cast
{
public:
    Cast_arm();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_CAST_ARM_H
