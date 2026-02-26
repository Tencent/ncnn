// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GROUPNORM_X86_H
#define LAYER_GROUPNORM_X86_H

#include "groupnorm.h"

namespace ncnn {

class GroupNorm_x86 : public GroupNorm
{
public:
    GroupNorm_x86();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_GROUPNORM_X86_H
