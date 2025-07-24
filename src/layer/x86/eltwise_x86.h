// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ELTWISE_X86_H
#define LAYER_ELTWISE_X86_H

#include "eltwise.h"

namespace ncnn {

class Eltwise_x86 : public Eltwise
{
public:
    Eltwise_x86();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_ELTWISE_X86_H
