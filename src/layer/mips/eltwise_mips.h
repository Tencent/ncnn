// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ELTWISE_MIPS_H
#define LAYER_ELTWISE_MIPS_H

#include "eltwise.h"

namespace ncnn {

class Eltwise_mips : public Eltwise
{
public:
    Eltwise_mips();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_ELTWISE_MIPS_H
