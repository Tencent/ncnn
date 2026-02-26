// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_INTERP_MIPS_H
#define LAYER_INTERP_MIPS_H

#include "interp.h"

namespace ncnn {

class Interp_mips : public Interp
{
public:
    Interp_mips();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_INTERP_MIPS_H
