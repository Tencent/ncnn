// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SLICE_MIPS_H
#define LAYER_SLICE_MIPS_H

#include "slice.h"

namespace ncnn {

class Slice_mips : public Slice
{
public:
    Slice_mips();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SLICE_MIPS_H
