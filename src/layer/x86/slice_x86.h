// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SLICE_X86_H
#define LAYER_SLICE_X86_H

#include "slice.h"

namespace ncnn {

class Slice_x86 : public Slice
{
public:
    Slice_x86();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SLICE_X86_H
