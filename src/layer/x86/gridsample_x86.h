// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GRIDSAMPLE_X86_H
#define LAYER_GRIDSAMPLE_X86_H

#include "gridsample.h"

namespace ncnn {

class GridSample_x86 : public GridSample
{
public:
    GridSample_x86();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_GRIDSAMPLE_X86_H
