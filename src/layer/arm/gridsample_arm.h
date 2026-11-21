// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GRIDSAMPLE_ARM_H
#define LAYER_GRIDSAMPLE_ARM_H

#include "gridsample.h"

namespace ncnn {

class GridSample_arm : public GridSample
{
public:
    GridSample_arm();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_GRIDSAMPLE_ARM_H
