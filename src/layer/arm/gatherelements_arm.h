// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GATHERELEMENTS_ARM_H
#define LAYER_GATHERELEMENTS_ARM_H

#include "gatherelements.h"

namespace ncnn {

class GatherElements_arm : public GatherElements
{
public:
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_GATHERELEMENTS_ARM_H
