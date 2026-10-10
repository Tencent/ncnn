// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ROIALIGN_ARM_H
#define LAYER_ROIALIGN_ARM_H

#include "roialign.h"

namespace ncnn {

class ROIAlign_arm : public ROIAlign
{
public:
    ROIAlign_arm();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_ROIALIGN_ARM_H
