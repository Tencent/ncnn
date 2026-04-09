// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ROIALIGN_MIPS_H
#define LAYER_ROIALIGN_MIPS_H

#include "roialign.h"

namespace ncnn {

class ROIAlign_mips : public ROIAlign
{
public:
    ROIAlign_mips();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_ROIALIGN_MIPS_H
