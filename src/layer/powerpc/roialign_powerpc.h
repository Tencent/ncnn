// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ROIALIGN_POWERPC_H
#define LAYER_ROIALIGN_POWERPC_H

#include "roialign.h"

namespace ncnn {

class ROIAlign_powerpc : public ROIAlign
{
public:
    ROIAlign_powerpc();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_ROIALIGN_POWERPC_H
