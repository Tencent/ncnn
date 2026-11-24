// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ROIPOOLING_POWERPC_H
#define LAYER_ROIPOOLING_POWERPC_H

#include "roipooling.h"

namespace ncnn {

class ROIPooling_powerpc : public ROIPooling
{
public:
    ROIPooling_powerpc();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_ROIPOOLING_POWERPC_H
