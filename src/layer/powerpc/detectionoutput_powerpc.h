// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DETECTIONOUTPUT_POWERPC_H
#define LAYER_DETECTIONOUTPUT_POWERPC_H

#include "detectionoutput.h"

namespace ncnn {

class DetectionOutput_powerpc : public DetectionOutput
{
public:
    DetectionOutput_powerpc();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_DETECTIONOUTPUT_POWERPC_H
