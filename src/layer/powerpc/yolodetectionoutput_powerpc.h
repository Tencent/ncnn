// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_YOLODETECTIONOUTPUT_POWERPC_H
#define LAYER_YOLODETECTIONOUTPUT_POWERPC_H

#include "yolodetectionoutput.h"

namespace ncnn {

class YoloDetectionOutput_powerpc : public YoloDetectionOutput
{
public:
    YoloDetectionOutput_powerpc();

    virtual int forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_YOLODETECTIONOUTPUT_POWERPC_H
