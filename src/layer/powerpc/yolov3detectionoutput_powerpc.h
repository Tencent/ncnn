// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_YOLOV3DETECTIONOUTPUT_POWERPC_H
#define LAYER_YOLOV3DETECTIONOUTPUT_POWERPC_H

#include "yolov3detectionoutput.h"

namespace ncnn {

class Yolov3DetectionOutput_powerpc : public Yolov3DetectionOutput
{
public:
    Yolov3DetectionOutput_powerpc();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_YOLOV3DETECTIONOUTPUT_POWERPC_H
