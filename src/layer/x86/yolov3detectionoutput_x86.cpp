// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause
#if __AVX__
#include <immintrin.h>
#endif

#include "yolov3detectionoutput_x86.h"

#include <float.h>

namespace ncnn {

#include "yolov3detectionoutput_fp32.h"

Yolov3DetectionOutput_x86::Yolov3DetectionOutput_x86()
{
}

int Yolov3DetectionOutput_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    return yolov3detectionoutput_fp32(*this, bottom_blobs, top_blobs, opt);
}

} // namespace ncnn
