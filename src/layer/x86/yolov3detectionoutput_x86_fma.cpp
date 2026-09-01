// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "yolov3detectionoutput_x86.h"

#include <float.h>

#include "cpu.h"

namespace ncnn {

#include "yolov3detectionoutput_fp32.h"

int yolov3detectionoutput_fp32_fma(const Yolov3DetectionOutput& self, const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt)
{
    return yolov3detectionoutput_fp32(self, bottom_blobs, top_blobs, opt);
}

} // namespace ncnn
