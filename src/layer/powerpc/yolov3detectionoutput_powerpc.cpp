// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "yolov3detectionoutput_powerpc.h"

namespace ncnn {

Yolov3DetectionOutput_powerpc::Yolov3DetectionOutput_powerpc()
{
    // the base class is not packing aware, support_packing stays false
}

int Yolov3DetectionOutput_powerpc::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    // Yolov3DetectionOutput is dominated by per-box scalar logic, qsort and NMS, delegate to the generic implementation
    // int8 / bf16 inputs fall through to the base class as well

    // defensive unpack, the base class is not packing aware so inputs are elempack 1 in practice
    bool need_unpack = false;
    for (size_t i = 0; i < bottom_blobs.size(); i++)
    {
        if (bottom_blobs[i].elempack != 1)
        {
            need_unpack = true;
            break;
        }
    }

    if (need_unpack)
    {
        std::vector<Mat> bottoms(bottom_blobs.size());
        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            if (bottom_blobs[i].elempack != 1)
            {
                Mat unpacked;
                convert_packing(bottom_blobs[i], unpacked, 1, opt);
                if (unpacked.empty())
                    return -100;

                bottoms[i] = unpacked;
            }
            else
            {
                bottoms[i] = bottom_blobs[i];
            }
        }

        return Yolov3DetectionOutput::forward(bottoms, top_blobs, opt);
    }

    return Yolov3DetectionOutput::forward(bottom_blobs, top_blobs, opt);
}

} // namespace ncnn
