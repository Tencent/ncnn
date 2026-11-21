// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "yolodetectionoutput_powerpc.h"

namespace ncnn {

YoloDetectionOutput_powerpc::YoloDetectionOutput_powerpc()
{
    // the base class is not packing aware, support_packing stays false
}

int YoloDetectionOutput_powerpc::forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
    // YoloDetectionOutput is dominated by per-box scalar logic, qsort and NMS, delegate to the generic implementation
    // int8 / bf16 inputs fall through to the base class as well

    // defensive unpack, the base class is not packing aware so inputs are elempack 1 in practice
    bool need_unpack = false;
    for (size_t i = 0; i < bottom_top_blobs.size(); i++)
    {
        if (bottom_top_blobs[i].elempack != 1)
        {
            need_unpack = true;
            break;
        }
    }

    if (need_unpack)
    {
        std::vector<Mat> bottoms(bottom_top_blobs.size());
        for (size_t i = 0; i < bottom_top_blobs.size(); i++)
        {
            if (bottom_top_blobs[i].elempack != 1)
            {
                Mat unpacked;
                convert_packing(bottom_top_blobs[i], unpacked, 1, opt);
                if (unpacked.empty())
                    return -100;

                bottoms[i] = unpacked;
            }
            else
            {
                bottoms[i] = bottom_top_blobs[i];
            }
        }

        int ret = YoloDetectionOutput::forward_inplace(bottoms, opt);
        if (ret == 0)
        {
            // propagate the results, Mat assignment is a refcounted shallow copy
            bottom_top_blobs = bottoms;
        }

        return ret;
    }

    return YoloDetectionOutput::forward_inplace(bottom_top_blobs, opt);
}

} // namespace ncnn
