// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "split.h"
#include "cpu.h"

namespace ncnn {

Split::Split()
{
    one_blob_only = false;
    support_inplace = false;
    support_packing = true;
    support_fp16_storage = cpu_support_arm_asimdhp() || cpu_support_riscv_zvfh();
    support_bf16_storage = true;
}

int Split::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& /*opt*/) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    for (size_t i = 0; i < top_blobs.size(); i++)
    {
        top_blobs[i] = bottom_blob;
    }

    return 0;
}

} // namespace ncnn
