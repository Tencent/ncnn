// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "argmax_powerpc.h"

namespace ncnn {

ArgMax_powerpc::ArgMax_powerpc()
{
    // base defaults kept: one_blob_only = true
    // the topk partial sort with index tracking is inherently sequential per
    // lane group, and the output is a small (topk, 1) index blob that is never
    // packed; keep all base defaults and only unpack a pack4 input
}

int ArgMax_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (bottom_blob.elembits() == 32 && bottom_blob.elempack != 1)
    {
        // unpack + base implementation, the output blob stays unpacked
        Mat bottom_blob_unpacked;
        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt);
        if (bottom_blob_unpacked.empty())
            return -100;

        return ArgMax::forward(bottom_blob_unpacked, top_blob, opt);
    }

    return ArgMax::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
