// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "diag_powerpc.h"

namespace ncnn {

Diag_powerpc::Diag_powerpc()
{
    // base defaults kept: one_blob_only = true, support_inplace = false
    // support_packing stays false: the diagonal output is a plain 1d/2d blob
    // without a packing axis, so there is no pack4 output shape to claim
}

int Diag_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (bottom_blob.elembits() == 32 && bottom_blob.elempack != 1)
    {
        // unpack + base implementation, the output blob stays unpacked
        Mat bottom_blob_unpacked;
        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt);
        if (bottom_blob_unpacked.empty())
            return -100;

        return Diag::forward(bottom_blob_unpacked, top_blob, opt);
    }

    return Diag::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
