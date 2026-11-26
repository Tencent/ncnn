// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "split_powerpc.h"

namespace ncnn {

Split_powerpc::Split_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
    support_bf16_storage = true;
#endif // __ALTIVEC__
}

int Split_powerpc::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    // reference assignment, packing is preserved, delegate to the base class
    return Split::forward(bottom_blobs, top_blobs, opt);
}

} // namespace ncnn
