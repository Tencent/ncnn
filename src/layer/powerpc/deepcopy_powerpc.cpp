// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "deepcopy_powerpc.h"

namespace ncnn {

DeepCopy_powerpc::DeepCopy_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
#endif // __ALTIVEC__
}

int DeepCopy_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // a clone is a single contiguous memcpy, already optimal on altivec
    return DeepCopy::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
