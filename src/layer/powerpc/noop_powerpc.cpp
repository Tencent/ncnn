// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "noop_powerpc.h"

namespace ncnn {

Noop_powerpc::Noop_powerpc()
{
#if __ALTIVEC__
    support_inplace = true;
    support_packing = true;
    support_bf16_storage = true;
#endif // __ALTIVEC__
}

int Noop_powerpc::forward_inplace(std::vector<Mat>& /*bottom_top_blobs*/, const Option& /*opt*/) const
{
    // no operation, packing is preserved
    return 0;
}

} // namespace ncnn
