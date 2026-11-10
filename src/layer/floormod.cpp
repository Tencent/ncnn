// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "floormod.h"

namespace ncnn {

FloorMod::FloorMod()
{
    floor_remainder = true;
}

int FloorMod::load_param(const ParamDict& pd)
{
    int reverse = pd.get(0, 0);
#if NCNN_VALIDATION
    if (reverse < 0 || reverse > 1)
        return -1;
#endif // NCNN_VALIDATION

    ParamDict pd_binary = pd;
    pd_binary.set(0, reverse ? Operation_RREMAINDER : Operation_REMAINDER);
    return BinaryOp::load_param(pd_binary);
}

} // namespace ncnn
