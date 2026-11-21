// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "spp_powerpc.h"

namespace ncnn {

SPP_powerpc::SPP_powerpc()
{
    // the spatial pyramid pooling is implemented as a sequence of pooling +
    // concat steps in the generic implementation, which is not packing aware
    // and already delegates to Pooling; keep all base defaults and delegate
}

int SPP_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    return SPP::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
