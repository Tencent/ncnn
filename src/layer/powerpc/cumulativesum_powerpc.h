// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CUMULATIVESUM_POWERPC_H
#define LAYER_CUMULATIVESUM_POWERPC_H

#include "cumulativesum.h"

namespace ncnn {

class CumulativeSum_powerpc : public CumulativeSum
{
public:
    CumulativeSum_powerpc();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_CUMULATIVESUM_POWERPC_H
