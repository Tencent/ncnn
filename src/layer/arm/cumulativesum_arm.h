// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CUMULATIVESUM_ARM_H
#define LAYER_CUMULATIVESUM_ARM_H

#include "cumulativesum.h"

namespace ncnn {

class CumulativeSum_arm : public CumulativeSum
{
public:
    CumulativeSum_arm();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_CUMULATIVESUM_ARM_H
