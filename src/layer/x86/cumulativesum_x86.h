// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CUMULATIVESUM_X86_H
#define LAYER_CUMULATIVESUM_X86_H

#include "cumulativesum.h"

namespace ncnn {

class CumulativeSum_x86 : public CumulativeSum
{
public:
    CumulativeSum_x86();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_CUMULATIVESUM_X86_H
