// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_THRESHOLD_POWERPC_H
#define LAYER_THRESHOLD_POWERPC_H

#include "threshold.h"

namespace ncnn {

class Threshold_powerpc : public Threshold
{
public:
    Threshold_powerpc();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_THRESHOLD_POWERPC_H
