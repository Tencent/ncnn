// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_POWER_POWERPC_H
#define LAYER_POWER_POWERPC_H

#include "power.h"

namespace ncnn {

class Power_powerpc : public Power
{
public:
    Power_powerpc();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_POWER_POWERPC_H
