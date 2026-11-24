// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_INSTANCENORM_POWERPC_H
#define LAYER_INSTANCENORM_POWERPC_H

#include "instancenorm.h"

namespace ncnn {

class InstanceNorm_powerpc : public InstanceNorm
{
public:
    InstanceNorm_powerpc();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_INSTANCENORM_POWERPC_H
