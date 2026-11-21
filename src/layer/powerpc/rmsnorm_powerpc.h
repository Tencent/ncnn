// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_RMSNORM_POWERPC_H
#define LAYER_RMSNORM_POWERPC_H

#include "rmsnorm.h"

namespace ncnn {

class RMSNorm_powerpc : public RMSNorm
{
public:
    RMSNorm_powerpc();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_RMSNORM_POWERPC_H
