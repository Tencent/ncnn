// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ERF_POWERPC_H
#define LAYER_ERF_POWERPC_H

#include "erf.h"

namespace ncnn {

class Erf_powerpc : public Erf
{
public:
    Erf_powerpc();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_ERF_POWERPC_H
