// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_BNLL_POWERPC_H
#define LAYER_BNLL_POWERPC_H

#include "bnll.h"

namespace ncnn {

class BNLL_powerpc : public BNLL
{
public:
    BNLL_powerpc();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_BNLL_POWERPC_H
