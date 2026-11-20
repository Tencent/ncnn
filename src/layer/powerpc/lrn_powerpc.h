// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_LRN_POWERPC_H
#define LAYER_LRN_POWERPC_H

#include "lrn.h"

namespace ncnn {

class LRN_powerpc : public LRN
{
public:
    LRN_powerpc();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_LRN_POWERPC_H
