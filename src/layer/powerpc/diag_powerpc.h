// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DIAG_POWERPC_H
#define LAYER_DIAG_POWERPC_H

#include "diag.h"

namespace ncnn {

class Diag_powerpc : public Diag
{
public:
    Diag_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_DIAG_POWERPC_H
