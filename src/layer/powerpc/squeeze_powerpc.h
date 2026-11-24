// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SQUEEZE_POWERPC_H
#define LAYER_SQUEEZE_POWERPC_H

#include "squeeze.h"

namespace ncnn {

class Squeeze_powerpc : public Squeeze
{
public:
    Squeeze_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SQUEEZE_POWERPC_H
