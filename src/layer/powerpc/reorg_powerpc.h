// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_REORG_POWERPC_H
#define LAYER_REORG_POWERPC_H

#include "reorg.h"

namespace ncnn {

class Reorg_powerpc : public Reorg
{
public:
    Reorg_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_REORG_POWERPC_H
