// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_FOLD_POWERPC_H
#define LAYER_FOLD_POWERPC_H

#include "fold.h"

namespace ncnn {

class Fold_powerpc : public Fold
{
public:
    Fold_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_FOLD_POWERPC_H
