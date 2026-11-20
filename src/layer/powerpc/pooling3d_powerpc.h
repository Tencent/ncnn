// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_POOLING3D_POWERPC_H
#define LAYER_POOLING3D_POWERPC_H

#include "pooling3d.h"

namespace ncnn {

class Pooling3D_powerpc : public Pooling3D
{
public:
    Pooling3D_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_POOLING3D_POWERPC_H
