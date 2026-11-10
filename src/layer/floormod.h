// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_FLOORMOD_H
#define LAYER_FLOORMOD_H

#include "binaryop.h"

namespace ncnn {

class FloorMod : public BinaryOp
{
public:
    FloorMod();
    virtual int load_param(const ParamDict& pd);
};

} // namespace ncnn

#endif // LAYER_FLOORMOD_H
