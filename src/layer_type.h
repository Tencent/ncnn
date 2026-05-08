// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_LAYER_TYPE_H
#define NCNN_LAYER_TYPE_H

namespace ncnn {

namespace LayerType {
enum LayerType
{
#include "layer_type_enum.h"
    CustomBit = (1 << 8),
};
} // namespace LayerType

} // namespace ncnn

#endif // NCNN_LAYER_TYPE_H
