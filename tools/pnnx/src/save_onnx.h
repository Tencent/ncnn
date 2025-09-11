// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_SAVE_ONNX_H
#define PNNX_SAVE_ONNX_H

#include "ir.h"

namespace pnnx {

int save_onnx(const Graph& g, const char* onnxpath, int fp16);

} // namespace pnnx

#endif // PNNX_SAVE_ONNX_H
