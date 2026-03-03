// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_LOAD_ONNX_H
#define PNNX_LOAD_ONNX_H

#include "ir.h"

namespace pnnx {

int load_onnx(const std::string& onnxpath, Graph& g,
              const std::vector<std::vector<int64_t> >& input_shapes,
              const std::vector<std::string>& input_types,
              const std::vector<std::vector<int64_t> >& input_shapes2,
              const std::vector<std::string>& input_types2);

} // namespace pnnx

#endif // PNNX_LOAD_ONNX_H
