// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "onnx-ml.pb.h"

namespace pnnx {

namespace onnx2pnnx {

void eliminate_initializer_input(onnx::ModelProto& model);

} // namespace onnx2pnnx

} // namespace pnnx
