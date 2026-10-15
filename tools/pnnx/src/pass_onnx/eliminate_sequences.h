// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <string>
#include <vector>

#include "onnx-ml.pb.h"

namespace pnnx {

namespace onnx2pnnx {

// Lower only statically enumerable sequences with fully supported consumers.
void eliminate_sequences(onnx::ModelProto& model, const std::string& external_data_path = std::string(), const std::vector<unsigned char>& external_data = std::vector<unsigned char>());

} // namespace onnx2pnnx

} // namespace pnnx
