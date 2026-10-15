// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <string>
#include <vector>

#include "onnx-ml.pb.h"

namespace pnnx {

namespace onnx2pnnx {

void materialize_external_split_sizes(onnx::ModelProto& model, const std::string& external_data_path, const std::vector<unsigned char>& external_data);
void fuse_constant_as_attribute(onnx::ModelProto& model, const std::string& external_data_path = std::string(), const std::vector<unsigned char>& external_data = std::vector<unsigned char>());

} // namespace onnx2pnnx

} // namespace pnnx
