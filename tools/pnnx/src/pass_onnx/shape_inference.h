// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "onnx-ml.pb.h"

namespace pnnx {

namespace onnx2pnnx {

void shape_inference(onnx::ModelProto& model, const std::string& external_data_path, const std::vector<unsigned char>& external_data,
                     const std::vector<std::vector<int64_t> >& input_shapes,
                     const std::vector<std::string>& input_types,
                     const std::vector<std::vector<int64_t> >& input_shapes2,
                     const std::vector<std::string>& input_types2);

} // namespace onnx2pnnx

} // namespace pnnx
