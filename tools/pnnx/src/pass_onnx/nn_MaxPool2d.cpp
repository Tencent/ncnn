// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "pass_onnx.h"
#include "ir.h"

#include "onnx-ml.pb.h"

namespace pnnx {

namespace onnx2pnnx {

class MaxPool2d : public FuseFunctionPass
{
public:
    const char* match_type_str() const
    {
        return "nn.MaxPool2d";
    }

    const char* type_str() const
    {
        return "nn.MaxPool2d";
    }

    void write(Operator* op, const OnnxFunctionProxy& function) const
    {
        const OnnxNodeProxy aten_max_pool_with_indices_onnx = function.typed_node("_aten_max_pool_with_indices_onnx");

        std::vector<int64_t> kernel_size = aten_max_pool_with_indices_onnx.attribute("kernel_size");
        std::vector<int64_t> dilation = aten_max_pool_with_indices_onnx.attribute("dilation");
        std::vector<int64_t> stride = aten_max_pool_with_indices_onnx.attribute("stride");
        std::vector<int64_t> padding = aten_max_pool_with_indices_onnx.attribute("padding");
        int64_t ceil_mode = aten_max_pool_with_indices_onnx.attribute("ceil_mode");

        if (padding.size() == 4)
        {
            padding = {padding[0], padding[1]};
        }

        op->params["kernel_size"] = kernel_size;
        op->params["dilation"] = dilation;
        op->params["stride"] = stride;
        op->params["padding"] = padding;
        op->params["ceil_mode"] = (ceil_mode != 0);
        op->params["return_indices"] = (function.function.output_size() != 1);
    }
};

REGISTER_GLOBAL_PNNX_FUSE_FUNCTION_PASS(MaxPool2d)

} // namespace onnx2pnnx

} // namespace pnnx
