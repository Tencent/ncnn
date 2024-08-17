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

class GELU : public FuseFunctionPass
{
public:
    const char* match_type_str() const
    {
        return "nn.GELU";
    }

    const char* type_str() const
    {
        return "nn.GELU";
    }

    void write(Operator* op, const OnnxFunctionProxy& function) const
    {
        bool approximate_none = function.has_typed_node("_aten_gelu_approximate_none");
        bool approximate_tanh = function.has_typed_node("_aten_gelu_approximate_tanh");

        if (approximate_none)
            op->params["approximate"] = "none";

        if (approximate_tanh)
            op->params["approximate"] = "tanh";
    }
};

REGISTER_GLOBAL_PNNX_FUSE_FUNCTION_PASS(GELU)

} // namespace onnx2pnnx

} // namespace pnnx
