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

#include "onnx.pb.h"

namespace pnnx {

namespace onnx2pnnx {

class AdaptiveAvgPool2d : public FuseFunctionPass
{
public:
    const char* match_type_str() const
    {
        return "nn.AdaptiveAvgPool2d";
    }

    const char* type_str() const
    {
        return "nn.AdaptiveAvgPool2d";
    }

    void write(Operator* op, const OnnxFunctionProxy& function) const
    {
        if (function.has_typed_node("aten_mean_dim") && function.function.node_size() == 2)
        {
            op->params["output_size"] = {1,1};
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_FUNCTION_PASS(AdaptiveAvgPool2d)

} // namespace onnx2pnnx

} // namespace pnnx
