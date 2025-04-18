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

class Linear : public FuseFunctionPass
{
public:
    const char* match_type_str() const
    {
        return "nn.Linear";
    }

    const char* type_str() const
    {
        return "nn.Linear";
    }

    void write(Operator* op, const OnnxFunctionProxy& function) const
    {
        const onnx::TensorProto& weight = function.initializer("weight");

        op->params["in_features"] = weight.dims(1);
        op->params["out_features"] = weight.dims(0);
        op->params["bias"] = function.has_initializer("bias");

        op->attrs["weight"] = weight;
        if (function.has_initializer("bias"))
        {
            op->attrs["bias"] = function.initializer("bias");
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_FUNCTION_PASS(Linear)

} // namespace onnx2pnnx

} // namespace pnnx
