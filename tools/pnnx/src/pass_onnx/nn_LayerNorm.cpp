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

class LayerNorm : public FuseFunctionPass
{
public:
    const char* match_type_str() const
    {
        return "nn.LayerNorm";
    }

    const char* type_str() const
    {
        return "nn.LayerNorm";
    }

    void write(Operator* op, const OnnxFunctionProxy& function) const
    {
        const int input_rank = op->inputs[0]->shape.size();

        const OnnxNodeProxy layernormalization = function.typed_node("LayerNormalization");

        int64_t axis = layernormalization.attribute("axis");

        if (axis < 0)
        {
            axis = input_rank + axis;
        }

        std::vector<int> normalized_shape;
        for (int i = axis; i < input_rank; i++)
        {
            normalized_shape.push_back(op->inputs[0]->shape[i]);
        }

        op->params["normalized_shape"] = normalized_shape;
        op->params["eps"] = layernormalization.attribute("epsilon");
        op->params["elementwise_affine"] = function.has_initializer("weight") && function.has_initializer("bias");

        if (function.has_initializer("weight") && function.has_initializer("bias"))
        {
            op->attrs["weight"] = function.initializer("weight");
            op->attrs["bias"] = function.initializer("bias");
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_FUNCTION_PASS(LayerNorm)

} // namespace onnx2pnnx

} // namespace pnnx
