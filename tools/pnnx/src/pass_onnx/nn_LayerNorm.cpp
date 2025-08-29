// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

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
