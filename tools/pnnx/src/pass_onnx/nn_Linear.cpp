// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

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
