// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_onnx.h"
#include "ir.h"

#include "onnx-ml.pb.h"

namespace pnnx {

namespace onnx2pnnx {

class BatchNorm3d : public FuseFunctionPass
{
public:
    const char* match_type_str() const
    {
        return "nn.BatchNorm3d";
    }

    const char* type_str() const
    {
        return "nn.BatchNorm3d";
    }

    void write(Operator* op, const OnnxFunctionProxy& function) const
    {
        float eps;
        if (function.has_typed_node("_aten_native_batch_norm_inference_onnx"))
        {
            const OnnxNodeProxy aten_native_batch_norm_inference_onnx = function.typed_node("_aten_native_batch_norm_inference_onnx");
            eps = aten_native_batch_norm_inference_onnx.attribute("eps");
        }
        else
        {
            const OnnxNodeProxy add_eps = function.named_node("aten_add_5");
            eps = function.find_producer(add_eps.node.input(1)).attribute("value");
        }

        const onnx::TensorProto& running_mean = function.initializer("running_mean");
        const onnx::TensorProto& running_var = function.initializer("running_var");

        op->params["num_features"] = running_mean.dims(0);
        op->params["eps"] = eps;
        op->params["affine"] = function.has_initializer("weight") && function.has_initializer("bias");

        op->attrs["running_mean"] = running_mean;
        op->attrs["running_var"] = running_var;
        if (function.has_initializer("weight") && function.has_initializer("bias"))
        {
            op->attrs["weight"] = function.initializer("weight");
            op->attrs["bias"] = function.initializer("bias");
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_FUNCTION_PASS(BatchNorm3d)

} // namespace onnx2pnnx

} // namespace pnnx
