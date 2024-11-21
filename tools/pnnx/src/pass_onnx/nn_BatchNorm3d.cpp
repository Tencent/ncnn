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
