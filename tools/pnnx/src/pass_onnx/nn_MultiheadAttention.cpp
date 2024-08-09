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

class MultiheadAttention : public FuseFunctionPass
{
public:
    const char* match_type_str() const
    {
        return "nn.MultiheadAttention";
    }

    const char* type_str() const
    {
        return "nn.MultiheadAttention";
    }

    void write(Operator* op, const OnnxFunctionProxy& function) const
    {
        const OnnxNodeProxy attention_scale = function.typed_node("_attention_scale");

        const OnnxNodeProxy reshape_heads = function.find_producer(attention_scale.node.input(0));

        const OnnxNodeProxy constant_shape = function.find_producer(reshape_heads.node.input(1));

        if (constant_shape.node.op_type() == "Constant")
        {
            std::vector<int64_t> shape = constant_shape.attribute("value");
            op->params["num_heads"] = shape[1];
        }

        const OnnxNodeProxy transpose = function.typed_node("Transpose");
        std::vector<int64_t> perm = transpose.attribute("perm");
        if (perm == std::vector<int64_t> {1, 0, 2})
        {
            op->params["batch_first"] = true;
        }
        else
        {
            op->params["batch_first"] = false;
        }

        op->params["add_zero_attn"] = false; // TODO

        if (function.has_typed_node("_aten_scaled_dot_product_attention_no_mask_onnx"))
        {
            // TODO handle attn_mask
        }

        if (function.has_initializer("in_proj_weight"))
        {
            const onnx::TensorProto& in_proj_weight = function.initializer("in_proj_weight");

            op->params["embed_dim"] = in_proj_weight.dims(1);
            op->params["kdim"] = in_proj_weight.dims(1);
            op->params["vdim"] = in_proj_weight.dims(1);
            op->attrs["in_proj_weight"] = in_proj_weight;
        }
        else
        {
            const onnx::TensorProto& q_proj_weight = function.initializer("q_proj_weight");
            const onnx::TensorProto& k_proj_weight = function.initializer("k_proj_weight");
            const onnx::TensorProto& v_proj_weight = function.initializer("v_proj_weight");

            op->params["embed_dim"] = q_proj_weight.dims(1);
            op->params["kdim"] = k_proj_weight.dims(1);
            op->params["vdim"] = v_proj_weight.dims(1);
            op->attrs["q_proj_weight"] = q_proj_weight;
            op->attrs["k_proj_weight"] = k_proj_weight;
            op->attrs["v_proj_weight"] = v_proj_weight;
        }

        op->attrs["out_proj.weight"] = function.initializer("weight");

        if (function.has_initializer("in_proj_bias") && function.has_initializer("bias"))
        {
            op->params["bias"] = true;
            op->attrs["in_proj_bias"] = function.initializer("in_proj_bias");
            op->attrs["out_proj.bias"] = function.initializer("bias");
        }
        else
        {
            op->params["bias"] = false;
        }

        if (function.has_initializer("bias_k") && function.has_initializer("bias_v"))
        {
            op->params["add_bias_kv"] = true;
            op->attrs["bias_k"] = function.initializer("bias_k");
            op->attrs["bias_v"] = function.initializer("bias_v");
        }
        else
        {
            op->params["add_bias_kv"] = false;
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_FUNCTION_PASS(MultiheadAttention)

} // namespace onnx2pnnx

} // namespace pnnx
