// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "pass_level2.h"

namespace pnnx {

class F_layer_norm : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
pnnx.Input              input_3     0 1 normalized_shape
prim::Constant          op_0        0 1 eps value=%eps
prim::Constant          op_1        0 1 cudnn_enabled value=*
aten::layer_norm        op_2        6 1 input normalized_shape weight bias eps cudnn_enabled out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.layer_norm";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_layer_norm, 10)

class F_layer_norm_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
LayerNormalization      op_0        3 1 input weight bias out axis=%axis epsilon=%epsilon
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.layer_norm";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const int input_rank = op->inputs[0]->shape.size();

        int axis = captured_params.at("axis").i;
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
        op->params["eps"] = captured_params.at("epsilon");
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_layer_norm_onnx, 10)

class F_layer_norm_onnx_1 : public F_layer_norm_onnx
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 6
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
LayerNormalization      op_0        3 3 input weight bias out Mean InvStdDev axis=%axis epsilon=%epsilon stash_type=%stash_type
pnnx.Output             output      3 0 out Mean InvStdDev
)PNNXIR";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        F_layer_norm_onnx::write(op, captured_params);

        // drop Mean and InvStdDev if not used
        if (op->outputs[1]->consumers.empty() && op->outputs[2]->consumers.empty())
        {
            op->outputs[1]->producer = 0;
            op->outputs[2]->producer = 0;
            op->outputs.resize(1);
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_layer_norm_onnx_1, 10)

class F_layer_norm_onnx_2 : public F_layer_norm_onnx
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
LayerNormalization      op_0        3 1 input weight bias out axis=%axis epsilon=%epsilon stash_type=%stash_type
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_layer_norm_onnx_2, 10)

class F_layer_norm_onnx_3 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
11 10
pnnx.Input              input       0 1 input
torch.mean              mean        1 1 input mean dim=%dim keepdim=True
aten::sub               op_1        2 1 input mean pnnx_1
prim::Constant          op_2        0 1 two value=2.000000e+00
aten::pow               op_3        2 1 pnnx_1 two pnnx_2
torch.mean              op_4        1 1 pnnx_2 var dim=%dim keepdim=True
prim::Constant          op_5        0 1 eps value=%eps
aten::add               op_6        2 1 var eps pnnx_4
aten::sqrt              op_7        1 1 pnnx_4 pnnx_5
aten::div               op_8        2 1 pnnx_1 pnnx_5 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.layer_norm";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        const Operator* op_mean = matched_operators.at("mean");
        const std::vector<int>& inputshape = op_mean->inputs[0]->shape;
        if (inputshape.empty())
            return false;

        // dim must be the last N dimensions
        std::vector<int> dim = captured_params.at("dim").ai;

        const int input_rank = (int)inputshape.size();
        const int dim_count = (int)dim.size();

        for (int i = 0; i < dim_count; i++)
        {
            if (dim[i] < 0)
                dim[i] += input_rank;

            if (dim[i] < input_rank - dim_count)
                return false;
        }

        std::vector<int> normalized_shape(dim_count);
        for (int i = 0; i < dim_count; i++)
        {
            normalized_shape[i] = inputshape[input_rank - dim_count + i];
        }

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& inputshape = op->inputs[0]->shape;
        const std::vector<int>& dim = captured_params.at("dim").ai;
        const int input_rank = (int)inputshape.size();
        const int dim_count = (int)dim.size();

        std::vector<int> normalized_shape(dim_count);
        for (int i = 0; i < dim_count; i++)
        {
            normalized_shape[i] = inputshape[input_rank - dim_count + i];
        }

        op->params["normalized_shape"] = normalized_shape;
        op->params["eps"] = captured_params.at("eps");
        op->params["weight"] = Parameter();
        op->params["bias"] = Parameter();
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_layer_norm_onnx_3, 30)

class F_layer_norm_onnx_4 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
15 14
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
torch.mean              mean        1 1 input mean dim=%dim keepdim=True
aten::sub               op_1        2 1 input mean pnnx_1
prim::Constant          op_2        0 1 two value=2.000000e+00
aten::pow               op_3        2 1 pnnx_1 two pnnx_2
torch.mean              op_4        1 1 pnnx_2 var dim=%dim keepdim=True
prim::Constant          op_5        0 1 eps value=%eps
aten::add               op_6        2 1 var eps pnnx_4
aten::sqrt              op_7        1 1 pnnx_4 pnnx_5
aten::div               op_8        2 1 pnnx_1 pnnx_5 pnnx_6
aten::mul               mul         2 1 pnnx_6 weight pnnx_7
aten::add               add         2 1 pnnx_7 bias out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.layer_norm";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        const Operator* op_mean = matched_operators.at("mean");
        const std::vector<int>& inputshape = op_mean->inputs[0]->shape;
        if (inputshape.empty())
            return false;

        // dim must be the last N dimensions
        std::vector<int> dim = captured_params.at("dim").ai;

        const int input_rank = (int)inputshape.size();
        const int dim_count = (int)dim.size();

        for (int i = 0; i < dim_count; i++)
        {
            if (dim[i] < 0)
                dim[i] += input_rank;

            if (dim[i] < input_rank - dim_count)
                return false;
        }

        std::vector<int> normalized_shape(dim_count);
        for (int i = 0; i < dim_count; i++)
        {
            normalized_shape[i] = inputshape[input_rank - dim_count + i];
        }

        // check weight and bias shape
        const Operator* op_mul = matched_operators.at("mul");
        const Operator* op_add = matched_operators.at("add");
        const std::vector<int>& weight_shape = op_mul->inputs[1]->shape;
        const std::vector<int>& bias_shape = op_add->inputs[1]->shape;

        if (weight_shape != normalized_shape)
            return false;

        if (bias_shape != normalized_shape)
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& inputshape = op->inputs[0]->shape;
        const std::vector<int>& dim = captured_params.at("dim").ai;
        const int input_rank = (int)inputshape.size();
        const int dim_count = (int)dim.size();

        std::vector<int> normalized_shape(dim_count);
        for (int i = 0; i < dim_count; i++)
        {
            normalized_shape[i] = inputshape[input_rank - dim_count + i];
        }

        op->params["normalized_shape"] = normalized_shape;
        op->params["eps"] = captured_params.at("eps");
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_layer_norm_onnx_4, 29)

} // namespace pnnx
