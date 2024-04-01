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

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class nn_Linear_0 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input #input=(1,%m,%in_features)f32
nn.Linear               op_0        1 1 input out in_features=%in_features out_features=%out_features bias=%bias
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Gemm";
    }

    const char* name_str() const
    {
        return "gemm";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        op->params["2"] = 0;
        op->params["3"] = 1;
        op->params["4"] = 0;
        op->params["5"] = 1;
        op->params["6"] = 1;
        op->params["7"] = captured_params.at("m");
        op->params["8"] = captured_params.at("out_features");
        op->params["9"] = captured_params.at("in_features");
        op->params["10"] = captured_params.at("bias").b ? 4 : -1;

        op->attrs["0"] = Attribute();
        op->attrs["0"].data = {0, 0, 0, 0};
        op->attrs["1"] = captured_attrs.at("op_0.weight");
        if (captured_params.at("bias").b)
        {
            op->attrs["2"] = Attribute();
            op->attrs["2"].data = {0, 0, 0, 0};
            op->attrs["3"] = captured_attrs.at("op_0.bias");
        }
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_Linear_0, 19)

class nn_Linear_01 : public nn_Linear_0
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input #input=(%m,%in_features)f32
nn.Linear               op_0        1 1 input out in_features=%in_features out_features=%out_features bias=%bias
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const int m = captured_params.at("m").i;

        if (m == 1)
            return false;

        return true;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_Linear_01, 19)

class nn_Linear_10 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input #input=(1,%m,%in_features)f32
pnnx.Input              input_1     0 1 bias
nn.Linear               op_0        2 1 input bias out in_features=%in_features out_features=%out_features bias=False
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Gemm";
    }

    const char* name_str() const
    {
        return "gemm";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        op->params["2"] = 0;
        op->params["3"] = 1;
        op->params["4"] = 0;
        op->params["5"] = 1;
        op->params["6"] = 0;
        op->params["7"] = captured_params.at("m");
        op->params["8"] = captured_params.at("out_features");
        op->params["9"] = captured_params.at("in_features");
        op->params["10"] = 4;

        op->attrs["0"] = Attribute();
        op->attrs["0"].data = {0, 0, 0, 0};
        op->attrs["1"] = captured_attrs.at("op_0.weight");
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_Linear_10, 19)

class nn_Linear_11 : public nn_Linear_10
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input #input=(%m,%in_features)f32
pnnx.Input              input_1     0 1 bias
nn.Linear               op_0        2 1 input bias out in_features=%in_features out_features=%out_features bias=False
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const int m = captured_params.at("m").i;

        if (m == 1)
            return false;

        return true;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_Linear_11, 19)

class nn_Linear : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.Linear               op_0        1 1 input out in_features=%in_features out_features=%out_features bias=%bias
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "InnerProduct";
    }

    const char* name_str() const
    {
        return "linear";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        op->params["0"] = captured_params.at("out_features");
        op->params["1"] = captured_params.at("bias").b ? 1 : 0;
        op->params["2"] = captured_attrs.at("op_0.weight").elemcount();

        op->attrs["0"] = Attribute();
        op->attrs["0"].data = {0, 0, 0, 0};
        op->attrs["1"] = captured_attrs.at("op_0.weight");
        if (captured_params.at("bias").b)
            op->attrs["2"] = captured_attrs.at("op_0.bias");
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_Linear, 20)

class nn_Linear_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 bias
nn.Linear               op_0        2 1 input bias out in_features=%in_features out_features=%out_features bias=False
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 bias
InnerProduct            linear      1 1 input a
BinaryOp                bias        2 1 a bias out 0=0
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        const int batch_index = ops.at("linear")->inputs[0]->params["__batch_index"].i;

        ops.at("linear")->params["0"] = captured_params.at("out_features");
        ops.at("linear")->params["1"] = 0;
        ops.at("linear")->params["2"] = captured_attrs.at("op_0.weight").elemcount();

        ops.at("linear")->attrs["0"] = Attribute();
        ops.at("linear")->attrs["0"].data = {0, 0, 0, 0};
        ops.at("linear")->attrs["1"] = captured_attrs.at("op_0.weight");

        ops.at("linear")->outputs[0]->params["__batch_index"] = batch_index;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_Linear_1, 20)

} // namespace ncnn

} // namespace pnnx
