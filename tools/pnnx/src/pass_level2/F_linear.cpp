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

class F_linear : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
aten::linear            op_0        3 1 input weight bias out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.linear";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_linear, 10)

class F_linear_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
aten::t                 op_0        1 1 weight 9
aten::matmul            op_1        2 1 input 9 a
prim::Constant          op_2        0 1 19 value=1
aten::add               op_3        3 1 a bias 19 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.linear";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_linear_1, 9)

class F_linear_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
aten::t                 op_0        1 1 weight 9
aten::matmul            op_1        2 1 input 9 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.linear";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        op->params["bias"] = Parameter();
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_linear_2, 10)

class F_linear_3 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
aten::t                 op_0        1 1 weight 14
prim::Constant          op_1        0 1 15 value=1
prim::Constant          op_2        0 1 30 value=1
aten::addmm             op_3        5 1 bias input 14 15 30 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.linear";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_linear_3, 10)

class F_linear_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
Gemm                    op_0        3 1 input weight bias out alpha=1.000000e+00 beta=1.000000e+00 transB=1
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.linear";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_linear_onnx, 10)

class F_linear_onnx_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 bias
pnnx.Attribute          weight      0 1 weight @data=(%in_features,%out_features)f32
Gemm                    gemm        3 1 input weight bias out alpha=1.000000e+00 beta=1.000000e+00
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 bias
pnnx.Attribute          weight      0 1 weight
F.linear                linear      3 1 input weight bias out $weight=weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const int in_features = captured_params.at("in_features").i;
        const int out_features = captured_params.at("out_features").i;

        auto weight = captured_attrs.at("weight.data").get_float32_data();
        std::vector<float> transposed_weight(in_features * out_features);
        for (int i = 0; i < out_features; i++)
        {
            float* wptr = (float*)transposed_weight.data() + in_features * i;
            for (int j = 0; j < in_features; j++)
            {
                wptr[j] = ((const float*)weight.data())[out_features * j + i];
            }
        }

        Operator* op_weight = ops.at("weight");
        op_weight->attrs["data"] = Attribute({out_features, in_features}, transposed_weight);
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_linear_onnx_1, 9)

class F_linear_onnx_2 : public F_linear_onnx_1
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Attribute          weight      0 1 weight @data=(%in_features,%out_features)f32
MatMul                  matmul      2 1 input weight out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Attribute          weight      0 1 weight
F.linear                linear      2 1 input weight out bias=None $weight=weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_linear_onnx_2, 9)

class F_linear_onnx_3 : public F_linear_onnx_1
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Attribute          weight      0 1 weight @data=(%in_features,%out_features)f32
pnnx.Attribute          bias        0 1 bias @data=(%out_features)f32
MatMul                  matmul      2 1 input weight mm
aten::add               add         2 1 mm bias out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Attribute          weight      0 1 weight
pnnx.Attribute          bias        0 1 bias
F.linear                linear      3 1 input weight bias out $weight=weight $bias=bias
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        F_linear_onnx_1::write(ops, captured_params, captured_attrs);

        Operator* op_bias = ops.at("bias");
        op_bias->attrs["data"] = captured_attrs.at("bias.data");
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_linear_onnx_3, 8)

class F_linear_onnx_4 : public F_linear_onnx_3
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Attribute          weight      0 1 weight @data=(%in_features,%out_features)f32
pnnx.Attribute          bias        0 1 bias @data=(%out_features)f32
MatMul                  matmul      2 1 input weight mm
aten::add               add         2 1 bias mm out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_linear_onnx_4, 8)

} // namespace pnnx
