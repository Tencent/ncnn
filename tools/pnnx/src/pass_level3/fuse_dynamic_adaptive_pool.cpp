// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "fuse_dynamic_adaptive_pool.h"

#include "pass_level2.h"

namespace pnnx {

class fuse_dynamic_adaptive_pool_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
9 8
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 dynamic_axis value=%dynamic_axis
aten::size              op_1        2 1 input dynamic_axis 4
prim::NumToTensor       op_2        1 1 4 5
aten::Int               op_3        1 1 5 outh
prim::Constant          op_4        0 1 outw value=%outw
prim::ListConstruct     op_5        2 1 outh outw output_size
F.adaptive_avg_pool2d   op_6        2 1 input output_size out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.adaptive_avg_pool2d";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        int dynamic_axis = captured_params.at("dynamic_axis").i;
        size_t input_rank = matched_operators.at("op_6")->inputs[0]->shape.size();
        return (input_rank == 3 && dynamic_axis == 1) || (input_rank == 4 && dynamic_axis == 2);
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int outw = captured_params.at("outw").i;
        op->params["output_size"] = std::vector<int>{0, outw};
    }
};

class fuse_dynamic_adaptive_pool_pass_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
9 8
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 dynamic_axis value=%dynamic_axis
aten::size              op_1        2 1 input dynamic_axis 4
prim::NumToTensor       op_2        1 1 4 5
aten::Int               op_3        1 1 5 outw
prim::Constant          op_4        0 1 outh value=%outh
prim::ListConstruct     op_5        2 1 outh outw output_size
F.adaptive_avg_pool2d   op_6        2 1 input output_size out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.adaptive_avg_pool2d";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        int dynamic_axis = captured_params.at("dynamic_axis").i;
        size_t input_rank = matched_operators.at("op_6")->inputs[0]->shape.size();
        return (input_rank == 3 && dynamic_axis == 2) || (input_rank == 4 && dynamic_axis == 3);
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int outh = captured_params.at("outh").i;
        op->params["output_size"] = std::vector<int>{outh, 0};
    }
};

void fuse_dynamic_adaptive_pool(Graph& graph)
{
    fuse_dynamic_adaptive_pool_pass a;
    fuse_dynamic_adaptive_pool_pass_1 b;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
}

} // namespace pnnx
