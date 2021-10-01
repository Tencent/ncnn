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

class torch_cat : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 a
pnnx.Input              input_1     0 1 b
torch.cat               op_0        2 1 a b out dim=%dim
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Concat";
    }

    const char* name_str() const
    {
        return "cat";
    }

    void write(const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs, Operator* op) const
    {
        int axis = captured_params.at("dim").i;
        op->params["0"] = axis > 0 ? axis - 1 : axis;
    }
};

class torch_cat_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 a
pnnx.Input              input_1     0 1 b
pnnx.Input              input_2     0 1 c
torch.cat               op_0        3 1 a b c out dim=%dim
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Concat";
    }

    const char* name_str() const
    {
        return "cat";
    }

    void write(const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs, Operator* op) const
    {
        int axis = captured_params.at("dim").i;
        op->params["0"] = axis > 0 ? axis - 1 : axis;
    }
};

class torch_cat_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 a
pnnx.Input              input_1     0 1 b
pnnx.Input              input_2     0 1 c
pnnx.Input              input_3     0 1 d
torch.cat               op_0        4 1 a b c d out dim=%dim
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Concat";
    }

    const char* name_str() const
    {
        return "cat";
    }

    void write(const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs, Operator* op) const
    {
        int axis = captured_params.at("dim").i;
        op->params["0"] = axis > 0 ? axis - 1 : axis;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_cat, 20)
REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_cat_1, 20)
REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_cat_2, 20)

} // namespace ncnn

} // namespace pnnx
