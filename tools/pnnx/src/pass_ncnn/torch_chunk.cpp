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

class torch_chunk : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 3
pnnx.Input              input       0 1 in
torch.chunk             op_0        1 2 in a b chunks=2 dim=%dim
pnnx.Output             output      2 0 a b
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Slice";
    }

    const char* name_str() const
    {
        return "chunk";
    }

    void write(const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs, Operator* op) const
    {
        int axis = captured_params.at("dim").i;
        if (axis == 0)
        {
            fprintf(stderr, "chunk along batch axis is not supported\n");
            return;
        }

        if (axis < 0)
        {
            int input_rank = op->inputs[0]->shape.size();
            axis = input_rank + axis;
        }

        op->params["0"] = {-233, -233};
        op->params["1"] = axis - 1;
    }
};

class torch_chunk_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 4
pnnx.Input              input       0 1 in
torch.chunk             op_0        1 3 in a b c chunks=3 dim=%dim
pnnx.Output             output      3 0 a b c
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Slice";
    }

    const char* name_str() const
    {
        return "chunk";
    }

    void write(const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs, Operator* op) const
    {
        int axis = captured_params.at("dim").i;
        if (axis == 0)
        {
            fprintf(stderr, "chunk along batch axis is not supported\n");
            return;
        }

        if (axis < 0)
        {
            int input_rank = op->inputs[0]->shape.size();
            axis = input_rank + axis;
        }

        op->params["0"] = {-233, -233, -233};
        op->params["1"] = axis - 1;
    }
};

class torch_chunk_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 5
pnnx.Input              input       0 1 in
torch.chunk             op_0        1 4 in a b c d chunks=4 dim=%dim
pnnx.Output             output      4 0 a b c d
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Slice";
    }

    const char* name_str() const
    {
        return "chunk";
    }

    void write(const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs, Operator* op) const
    {
        int axis = captured_params.at("dim").i;
        if (axis == 0)
        {
            fprintf(stderr, "chunk along batch axis is not supported\n");
            return;
        }

        if (axis < 0)
        {
            int input_rank = op->inputs[0]->shape.size();
            axis = input_rank + axis;
        }

        op->params["0"] = {-233, -233, -233, -233};
        op->params["1"] = axis - 1;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_chunk, 20)
REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_chunk_1, 20)
REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_chunk_2, 20)

} // namespace ncnn

} // namespace pnnx
