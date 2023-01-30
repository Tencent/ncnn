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

#include "fuse_permute_gridsample.h"

#include "pass_level2.h"

#include <float.h>

namespace pnnx {

namespace ncnn {

class fuse_permute_gridsample_4d_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_a     0 1 a
pnnx.Input              input_b     0 1 b
Permute                 op_0        1 1 b b1 0=3 1=2
GridSample              op_1        2 1 a b1 out 0=%c 1=%d 2=%e 3=0
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "GridSample";
    }

    const char* name_str() const
    {
        return "permutegridsample";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int mode = 0;
        int padding_mode = 0;
        int align_corner = 0;
        if (captured_params.at("c").type == 2)
            mode = captured_params.at("c").i;
        if (captured_params.at("d").type == 2)
            padding_mode = captured_params.at("d").i;
        if (captured_params.at("e").type == 2)
            align_corner = captured_params.at("e").i;

        op->params["0"] = mode;
        op->params["1"] = padding_mode;
        op->params["2"] = align_corner;
        op->params["3"] = 1;
    }
};

class fuse_permute_gridsample_5d_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_a     0 1 a
pnnx.Input              input_b     0 1 b
Permute                 op_0        1 1 b b1 0=9 1=3
GridSample              op_1        2 1 a b1 out 0=%c 1=%d 2=%e 3=0
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "GridSample";
    }

    const char* name_str() const
    {
        return "permutegridsample";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int mode = 0;
        int padding_mode = 0;
        int align_corner = 0;
        if (captured_params.at("c").type == 2)
            mode = captured_params.at("c").i;
        if (captured_params.at("d").type == 2)
            padding_mode = captured_params.at("d").i;
        if (captured_params.at("e").type == 2)
            align_corner = captured_params.at("e").i;

        op->params["0"] = mode;
        op->params["1"] = padding_mode;
        op->params["2"] = align_corner;
        op->params["3"] = 1;
    }
};

void fuse_permute_gridsample(Graph& graph)
{
    fuse_permute_gridsample_4d_pass a;
    fuse_permute_gridsample_5d_pass b;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
}

} // namespace ncnn

} // namespace pnnx
