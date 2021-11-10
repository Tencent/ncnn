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

#include "fuse_slice_indices.h"

#include <string.h>
#include "pass_level2.h"

namespace pnnx {

class fuse_slice_indices_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
Tensor.slice            op_0        1 1 input a dim=%dim0 end=%end0 start=%start0 step=%step0
Tensor.slice            op_1        1 1 a b dim=%dim1 end=%end1 start=%start1 step=%step1
Tensor.slice            op_2        1 1 b c dim=%dim2 end=%end2 start=%start2 step=%step2
Tensor.slice            op_3        1 1 c d dim=%dim3 end=%end3 start=%start3 step=%step3
Tensor.slice            op_4        1 1 d out dim=%dim4 end=%end4 start=%start4 step=%step4
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.slice";
    }

    const char* name_str() const
    {
        return "slice";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        int dim0 = captured_params.at("dim0").i;
        int dim1 = captured_params.at("dim1").i;
        int dim2 = captured_params.at("dim2").i;
        int dim3 = captured_params.at("dim3").i;
        int dim4 = captured_params.at("dim4").i;

        return dim0 < dim1 && dim1 < dim2 && dim2 < dim3 && dim3 < dim4;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int dim0 = captured_params.at("dim0").i;
        int dim1 = captured_params.at("dim1").i;
        int dim2 = captured_params.at("dim2").i;
        int dim3 = captured_params.at("dim3").i;
        int dim4 = captured_params.at("dim4").i;

        int start0 = captured_params.at("start0").i;
        int start1 = captured_params.at("start1").i;
        int start2 = captured_params.at("start2").i;
        int start3 = captured_params.at("start3").i;
        int start4 = captured_params.at("start4").i;

        int end0 = captured_params.at("end0").i;
        int end1 = captured_params.at("end1").i;
        int end2 = captured_params.at("end2").i;
        int end3 = captured_params.at("end3").i;
        int end4 = captured_params.at("end4").i;

        int step0 = captured_params.at("step0").i;
        int step1 = captured_params.at("step1").i;
        int step2 = captured_params.at("step2").i;
        int step3 = captured_params.at("step3").i;
        int step4 = captured_params.at("step4").i;

        op->params["dims"] = Parameter{dim0, dim1, dim2, dim3, dim4};
        op->params["starts"] = Parameter{start0, start1, start2, start3, start4};
        op->params["ends"] = Parameter{end0, end1, end2, end3, end4};
        op->params["steps"] = Parameter{step0, step1, step2, step3, step4};
    }
};

class fuse_slice_indices_pass_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input       0 1 input
Tensor.slice            op_0        1 1 input a dim=%dim0 end=%end0 start=%start0 step=%step0
Tensor.slice            op_1        1 1 a b dim=%dim1 end=%end1 start=%start1 step=%step1
Tensor.slice            op_2        1 1 b c dim=%dim2 end=%end2 start=%start2 step=%step2
Tensor.slice            op_3        1 1 c out dim=%dim3 end=%end3 start=%start3 step=%step3
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.slice";
    }

    const char* name_str() const
    {
        return "slice";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        int dim0 = captured_params.at("dim0").i;
        int dim1 = captured_params.at("dim1").i;
        int dim2 = captured_params.at("dim2").i;
        int dim3 = captured_params.at("dim3").i;

        return dim0 < dim1 && dim1 < dim2 && dim2 < dim3;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int dim0 = captured_params.at("dim0").i;
        int dim1 = captured_params.at("dim1").i;
        int dim2 = captured_params.at("dim2").i;
        int dim3 = captured_params.at("dim3").i;

        int start0 = captured_params.at("start0").i;
        int start1 = captured_params.at("start1").i;
        int start2 = captured_params.at("start2").i;
        int start3 = captured_params.at("start3").i;

        int end0 = captured_params.at("end0").i;
        int end1 = captured_params.at("end1").i;
        int end2 = captured_params.at("end2").i;
        int end3 = captured_params.at("end3").i;

        int step0 = captured_params.at("step0").i;
        int step1 = captured_params.at("step1").i;
        int step2 = captured_params.at("step2").i;
        int step3 = captured_params.at("step3").i;

        op->params["dims"] = Parameter{dim0, dim1, dim2, dim3};
        op->params["starts"] = Parameter{start0, start1, start2, start3};
        op->params["ends"] = Parameter{end0, end1, end2, end3};
        op->params["steps"] = Parameter{step0, step1, step2, step3};
    }
};

class fuse_slice_indices_pass_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
Tensor.slice            op_0        1 1 input a dim=%dim0 end=%end0 start=%start0 step=%step0
Tensor.slice            op_1        1 1 a b dim=%dim1 end=%end1 start=%start1 step=%step1
Tensor.slice            op_2        1 1 b out dim=%dim2 end=%end2 start=%start2 step=%step2
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.slice";
    }

    const char* name_str() const
    {
        return "slice";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        int dim0 = captured_params.at("dim0").i;
        int dim1 = captured_params.at("dim1").i;
        int dim2 = captured_params.at("dim2").i;

        return dim0 < dim1 && dim1 < dim2;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int dim0 = captured_params.at("dim0").i;
        int dim1 = captured_params.at("dim1").i;
        int dim2 = captured_params.at("dim2").i;

        int start0 = captured_params.at("start0").i;
        int start1 = captured_params.at("start1").i;
        int start2 = captured_params.at("start2").i;

        int end0 = captured_params.at("end0").i;
        int end1 = captured_params.at("end1").i;
        int end2 = captured_params.at("end2").i;

        int step0 = captured_params.at("step0").i;
        int step1 = captured_params.at("step1").i;
        int step2 = captured_params.at("step2").i;

        op->params["dims"] = Parameter{dim0, dim1, dim2};
        op->params["starts"] = Parameter{start0, start1, start2};
        op->params["ends"] = Parameter{end0, end1, end2};
        op->params["steps"] = Parameter{step0, step1, step2};
    }
};

class fuse_slice_indices_pass_3 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
Tensor.slice            op_0        1 1 input a dim=%dim0 end=%end0 start=%start0 step=%step0
Tensor.slice            op_1        1 1 a out dim=%dim1 end=%end1 start=%start1 step=%step1
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.slice";
    }

    const char* name_str() const
    {
        return "slice";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        int dim0 = captured_params.at("dim0").i;
        int dim1 = captured_params.at("dim1").i;

        return dim0 < dim1;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int dim0 = captured_params.at("dim0").i;
        int dim1 = captured_params.at("dim1").i;

        int start0 = captured_params.at("start0").i;
        int start1 = captured_params.at("start1").i;

        int end0 = captured_params.at("end0").i;
        int end1 = captured_params.at("end1").i;

        int step0 = captured_params.at("step0").i;
        int step1 = captured_params.at("step1").i;

        op->params["dims"] = Parameter{dim0, dim1};
        op->params["starts"] = Parameter{start0, start1};
        op->params["ends"] = Parameter{end0, end1};
        op->params["steps"] = Parameter{step0, step1};
    }
};

class fuse_slice_indices_pass_4 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Tensor.slice            op_0        1 1 input out dim=%dim0 end=%end0 start=%start0 step=%step0
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.slice";
    }

    const char* name_str() const
    {
        return "slice";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int dim0 = captured_params.at("dim0").i;
        int start0 = captured_params.at("start0").i;
        int end0 = captured_params.at("end0").i;
        int step0 = captured_params.at("step0").i;

        op->params["dims"] = Parameter{dim0};
        op->params["starts"] = Parameter{start0};
        op->params["ends"] = Parameter{end0};
        op->params["steps"] = Parameter{step0};
    }
};

void fuse_slice_indices(Graph& graph)
{
    fuse_slice_indices_pass a;
    fuse_slice_indices_pass_1 b;
    fuse_slice_indices_pass_2 c;
    fuse_slice_indices_pass_3 d;
    fuse_slice_indices_pass_4 e;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
    pnnx_graph_rewrite(graph, &c, opindex);
    pnnx_graph_rewrite(graph, &d, opindex);
    pnnx_graph_rewrite(graph, &e, opindex);
}

} // namespace pnnx
