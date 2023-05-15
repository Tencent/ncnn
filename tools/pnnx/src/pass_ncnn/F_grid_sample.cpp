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

#include "pass_ncnn.h"

namespace pnnx {
namespace ncnn {

class F_grid_sample : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0       0 1 input0
pnnx.Input              input_1       0 1 input1
F.grid_sample           op_0          2 1 input0 input1 out mode=%mode padding_mode=%padding_mode align_corners=%align_corners
pnnx.Output             output        1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "GridSample";
    }

    const char* name_str() const
    {
        return "gridsample";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::string& mode = captured_params.at("mode").s;
        if (mode == "bilinear")
            op->params["0"] = 1;
        if (mode == "nearest")
            op->params["0"] = 2;
        if (mode == "bicubic")
            op->params["0"] = 3;

        const std::string& padding_mode = captured_params.at("padding_mode").s;
        if (padding_mode == "zeros")
            op->params["1"] = 1;
        if (padding_mode == "border")
            op->params["1"] = 2;
        if (padding_mode == "reflection")
            op->params["1"] = 3;

        op->params["2"] = captured_params.at("align_corners").b ? 1 : 0;
        op->params["3"] = 0;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_grid_sample, 20)

class F_grid_sample_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_a     0 1 a
pnnx.Input              input_b     0 1 b
torch.permute           op_0        1 1 b b1 dims=%dims
F.grid_sample           op_1        2 1 a b1 out mode=%mode padding_mode=%padding_mode align_corners=%align_corners
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

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& dims = captured_params.at("dims").ai;

        if ((dims == std::vector<int>{1, 2, 0}) || (dims == std::vector<int>{1, 2, 3, 0}))
            return true;
        if ((dims == std::vector<int>{0, 2, 3, 1}) || (dims == std::vector<int>{0, 2, 3, 4, 1}))
            return true;
        return false;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::string& mode = captured_params.at("mode").s;
        if (mode == "bilinear")
            op->params["0"] = 1;
        if (mode == "nearest")
            op->params["0"] = 2;
        if (mode == "bicubic")
            op->params["0"] = 3;

        const std::string& padding_mode = captured_params.at("padding_mode").s;
        if (padding_mode == "zeros")
            op->params["1"] = 1;
        if (padding_mode == "border")
            op->params["1"] = 2;
        if (padding_mode == "reflection")
            op->params["1"] = 3;

        op->params["2"] = captured_params.at("align_corners").b ? 1 : 0;

        const int batch_index = op->inputs[1]->params["__batch_index"].i;

        const std::vector<int>& dims = captured_params.at("dims").ai;

        int input_rank = (int)op->inputs[0]->shape.size();

        if (input_rank == 0)
        {
            // assume input is fine
            input_rank = (int)dims.size();
        }

        if (batch_index >= 0 && batch_index < input_rank)
            input_rank -= 1;

        if (input_rank > 4)
        {
            fprintf(stderr, "permute %d-rank tensor is not supported yet!\n", input_rank);
            return;
        }

        // drop permute batch index
        std::vector<int> new_dims;
        for (int i = 0; i < (int)dims.size(); i++)
        {
            if (dims[i] == batch_index)
                continue;

            int new_dim = dims[i] > batch_index ? dims[i] - 1 : dims[i];
            new_dims.push_back(new_dim);
        }

        if (input_rank != (int)new_dims.size())
        {
            fprintf(stderr, "permute %d-rank tensor with %d-rank dims is not possible\n", input_rank, (int)new_dims.size());
            return;
        }

        if ((input_rank == 3 && new_dims == std::vector<int>{1, 2, 0}) || (input_rank == 4 && new_dims == std::vector<int>{1, 2, 3, 0}))
            op->params["3"] = 1;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_grid_sample_1, 19)

} // namespace ncnn

} // namespace pnnx
