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

class torch_permute : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.permute           op_0        1 1 input out dims=%dims
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Permute";
    }

    const char* name_str() const
    {
        return "permute";
    }

    void write(const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs, Operator* op) const
    {
        op->params["0"] = 0;

        const std::vector<int>& dims = captured_params.at("dims").ai;

        int input_rank = op->inputs[0]->shape.size();

        if (input_rank > 5)
        {
            fprintf(stderr, "permute %d-rank tensor is not supported yet!\n", input_rank);
            return;
        }

        if (dims.size() != input_rank)
        {
            fprintf(stderr, "permute %d-rank tensor with %d-rank dims is not possible\n", input_rank, (int)dims.size());
            return;
        }

        if (input_rank == 2)
        {
            // noop
        }
        if (input_rank == 3)
        {
            if (dims[1] == 2 && dims[2] == 1) op->params["0"] = 1;
            else if (dims[0] == 2 && dims[1] == 0 && dims[2] == 1) op->params["0"] = 1;
        }
        if (input_rank == 4)
        {
            if (dims[1] == 1 && dims[2] == 3 && dims[3] == 2) op->params["0"] = 1;
            else if (dims[1] == 2 && dims[2] == 1 && dims[3] == 3) op->params["0"] = 2;
            else if (dims[1] == 2 && dims[2] == 3 && dims[3] == 1) op->params["0"] = 3;
            else if (dims[1] == 3 && dims[2] == 1 && dims[3] == 2) op->params["0"] = 4;
            else if (dims[1] == 3 && dims[2] == 2 && dims[3] == 1) op->params["0"] = 5;
        }
        if (input_rank == 5)
        {
            if (dims[1] == 1 && dims[2] == 3 && dims[3] == 4 && dims[4] == 2) op->params["0"] = 1;
            else if (dims[1] == 2 && dims[2] == 1 && dims[3] == 3 && dims[4] == 4) op->params["0"] = 2;
            else if (dims[1] == 2 && dims[2] == 3 && dims[3] == 4 && dims[4] == 1) op->params["0"] = 3;
            else if (dims[1] == 3 && dims[2] == 4 && dims[3] == 1 && dims[4] == 2) op->params["0"] = 4;
            else if (dims[1] == 3 && dims[2] == 4 && dims[3] == 2 && dims[4] == 1) op->params["0"] = 5;
            else fprintf(stderr, "unsupported permute dims!\n");
        }
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_permute, 20)

} // namespace ncnn

} // namespace pnnx
