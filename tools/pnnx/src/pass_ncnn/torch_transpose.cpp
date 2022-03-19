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

class torch_transpose : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.transpose         op_0        1 1 input out dim0=%dim0 dim1=%dim1
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Permute";
    }

    const char* name_str() const
    {
        return "transpose";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["0"] = 0;

        const int batch_index = op->inputs[0]->params["__batch_index"].i;

        int dim0 = captured_params.at("dim0").i;
        int dim1 = captured_params.at("dim1").i;

        int input_rank = op->inputs[0]->shape.size();

        if (dim0 < 0)
        {
            dim0 = input_rank + dim0;
        }
        if (dim1 < 0)
        {
            dim1 = input_rank + dim1;
        }

        if (dim0 == batch_index || dim1 == batch_index)
        {
            fprintf(stderr, "permute across batch dim is not supported yet!\n");
            return;
        }

        if (batch_index >= 0 && batch_index < input_rank)
            input_rank -= 1;

        if (input_rank > 4)
        {
            fprintf(stderr, "permute %d-rank tensor is not supported yet!\n", input_rank);
            return;
        }

        if (dim0 > batch_index)
            dim0 -= 1;
        if (dim1 > batch_index)
            dim1 -= 1;

        if (input_rank == 1)
        {
            // noop
            op->type = "Noop";
        }
        if (input_rank == 2)
        {
            if (dim0 == 0 && dim1 == 1) op->params["0"] = 1;
            if (dim0 == 1 && dim1 == 0) op->params["0"] = 1;
        }
        if (input_rank == 3)
        {
            if (dim0 == 0 && dim1 == 1) op->params["0"] = 2;
            if (dim0 == 1 && dim1 == 0) op->params["0"] = 2;
            if (dim0 == 0 && dim1 == 2) op->params["0"] = 5;
            if (dim0 == 2 && dim1 == 0) op->params["0"] = 5;
            if (dim0 == 1 && dim1 == 2) op->params["0"] = 1;
            if (dim0 == 2 && dim1 == 1) op->params["0"] = 1;
        }
        if (input_rank == 4)
        {
            if (dim0 == 0 && dim1 == 1) op->params["0"] = 6;
            if (dim0 == 1 && dim1 == 0) op->params["0"] = 6;
            if (dim0 == 0 && dim1 == 2) op->params["0"] = 14;
            if (dim0 == 2 && dim1 == 0) op->params["0"] = 14;
            if (dim0 == 0 && dim1 == 3) op->params["0"] = 21;
            if (dim0 == 3 && dim1 == 0) op->params["0"] = 21;
            if (dim0 == 1 && dim1 == 2) op->params["0"] = 2;
            if (dim0 == 2 && dim1 == 1) op->params["0"] = 2;
            if (dim0 == 1 && dim1 == 3) op->params["0"] = 5;
            if (dim0 == 3 && dim1 == 1) op->params["0"] = 5;
            if (dim0 == 2 && dim1 == 3) op->params["0"] = 1;
            if (dim0 == 3 && dim1 == 2) op->params["0"] = 1;
        }
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_transpose, 20)

} // namespace ncnn

} // namespace pnnx
