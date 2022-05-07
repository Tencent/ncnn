// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "convert_torch_einsum.h"

namespace pnnx {

namespace ncnn {

void convert_torch_einsum(Graph& graph)
{
    int op_index = 0;

    for (Operator* op : graph.ops)
    {
        if (op->type != "torch.einsum")
            continue;

        op->type = "Einsum";
        op->name = std::string("einsum_") + std::to_string(op_index++);

        const int batch_index = op->inputs[0]->params["__batch_index"].i;

        // TODO drop batch index in equation

        const std::string equation = op->params.at("equation").s;

        std::vector<int> equation_int(equation.size());
        for (size_t i = 0; i < equation.size(); i++)
        {
            equation_int[i] = equation[i];
        }

        op->params["0"] = equation_int;

        op->params.erase("equation");
    }
}

} // namespace ncnn

} // namespace pnnx
