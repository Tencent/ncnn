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

        std::string equation = op->params.at("equation").s;

        const int batch_index = op->inputs[0]->params["__batch_index"].i;

        if (batch_index != 233)
        {
            // drop batch index in equation
            char batch_x = 'i' + batch_index;

            std::string new_equation;
            for (size_t i = 0; i < equation.size(); i++)
            {
                if (equation[i] == batch_x)
                    continue;

                char x = equation[i];
                if (x > batch_x)
                    x -= 1;

                new_equation.push_back(x);
            }

            equation = new_equation;
        }

        op->name = std::string("einsum_") + std::to_string(op_index++);

        if (equation == "ij->" || equation == "ijk->" || equation == "ijkl->")
        {
            // reduce sum
            op->type = "Reduction";

            op->params["0"] = 0;
            op->params["1"] = 1;
            op->params["4"] = 0;
        }
        else if (equation == "ii")
        {
            // trace
            op->type = "Einsum";

            std::vector<int> equation_int(equation.size());
            for (size_t i = 0; i < equation.size(); i++)
            {
                equation_int[i] = equation[i];
            }

            op->params["0"] = equation_int;
        }
        else if (equation.find("->") == std::string::npos)
        {
            // permute
            op->type = "Permute";

            if (equation == "ji")
                op->params["0"] = 1;

            if (equation == "ikj")
                op->params["0"] = 1;
            else if (equation == "jik")
                op->params["0"] = 2;
            else if (equation == "kij")
                op->params["0"] = 3;
            else if (equation == "jki")
                op->params["0"] = 4;
            else if (equation == "kji")
                op->params["0"] = 5;

            if (equation == "ijlk")
                op->params["0"] = 1;
            else if (equation == "ikjl")
                op->params["0"] = 2;
            else if (equation == "iljk")
                op->params["0"] = 3;
            else if (equation == "iklj")
                op->params["0"] = 4;
            else if (equation == "ilkj")
                op->params["0"] = 5;
            else if (equation == "jikl")
                op->params["0"] = 6;
            else if (equation == "jilk")
                op->params["0"] = 7;
            else if (equation == "kijl")
                op->params["0"] = 8;
            else if (equation == "lijk")
                op->params["0"] = 9;
            else if (equation == "kilj")
                op->params["0"] = 10;
            else if (equation == "likj")
                op->params["0"] = 11;
            else if (equation == "jkil")
                op->params["0"] = 12;
            else if (equation == "jlik")
                op->params["0"] = 13;
            else if (equation == "kjil")
                op->params["0"] = 14;
            else if (equation == "ljik")
                op->params["0"] = 15;
            else if (equation == "klij")
                op->params["0"] = 16;
            else if (equation == "lkij")
                op->params["0"] = 17;
            else if (equation == "jkli")
                op->params["0"] = 18;
            else if (equation == "jlki")
                op->params["0"] = 19;
            else if (equation == "kjli")
                op->params["0"] = 20;
            else if (equation == "ljki")
                op->params["0"] = 21;
            else if (equation == "klji")
                op->params["0"] = 22;
            else if (equation == "lkji")
                op->params["0"] = 23;
        }
        else
        {
            op->type = "Einsum";

            std::vector<int> equation_int(equation.size());
            for (size_t i = 0; i < equation.size(); i++)
            {
                equation_int[i] = equation[i];
            }

            op->params["0"] = equation_int;
        }

        op->params.erase("equation");
    }
}

} // namespace ncnn

} // namespace pnnx
