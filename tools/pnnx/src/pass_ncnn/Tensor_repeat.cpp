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

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class Tensor_repeat : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Tensor.repeat           op_0        1 1 input out sizes=%sizes
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tile";
    }

    const char* name_str() const
    {
        return "repeat";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& sizes = captured_params.at("sizes").ai;

        const int batch_index = op->outputs[0]->params["__batch_index"].i;

        if (batch_index != 0 && batch_index != 233)
        {
            fprintf(stderr, "repeat tensor with batch index %d is not supported yet!\n", batch_index);
        }

        // drop sizes batch index
        std::vector<int> new_sizes;
        for (int i = 0; i < (int)sizes.size(); i++)
        {
            if (i == batch_index && sizes[i] == 1)
                continue;

            new_sizes.push_back(sizes[i]);
        }

        if (new_sizes.size() == 5 && batch_index == 233)
        {
            if (new_sizes[0] == 1)
            {
                fprintf(stderr, "assume repeat 5-rank tensor has batch_index 0\n");
                new_sizes.erase(new_sizes.begin());
            }
        }

        const int sizes_rank = (int)new_sizes.size();

        if (sizes_rank > 5)
        {
            fprintf(stderr, "repeat to %d-rank tensor is not supported yet!\n", sizes_rank);
            return;
        }

        op->params["2"] = new_sizes;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(Tensor_repeat, 20)

} // namespace ncnn

} // namespace pnnx
