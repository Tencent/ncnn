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

class nn_Embedding : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.Embedding            op_0        1 1 input out embedding_dim=%embedding_dim num_embeddings=%num_embeddings sparse=False @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Embed";
    }

    const char* name_str() const
    {
        return "embed";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        op->params["0"] = captured_params.at("embedding_dim");
        op->params["1"] = captured_params.at("num_embeddings");
        op->params["2"] = 0;
        op->params["3"] = (int)(captured_attrs.at("op_0.weight").data.size() / sizeof(float));

        op->attrs["0"] = Attribute();
        op->attrs["0"].data = {0, 0, 0, 0};
        op->attrs["1"] = captured_attrs.at("op_0.weight");
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_Embedding, 20)

} // namespace ncnn

} // namespace pnnx
