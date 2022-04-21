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

#include "pass_level1.h"

#include "../utils.h"

namespace pnnx {

class GRU : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.rnn.GRU";
    }

    const char* type_str() const
    {
        return "nn.GRU";
    }

    void write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph, const torch::jit::Module& mod) const
    {
        //         mod.dump(true, true, true);

        //         graph->dump();

        const torch::jit::Node* gru = find_node_by_kind(graph, "aten::gru");

        const torch::jit::Node* return_tuple = find_node_by_kind(graph, "prim::TupleConstruct");
        if (return_tuple && return_tuple->inputs().size() == 2 && gru->outputs().size() == 2
                && return_tuple->inputs()[0] == gru->outputs()[1] && return_tuple->inputs()[1] == gru->outputs()[0])
        {
            // mark the swapped output tuple
            // we would restore the fine order in pass_level3/fuse_rnn_unpack
            fprintf(stderr, "swapped detected !\n");
            op->params["pnnx_rnn_output_swapped"] = 1;
        }

        //         for (auto aa : gru->schema().arguments())
        //         {
        //             fprintf(stderr, "arg %s\n", aa.name().c_str());
        //         }

        const auto& weight_ih_l0 = mod.attr("weight_ih_l0").toTensor();

        op->params["input_size"] = weight_ih_l0.size(1);
        op->params["hidden_size"] = weight_ih_l0.size(0) / 3;
        op->params["num_layers"] = gru->namedInput("num_layers");
        op->params["bias"] = gru->namedInput("has_biases");
        op->params["batch_first"] = gru->namedInput("batch_first");
        op->params["bidirectional"] = gru->namedInput("bidirectional");

        const int num_layers = op->params["num_layers"].i;
        const bool bias = op->params["bias"].b;
        const bool bidirectional = op->params["bidirectional"].b;

        for (int k = 0; k < num_layers; k++)
        {
            std::string weight_ih_lk_key = std::string("weight_ih_l") + std::to_string(k);
            std::string weight_hh_lk_key = std::string("weight_hh_l") + std::to_string(k);

            op->attrs[weight_ih_lk_key] = mod.attr(weight_ih_lk_key).toTensor();
            op->attrs[weight_hh_lk_key] = mod.attr(weight_hh_lk_key).toTensor();

            if (bias)
            {
                std::string bias_ih_lk_key = std::string("bias_ih_l") + std::to_string(k);
                std::string bias_hh_lk_key = std::string("bias_hh_l") + std::to_string(k);

                op->attrs[bias_ih_lk_key] = mod.attr(bias_ih_lk_key).toTensor();
                op->attrs[bias_hh_lk_key] = mod.attr(bias_hh_lk_key).toTensor();
            }

            if (bidirectional)
            {
                std::string weight_ih_lk_reverse_key = std::string("weight_ih_l") + std::to_string(k) + "_reverse";
                std::string weight_hh_lk_reverse_key = std::string("weight_hh_l") + std::to_string(k) + "_reverse";

                op->attrs[weight_ih_lk_reverse_key] = mod.attr(weight_ih_lk_reverse_key).toTensor();
                op->attrs[weight_hh_lk_reverse_key] = mod.attr(weight_hh_lk_reverse_key).toTensor();

                if (bias)
                {
                    std::string bias_ih_lk_reverse_key = std::string("bias_ih_l") + std::to_string(k) + "_reverse";
                    std::string bias_hh_lk_reverse_key = std::string("bias_hh_l") + std::to_string(k) + "_reverse";

                    op->attrs[bias_ih_lk_reverse_key] = mod.attr(bias_ih_lk_reverse_key).toTensor();
                    op->attrs[bias_hh_lk_reverse_key] = mod.attr(bias_hh_lk_reverse_key).toTensor();
                }
            }
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(GRU)

} // namespace pnnx
