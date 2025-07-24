// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class LSTM : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.rnn.LSTM";
    }

    const char* type_str() const
    {
        return "nn.LSTM";
    }

    void write(Operator* op, const TorchGraphProxy& graph, const TorchModuleProxy& mod) const
    {
        // mod.dump(true, true, true);
        //
        // graph->dump();

        const TorchNodeProxy* lstm = graph.find_node_by_kind("aten::lstm");

        const TorchNodeProxy* return_tuple = graph.find_node_by_kind("prim::TupleConstruct");
        if (return_tuple && return_tuple->input_count() == 3 && lstm->output_count() == 3
                && return_tuple->input(0) == lstm->output(1) && return_tuple->input(1) == lstm->output(2) && return_tuple->input(2) == lstm->output(0))
        {
            // mark the swapped output tuple
            // we would restore the fine order in pass_level3/fuse_rnn_unpack
            fprintf(stderr, "swapped detected !\n");
            op->params["pnnx_rnn_output_swapped"] = 1;
        }

        // for (auto aa : lstm->schema().arguments())
        // {
        //     fprintf(stderr, "arg %s\n", aa.name().c_str());
        // }

        const TorchTensorProxy& weight_ih_l0 = mod.attr("weight_ih_l0");
        const TorchTensorProxy& weight_hh_l0 = mod.attr("weight_hh_l0");

        op->params["input_size"] = weight_ih_l0.size(1);
        op->params["hidden_size"] = weight_ih_l0.size(0) / 4;
        op->params["num_layers"] = lstm->namedInput("num_layers");
        op->params["bias"] = lstm->namedInput("has_biases");
        op->params["batch_first"] = lstm->namedInput("batch_first");
        op->params["bidirectional"] = lstm->namedInput("bidirectional");
        op->params["proj_size"] = weight_ih_l0.size(0) / 4 == weight_hh_l0.size(1) ? 0 : weight_hh_l0.size(1);

        const int num_layers = op->params["num_layers"].i;
        const bool bias = op->params["bias"].b;
        const bool bidirectional = op->params["bidirectional"].b;
        const int proj_size = op->params["proj_size"].i;

        for (int k = 0; k < num_layers; k++)
        {
            std::string weight_ih_lk_key = std::string("weight_ih_l") + std::to_string(k);
            std::string weight_hh_lk_key = std::string("weight_hh_l") + std::to_string(k);

            op->attrs[weight_ih_lk_key] = mod.attr(weight_ih_lk_key);
            op->attrs[weight_hh_lk_key] = mod.attr(weight_hh_lk_key);

            if (bias)
            {
                std::string bias_ih_lk_key = std::string("bias_ih_l") + std::to_string(k);
                std::string bias_hh_lk_key = std::string("bias_hh_l") + std::to_string(k);

                op->attrs[bias_ih_lk_key] = mod.attr(bias_ih_lk_key);
                op->attrs[bias_hh_lk_key] = mod.attr(bias_hh_lk_key);
            }

            if (proj_size > 0)
            {
                std::string weight_hr_lk_key = std::string("weight_hr_l") + std::to_string(k);

                op->attrs[weight_hr_lk_key] = mod.attr(weight_hr_lk_key);
            }

            if (bidirectional)
            {
                std::string weight_ih_lk_reverse_key = std::string("weight_ih_l") + std::to_string(k) + "_reverse";
                std::string weight_hh_lk_reverse_key = std::string("weight_hh_l") + std::to_string(k) + "_reverse";

                op->attrs[weight_ih_lk_reverse_key] = mod.attr(weight_ih_lk_reverse_key);
                op->attrs[weight_hh_lk_reverse_key] = mod.attr(weight_hh_lk_reverse_key);

                if (bias)
                {
                    std::string bias_ih_lk_reverse_key = std::string("bias_ih_l") + std::to_string(k) + "_reverse";
                    std::string bias_hh_lk_reverse_key = std::string("bias_hh_l") + std::to_string(k) + "_reverse";

                    op->attrs[bias_ih_lk_reverse_key] = mod.attr(bias_ih_lk_reverse_key);
                    op->attrs[bias_hh_lk_reverse_key] = mod.attr(bias_hh_lk_reverse_key);
                }

                if (proj_size > 0)
                {
                    std::string weight_hr_lk_reverse_key = std::string("weight_hr_l") + std::to_string(k) + "_reverse";

                    op->attrs[weight_hr_lk_reverse_key] = mod.attr(weight_hr_lk_reverse_key);
                }
            }
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(LSTM)

} // namespace pnnx
