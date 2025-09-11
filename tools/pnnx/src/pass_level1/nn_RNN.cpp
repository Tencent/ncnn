// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class RNN : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.rnn.RNN";
    }

    const char* type_str() const
    {
        return "nn.RNN";
    }

    void write(Operator* op, const TorchGraphProxy& graph, const TorchModuleProxy& mod) const
    {
        //         mod.dump(true, true, true);

        //         graph->dump();

        const TorchNodeProxy* rnn = graph.find_node_by_kind("aten::rnn_tanh");
        const TorchNodeProxy* rnn_relu = graph.find_node_by_kind("aten::rnn_relu");

        if (rnn_relu)
        {
            rnn = rnn_relu;
        }

        const TorchNodeProxy* return_tuple = graph.find_node_by_kind("prim::TupleConstruct");
        if (return_tuple && return_tuple->input_count() == 2 && rnn->output_count() == 2
                && return_tuple->input(0) == rnn->output(1) && return_tuple->input(1) == rnn->output(0))
        {
            // mark the swapped output tuple
            // we would restore the fine order in pass_level3/fuse_rnn_unpack
            fprintf(stderr, "swapped detected !\n");
            op->params["pnnx_rnn_output_swapped"] = 1;
        }

        //         for (auto aa : rnn->schema().arguments())
        //         {
        //             fprintf(stderr, "arg %s\n", aa.name().c_str());
        //         }

        const TorchTensorProxy& weight_ih_l0 = mod.attr("weight_ih_l0");

        op->params["input_size"] = weight_ih_l0.size(1);
        op->params["hidden_size"] = weight_ih_l0.size(0);
        op->params["num_layers"] = rnn->namedInput("num_layers");
        op->params["nonlinearity"] = rnn_relu ? "relu" : "tanh";
        op->params["bias"] = rnn->namedInput("has_biases");
        op->params["batch_first"] = rnn->namedInput("batch_first");
        op->params["bidirectional"] = rnn->namedInput("bidirectional");

        const int num_layers = op->params["num_layers"].i;
        const bool bias = op->params["bias"].b;
        const bool bidirectional = op->params["bidirectional"].b;

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
            }
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(RNN)

} // namespace pnnx
