// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// #include "pass_level1.h"
//
// #include <torch/csrc/api/include/torch/torch.h>
//
// #include "../utils.h"

#include "fuse_module_pass.h"

namespace pnnx {

class MultiheadAttention : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.activation.MultiheadAttention";
    }

    const char* type_str() const
    {
        return "nn.MultiheadAttention";
    }

    void write(Operator* op, const TorchGraphProxy& graph, const TorchModuleProxy& mod) const
    {
        // mod.dump(false, false, false);
        // graph->dump();

        const TorchNodeProxy* multi_head_attention = graph.find_node_by_kind("aten::_native_multi_head_attention");
        if (multi_head_attention)
        {
            op->params["num_heads"] = multi_head_attention->namedInput("num_head");
            op->params["batch_first"] = true;
            op->params["add_zero_attn"] = false;

            if (multi_head_attention->hasNamedInput("mask") && multi_head_attention->namedInput("mask") == graph.input(graph.input_count() - 1))
            {
                size_t input_count = op->inputs.size();
                op->inputnames.resize(input_count);
                op->inputnames[input_count - 1] = "attn_mask";
            }
        }
        else
        {
            const TorchNodeProxy* div_num_heads = graph.find_node_by_kind("aten::div");
            const TorchNodeProxy* div_num_heads_18 = graph.find_node_by_kind("aten::floor_divide");
            if (div_num_heads_18)
            {
                div_num_heads = div_num_heads_18;
            }

            // const TorchNodeProxy* div_num_heads_input_1 = graph.find_producer_node_by_value(div_num_heads->input(1));

            // op->params["num_heads"] = (int)div_num_heads_input_1->t(torch::jit::attr::value).item<int64_t>();
            op->params["num_heads"] = div_num_heads->input(1);

            const TorchNodeProxy* transpose_batch_seq = graph.find_node_by_kind("aten::transpose");

            Parameter transpose_dim0 = transpose_batch_seq->input(1);
            Parameter transpose_dim1 = transpose_batch_seq->input(2);
            if (transpose_dim0.i == 1 && transpose_dim1.i == 0)
            {
                op->params["batch_first"] = true;
            }

            const TorchNodeProxy* add_zero_attn = graph.find_node_by_kind("aten::zeros");
            if (add_zero_attn)
            {
                op->params["add_zero_attn"] = true;
            }
            else
            {
                op->params["add_zero_attn"] = false;
            }

            const TorchNodeProxy* scaled_dot_product_attention = graph.find_node_by_kind("aten::scaled_dot_product_attention");
            if (scaled_dot_product_attention)
            {
                if (!scaled_dot_product_attention->is_input_none(3))
                {
                    size_t input_count = op->inputs.size();
                    op->inputnames.resize(input_count);
                    op->inputnames[input_count - 1] = "attn_mask";
                }
            }

            // find attention mask addition pattern pre torch-2.1
            const TorchNodeProxy* has_attn_mask = graph.find_node_by_kind("aten::baddbmm");
            if (has_attn_mask)
            {
                size_t input_count = op->inputs.size();
                op->inputnames.resize(input_count);
                op->inputnames[input_count - 1] = "attn_mask";
            }

            // find attention mask addition pattern pre torch-1.12
            // attn = torch.bmm(Q, K)
            // input0 = torch.add_(attn, attn_mask)
            // attn0 = torch.softmax(input0, -1)
            const TorchNodeProxy* softmax = graph.find_node_by_kind("aten::softmax");
            if (softmax)
            {
                const TorchNodeProxy* add_ = graph.find_producer_node_by_value(softmax->input(0));
                if (add_ && add_->kind() == "aten::add_")
                {
                    const TorchNodeProxy* bmm = graph.find_producer_node_by_value(add_->input(0));
                    if (bmm && bmm->kind() == "aten::bmm")
                    {
                        size_t input_count = op->inputs.size();
                        op->inputnames.resize(input_count);
                        op->inputnames[input_count - 1] = "attn_mask";
                    }
                }
            }
        }

        if (mod.hasattr("in_proj_weight"))
        {
            const TorchTensorProxy& in_proj_weight = mod.attr("in_proj_weight");

            op->params["embed_dim"] = in_proj_weight.size(1);
            op->params["kdim"] = in_proj_weight.size(1);
            op->params["vdim"] = in_proj_weight.size(1);
            op->attrs["in_proj_weight"] = in_proj_weight;
        }
        else
        {
            const TorchTensorProxy& q_proj_weight = mod.attr("q_proj_weight");
            const TorchTensorProxy& k_proj_weight = mod.attr("k_proj_weight");
            const TorchTensorProxy& v_proj_weight = mod.attr("v_proj_weight");

            op->params["embed_dim"] = q_proj_weight.size(1);
            op->params["kdim"] = k_proj_weight.size(1);
            op->params["vdim"] = v_proj_weight.size(1);
            op->attrs["q_proj_weight"] = q_proj_weight;
            op->attrs["k_proj_weight"] = k_proj_weight;
            op->attrs["v_proj_weight"] = v_proj_weight;
        }

        const TorchTensorProxy& out_proj_weight = mod.attr("out_proj.weight");

        op->attrs["out_proj.weight"] = out_proj_weight;

        if (mod.hasattr("in_proj_bias") && mod.hasattr("out_proj.bias"))
        {
            // bias=True
            const TorchTensorProxy& in_proj_bias = mod.attr("in_proj_bias");
            const TorchTensorProxy& out_proj_bias = mod.attr("out_proj.bias");

            op->params["bias"] = true;
            op->attrs["in_proj_bias"] = in_proj_bias;
            op->attrs["out_proj.bias"] = out_proj_bias;
        }
        else
        {
            op->params["bias"] = false;

            // the output projection bias always there no matter bias is False in pytorch 1.8
            // this behavior changes since https://github.com/pytorch/pytorch/commit/58d1b3639bc07f9519de18e5a18e575f260c7eeb
            if (mod.hasattr("out_proj.bias"))
            {
                const TorchTensorProxy& out_proj_bias = mod.attr("out_proj.bias");
                op->attrs["out_proj.bias"] = out_proj_bias;
            }
        }

        if (mod.hasattr("bias_k") && mod.hasattr("bias_v"))
        {
            // add_bias_kv=True
            const TorchTensorProxy& bias_k = mod.attr("bias_k");
            const TorchTensorProxy& bias_v = mod.attr("bias_v");

            op->params["add_bias_kv"] = true;
            op->attrs["bias_k"] = bias_k;
            op->attrs["bias_v"] = bias_v;
        }
        else
        {
            op->params["add_bias_kv"] = false;
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(MultiheadAttention)

} // namespace pnnx
