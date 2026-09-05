// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// pt2 (torch.export) path: aten::gru / aten::lstm / aten::rnn_tanh / aten::rnn_relu
// are single ops whose weights arrive as a list through prim::ListConstruct
// (load_exportedprogram also expands has_biases/num_layers/dropout/train/
// bidirectional/batch_first into constant inputs). Convert them to
// nn.GRU / nn.LSTM / nn.RNN (num_layers may be >1), which pass_level5
// unroll_rnn_op expands to single layers and pass_ncnn then turns into ncnn
// layers.

#include "torch_rnn_pt2.h"

#include <algorithm>
#include <string>

namespace pnnx {

void torch_rnn_pt2(Graph& graph)
{
    for (int i = (int)graph.ops.size() - 1; i >= 0; i--)
    {
        Operator* op = graph.ops[i];

        std::string new_type;
        std::string nonlinearity;
        if (op->type == "aten::gru")
            new_type = "nn.GRU";
        else if (op->type == "aten::lstm")
            new_type = "nn.LSTM";
        else if (op->type == "aten::rnn_tanh")
        {
            new_type = "nn.RNN";
            nonlinearity = "tanh";
        }
        else if (op->type == "aten::rnn_relu")
        {
            new_type = "nn.RNN";
            nonlinearity = "relu";
        }
        else
            continue;

        // locate the weights list input (producer = prim::ListConstruct with
        // all inputs being pnnx.Attribute; LSTM's hx is also a ListConstruct of
        // tensors so it must be skipped)
        int list_index = -1;
        for (size_t j = 0; j < op->inputs.size(); j++)
        {
            // for aten::lstm the initial state ListConstruct[hx, cx] sits at
            // input index 1 and may itself be all pnnx.Attribute (buffer
            // states); skip it so it is never mistaken for the weights list
            if (new_type == "nn.LSTM" && j == 1)
                continue;

            Operator* prod = op->inputs[j]->producer;
            if (prod && prod->type == "prim::ListConstruct")
            {
                bool all_attr = true;
                for (auto in : prod->inputs)
                {
                    if (!in->producer || in->producer->type != "pnnx.Attribute")
                    {
                        all_attr = false;
                        break;
                    }
                }
                if (all_attr)
                {
                    list_index = (int)j;
                    break;
                }
            }
        }
        if (list_index == -1)
            continue;

        Operator* list_cons = op->inputs[list_index]->producer;

        // the weight list: ListConstruct inputs must be pnnx.Attribute
        std::vector<Attribute> weights;
        bool ok = true;
        for (auto in : list_cons->inputs)
        {
            if (!in->producer || in->producer->type != "pnnx.Attribute" || in->producer->attrs.find("data") == in->producer->attrs.end())
            {
                ok = false;
                break;
            }
            weights.push_back(in->producer->attrs.at("data"));
        }
        if (!ok || weights.empty())
            continue;

        // the frontend expands has_biases/num_layers/dropout/train/bidirectional/
        // batch_first into prim::Constant inputs after the ListConstruct (in
        // aten.* schema order)
        bool has_biases = false;
        bool bidirectional = false;
        bool batch_first = false;
        int num_layers = 1;

        // prefer params when the frontend already parsed them
        if (op->params.count("has_biases"))
            has_biases = op->params.at("has_biases").b;
        if (op->params.count("bidirectional"))
            bidirectional = op->params.at("bidirectional").b;
        if (op->params.count("batch_first"))
            batch_first = op->params.at("batch_first").b;
        if (op->params.count("num_layers"))
            num_layers = op->params.at("num_layers").i;

        if (!op->params.count("num_layers") || !op->params.count("has_biases") || !op->params.count("bidirectional") || !op->params.count("batch_first"))
        {
            // read from the constant inputs
            std::vector<const Operator*> const_inputs;
            for (size_t j = list_index + 1; j < op->inputs.size(); j++)
            {
                Operator* prod = op->inputs[j]->producer;
                if (prod && prod->type == "prim::Constant")
                    const_inputs.push_back(prod);
            }
            // schema order: has_biases, num_layers, dropout, train, bidirectional, batch_first
            if (const_inputs.size() >= 1 && const_inputs[0]->has_param("value") && const_inputs[0]->params.at("value").type == 1)
                has_biases = const_inputs[0]->params.at("value").b;
            if (const_inputs.size() >= 2 && const_inputs[1]->has_param("value") && const_inputs[1]->params.at("value").type == 2)
                num_layers = const_inputs[1]->params.at("value").i;
            if (const_inputs.size() >= 5 && const_inputs[4]->has_param("value") && const_inputs[4]->params.at("value").type == 1)
                bidirectional = const_inputs[4]->params.at("value").b;
            if (const_inputs.size() >= 6 && const_inputs[5]->has_param("value") && const_inputs[5]->params.at("value").type == 1)
                batch_first = const_inputs[5]->params.at("value").b;
        }

        // validate weight count: 4 (unidirectional) or 8 (bidirectional) per
        // layer, minus 2 when no bias; LSTM with proj_size adds one weight_hr
        // per layer (per direction)
        bool has_proj = false;
        int proj_size = 0;
        const int per_layer_base = has_biases ? (bidirectional ? 8 : 4) : (bidirectional ? 4 : 2);
        if ((int)weights.size() != per_layer_base * num_layers)
        {
            if (new_type == "nn.LSTM")
            {
                const int per_layer_proj = per_layer_base + (bidirectional ? 2 : 1);
                if ((int)weights.size() == per_layer_proj * num_layers)
                    has_proj = true;
                else
                    continue;
            }
            else
                continue;
        }
        const Attribute& wih0 = weights[0];
        const Attribute& whh0 = weights[1];

        // parse per-layer weights and store them in attrs under nn.* names
        int idx = 0;
        for (int l = 0; l < num_layers; l++)
        {
            op->attrs["weight_ih_l" + std::to_string(l)] = weights[idx++];
            op->attrs["weight_hh_l" + std::to_string(l)] = weights[idx++];
            if (has_biases)
            {
                op->attrs["bias_ih_l" + std::to_string(l)] = weights[idx++];
                op->attrs["bias_hh_l" + std::to_string(l)] = weights[idx++];
            }
            if (has_proj)
                op->attrs["weight_hr_l" + std::to_string(l)] = weights[idx++];
            if (bidirectional)
            {
                op->attrs["weight_ih_l" + std::to_string(l) + "_reverse"] = weights[idx++];
                op->attrs["weight_hh_l" + std::to_string(l) + "_reverse"] = weights[idx++];
                if (has_biases)
                {
                    op->attrs["bias_ih_l" + std::to_string(l) + "_reverse"] = weights[idx++];
                    op->attrs["bias_hh_l" + std::to_string(l) + "_reverse"] = weights[idx++];
                }
                if (has_proj)
                    op->attrs["weight_hr_l" + std::to_string(l) + "_reverse"] = weights[idx++];
            }
        }

        // infer input_size / hidden_size from the weights
        // GRU/RNN: weight_ih_l0 = (3*hidden, input_size), weight_hh_l0 = (3*hidden, hidden)
        // LSTM: weight_ih_l0 = (4*hidden, input_size), weight_hh_l0 = (4*hidden, hidden or proj)
        if (wih0.shape.size() < 2 || whh0.shape.size() < 2)
            continue;
        op->params["input_size"] = wih0.shape[1];
        if (new_type == "nn.LSTM")
            op->params["hidden_size"] = wih0.shape[0] / 4;
        else
            op->params["hidden_size"] = whh0.shape[1];
        op->params["num_layers"] = num_layers;
        op->params["bias"] = has_biases;
        op->params["batch_first"] = batch_first;
        op->params["bidirectional"] = bidirectional;

        if (new_type == "nn.RNN")
            op->params["nonlinearity"] = nonlinearity;

        if (new_type == "nn.LSTM")
        {
            if (has_proj)
            {
                // weight_hr_l0 = (proj_size, hidden)
                const Attribute& whr0 = op->attrs.at("weight_hr_l0");
                if (whr0.shape.size() < 1)
                    continue;
                proj_size = whr0.shape[0];
            }
            op->params["proj_size"] = proj_size;
        }

        // check whether an operand comes from an all-zero constant (initial
        // state that can be omitted implicitly). At the pass_level2 stage zeros
        // are still aten::zeros ops (pnnx.Attribute is only folded at
        // pass_level3). Passing explicit all-zero hx/cx makes ncnn LSTM/GRU
        // emit a 4D hidden ((1,dirs,1,hidden)) which differs from the ts path
        // (implicit zeros, 3D hidden) and misaligns multi-layer state passing.
        auto is_zero_init = [](Operand* operand) {
            Operator* prod = operand ? operand->producer : 0;
            if (!prod)
                return false;
            if (prod->type == "aten::zeros")
                return true;
            if (prod->type != "pnnx.Attribute")
                return false;
            const auto it = prod->attrs.find("data");
            if (it == prod->attrs.end())
                return false;
            const Attribute& a = it->second;
            if (a.type != 1) // f32 only
                return false;
            const float* d = (const float*)a.data.data();
            const size_t n = a.data.size() / sizeof(float);
            for (size_t k = 0; k < n; k++)
            {
                if (d[k] != 0.f)
                    return false;
            }
            return true;
        };

        // detach op from the consumer lists of inputs[j]
        auto detach_inputs_from = [&op](size_t from) {
            for (size_t j = from; j < op->inputs.size(); j++)
            {
                op->inputs[j]->consumers.erase(std::find(op->inputs[j]->consumers.begin(), op->inputs[j]->consumers.end(), op));
            }
        };

        // rebuild inputs:
        //  GRU/RNN: [input, hx] (the frontend passes hx as a single input;
        //           all-zero constants are omitted)
        //  LSTM:    the frontend loads hx as ListConstruct[hx, cx]; split into
        //           [input, hx, cx]
        if (new_type == "nn.LSTM")
        {
            if (op->inputs.size() < 3)
                continue;
            Operator* hx_list = op->inputs[1]->producer;
            if (!hx_list || hx_list->type != "prim::ListConstruct" || hx_list->inputs.size() < 2)
                continue;

            Operand* hx = hx_list->inputs[0];
            Operand* cx = hx_list->inputs[1];

            std::vector<Operand*> keep;
            keep.push_back(op->inputs[0]);
            const bool zero_hx = is_zero_init(hx);
            const bool zero_cx = is_zero_init(cx);
            if (!(zero_hx && zero_cx))
            {
                // keep the hidden and cell states paired: ncnn nn.LSTM only has a
                // 1-input (both implicitly zero) or a 3-input pattern. Omitting a
                // single all-zero state would shift the other one into the hidden
                // slot and misalign the layer / state passing.
                keep.push_back(hx);
                keep.push_back(cx);
            }

            detach_inputs_from(1);
            op->inputs = keep;
            for (size_t j = 1; j < op->inputs.size(); j++)
                op->inputs[j]->consumers.push_back(op);
        }
        else
        {
            if (op->inputs.size() < 2)
                continue;
            std::vector<Operand*> keep;
            keep.push_back(op->inputs[0]);
            if (!is_zero_init(op->inputs[1]))
                keep.push_back(op->inputs[1]);

            detach_inputs_from(1);
            op->inputs = keep;
            for (size_t j = 1; j < op->inputs.size(); j++)
                op->inputs[j]->consumers.push_back(op);
        }

        // only switch the op type after the input rebuild fully succeeded, so
        // a failed rebuild never leaves an nn.* node with aten-style inputs
        op->type = new_type;
        op->inputnames.clear();
    }
}

} // namespace pnnx
