// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_reshape_activation_reshape.h"

#include <algorithm>
#include <set>

namespace pnnx {

static bool is_reshape_like_op(const Operator* op)
{
    return op->type == "Tensor.reshape" || op->type == "torch.squeeze" || op->type == "torch.unsqueeze";
}

// a reshape-like op that changes the batch axis position is batch-sensitive:
// ncnn carries the batch dim dynamically at runtime, so such ops can only be
// fused when the batch axis is known and stays in place. a squeeze on the
// batch axis with a statically-known non-1 batch is turned into a Noop by the
// ncnn conversion and is safe to fuse.
static bool is_batch_sensitive_reshape_like_op(const Operator* op)
{
    if (op->type == "torch.squeeze")
    {
        const bool batch_index_known = op->inputs[0]->params.find("__batch_index") != op->inputs[0]->params.end();
        if (!batch_index_known)
            return true;

        const int batch_index = op->inputs[0]->params.at("__batch_index").i;
        if (batch_index < 0 || batch_index == 233)
            return false;

        const std::vector<int>& input_shape = op->inputs[0]->shape;

        if (!op->has_param("dim"))
        {
            // squeeze all dims may drop the dynamic batch dim
            return true;
        }

        std::vector<int> dims;
        if (op->params.at("dim").type == 2)
            dims.push_back(op->params.at("dim").i);
        else
            dims = op->params.at("dim").ai;

        for (int dim : dims)
        {
            if (dim < 0)
                dim += (int)input_shape.size();

            if (dim < 0 || dim >= (int)input_shape.size())
                return true;

            if (dim == batch_index)
            {
                // ncnn turns the batch-axis squeeze into a Noop unless the batch
                // is statically size 1 or unknown, only then the batch layout is
                // affected
                if (input_shape[dim] == -1 || input_shape[dim] == 1)
                    return true;

                continue;
            }

            // a size-1 dim removed before the batch axis shifts the batch axis
            if (dim < batch_index && input_shape[dim] == 1)
                return true;
        }

        return false;
    }
    else if (op->type == "torch.unsqueeze")
    {
        const bool batch_index_known = op->inputs[0]->params.find("__batch_index") != op->inputs[0]->params.end();
        if (!batch_index_known)
            return true;

        const int batch_index = op->inputs[0]->params.at("__batch_index").i;
        if (batch_index < 0 || batch_index == 233)
            return false;

        if (op->params.find("dim") == op->params.end())
            return true;

        int dim = op->params.at("dim").i;
        if (dim < 0)
            dim += (int)op->inputs[0]->shape.size() + 1;

        // inserting a dim at or before the batch position shifts the batch axis
        return dim <= batch_index;
    }

    return false;
}

static bool is_dim_insensitive_activation(const Operator* op)
{
    static const std::set<std::string> activation_types = {
        "F.relu", "nn.ReLU",
        "F.relu6", "nn.ReLU6",
        "F.hardtanh", "nn.Hardtanh",
        "F.threshold", "nn.Threshold",
        "F.leaky_relu", "nn.LeakyReLU",
        "F.elu", "nn.ELU",
        "F.celu", "nn.CELU",
        "F.selu", "nn.SELU",
        "F.sigmoid", "nn.Sigmoid",
        "F.tanh", "nn.Tanh",
        "F.silu", "nn.SiLU",
        "F.gelu", "nn.GELU",
        "F.hardsigmoid", "nn.Hardsigmoid",
        "F.hardswish", "nn.Hardswish",
        "F.mish", "nn.Mish",
        "F.logsigmoid", "nn.LogSigmoid",
        "F.softplus", "nn.Softplus",
        "F.softsign", "nn.Softsign",
        "F.tanhshrink", "nn.Tanhshrink",
        "F.hardshrink", "nn.Hardshrink",
        "F.softshrink", "nn.Softshrink",
        "torch.clamp",
    };

    return activation_types.find(op->type) != activation_types.end();
}

// compute the output shape of a reshape-like op chain applied to `shape`
// returns false when the shape cannot be determined safely
static bool compose_reshape_chain_shape(const std::vector<Operator*>& chain, std::vector<int>& shape)
{
    for (const Operator* op : chain)
    {
        if (op->type == "Tensor.reshape")
        {
            if (op->inputs.size() != 1)
                return false;

            if (op->params.find("shape") == op->params.end())
                return false;

            const std::vector<int>& params = op->params.at("shape").ai;

            std::vector<int> outshape;
            outshape.reserve(params.size());

            int dynamic_dim_count = 0;
            for (size_t i = 0; i < params.size(); i++)
            {
                int dimsize = params[i];

                if (dimsize == 0)
                {
                    // copy input dim i
                    if (i >= shape.size() || shape[i] == -1)
                        return false;

                    dimsize = shape[i];
                }

                if (dimsize == -1)
                    dynamic_dim_count += 1;

                if (dynamic_dim_count > 1)
                    return false;

                outshape.push_back(dimsize);
            }

            shape = outshape;
        }
        else if (op->type == "torch.unsqueeze")
        {
            if (op->inputs.size() != 1)
                return false;

            if (op->params.find("dim") == op->params.end())
                return false;

            int dim = op->params.at("dim").i;
            if (dim < 0)
                dim += (int)shape.size() + 1;

            if (dim < 0 || dim > (int)shape.size())
                return false;

            shape.insert(shape.begin() + dim, 1);
        }
        else if (op->type == "torch.squeeze")
        {
            if (op->inputs.size() != 1)
                return false;

            if (op->params.find("dim") == op->params.end())
            {
                // squeeze all size-1 dims
                std::vector<int> outshape;
                outshape.reserve(shape.size());

                for (size_t i = 0; i < shape.size(); i++)
                {
                    if (shape[i] == -1)
                        return false;

                    if (shape[i] != 1)
                        outshape.push_back(shape[i]);
                }

                shape = outshape;
            }
            else
            {
                if (op->params.at("dim").type != 2)
                    return false;

                int dim = op->params.at("dim").i;
                if (dim < 0)
                    dim += (int)shape.size();

                if (dim < 0 || dim >= (int)shape.size())
                    return false;

                if (shape[dim] == -1)
                    return false;

                if (shape[dim] == 1)
                    shape.erase(shape.begin() + dim);
            }
        }
        else
        {
            return false;
        }
    }

    return true;
}

void fuse_reshape_activation_reshape(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (int i = (int)graph.ops.size() - 1; i > 0; i--)
        {
            Operator* op = graph.ops[i];

            // look for reshape-like chain + dim-insensitive activation + reshape-like chain
            if (!is_dim_insensitive_activation(op))
                continue;

            if (op->inputs.size() != 1 || op->outputs.size() != 1)
                continue;

            // collect the upstream reshape-like chain
            std::vector<Operator*> upstream;
            const Operand* in0 = op->inputs[0];
            while (in0->producer && in0->consumers.size() == 1 && is_reshape_like_op(in0->producer) && !is_batch_sensitive_reshape_like_op(in0->producer))
            {
                upstream.push_back(in0->producer);
                in0 = in0->producer->inputs[0];
            }

            if (upstream.empty())
                continue;

            // collect the downstream reshape-like chain
            std::vector<Operator*> downstream;
            const Operand* out0 = op->outputs[0];
            while (out0->consumers.size() == 1 && is_reshape_like_op(out0->consumers[0]) && !is_batch_sensitive_reshape_like_op(out0->consumers[0]))
            {
                Operator* op0 = out0->consumers[0];
                downstream.push_back(op0);
                out0 = op0->outputs[0];
            }

            if (downstream.empty())
                continue;

            // compute the merged reshape output shape
            std::vector<int> outshape;
            const std::vector<int>& final_shape = downstream.back()->outputs[0]->shape;
            if (!final_shape.empty())
            {
                outshape = final_shape;
            }
            else
            {
                // try to infer the activation input/output shape
                outshape = op->outputs[0]->shape;
                if (outshape.empty())
                    outshape = op->inputs[0]->shape;

                if (outshape.empty())
                {
                    // infer from the upstream chain and its input shape
                    std::vector<Operator*> upstream_chain(upstream.rbegin(), upstream.rend());
                    std::vector<int> in_shape = upstream.back()->inputs[0]->shape;
                    if (!in_shape.empty() && compose_reshape_chain_shape(upstream_chain, in_shape))
                        outshape = in_shape;
                }

                if (outshape.empty())
                {
                    // fully dynamic shape, only safe when the downstream chain is all plain reshapes
                    bool all_plain_reshape = true;
                    for (const Operator* op0 : downstream)
                    {
                        if (op0->type != "Tensor.reshape")
                        {
                            all_plain_reshape = false;
                            break;
                        }
                    }

                    if (!all_plain_reshape)
                        continue;
                }

                if (!compose_reshape_chain_shape(downstream, outshape))
                    continue;

                if (outshape.empty())
                    continue;
            }

            // the merged reshape param must not contain more than one dynamic dim
            int dynamic_dim_count = 0;
            for (int v : outshape)
            {
                if (v == -1)
                    dynamic_dim_count += 1;
            }

            if (dynamic_dim_count > 1)
                continue;

            matched = true;

            // capture the batch layout of the removed downstream reshape chain output,
            // the surviving reshape output now feeds its consumers
            const Operand* downstream_out = downstream.back()->outputs[0];
            bool downstream_out_has_batch_index = downstream_out->params.find("__batch_index") != downstream_out->params.end();
            bool downstream_out_has_ncnn_batch_axis = downstream_out->params.find("__ncnn_batch_axis") != downstream_out->params.end();
            const int downstream_out_batch_index = downstream_out_has_batch_index ? downstream_out->params.at("__batch_index").i : 0;
            const int downstream_out_ncnn_batch_axis = downstream_out_has_ncnn_batch_axis ? downstream_out->params.at("__ncnn_batch_axis").i : 0;

            // keep the first upstream reshape-like op as the merged reshape
            Operator* reshape = upstream[0];

            reshape->type = "Tensor.reshape";
            reshape->params.clear();
            reshape->params["shape"] = outshape;

            // drop the dynamic shape input if any
            if (reshape->inputs.size() == 2)
            {
                reshape->inputs[1]->remove_consumer(reshape);
                reshape->inputs.resize(1);
            }

            auto remove_op = [&graph](Operator* op0) {
                for (auto& x : op0->inputs)
                {
                    x->remove_consumer(op0);
                }

                Operand* op0_in = op0->inputs[0];
                Operand* op0_out = op0->outputs[0];

                for (auto& x : op0_out->consumers)
                {
                    for (size_t j = 0; j < x->inputs.size(); j++)
                    {
                        if (x->inputs[j] == op0_out)
                            x->inputs[j] = op0_in;
                    }

                    op0_in->consumers.push_back(x);
                }

                op0_in->name = op0_out->name;

                op0_out->producer = 0;
                op0_out->consumers.clear();

                graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), op0_out));
                delete op0_out;

                op0->inputs.clear();
                op0->outputs.clear();

                graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op0));
                delete op0;
            };

            // remove the remaining upstream reshape-like ops
            for (int j = (int)upstream.size() - 1; j > 0; j--)
            {
                remove_op(upstream[j]);
            }

            // remove the downstream reshape-like ops
            for (int j = (int)downstream.size() - 1; j >= 0; j--)
            {
                remove_op(downstream[j]);
            }

            // update the shapes of the surviving operands
            reshape->outputs[0]->shape = outshape;
            op->outputs[0]->shape = outshape;

            // the surviving reshape output carries the batch layout of the removed
            // downstream reshape chain output for its new consumers
            if (downstream_out_has_batch_index)
                reshape->outputs[0]->params["__batch_index"] = downstream_out_batch_index;
            if (downstream_out_has_ncnn_batch_axis)
                reshape->outputs[0]->params["__ncnn_batch_axis"] = downstream_out_ncnn_batch_axis;

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
