// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

#include <algorithm>
#include <cmath>
#include <stdint.h>
#include <string.h>

#include "pass_ncnn/convert_attribute.h"
#include "pass_ncnn/convert_batch_layout.h"
#include "pass_ncnn/convert_custom_op.h"
#include "pass_ncnn/convert_module_op.h"
#include "pass_ncnn/convert_half_to_float.h"
#include "pass_ncnn/convert_input.h"
#include "pass_ncnn/convert_reshape_interp_expression.h"
#include "pass_ncnn/convert_slice_expression.h"
#include "pass_ncnn/convert_torch_cat.h"
#include "pass_ncnn/convert_torch_chunk.h"
#include "pass_ncnn/convert_torch_einsum.h"
#include "pass_ncnn/convert_torch_split.h"
#include "pass_ncnn/convert_torch_stack.h"
#include "pass_ncnn/convert_torch_tensor_split.h"
#include "pass_ncnn/convert_torch_unbind.h"
#include "pass_ncnn/convert_Tensor_select.h"
#include "pass_ncnn/convert_Tensor_slice.h"
#include "pass_ncnn/convert_Tensor_slice_copy.h"
#include "pass_ncnn/eliminate_output.h"
#include "pass_ncnn/expand_expression.h"
#include "pass_ncnn/fuse_convert_shufflechannel_slice.h"
#include "pass_ncnn/fuse_convert_rotaryembed.h"
#include "pass_ncnn/insert_split.h"
#include "pass_ncnn/chain_multi_output.h"
#include "pass_ncnn/solve_batch_index.h"

#include "pass_ncnn/eliminate_noop.h"
#include "pass_ncnn/fuse_convolution_activation.h"
#include "pass_ncnn/fuse_convolution1d_activation.h"
#include "pass_ncnn/fuse_convolutiondepthwise_activation.h"
#include "pass_ncnn/fuse_convolutiondepthwise1d_activation.h"
#include "pass_ncnn/fuse_deconvolution_activation.h"
#include "pass_ncnn/fuse_deconvolutiondepthwise_activation.h"
#include "pass_ncnn/fuse_innerproduct_activation.h"
#include "pass_ncnn/fuse_padding_convolution.h"
#include "pass_ncnn/fuse_padding_convolutiondepthwise.h"
#include "pass_ncnn/fuse_transpose_matmul.h"
#include "pass_ncnn/fuse_binaryop_eltwise.h"
#include "pass_ncnn/eliminate_reshape_binaryop_broadcast.h"
#include "pass_ncnn/insert_reshape_numpy_binaryop_broadcast.h"
#include "pass_ncnn/insert_reshape_linear.h"
#include "pass_ncnn/insert_reshape_pooling.h"
#include "pass_ncnn/legalize_global_pooling_layout.h"

#include "pass_level4/attribute_pooling.h"
#include "pass_level4/dead_code_elimination.h"
#include "pass_level4/canonicalize.h"
#include "pass_level5/attribute_unpooling.h"
#include "pass_level5/eliminate_maxpool_indices.h"
#include "pass_level5/unroll_rnn_op.h"

namespace pnnx {

static std::map<int, std::vector<const GraphRewriterPass*> > g_global_pnnx_ncnn_graph_rewriter_passes;

NcnnGraphRewriterPassRegister::NcnnGraphRewriterPassRegister(const GraphRewriterPass* _pass, int priority)
    : pass(_pass)
{
    if (g_global_pnnx_ncnn_graph_rewriter_passes.find(priority) == g_global_pnnx_ncnn_graph_rewriter_passes.end())
    {
        g_global_pnnx_ncnn_graph_rewriter_passes[priority] = std::vector<const GraphRewriterPass*>();
    }

    g_global_pnnx_ncnn_graph_rewriter_passes[priority].push_back(pass);
}

NcnnGraphRewriterPassRegister::~NcnnGraphRewriterPassRegister()
{
    delete pass;
}

static bool is_ncnn_embedding_index(const Operand* r)
{
    if (r->type != 4 && r->type != 5)
        return false;
    if (r->producer->type != "pnnx.Input" && r->producer->type != "pnnx.Attribute")
        return false;
    if (r->shape.empty() || r->shape.size() > 2 || r->consumers.empty())
        return false;
    for (int d : r->shape)
        if (d <= 0)
            return false;

    int num_embeddings = INT_MAX;
    for (const Operator* op : r->consumers)
    {
        if (op->inputs.empty() || op->inputs[0] != r)
            return false;

        const Attribute* weight = 0;
        if (op->type == "nn.Embedding" && op->has_attr("weight"))
            weight = &op->attrs.at("weight");
        if (op->type == "F.embedding" && op->inputs.size() == 2)
        {
            const Operator* producer = op->inputs[1]->producer;
            if (producer->type == "pnnx.Attribute" && producer->has_attr("data"))
                weight = &producer->attrs.at("data");
        }
        if (!weight || (weight->type != 1 && weight->type != 3) || weight->shape.size() != 2 || weight->shape[0] <= 0 || weight->shape[1] <= 0)
            return false;
        num_embeddings = std::min(num_embeddings, weight->shape[0]);
    }

    // Embed reads i32 indices. save_ncnn already generates torch.int inputs for
    // i64 frontend inputs; valid indices into these static tables fit in i32.
    // Constant indices use the writer's i64 -> i32 conversion, so check their
    // actual values rather than allowing its out-of-range saturation.
    if (r->producer->type == "pnnx.Attribute")
    {
        if (!r->producer->has_attr("data"))
            return false;
        const Attribute& attr = r->producer->attrs.at("data");
        if (attr.type != r->type)
            return false;
        const size_t elemsize = r->type == 5 ? sizeof(int64_t) : sizeof(int32_t);
        if (attr.data.size() % elemsize != 0)
            return false;
        for (size_t i = 0; i < attr.data.size(); i += elemsize)
        {
            int64_t index;
            if (r->type == 5)
                memcpy(&index, attr.data.data() + i, sizeof(int64_t));
            else
            {
                int32_t index32;
                memcpy(&index32, attr.data.data() + i, sizeof(int32_t));
                index = index32;
            }
            if (index < 0 || index >= num_embeddings)
                return false;
        }
    }
    return true;
}

int check_ncnn(const Graph& g, std::string& error)
{
    for (const Operator* op : g.ops)
    {
        std::string reason;
        for (const auto& param : op->params)
        {
            if (param.second.type == 3 && !std::isfinite(param.second.f))
                reason = "non-finite parameter";
            if (op->type == "pnnx.Expression" && param.first == "expr"
                    && (param.second.s.find("nan") != std::string::npos || param.second.s.find("inf") != std::string::npos))
                reason = "non-finite expression";
        }
        if ((op->type == "F.interpolate" || op->type == "nn.Upsample") && op->has_param("mode") && op->params.at("mode").s == "nearest-exact")
            reason = "nearest-exact interpolation";
        if (op->type.compare(0, 15, "nn.InstanceNorm") == 0 && op->has_param("track_running_stats") && op->params.at("track_running_stats").b)
            reason = "tracked InstanceNorm";
        if (op->has_param("dtype") && op->params.at("dtype").type != 0)
            reason = "explicit computation dtype";
        for (const auto& attr : op->attrs)
        {
            // convert_half_to_float handles f16 weights before ncnn rewrites.
            const bool index = op->type == "pnnx.Attribute" && op->outputs.size() == 1 && is_ncnn_embedding_index(op->outputs[0]);
            if (attr.second.type != 1 && attr.second.type != 3 && !index)
                reason = "tensor attribute dtype";
            if (attr.second.shape.empty() || std::find(attr.second.shape.begin(), attr.second.shape.end(), 0) != attr.second.shape.end())
                reason = "scalar or empty tensor attribute";
        }
        if (op->type == "pnnx.Input")
        {
            for (const Operand* r : op->outputs)
            {
                if (r->type != 1 && !is_ncnn_embedding_index(r))
                    reason = "non-FP32 runtime input";
                if (r->shape.empty() || std::find(r->shape.begin(), r->shape.end(), 0) != r->shape.end())
                    reason = "scalar or empty runtime input";
            }
        }
        if (op->type != "pnnx.Input" && op->type != "pnnx.Attribute")
        {
            for (const Operand* r : op->outputs)
            {
                // Integer storage is supported only for the direct index path
                // above, not general integer arithmetic or dtype conversion.
                if (r->type != 0 && r->type != 1)
                    reason = "non-FP32 computation";
            }
        }
        if (op->type == "pnnx.Output")
        {
            for (const Operand* r : op->inputs)
            {
                if (r->type != 1)
                    reason = "non-FP32 user output";
                if (r->shape.empty() || std::find(r->shape.begin(), r->shape.end(), 0) != r->shape.end())
                    reason = "scalar or empty user output";
            }
        }
        if (op->type.find("pool") != std::string::npos && op->has_param("stride") && op->params.at("stride").type == 5 && op->params.at("stride").ai.empty())
            reason = "empty pooling stride";
        if (!reason.empty())
        {
            error = "unsupported " + reason + " at " + op->type + " (" + op->name + ")";
            return -1;
        }
    }
    return 0;
}

void pass_ncnn(Graph& g, const std::vector<std::string>& module_operators)
{
    unroll_rnn_op(g);

    eliminate_maxpool_indices(g);

    attribute_unpooling(g);

    ncnn::fuse_convert_rotaryembed(g);

    ncnn::expand_expression(g);

    ncnn::chain_multi_output(g);

    // solve torch batch axis first, then legalize how ncnn carries it
    ncnn::solve_batch_index(g);
    ncnn::convert_batch_layout(g);

    ncnn::convert_half_to_float(g);

    ncnn::insert_reshape_numpy_binaryop_broadcast(g);
    ncnn::insert_reshape_pooling(g);
    ncnn::legalize_global_pooling_layout(g);
    ncnn::insert_reshape_linear(g);

    ncnn::fuse_convert_shufflechannel_slice(g);

    ncnn::convert_torch_cat(g);
    ncnn::convert_torch_chunk(g);
    ncnn::convert_torch_stack(g);
    ncnn::convert_torch_split(g);
    ncnn::convert_torch_unbind(g);
    ncnn::convert_torch_tensor_split(g);
    ncnn::convert_torch_einsum(g);

    ncnn::convert_reshape_interp_expression(g);
    ncnn::convert_slice_expression(g);

    ncnn::convert_Tensor_select(g);
    ncnn::convert_Tensor_slice(g);
    ncnn::convert_Tensor_slice_copy(g);

    // slice        -> crop + reshape
    // slice_copy   -> reshape + copyto

    int opindex = 0;
    for (auto x : g_global_pnnx_ncnn_graph_rewriter_passes)
    {
        for (auto rewriter : x.second)
        {
            pnnx_graph_rewrite(g, rewriter, opindex);
        }
    }

    ncnn::eliminate_noop(g);

    ncnn::eliminate_reshape_binaryop_broadcast(g);

    ncnn::insert_split(g);

    ncnn::fuse_transpose_matmul(g);
    ncnn::fuse_binaryop_eltwise(g);
    ncnn::fuse_padding_convolution(g);
    ncnn::fuse_padding_convolutiondepthwise(g);
    ncnn::fuse_convolution_activation(g);
    ncnn::fuse_convolution1d_activation(g);
    ncnn::fuse_convolutiondepthwise_activation(g);
    ncnn::fuse_convolutiondepthwise1d_activation(g);
    ncnn::fuse_deconvolution_activation(g);
    ncnn::fuse_deconvolutiondepthwise_activation(g);
    ncnn::fuse_innerproduct_activation(g);

    attribute_pooling(g);

    ncnn::insert_split(g);

    dead_code_elimination(g);

    canonicalize(g);

    ncnn::convert_custom_op(g);
    ncnn::convert_module_op(g, module_operators);

    ncnn::convert_attribute(g);

    ncnn::convert_input(g);

    ncnn::eliminate_output(g);
}

} // namespace pnnx
