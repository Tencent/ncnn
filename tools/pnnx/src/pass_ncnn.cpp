// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

#include "pass_ncnn/convert_attribute.h"
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
#include "pass_ncnn/insert_reshape_numpy_binaryop_broadcast.h"
#include "pass_ncnn/insert_reshape_linear.h"
#include "pass_ncnn/insert_reshape_pooling.h"
#include "pass_ncnn/insert_reshape_global_pooling.h"

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

void pass_ncnn(Graph& g, const std::vector<std::string>& module_operators)
{
    unroll_rnn_op(g);

    eliminate_maxpool_indices(g);

    attribute_unpooling(g);

    ncnn::expand_expression(g);

    ncnn::chain_multi_output(g);

    ncnn::solve_batch_index(g);

    ncnn::convert_half_to_float(g);

    ncnn::insert_reshape_numpy_binaryop_broadcast(g);
    ncnn::insert_reshape_pooling(g);
    ncnn::insert_reshape_linear(g);
    ncnn::insert_reshape_global_pooling(g);

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
