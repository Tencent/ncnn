// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level5.h"

#include "pass_level5/attribute_unpooling.h"
#include "pass_level5/fold_constants.h"
#include "pass_level5/eliminate_dropout.h"
#include "pass_level5/eliminate_identity_operator.h"
#include "pass_level5/eliminate_noop_cat.h"
#include "pass_level5/eliminate_noop_einsum.h"
#include "pass_level5/eliminate_noop_expand.h"
#include "pass_level5/eliminate_noop_expression.h"
#include "pass_level5/eliminate_noop_pad.h"
#include "pass_level5/eliminate_noop_upsample.h"
#include "pass_level5/eliminate_noop_slice.h"
#include "pass_level5/eliminate_noop_reshape.h"
#include "pass_level5/eliminate_reshape_shape_expression.h"
#include "pass_level5/eliminate_type_as.h"
#include "pass_level5/eval_expression.h"
#include "pass_level5/fuse_adjacent_reshape.h"
#include "pass_level5/fuse_channel_shuffle.h"
#include "pass_level5/fuse_constant_expression.h"
#include "pass_level5/fuse_conv1d_batchnorm1d.h"
#include "pass_level5/fuse_conv2d_batchnorm2d.h"
#include "pass_level5/fuse_conv3d_batchnorm3d.h"
#include "pass_level5/fuse_convtranspose1d_batchnorm1d.h"
#include "pass_level5/fuse_convtranspose2d_batchnorm2d.h"
#include "pass_level5/fuse_convtranspose3d_batchnorm3d.h"
#include "pass_level5/fuse_layernorm.h"
#include "pass_level5/fuse_linear_batchnorm1d.h"
#include "pass_level5/fuse_multiheadattention.h"
#include "pass_level5/fuse_multiheadattention_sameqkv.h"
#include "pass_level5/fuse_pad_conv1d.h"
#include "pass_level5/fuse_pad_conv2d.h"
#include "pass_level5/fuse_rmsnorm.h"
#include "pass_level5/fuse_scaled_dot_product_attention.h"
#include "pass_level5/fuse_select_to_unbind.h"
#include "pass_level5/fuse_silu.h"
#include "pass_level5/fuse_slice_copy.h"
#include "pass_level5/fuse_slice_indices.h"
#include "pass_level5/fuse_slice_to_tensor_split.h"
#include "pass_level5/fuse_slice_squeeze_to_select.h"
#include "pass_level5/fuse_static_batchnorm.h"
#include "pass_level5/fuse_static_conv.h"
#include "pass_level5/fuse_static_convtranspose.h"
#include "pass_level5/fuse_static_embedding.h"
#include "pass_level5/fuse_static_groupnorm.h"
#include "pass_level5/fuse_static_instancenorm.h"
#include "pass_level5/fuse_static_layernorm.h"
#include "pass_level5/fuse_static_linear.h"
#include "pass_level5/fuse_static_prelu.h"
#include "pass_level5/fuse_static_rmsnorm.h"
#include "pass_level5/fuse_transformers_multiheadattention.h"
#include "pass_level5/fuse_transformers_scaled_dot_product_attention.h"
#include "pass_level5/normalize_einsum_equation.h"
#include "pass_level4/attribute_pooling.h"
#include "pass_level4/dead_code_elimination.h"
#include "pass_level4/canonicalize.h"
#include "pass_level3/fuse_index_expression.h"
#include "pass_level5/fuse_pixel_shuffle.h"
#include "pass_level5/fuse_pixel_unshuffle.h"

namespace pnnx {

void pass_level5(Graph& g, const std::set<std::string>& foldable_constants, const std::string& foldable_constants_zippath)
{
    eval_expression(g);

    fuse_constant_expression(g);

    fold_constants(g, foldable_constants, foldable_constants_zippath);

    eliminate_noop_expression(g);

    eliminate_noop_slice(g);

    fuse_slice_squeeze_to_select(g);

    fuse_slice_indices(g);

    normalize_einsum_equation(g);

    eliminate_noop_einsum(g);

    eliminate_identity_operator(g);

    fuse_select_to_unbind(g);

    fuse_slice_to_tensor_split(g);

    fuse_slice_copy(g);

    attribute_unpooling(g);

    fuse_static_batchnorm(g);
    fuse_static_groupnorm(g);
    fuse_static_instancenorm(g);
    fuse_static_layernorm(g);
    fuse_static_rmsnorm(g);

    fuse_static_conv(g);
    fuse_static_convtranspose(g);
    fuse_static_linear(g);
    fuse_static_embedding(g);
    fuse_static_prelu(g);

    fuse_conv1d_batchnorm1d(g);
    fuse_conv2d_batchnorm2d(g);
    fuse_conv3d_batchnorm3d(g);
    fuse_convtranspose1d_batchnorm1d(g);
    fuse_convtranspose2d_batchnorm2d(g);
    fuse_convtranspose3d_batchnorm3d(g);
    fuse_linear_batchnorm1d(g);

    fuse_pad_conv1d(g);
    fuse_pad_conv2d(g);

    eliminate_noop_pad(g);

    eliminate_noop_cat(g);

    eliminate_dropout(g);
    eliminate_type_as(g);

    eliminate_noop_upsample(g);

    // need to execute before fuse_adjacent_reshape
    fuse_pixel_shuffle(g);
    fuse_pixel_unshuffle(g);

    fuse_adjacent_reshape(g);

    eliminate_noop_reshape(g);

    eliminate_reshape_shape_expression(g);
    eliminate_noop_expand(g);

    fuse_channel_shuffle(g);
    fuse_layernorm(g);
    fuse_rmsnorm(g);

    fuse_transformers_multiheadattention(g);
    fuse_multiheadattention(g);

    fuse_multiheadattention_sameqkv(g);

    fuse_transformers_scaled_dot_product_attention(g);
    fuse_scaled_dot_product_attention(g);

    fuse_silu(g);

    fuse_index_expression(g);

    attribute_pooling(g);

    dead_code_elimination(g);

    canonicalize(g);
}

} // namespace pnnx
