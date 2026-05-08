// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_static_embedding.h"

#include "pass_level2.h"

namespace pnnx {

class fuse_static_Fembedding_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @data=(%num_embeddings,%embedding_dim)f32
F.embedding             op_0        2 1 input weight out scale_grad_by_freq=* sparse=%sparse
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.Embedding            embedding   1 1 input out num_embeddings=%num_embeddings embedding_dim=%embedding_dim sparse=%sparse @weight=%op_weight.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

void fuse_static_embedding(Graph& graph)
{
    fuse_static_Fembedding_pass a;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
}

} // namespace pnnx
