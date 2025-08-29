// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class Embedding : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.sparse.Embedding";
    }

    const char* type_str() const
    {
        return "nn.Embedding";
    }

    void write(Operator* op, const TorchGraphProxy& graph, const TorchModuleProxy& mod) const
    {
        const TorchNodeProxy* embedding = graph.find_node_by_kind("aten::embedding");

        const TorchTensorProxy& weight = mod.attr("weight");

        op->params["num_embeddings"] = weight.size(0);
        op->params["embedding_dim"] = weight.size(1);

        // op->params["padding_idx"] = embedding->namedInput("padding_idx");
        // op->params["scale_grad_by_freq"] = embedding->namedInput("scale_grad_by_freq");
        op->params["sparse"] = embedding->namedInput("sparse");

        op->attrs["weight"] = weight;
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(Embedding)

} // namespace pnnx
