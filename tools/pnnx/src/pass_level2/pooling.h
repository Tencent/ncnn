// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_POOLING_H
#define PNNX_POOLING_H

#include "ir.h"

namespace pnnx {

static inline bool match_pooling_stride(const Operator* stride)
{
    return (stride->type == "prim::Constant" && stride->has_param("value") && stride->params.at("value").type == 5)
           || (stride->type == "prim::ListConstruct" && stride->inputs.empty());
}

static inline void write_pooling_stride(Operator* op)
{
    const Operator* stride = op->inputs[1]->producer;
    if (stride->type == "prim::ListConstruct" || stride->params.at("value").ai.empty())
        op->params["stride"] = op->params.at("kernel_size");
    else
        op->params["stride"] = stride->params.at("value");

    op->inputs[1]->remove_consumer(op);
    op->inputs.erase(op->inputs.begin() + 1);
    op->inputnames.resize(1);
}

} // namespace pnnx

#endif // PNNX_POOLING_H
