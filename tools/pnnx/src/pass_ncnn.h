// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_PASS_NCNN_H
#define PNNX_PASS_NCNN_H

#include "ir.h"

#include "pass_level2.h"

namespace pnnx {

class NcnnGraphRewriterPassRegister
{
public:
    NcnnGraphRewriterPassRegister(const GraphRewriterPass* pass, int priority);
    ~NcnnGraphRewriterPassRegister();
    const GraphRewriterPass* pass;
};

#define REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(CLASS, PRIORITY) \
    static NcnnGraphRewriterPassRegister g_global_pnnx_ncnngraphrewriterpass_##CLASS##_register(new CLASS, PRIORITY);

void pass_ncnn(Graph& g, const std::vector<std::string>& module_operators);

} // namespace pnnx

#endif // PNNX_PASS_NCNN_H
