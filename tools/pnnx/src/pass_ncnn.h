// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

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

void pass_ncnn(Graph& g);

} // namespace pnnx

#endif // PNNX_PASS_NCNN_H
