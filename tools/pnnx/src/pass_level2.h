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

#ifndef PNNX_PASS_LEVEL2_H
#define PNNX_PASS_LEVEL2_H

#include "ir.h"

namespace pnnx {

class GraphRewriterPass
{
public:
    virtual ~GraphRewriterPass();

    virtual const char* match_pattern_graph() const = 0;

    virtual const char* replace_pattern_graph() const;

    virtual const char* type_str() const;

    virtual const char* name_str() const;

    virtual bool match(const std::map<std::string, Parameter>& captured_params) const;

    virtual bool match(const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const;

    virtual bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const;

    virtual void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const;

    virtual void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const;

    virtual void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params) const;

    virtual void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const;
};

class GraphRewriterPassRegister
{
public:
    GraphRewriterPassRegister(const GraphRewriterPass* pass, int priority);
    ~GraphRewriterPassRegister();
    const GraphRewriterPass* pass;
};

#define REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(CLASS, PRIORITY) \
    static GraphRewriterPassRegister g_global_pnnx_graphrewriterpass_##CLASS##_register(new CLASS, PRIORITY);

void pnnx_graph_rewrite(Graph& graph, const GraphRewriterPass* pass, int& opindex);

void pass_level2(Graph& g);

} // namespace pnnx

#endif // PNNX_PASS_LEVEL2_H
