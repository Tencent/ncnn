// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

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
