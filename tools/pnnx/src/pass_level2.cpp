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

#include "pass_level2.h"

#include <algorithm>
#include <map>
#include <unordered_map>

namespace pnnx {

GraphRewriterPass::~GraphRewriterPass()
{
}

const char* GraphRewriterPass::name_str() const
{
    return type_str();
}

bool GraphRewriterPass::match(const std::map<std::string, Parameter>& /*captured_params*/) const
{
    return true;
}

bool GraphRewriterPass::match(const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
{
    return match(captured_params);
}

void GraphRewriterPass::write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
{
    for (auto x : captured_params)
    {
        op->params[x.first] = x.second;
    }
}

void GraphRewriterPass::write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
{
    write(op, captured_params);
}

static std::map<int, std::vector<const GraphRewriterPass*> > g_global_pnnx_graph_rewriter_passes;

GraphRewriterPassRegister::GraphRewriterPassRegister(const GraphRewriterPass* _pass, int priority)
    : pass(_pass)
{
    if (g_global_pnnx_graph_rewriter_passes.find(priority) == g_global_pnnx_graph_rewriter_passes.end())
    {
        g_global_pnnx_graph_rewriter_passes[priority] = std::vector<const GraphRewriterPass*>();
    }

    g_global_pnnx_graph_rewriter_passes[priority].push_back(pass);
}

GraphRewriterPassRegister::~GraphRewriterPassRegister()
{
    delete pass;
}

static bool match_parameter(const Parameter& a, const Parameter& b, std::map<std::string, Parameter>& captured_params)
{
    if (b.type == 4 && b.s[0] == '%')
    {
        // captured parameter
        captured_params[b.s.substr(1)] = a;
        return true;
    }

    if (b.type == 4 && b.s == "*")
    {
        // ignored parameter
        return true;
    }

    if (a.type != b.type)
        return false;

    const int type = a.type;

    if (type == 0)
    {
        return true;
    }
    if (type == 1)
    {
        return a.b == b.b;
    }
    if (type == 2)
    {
        return a.i == b.i;
    }
    if (type == 3)
    {
        return a.f == b.f;
    }
    if (type == 4)
    {
        return a.s == b.s;
    }
    if (type == 5)
    {
        if (a.ai.size() != b.ai.size())
            return false;

        for (size_t i = 0; i < a.ai.size(); i++)
        {
            if (a.ai[i] != b.ai[i])
                return false;
        }

        return true;
    }
    if (type == 6)
    {
        if (a.af.size() != b.af.size())
            return false;

        for (size_t i = 0; i < a.af.size(); i++)
        {
            if (a.af[i] != b.af[i])
                return false;
        }

        return true;
    }
    if (type == 7)
    {
        if (a.as.size() != b.as.size())
            return false;

        for (size_t i = 0; i < a.as.size(); i++)
        {
            if (a.as[i] != b.as[i])
                return false;
        }

        return true;
    }

    // unknown
    return false;
}

static bool match_operator(const Operator* a, const Operator* b, std::map<std::string, Parameter>& captured_params, std::map<std::string, Attribute>& captured_attrs)
{
    if (a->type != b->type)
        return false;

    if (a->inputs.size() != b->inputs.size())
        return false;

    if (a->outputs.size() != b->outputs.size())
        return false;

    // match params
    if (b->params.size() == 1 && b->params.find("%*") != b->params.end() && b->params.at("%*").type == 4 && b->params.at("%*").s == "%*")
    {
        for (const auto& p : a->params)
        {
            const std::string& pkey = p.first;
            const Parameter& pp = p.second;

            // capture all parameters
            captured_params[b->name + '.' + pkey] = pp;
        }
    }
    else
    {
        if (a->params.size() != b->params.size())
            return false;

        for (const auto& p : a->params)
        {
            const std::string& akey = p.first;
            const Parameter& ap = p.second;

            if (b->params.find(akey) == b->params.end())
                return false;

            if (!match_parameter(ap, b->params.at(akey), captured_params))
                return false;
        }
    }

    for (const auto& p : a->attrs)
    {
        const std::string& akey = p.first;
        const Attribute& aa = p.second;

        // capture all attributes
        captured_attrs[b->name + '.' + akey] = aa;
    }

    return true;
}

static bool match(const Operator* anchor, const Operator* pattern, std::unordered_map<std::string, const Operator*>& matched_operators, std::unordered_map<std::string, const Operand*>& matched_inputs, std::map<std::string, Parameter>& captured_params, std::map<std::string, Attribute>& captured_attrs)
{
    if (!match_operator(anchor, pattern, captured_params, captured_attrs))
        return false;

    for (size_t i = 0; i < pattern->outputs.size(); i++)
    {
        if (pattern->outputs[i]->consumers.size() == 1 && pattern->outputs[i]->consumers[0]->type == "pnnx.Output")
            continue;

        if (anchor->outputs[i]->consumers.size() != pattern->outputs[i]->consumers.size())
            return false;
    }

    matched_operators[pattern->name] = anchor;

    // lets match
    for (size_t i = 0; i < pattern->inputs.size(); i++)
    {
        const Operator* anchor2 = anchor->inputs[i]->producer;
        const Operator* pattern2 = pattern->inputs[i]->producer;

        if (pattern2->type == "pnnx.Input")
        {
            if (matched_inputs.find(pattern->inputs[i]->name) == matched_inputs.end())
            {
                matched_inputs[pattern->inputs[i]->name] = anchor->inputs[i];
            }
            else if (matched_inputs[pattern->inputs[i]->name] != anchor->inputs[i])
            {
                return false;
            }
            continue;
        }

        if (!match(anchor2, pattern2, matched_operators, matched_inputs, captured_params, captured_attrs))
            return false;
    }

    return true;
}

void pnnx_graph_rewrite(Graph& graph, const GraphRewriterPass* pass, int& opindex)
{
    Graph pattern_graph;
    pattern_graph.parse(pass->match_pattern_graph());

    // collect pattern inputs and outputs order
    std::vector<std::string> pattern_graph_inputs;
    std::vector<std::string> pattern_graph_outputs;
    std::vector<const Operator*> pattern_graph_output_operators;
    for (const auto& x : pattern_graph.ops)
    {
        if (x->type == "pnnx.Input")
        {
            for (const auto& y : x->outputs)
                pattern_graph_inputs.push_back(y->name);
        }
        if (x->type == "pnnx.Output")
        {
            pattern_graph_output_operators.push_back(x);
            for (const auto& y : x->inputs)
                pattern_graph_outputs.push_back(y->name);
        }
    }

    std::vector<Operator*> new_ops;

    while (1)
    {
        const int graph_op_count = (int)graph.ops.size();

        bool matched = true;

        // lets match from output
        std::unordered_map<std::string, const Operator*> matched_operators;
        std::unordered_map<std::string, const Operand*> matched_inputs;
        std::unordered_map<std::string, const Operand*> matched_outputs;
        std::map<std::string, Parameter> captured_params;
        std::map<std::string, Attribute> captured_attrs;

        // pattern match from end to beginning
        int q = graph_op_count - 1;
        for (; q >= 1; q--)
        {
            for (const Operator* pattern : pattern_graph_output_operators)
            {
                for (size_t i = 0; i < pattern->inputs.size(); i++)
                {
                    const Operator* pattern2 = pattern->inputs[i]->producer;

                    int j = q;
                    for (; j >= 0; j--)
                    {
                        const Operator* anchor = graph.ops[j];

                        std::unordered_map<std::string, const Operator*> matched_operators2;
                        std::unordered_map<std::string, const Operand*> matched_inputs2;
                        std::map<std::string, Parameter> captured_params2;
                        std::map<std::string, Attribute> captured_attrs2;
                        if (!match(anchor, pattern2, matched_operators2, matched_inputs2, captured_params2, captured_attrs2))
                            continue;

                        bool submatch_matched = true;
                        for (auto x : matched_operators2)
                        {
                            // check these matched operators are same with previous matched ones
                            if (matched_operators.find(x.first) != matched_operators.end())
                            {
                                if (matched_operators[x.first] != x.second)
                                {
                                    // unmatched two sub-matches
                                    submatch_matched = false;
                                    break;
                                }
                            }
                            else
                            {
                                matched_operators[x.first] = x.second;
                            }
                        }

                        if (!submatch_matched)
                            continue;

                        for (auto x : matched_inputs2)
                        {
                            if (matched_inputs.find(x.first) == matched_inputs.end())
                            {
                                matched_inputs[x.first] = x.second;
                            }
                        }
                        for (auto x : captured_params2)
                        {
                            captured_params[x.first] = x.second;
                        }
                        for (auto x : captured_attrs2)
                        {
                            captured_attrs[x.first] = x.second;
                        }

                        // match !
                        matched_outputs[pattern->inputs[i]->name] = anchor->outputs[i];
                        break;
                    }

                    if (j == -1)
                    {
                        matched = false;
                        break;
                    }
                }

                if (!matched)
                    break;
            }

            if (matched && !pass->match(captured_params, captured_attrs))
            {
                matched_operators.clear();
                matched_inputs.clear();
                matched_outputs.clear();
                captured_params.clear();
                captured_attrs.clear();
                continue;
            }

            break;
        }

        if (!matched)
            break;

        //         fprintf(stderr, "matched !\n");

        // lets replace

        // remove all matched_operators
        for (auto& _x : matched_operators)
        {
            //             fprintf(stderr, "remove %s\n", _x.second->name.c_str());

            Operator* x = (Operator*)_x.second;
            for (auto& r : x->inputs)
            {
                r->remove_consumer(x);
            }

            x->inputs.clear();

            for (auto& r : x->outputs)
            {
                r->producer = 0;
            }

            x->outputs.clear();

            graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), x));

            delete _x.second;
        }

        // insert new operator before all output consumers
        const Operator* cur = 0;
        {
            int cur_index = graph.ops.size() - 1;
            for (auto& o : matched_outputs)
            {
                for (auto& c : o.second->consumers)
                {
                    int c_index = std::find(graph.ops.begin(), graph.ops.end(), c) - graph.ops.begin();
                    cur_index = std::min(cur_index, c_index);
                }
            }

            cur = graph.ops[cur_index];
        }

        Operator* op = graph.new_operator_before(pass->type_str(), std::string(pass->name_str()), cur);

        for (const auto& k : pattern_graph_inputs)
        {
            Operand* r = (Operand*)matched_inputs.at(k);
            r->consumers.push_back(op);
            op->inputs.push_back(r);

            op->inputnames.push_back(k);
        }

        for (const auto& k : pattern_graph_outputs)
        {
            Operand* r = (Operand*)matched_outputs.at(k);
            r->producer = op;
            op->outputs.push_back(r);
        }

        pass->write(op, captured_params, captured_attrs);

        new_ops.push_back(op);
    }

    // assign new op name number
    for (int i = (int)new_ops.size() - 1; i >= 0; i--)
    {
        new_ops[i]->name = new_ops[i]->name + "_" + std::to_string(opindex++);
    }
}

void pass_level2(Graph& g)
{
    int opindex = 0;
    for (auto x : g_global_pnnx_graph_rewriter_passes)
    {
        for (auto rewriter : x.second)
        {
            pnnx_graph_rewrite(g, rewriter, opindex);
        }
    }
}

} // namespace pnnx
