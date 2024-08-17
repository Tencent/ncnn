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
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace pnnx {

GraphRewriterPass::~GraphRewriterPass()
{
}

const char* GraphRewriterPass::replace_pattern_graph() const
{
    return 0;
}

const char* GraphRewriterPass::type_str() const
{
    fprintf(stderr, "GraphRewriterPass type_str() should be implemented\n");
    return "unk";
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

bool GraphRewriterPass::match(const std::map<std::string, const Operator*>& /*matched_operators*/, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
{
    return match(captured_params, captured_attrs);
}

void GraphRewriterPass::write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
{
    if (replace_pattern_graph() == 0)
    {
        for (auto x : captured_params)
        {
            op->params[x.first] = x.second;
        }

        return;
    }

    for (auto x : op->params)
    {
        if (x.second.type != 4)
            continue;

        std::string str = x.second.s;
        if (str.find('%') == std::string::npos)
            continue;

        // search % token and replace with captured
        size_t pos = str.find('%');
        while (pos != std::string::npos)
        {
            // %xyz
            char buf[256];
            sscanf(str.c_str() + pos + 1, "%255[^][,() ]", buf);
            std::string key(buf);

            if (captured_params.find(key) == captured_params.end())
            {
                fprintf(stderr, "replace pattern param %%%s missing captured\n", key.c_str());
                return;
            }

            // replace %xyz with encoded_str
            std::string encoded_str = Parameter::encode_to_string(captured_params.at(key));
            str.replace(pos, key.size() + 1, encoded_str);

            pos = str.find('%', pos + 1);
        }

        op->params[x.first] = Parameter::parse_from_string(str);
    }

    for (size_t i = 0; i < op->inputs.size(); i++)
    {
        Operand* operand = op->inputs[i];
        std::vector<int>& shape = operand->shape;
        for (size_t j = 0; j < shape.size(); j++)
        {
            int ai = shape[j];
            if (ai == -233)
            {
                std::string key = operand->params.at(std::string("__shape__") + std::to_string(j)).s;

                if (captured_params.find(key) == captured_params.end())
                {
                    fprintf(stderr, "replace pattern param %%%s missing captured\n", key.c_str());
                    return;
                }

                shape[j] = captured_params.at(key).i;
            }
        }
    }

    for (size_t i = 0; i < op->outputs.size(); i++)
    {
        Operand* operand = op->outputs[i];
        std::vector<int>& shape = operand->shape;
        for (size_t j = 0; j < shape.size(); j++)
        {
            int ai = shape[j];
            if (ai == -233)
            {
                std::string key = operand->params.at(std::string("__shape__") + std::to_string(j)).s;

                if (captured_params.find(key) == captured_params.end())
                {
                    fprintf(stderr, "replace pattern param %%%s missing captured\n", key.c_str());
                    return;
                }

                shape[j] = captured_params.at(key).i;
            }
        }
    }
}

void GraphRewriterPass::write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
{
    write(op, captured_params);

    for (auto x : op->attrs)
    {
        if (x.second.type != 0)
            continue;

        std::string key((const char*)x.second.data.data());
        if (key.empty())
            continue;

        op->attrs[x.first] = captured_attrs.at(key);
    }
}

void GraphRewriterPass::write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params) const
{
    for (auto x : ops)
    {
        Operator* op = x.second;
        write(op, captured_params);
    }
}

void GraphRewriterPass::write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
{
    write(ops, captured_params);

    for (auto x : ops)
    {
        Operator* op = x.second;
        for (auto x : op->attrs)
        {
            if (x.second.type != 0)
                continue;

            std::string key(x.second.data.begin(), x.second.data.end());
            if (key.empty() || key[0] != '%')
                continue;

            op->attrs[x.first] = captured_attrs.at(key.substr(1));
        }
    }
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

static bool token_is_argument(const std::string& t)
{
    if (t[0] != '@' || t.size() < 2)
        return false;

    for (size_t i = 1; i < t.size(); i++)
    {
        if (t[i] < '0' || t[i] > '9')
            return false;
    }

    return true;
}

static bool match_expression(const Operator* a, const Operator* b, std::map<std::string, Parameter>& captured_params)
{
    if (a->params.size() != 1 || a->params.find("expr") == a->params.end())
        return false;

    if (b->params.size() != 1 || b->params.find("expr") == b->params.end())
        return false;

    const std::string& a_expr = a->params.at("expr").s;
    const std::string& b_expr = b->params.at("expr").s;

    if (a_expr == b_expr)
        return true;

    // split into tokens
    std::vector<std::string> a_tokens;
    std::vector<std::string> b_tokens;
    {
        std::string t;
        for (size_t i = 0; i < a_expr.size(); i++)
        {
            char ch = a_expr[i];

            if (ch == '[') // list
            {
                t += ch;
                a_tokens.push_back(t);
                t.clear();
            }
            else if (ch == '(' || ch == ')' || ch == ',' || ch == ']')
            {
                if (!t.empty())
                {
                    a_tokens.push_back(t);
                    t.clear();
                }
            }
            else
            {
                t += ch;
            }
        }

        if (!t.empty())
        {
            a_tokens.push_back(t);
        }
    }
    {
        std::string t;
        for (size_t i = 0; i < b_expr.size(); i++)
        {
            char ch = b_expr[i];

            if (ch == '[') // list
            {
                t += ch;
                b_tokens.push_back(t);
                t.clear();
            }
            else if (ch == '(' || ch == ')' || ch == ',' || ch == ']')
            {
                if (!t.empty())
                {
                    b_tokens.push_back(t);
                    t.clear();
                }
            }
            else
            {
                t += ch;
            }
        }

        if (!t.empty())
        {
            b_tokens.push_back(t);
        }
    }

    if (a_tokens.size() != b_tokens.size())
        return false;

    // capture values
    for (size_t i = 0; i < a_tokens.size(); i++)
    {
        const std::string& at = a_tokens[i];
        const std::string& bt = b_tokens[i];

        if (at == bt)
            continue;

        if (bt[0] != '%')
            return false;

        if (token_is_argument(at))
            return false;

        std::string key = bt.substr(1);

        captured_params[key] = Parameter::parse_from_string(at);
    }

    return true;
}

static bool match_parameter(const Parameter& a, const Parameter& b, std::map<std::string, Parameter>& captured_params)
{
    if (b.type == 4 && b.s[0] == '%')
    {
        std::string key = b.s.substr(1);
        if (captured_params.find(key) != captured_params.end())
        {
            // match previous captured parameter
            return captured_params.at(key) == a;
        }

        // captured parameter
        captured_params[key] = a;
        return true;
    }

    if (b.type == 4 && b.s == "*")
    {
        // ignored parameter
        return true;
    }

    if (b.type == 4 && (b.s[0] == '(' || b.s[0] == '[') && b.s.find('%') != std::string::npos)
    {
        // list with pattern
        if (a.type != 5 && a.type != 6 && a.type != 7)
            return false;

        std::string lc = b.s.substr(1, b.s.size() - 2);
        std::istringstream lcss(lc);

        size_t i = 0;
        while (!lcss.eof())
        {
            std::string elem;
            std::getline(lcss, elem, ',');

            if (elem[0] == '%')
            {
                std::string key = elem.substr(1);
                if (captured_params.find(key) != captured_params.end())
                {
                    // match previous captured parameter
                    if (a.type == 5 && captured_params.at(key).i != a.ai[i])
                        return false;
                    if (a.type == 6 && captured_params.at(key).f != a.af[i])
                        return false;
                    if (a.type == 7 && captured_params.at(key).s != a.as[i])
                        return false;
                }

                // captured parameter
                if (a.type == 5)
                    captured_params[key] = a.ai[i];
                if (a.type == 6)
                    captured_params[key] = a.af[i];
                if (a.type == 7)
                    captured_params[key] = a.as[i];
            }
            else if ((elem[0] != '-' && (elem[0] < '0' || elem[0] > '9')) || (elem[0] == '-' && (elem[1] < '0' || elem[1] > '9')))
            {
                // string
                if (a.type != 7)
                    return false;

                if (a.as[i] != elem)
                    return false;
            }
            else if (elem.find('.') != std::string::npos || elem.find('e') != std::string::npos)
            {
                // float
                if (a.type != 6)
                    return false;

                if (a.af[i] != std::stof(elem))
                    return false;
            }
            else
            {
                // integer
                if (a.type != 5)
                    return false;

                if (a.ai[i] != std::stoi(elem))
                    return false;
            }

            i++;
        }

        return true;
    }

    if (a.type != b.type)
    {
        if (a.type == 2 && b.type == 3)
            return a.i == b.f;

        if (a.type == 3 && b.type == 2)
            return a.f == b.i;

        return false;
    }

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

static bool match_attribute(const Attribute& a, const Attribute& b, std::map<std::string, Parameter>& captured_params, const std::string& attrname, std::map<std::string, Attribute>& captured_attrs)
{
    // @data
    // @data=(1,2,3,4)f32
    // @data=%op1.data

    if (b.type == 0)
    {
        std::string bs(b.data.begin(), b.data.end());
        if (bs.empty())
        {
            // capture any shape
            captured_attrs[attrname] = a;
            return true;
        }

        if (bs[0] == '%')
        {
            // the captured replace
            return true;
        }

        fprintf(stderr, "malformed attribute pattern %s\n", bs.c_str());
        return false;
    }

    const std::vector<int>& a_shape = a.shape;
    const std::vector<int>& b_shape = b.shape;
    if (b_shape.empty())
        return false;

    if (a_shape.empty())
        return false;

    if (a_shape.size() != b_shape.size())
        return false;

    for (size_t j = 0; j < a_shape.size(); j++)
    {
        int ai = a_shape[j];
        int bi = b_shape[j];
        if (ai == bi)
            continue;

        if (bi == -1)
            continue;

        if (bi > 0)
            return false;

        if (bi != -233)
            return false;

        std::string key = b.params.at(std::string("__shape__") + std::to_string(j)).s;

        if (captured_params.find(key) != captured_params.end())
        {
            // match previous captured parameter
            if (captured_params.at(key).i != ai)
                return false;
        }

        // captured parameter
        captured_params[key] = ai;
    }

    captured_attrs[attrname] = a;
    return true;
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
    else if (a->type == "pnnx.Expression")
    {
        if (!match_expression(a, b, captured_params))
            return false;
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

    // match shapes
    for (size_t i = 0; i < a->inputs.size(); i++)
    {
        int a_type = a->inputs[i]->type;
        int b_type = b->inputs[i]->type;
        if (b_type != 0 && a_type != b_type)
            return false;

        const std::vector<int>& a_shape = a->inputs[i]->shape;
        const std::vector<int>& b_shape = b->inputs[i]->shape;
        if (b_shape.empty())
            continue;

        if (a_shape.empty())
            return false;

        if (a_shape.size() != b_shape.size())
            return false;

        for (size_t j = 0; j < a_shape.size(); j++)
        {
            int ai = a_shape[j];
            int bi = b_shape[j];
            if (ai == bi)
                continue;

            if (bi == -1)
                continue;

            if (bi > 0)
                return false;

            if (bi != -233)
                return false;

            std::string key = b->inputs[i]->params.at(std::string("__shape__") + std::to_string(j)).s;

            if (captured_params.find(key) != captured_params.end())
            {
                // match previous captured parameter
                if (captured_params.at(key).i != ai)
                    return false;
            }

            // captured parameter
            captured_params[key] = ai;
        }
    }

    for (size_t i = 0; i < a->outputs.size(); i++)
    {
        int a_type = a->outputs[i]->type;
        int b_type = b->outputs[i]->type;
        if (b_type != 0 && a_type != b_type)
            return false;

        const std::vector<int>& a_shape = a->outputs[i]->shape;
        const std::vector<int>& b_shape = b->outputs[i]->shape;
        if (b_shape.empty())
            continue;

        if (a_shape.empty())
            return false;

        if (a_shape.size() != b_shape.size())
            return false;

        for (size_t j = 0; j < a_shape.size(); j++)
        {
            int ai = a_shape[j];
            int bi = b_shape[j];
            if (ai == bi)
                continue;

            if (bi == -1)
                continue;

            if (bi > 0)
                return false;

            if (bi != -233)
                return false;

            std::string key = b->outputs[i]->params.at(std::string("__shape__") + std::to_string(j)).s;

            if (captured_params.find(key) != captured_params.end())
            {
                // match previous captured parameter
                if (captured_params.at(key).i != ai)
                    return false;
            }

            // captured parameter
            captured_params[key] = ai;
        }
    }

    for (const auto& p : a->attrs)
    {
        const std::string& akey = p.first;
        const Attribute& aa = p.second;

        std::string attrname = b->name + '.' + akey;

        if (b->attrs.find(akey) == b->attrs.end())
        {
            // capture all attributes
            captured_attrs[attrname] = aa;
        }
        else
        {
            if (!match_attribute(aa, b->attrs.at(akey), captured_params, attrname, captured_attrs))
                return false;
        }
    }

    return true;
}

static bool match(const Operator* anchor, const Operator* pattern, std::map<std::string, const Operator*>& matched_operators, std::map<std::string, const Operand*>& matched_inputs, std::map<std::string, const Operand*>& matched_outputs, std::map<std::string, Parameter>& captured_params, std::map<std::string, Attribute>& captured_attrs)
{
    if (!match_operator(anchor, pattern, captured_params, captured_attrs))
        return false;

    for (size_t i = 0; i < pattern->outputs.size(); i++)
    {
        if (pattern->outputs[i]->consumers.size() == 1 && pattern->outputs[i]->consumers[0]->type == "pnnx.Output")
        {
            if (matched_outputs.find(pattern->outputs[i]->name) == matched_outputs.end())
            {
                matched_outputs[pattern->outputs[i]->name] = anchor->outputs[i];
            }
            else if (matched_outputs[pattern->outputs[i]->name] != anchor->outputs[i])
            {
                return false;
            }
            continue;
        }

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

        if (!match(anchor2, pattern2, matched_operators, matched_inputs, matched_outputs, captured_params, captured_attrs))
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
        std::map<std::string, const Operator*> matched_operators;
        std::map<std::string, const Operand*> matched_inputs;
        std::map<std::string, const Operand*> matched_outputs;
        std::map<std::string, Parameter> captured_params;
        std::map<std::string, Attribute> captured_attrs;

        // pattern match from end to beginning
        int q = graph_op_count - 1;
        for (; q >= 1; q--)
        {
            matched = true;

            for (const Operator* pattern : pattern_graph_output_operators)
            {
                for (size_t i = 0; i < pattern->inputs.size(); i++)
                {
                    const Operator* pattern2 = pattern->inputs[i]->producer;

                    int j = q;
                    for (; j >= 0; j--)
                    {
                        const Operator* anchor = graph.ops[j];

                        std::map<std::string, const Operator*> matched_operators2;
                        std::map<std::string, const Operand*> matched_inputs2;
                        std::map<std::string, const Operand*> matched_outputs2;
                        std::map<std::string, Parameter> captured_params2;
                        std::map<std::string, Attribute> captured_attrs2;
                        if (!match(anchor, pattern2, matched_operators2, matched_inputs2, matched_outputs2, captured_params2, captured_attrs2))
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
                        for (auto x : matched_outputs2)
                        {
                            if (matched_outputs.find(x.first) == matched_outputs.end())
                            {
                                matched_outputs[x.first] = x.second;
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

            if (matched && !pass->match(matched_operators, captured_params, captured_attrs))
            {
                matched_operators.clear();
                matched_inputs.clear();
                matched_outputs.clear();
                captured_params.clear();
                captured_attrs.clear();
                matched = false;
                continue;
            }

            break;
        }

        if (!matched)
            break;

        //         fprintf(stderr, "matched !\n");

        // lets replace

        // remove all operands inside matched graph
        std::map<std::string, Operand*> operands_to_remove;
        for (auto& _x : matched_operators)
        {
            Operator* x = (Operator*)_x.second;
            for (auto& r : x->inputs)
            {
                r->remove_consumer(x);

                bool is_input = false;
                for (auto& r2 : matched_inputs)
                {
                    if (r2.second == r)
                    {
                        is_input = true;
                        break;
                    }
                }

                if (!is_input)
                    operands_to_remove[r->name] = r;
            }

            x->inputs.clear();

            for (auto& r : x->outputs)
            {
                r->producer = 0;

                bool is_output = false;
                for (auto& r2 : matched_outputs)
                {
                    if (r2.second == r)
                    {
                        is_output = true;
                        break;
                    }
                }

                if (!is_output)
                    operands_to_remove[r->name] = r;
            }

            x->outputs.clear();
        }
        for (auto& _x : operands_to_remove)
        {
            Operand* r = _x.second;
            graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), r));
            delete r;
        }

        // insert new operator at the last matched one
        const Operator* cur = 0;
        {
            int cur_index = 1;
            for (auto& o : matched_operators)
            {
                int c_index = std::find(graph.ops.begin(), graph.ops.end(), o.second) - graph.ops.begin();
                cur_index = std::max(cur_index, c_index + 1);
            }

            cur_index = std::min(cur_index, (int)graph.ops.size() - 1);
            cur = graph.ops[cur_index];
        }

        // remove all matched_operators
        for (auto& _x : matched_operators)
        {
            //             fprintf(stderr, "remove %s\n", _x.second->name.c_str());

            Operator* x = (Operator*)_x.second;

            graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), x));

            delete _x.second;
        }

        if (pass->replace_pattern_graph() == 0)
        {
            // insert single
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
        else
        {
            // insert multiple
            Graph replace_graph;
            replace_graph.parse(pass->replace_pattern_graph());

            // move operators and operands from replace_graph to graph except input and output
            std::map<std::string, Operator*> ops;
            for (size_t i = 0; i < replace_graph.ops.size(); i++)
            {
                Operator* op = replace_graph.ops[i];
                if (op->type == "pnnx.Input" || op->type == "pnnx.Output")
                    continue;

                graph.ops.insert(std::find(graph.ops.begin(), graph.ops.end(), cur), op);
                replace_graph.ops[i] = 0;
                ops[op->name] = op;
            }

            for (size_t i = 0; i < replace_graph.operands.size(); i++)
            {
                Operand* r = replace_graph.operands[i];
                if (r->producer->type == "pnnx.Input" || (r->consumers.size() == 1 && r->consumers[0]->type == "pnnx.Output"))
                    continue;

                graph.operands.push_back(r);
                replace_graph.operands[i] = 0;
            }

            replace_graph.ops.erase(std::remove(replace_graph.ops.begin(), replace_graph.ops.end(), (Operator*)0), replace_graph.ops.end());
            replace_graph.operands.erase(std::remove(replace_graph.operands.begin(), replace_graph.operands.end(), (Operand*)0), replace_graph.operands.end());

            for (size_t i = 0; i < pattern_graph_inputs.size(); i++)
            {
                const std::string& k = pattern_graph_inputs[i];
                Operand* r = (Operand*)matched_inputs.at(k);
                const Operand* rr = replace_graph.get_operand(k);

                for (auto x : rr->consumers)
                {
                    r->consumers.push_back(x);

                    x->inputnames.resize(x->inputs.size());
                    for (size_t j = 0; j < x->inputs.size(); j++)
                    {
                        if (x->inputs[j]->name == k)
                        {
                            x->inputs[j] = r;
                            x->inputnames[j] = k;
                            break;
                        }
                    }
                }
            }

            for (size_t i = 0; i < pattern_graph_outputs.size(); i++)
            {
                const std::string& k = pattern_graph_outputs[i];
                Operand* r = (Operand*)matched_outputs.at(k);
                const Operand* rr = replace_graph.get_operand(k);

                r->producer = rr->producer;

                for (size_t j = 0; j < r->producer->outputs.size(); j++)
                {
                    if (r->producer->outputs[j]->name == k)
                    {
                        r->producer->outputs[j] = r;
                        break;
                    }
                }
            }

            pass->write(ops, captured_params, captured_attrs);

            for (auto x : ops)
            {
                new_ops.push_back(x.second);
            }
        }
    }

    // assign new op name number
    for (int i = (int)new_ops.size() - 1; i >= 0; i--)
    {
        new_ops[i]->name = new_ops[i]->name + "_" + std::to_string(opindex++);
    }
}

static bool is_alias_op(const Operator* op)
{
    if (op->type == "aten::slice" || op->type == "aten::select")
        return true;

    if (op->type == "aten::view")
        return true;

    return false;
}

static void functionize(Graph& graph)
{
    // graph.save("0.param", "0.bin");

    // 1. create shadow view/slice/select/... for each consumer
    // 2. replace inplace op, append copy
    // 3. tag operand alias for view/slice/select/... output
    // 4. scan inplace op, collect affacted alias
    //     5. look for any op after the inplace op with alias input
    //     6. collect ops on the chain back to alias
    //     7. move chain after copy op
    //     8. update all alias uses after copy op, retag alias
    // 9. clear all alias tag

    // 1. create shadow view/slice/select/... for each consumer
    {
        for (int i = (int)graph.ops.size() - 1; i >= 0; i--)
        {
            Operator* op = graph.ops[i];

            if (!is_alias_op(op))
                continue;

            Operand* out0 = op->outputs[0];

            if (out0->consumers.size() == 1)
                continue;

            bool all_consumers_are_same = true;
            for (size_t j = 1; j < out0->consumers.size(); j++)
            {
                if (out0->consumers[j] != out0->consumers[0])
                {
                    all_consumers_are_same = false;
                    break;
                }
            }
            if (all_consumers_are_same)
                continue;

            for (int j = (int)out0->consumers.size() - 1; j > 0; j--)
            {
                Operator* op1 = out0->consumers[j];

                Operator* op_shadow = graph.new_operator_after(op->type, op->name + "_pnnxshadow_" + std::to_string(j), op);

                Operand* shadow_out = graph.new_operand(op_shadow->name + "_out");

                op_shadow->inputs = op->inputs;
                op_shadow->params = op->params;
                op_shadow->outputs.push_back(shadow_out);

                for (Operand* x : op->inputs)
                {
                    x->consumers.push_back(op_shadow);
                }

                shadow_out->producer = op_shadow;
                shadow_out->type = out0->type;
                shadow_out->shape = out0->shape;
                shadow_out->params = out0->params;

                shadow_out->consumers.push_back(op1);

                for (size_t k = 0; k < op1->inputs.size(); k++)
                {
                    if (op1->inputs[k] == out0)
                        op1->inputs[k] = shadow_out;
                }
            }

            out0->consumers.resize(1);
        }
    }

    // graph.save("1.param", "1.bin");

    // 2. replace inplace op, append copy
    // 3. tag operand alias for view/slice/select/... output
    {
        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            bool is_inplace_op = op->type.size() > 2 && op->type[op->type.size() - 2] != '_' && op->type[op->type.size() - 1] == '_';

            if (op->type != "aten::copy_" && !is_alias_op(op) && !is_inplace_op)
                continue;

            Operand* in = op->inputs[0];

            int alias_index;
            if (in->params.find("__alias__") != in->params.end())
            {
                alias_index = in->params.at("__alias__").i;
            }
            else
            {
                alias_index = std::find(graph.operands.begin(), graph.operands.end(), in) - graph.operands.begin();
            }

            if (op->type == "aten::copy_")
            {
                op->outputs[0]->params["__alias__"] = alias_index;
                // fprintf(stderr, "operand %s is alias of %s\n", op->outputs[0]->name.c_str(), graph.operands[alias_index]->name.c_str());

                // set copy output shape as the alias one
                op->outputs[0]->type = graph.operands[alias_index]->type;
                op->outputs[0]->shape = graph.operands[alias_index]->shape;

                continue;
            }

            if (is_alias_op(op))
            {
                op->outputs[0]->params["__alias__"] = alias_index;
                // fprintf(stderr, "operand %s is alias of %s\n", op->outputs[0]->name.c_str(), graph.operands[alias_index]->name.c_str());
                continue;
            }

            if (is_inplace_op)
            {
                // replace with non-inplace version, create copy op
                op->type = op->type.substr(0, op->type.size() - 1);

                // append aten::copy_
                if (graph.operands[alias_index]->consumers.size() > 1)
                {
                    Operand* in0 = op->inputs[0];
                    Operand* out0 = op->outputs[0];

                    Operator* op_copy = graph.new_operator_after("aten::copy_", op->name + "_copy", op);
                    Operand* copy_out = graph.new_operand(op->name + "_copy_out");

                    op_copy->inputs.push_back(in0);
                    op_copy->inputs.push_back(out0);
                    in0->consumers.push_back(op_copy);
                    out0->consumers.push_back(op_copy);

                    op_copy->outputs.push_back(copy_out);
                    copy_out->producer = op_copy;
                }
            }
        }
    }

    // graph.save("3.param", "3.bin");

    // 4. scan inplace copy op, collect affacted alias
    {
        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "aten::copy_")
                continue;

            op->type = "aten::copy";

            Operand* out0 = op->outputs[0];

            // inplace op output always alias with the input
            const int alias_index = out0->params.at("__alias__").i;
            Operand* alias_in0 = graph.operands[alias_index];

            // fprintf(stderr, "\n---> %s  for %s\n", op->name.c_str(), alias_in0->name.c_str());

            size_t i_advanced = 0;

            // 5. look for any op after the inplace op with alias input
            for (size_t j = i + 1; j < graph.ops.size(); j++)
            {
                Operator* op1 = graph.ops[j];

                bool affacted = false;
                for (Operand* x : op1->inputs)
                {
                    if (x == alias_in0)
                    {
                        affacted = true;
                        break;
                    }

                    if (x->params.find("__alias__") == x->params.end())
                        continue;

                    int alias_index_1 = x->params.at("__alias__").i;
                    if (alias_index_1 == alias_index)
                    {
                        affacted = true;
                        break;
                    }
                }

                if (!affacted)
                    continue;

                // 6. collect ops on the chain back to alias
                std::set<size_t> chainsx_op_indexes;
                {
                    size_t op1_index = std::find(graph.ops.begin(), graph.ops.end(), op1) - graph.ops.begin();

                    if (op1_index < i - i_advanced)
                    {
                        chainsx_op_indexes.insert(op1_index);
                        // fprintf(stderr, "affacted op %s for %s\n", op1->name.c_str(), graph.operands[alias_index]->name.c_str());
                    }

                    while (1)
                    {
                        Operand* x = op1->inputs[0];
                        if (x->params.find("__alias__") == x->params.end())
                            break;

                        int alias_index_1 = x->params.at("__alias__").i;
                        if (alias_index_1 != alias_index)
                            break;

                        op1 = x->producer;
                        size_t op1_index = std::find(graph.ops.begin(), graph.ops.end(), op1) - graph.ops.begin();

                        if (op1_index < i - i_advanced)
                        {
                            chainsx_op_indexes.insert(op1_index);
                            // fprintf(stderr, "affacted op %s for %s   chained\n", op1->name.c_str(), graph.operands[alias_index]->name.c_str());
                        }
                    }
                }

                // 7. move chain after copy op
                {
                    int k = 0;
                    for (size_t doi : chainsx_op_indexes)
                    {
                        doi -= k;
                        // fprintf(stderr, "---> move %s after %s\n", graph.ops[doi]->name.c_str(), graph.ops[i - i_advanced]->name.c_str());

                        for (size_t l = doi; l < i - i_advanced; l++)
                        {
                            std::swap(graph.ops[l], graph.ops[l + 1]);
                        }

                        k += 1;
                    }

                    i_advanced += chainsx_op_indexes.size();
                }

                // 8. update all alias uses after copy op, retag alias
                out0->params.erase("__alias__");
                const int new_alias_index = std::find(graph.operands.begin(), graph.operands.end(), out0) - graph.operands.begin();
                for (size_t k = i - i_advanced + 1; k < graph.ops.size(); k++)
                {
                    Operator* op2 = graph.ops[k];

                    // bool use_in0 = false;
                    for (size_t l = 0; l < op2->inputs.size(); l++)
                    {
                        if (op2->inputs[l] == alias_in0)
                        {
                            // fprintf(stderr, "---> replace %s input %s to %s\n", op2->name.c_str(), op2->inputs[l]->name.c_str(), out0->name.c_str());

                            op2->inputs[l] = out0;
                            alias_in0->remove_consumer(op2);
                            out0->consumers.push_back(op2);
                        }
                    }

                    for (Operand* x : op2->outputs)
                    {
                        if (x->params.find("__alias__") != x->params.end() && x->params.at("__alias__").i == alias_index)
                        {
                            x->params["__alias__"] = new_alias_index;
                        }
                    }
                }

                // rewind to the updated copy operator
                j -= chainsx_op_indexes.size();
            }
        }
    }

    // graph.save("4.param", "4.bin");

    // 9. clear all alias tag
    {
        for (Operand* x : graph.operands)
        {
            x->params.erase("__alias__");
        }
    }
}

void pass_level2(Graph& g)
{
    functionize(g);

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
