// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "eliminate_noop_math.h"

#include <algorithm>
#include "pass_level2.h"
#include "pass_level4/dead_code_elimination.h"

namespace pnnx {

static bool constant_is_all_constant(const Operator* op_constant, float vf, int vi)
{
    const Parameter& param = op_constant->params.at("value");

    if (param.type == 2)
    {
        if (param.i != vi)
            return false;
    }
    else if (param.type == 3)
    {
        if (param.f != vf)
            return false;
    }
    else
    {
        // unsupported data type
        return false;
    }

    return true;
}

static bool attribute_is_all_constant(const Operator* op_attr, float vf, int vi)
{
    const Attribute& attr = op_attr->attrs.begin()->second;

    if (attr.shape.empty())
    {
        fprintf(stderr, "shape empty!\n");
        return false;
    }

    int size = attr.shape[0];
    for (size_t i = 1; i < attr.shape.size(); i++)
    {
        size *= attr.shape[i];
    }

    if (attr.type == 1)
    {
        const float* p = (const float*)attr.data.data();
        for (int i = 0; i < size; i++)
        {
            if (p[i] != vf)
                return false;
        }
    }
    else if (attr.type == 2)
    {
        const double* p = (const double*)attr.data.data();
        for (int i = 0; i < size; i++)
        {
            if (p[i] != vf)
                return false;
        }
    }
    else if (attr.type == 4)
    {
        const int* p = (const int*)attr.data.data();
        for (int i = 0; i < size; i++)
        {
            if (p[i] != vi)
                return false;
        }
    }
    else if (attr.type == 5)
    {
        const int64_t* p = (const int64_t*)attr.data.data();
        for (int i = 0; i < size; i++)
        {
            if (p[i] != vi)
                return false;
        }
    }
    else if (attr.type == 7)
    {
        const signed char* p = (const signed char*)attr.data.data();
        for (int i = 0; i < size; i++)
        {
            if (p[i] != vi)
                return false;
        }
    }
    else if (attr.type == 8)
    {
        const unsigned char* p = (const unsigned char*)attr.data.data();
        for (int i = 0; i < size; i++)
        {
            if (p[i] != vi)
                return false;
        }
    }
    else
    {
        // unsupported data type
        return false;
    }

    return true;
}

static bool operator_is_all_constant(const Operator* op, float vf, int vi)
{
    if (op->type == "pnnx.Attribute")
        return attribute_is_all_constant(op, vf, vi);

    if (op->type == "prim::Constant")
        return constant_is_all_constant(op, vf, vi);

    return false;
}

void eliminate_noop_math(Graph& graph)
{
    for (;;)
    {
        bool need_eliminate = false;

        // build expression via reverse order
        for (int i = (int)graph.ops.size() - 1; i >= 0; i--)
        {
            Operator* op = graph.ops[i];

            int identity_input_id = 0;
            if (op->type == "aten::add" || op->type == "aten::add_")
            {
                Operator* op0 = op->inputs[0]->producer;
                Operator* op1 = op->inputs[1]->producer;
                Operator* op2 = op->inputs[2]->producer;

                if (operator_is_all_constant(op1, 0.f, 0))
                {
                    // x <= a + 0 * c
                    need_eliminate = true;
                    identity_input_id = 0;
                }
                else if (operator_is_all_constant(op2, 0.f, 0))
                {
                    // x <= a + b * 0
                    need_eliminate = true;
                    identity_input_id = 0;
                }
                else if (operator_is_all_constant(op0, 0.f, 0) && operator_is_all_constant(op0, 1.f, 1))
                {
                    // x <= 0 + b * 1
                    need_eliminate = true;
                    identity_input_id = 1;
                }
            }
            if (op->type == "aten::sub")
            {
                Operator* op1 = op->inputs[1]->producer;
                Operator* op2 = op->inputs[2]->producer;

                if (operator_is_all_constant(op1, 0.f, 0))
                {
                    // x <= a - 0 * c
                    need_eliminate = true;
                    identity_input_id = 0;
                }
                else if (operator_is_all_constant(op2, 0.f, 0))
                {
                    // x <= a - b * 0
                    need_eliminate = true;
                    identity_input_id = 0;
                }
            }
            if (op->type == "aten::rsub")
            {
                Operator* op0 = op->inputs[0]->producer;
                Operator* op1 = op->inputs[1]->producer;
                Operator* op2 = op->inputs[2]->producer;

                if (operator_is_all_constant(op0, 0.f, 0) && operator_is_all_constant(op2, 1.f, 1))
                {
                    // x <= b * 1 - 0
                    need_eliminate = true;
                    identity_input_id = 1;
                }
                else if (operator_is_all_constant(op0, 0.f, 0) && operator_is_all_constant(op1, 1.f, 1))
                {
                    // x <= 1 * c - 0
                    need_eliminate = true;
                    identity_input_id = 2;
                }
            }
            if (op->type == "aten::mul")
            {
                Operator* op0 = op->inputs[0]->producer;
                Operator* op1 = op->inputs[1]->producer;

                if (operator_is_all_constant(op0, 1.f, 1))
                {
                    // x <= 1 * b
                    need_eliminate = true;
                    identity_input_id = 1;
                }
                if (operator_is_all_constant(op1, 1.f, 1))
                {
                    // x <= a * 1
                    need_eliminate = true;
                    identity_input_id = 0;
                }
            }
            if (op->type == "aten::div" || op->type == "aten::div_")
            {
                Operator* op1 = op->inputs[1]->producer;

                if (operator_is_all_constant(op1, 1.f, 1))
                {
                    // x <= a / 1
                    need_eliminate = true;
                    identity_input_id = 0;
                }
            }
            if (op->type == "aten::pow")
            {
                Operator* op1 = op->inputs[1]->producer;

                if (operator_is_all_constant(op1, 1.f, 1))
                {
                    // x <= x ^ 1
                    need_eliminate = true;
                    identity_input_id = 0;
                }
            }

            if (!need_eliminate)
                continue;

            fprintf(stderr, "eliminate_noop_math %s %s\n", op->type.c_str(), op->name.c_str());

            for (auto& x : op->inputs)
            {
                x->remove_consumer(op);
            }

            Operand* math_out = op->outputs[0];

            for (auto& x : math_out->consumers)
            {
                for (size_t j = 0; j < x->inputs.size(); j++)
                {
                    if (x->inputs[j] == math_out)
                        x->inputs[j] = op->inputs[identity_input_id];
                }

                op->inputs[identity_input_id]->consumers.push_back(x);
            }

            math_out->producer = 0;
            math_out->consumers.clear();

            graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), math_out));
            delete math_out;

            op->inputs.clear();
            op->outputs.clear();

            graph.ops.erase(graph.ops.begin() + i);
            delete op;

            break;
        }

        if (!need_eliminate)
            break;
    }

    // dce
    dead_code_elimination(graph);
}

} // namespace pnnx
