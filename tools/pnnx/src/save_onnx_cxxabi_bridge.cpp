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

#include "ir.h"

namespace pnnx {

const char* get_operand_name(const Operand* x)
{
    return x->name.c_str();
}

const char* get_operator_type(const Operator* op)
{
    return op->type.c_str();
}

const char* get_operator_name(const Operator* op)
{
    return op->name.c_str();
}

std::vector<const char*> get_operator_params_keys(const Operator* op)
{
    std::vector<const char*> keys;
    for (const auto& it : op->params)
    {
        const std::string& key = it.first;
        keys.push_back(key.c_str());
    }
    return keys;
}

std::vector<const char*> get_operator_attrs_keys(const Operator* op)
{
    std::vector<const char*> keys;
    for (const auto& it : op->attrs)
    {
        const std::string& key = it.first;
        keys.push_back(key.c_str());
    }
    return keys;
}

const Parameter& get_operator_param(const Operator* op, const char* key)
{
    return op->params.at(key);
}

const Attribute& get_operator_attr(const Operator* op, const char* key)
{
    return op->attrs.at(key);
}

const char* get_param_s(const Parameter& p)
{
    return p.s.c_str();
}

std::vector<const char*> get_param_as(const Parameter& p)
{
    std::vector<const char*> as;
    for (const auto& s : p.as)
    {
        as.push_back(s.c_str());
    }
    return as;
}

} // namespace pnnx
