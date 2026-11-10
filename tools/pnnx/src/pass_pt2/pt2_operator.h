// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_PT2_OPERATOR_H
#define PNNX_PT2_OPERATOR_H

#include "json.h"

namespace pnnx {

struct Pt2Argument
{
    std::string name;
    JsonValue value;
};

struct Pt2Operator
{
    std::string type;
    // values, including defaults, retain the serialized Argument union and enum encoding
    std::vector<Pt2Argument> inputs;
    std::vector<JsonValue> outputs;
    // Tensor arguments may be serialized as Python numbers by the exporter.
    std::vector<size_t> tensor_inputs;
    // schema input indices; aliases has one entry per schema return, including list returns
    std::vector<size_t> writes;
    std::vector<std::vector<size_t> > aliases;
};

int bind_pt2_operator(const JsonValue& node, Pt2Operator& op);

} // namespace pnnx

#endif // PNNX_PT2_OPERATOR_H
