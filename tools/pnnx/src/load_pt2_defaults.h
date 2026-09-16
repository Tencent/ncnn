// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// default-kwargs restoration for the pt2 loader (split out of
// load_exportedprogram.cpp).
//
// torch.export graph JSON omits schema default arguments; this module fills
// them back in by parameter name (append_default_kwargs) so the translated
// ops carry every input the pass_level2 patterns expect, in canonical order.

#ifndef PNNX_LOAD_PT2_DEFAULTS_H
#define PNNX_LOAD_PT2_DEFAULTS_H

#include <string>
#include <vector>

namespace pnnx {

class Graph;
class Operator;
class Parameter;

// create a prim::Constant operator and wire it as an input of the consumer
// (must be inserted before the consumer so pass_level3 fuse_expression handles
// the consumer first while the constant is still a prim::Constant)
void new_constant(Graph& g, Operator* consumer, const Parameter& value, int& constant_index);

// append default scalar inputs for aten operators that omitted default kwargs
void append_default_kwargs(Graph& g, Operator* op, const std::string& type, const std::vector<std::string>& inputnames, int& constant_index);

} // namespace pnnx

#endif // PNNX_LOAD_PT2_DEFAULTS_H
