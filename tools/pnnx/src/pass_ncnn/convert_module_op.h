// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

void convert_module_op(Graph& graph, const std::vector<std::string>& module_operators);

} // namespace ncnn

} // namespace pnnx
