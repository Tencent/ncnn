// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

void eliminate_output(Graph& graph, bool preserve_output_set = false);

} // namespace ncnn

} // namespace pnnx
