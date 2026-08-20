// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_LOAD_PT2_H
#define PNNX_LOAD_PT2_H

#include <stdint.h>

#include <string>
#include <vector>

namespace pnnx {

class Graph;

int load_pt2(const std::string& path, Graph& graph, const std::vector<std::vector<int64_t> >& input_shapes, const std::vector<std::vector<int64_t> >& input_shapes2);

} // namespace pnnx

#endif // PNNX_LOAD_PT2_H
