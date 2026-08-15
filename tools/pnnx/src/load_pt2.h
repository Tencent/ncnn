// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_LOAD_PT2_H
#define PNNX_LOAD_PT2_H

#include <string>

namespace pnnx {

class Graph;

int load_pt2(const std::string& path, Graph& graph);

} // namespace pnnx

#endif // PNNX_LOAD_PT2_H
