// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_PT2_GRAPH_LOWERING_H
#define PNNX_PT2_GRAPH_LOWERING_H

#include <string>

namespace pnnx {

class Graph;
struct Pt2Program;
struct Pt2Weights;

int lower_pt2_graph(const Pt2Program& program, Pt2Weights& weights, Graph& graph, std::string& error);

} // namespace pnnx

#endif // PNNX_PT2_GRAPH_LOWERING_H
