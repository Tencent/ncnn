// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_EXPORTED_PROGRAM_GRAPH_H
#define PNNX_EXPORTED_PROGRAM_GRAPH_H

#include "exported_program_schema.h"

#include <string>

namespace pnnx {

int normalize_exported_program_graph(const ExportedGraph& graph, ExportedGraph& normalized_graph, std::string& error);

} // namespace pnnx

#endif // PNNX_EXPORTED_PROGRAM_GRAPH_H
