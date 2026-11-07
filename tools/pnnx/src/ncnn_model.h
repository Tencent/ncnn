// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_NCNN_MODEL_H
#define PNNX_NCNN_MODEL_H

#include "ir.h"

namespace pnnx {

// Returns true when path names a textual ncnn param file.
bool is_ncnn_param_file(const std::string& path);

// Import/export the textual ncnn param and raw ncnn bin formats.
int load_ncnn_model(const std::string& parampath, const std::string& binpath, Graph& graph);
// Add the temporary pnnx graph boundaries needed by pass_ncnn.  The native
// adapter keeps these operators in the in-memory graph only; pass_ncnn turns
// them back into native Input layers and removes the temporary outputs.
int prepare_ncnn_graph_for_passes(Graph& graph);
int save_ncnn_model(const Graph& graph, const std::string& parampath, const std::string& binpath, int storage_type, const std::string& cutstart = std::string(), const std::string& cutend = std::string(), int cutstart_index = -1, int cutend_index = -1);

} // namespace pnnx

#endif // PNNX_NCNN_MODEL_H
