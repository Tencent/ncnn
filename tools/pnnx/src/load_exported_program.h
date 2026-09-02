// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_LOAD_EXPORTED_PROGRAM_H
#define PNNX_LOAD_EXPORTED_PROGRAM_H

#include <string>

#include "exported_program.h"
#include "ir.h"

namespace pnnx {

int load_exported_program(const std::string& path, Graph& graph);
int import_exported_program_inputs(const pt2::ExportedProgramArchive& archive, Graph& graph, std::string& error);

} // namespace pnnx

#endif // PNNX_LOAD_EXPORTED_PROGRAM_H