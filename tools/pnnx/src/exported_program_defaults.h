// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_EXPORTED_PROGRAM_DEFAULTS_H
#define PNNX_EXPORTED_PROGRAM_DEFAULTS_H

#include <string>

#include "exported_program.h"

namespace pnnx {
namespace pt2 {

bool append_default_arguments(ExportedProgram& program, std::string& error);
// Checks effects and common schema types, then orders arguments and fills
// dispatcher defaults. Also used by the public in-memory node importer.
bool normalize_exported_program_node(Node& node, std::string& error);

} // namespace pt2
} // namespace pnnx

#endif // PNNX_EXPORTED_PROGRAM_DEFAULTS_H