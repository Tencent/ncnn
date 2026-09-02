// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_LOAD_EXPORTED_PROGRAM_H
#define PNNX_LOAD_EXPORTED_PROGRAM_H

#include <string>

#include "ir.h"

namespace pnnx {

int load_exported_program(const std::string& path, Graph& graph);

} // namespace pnnx

#endif // PNNX_LOAD_EXPORTED_PROGRAM_H