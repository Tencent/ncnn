// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_LOAD_EXPORTED_PROGRAM_H
#define PNNX_LOAD_EXPORTED_PROGRAM_H

#include "ir.h"
#include "model_format.h"

#include <string>

namespace pnnx {

int load_exported_program(const std::string& pt2path, const ModelFormatInfo& format_info, Graph& graph, std::string& error);

} // namespace pnnx

#endif // PNNX_LOAD_EXPORTED_PROGRAM_H
