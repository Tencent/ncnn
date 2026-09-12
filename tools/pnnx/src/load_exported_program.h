// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_LOAD_EXPORTED_PROGRAM_H
#define PNNX_LOAD_EXPORTED_PROGRAM_H

#include "ir.h"

namespace pnnx {

bool model_file_is_exported_program(const std::string& path);

int load_exported_program(const std::string& ptpath, Graph& g,
                          const std::vector<std::vector<int64_t> >& input_shapes,
                          const std::vector<std::string>& input_types);

} // namespace pnnx

#endif // PNNX_LOAD_EXPORTED_PROGRAM_H
