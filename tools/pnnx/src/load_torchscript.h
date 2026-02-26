// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_LOAD_TORCHSCRIPT_H
#define PNNX_LOAD_TORCHSCRIPT_H

#include "ir.h"

namespace pnnx {

int load_torchscript(const std::string& ptpath, Graph& g,
                     const std::string& device,
                     const std::vector<std::vector<int64_t> >& input_shapes,
                     const std::vector<std::string>& input_types,
                     const std::vector<std::vector<int64_t> >& input_shapes2,
                     const std::vector<std::string>& input_types2,
                     const std::vector<std::string>& customop_modules,
                     const std::vector<std::string>& module_operators,
                     const std::string& foldable_constants_zippath,
                     std::set<std::string>& foldable_constants);

} // namespace pnnx

#endif // PNNX_LOAD_TORCHSCRIPT_H
