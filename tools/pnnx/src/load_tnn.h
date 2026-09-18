// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_LOAD_TNN_H
#define PNNX_LOAD_TNN_H

#include "ir.h"

namespace pnnx {

int load_tnn(const std::string& tnnpath, Graph& g,
             const std::vector<std::vector<int64_t> >& input_shapes,
             const std::vector<std::string>& input_types);

} // namespace pnnx

#endif // PNNX_LOAD_TNN_H
