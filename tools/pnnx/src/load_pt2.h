// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_LOAD_PT2_H
#define PNNX_LOAD_PT2_H

#include "ir.h"

namespace pnnx {

// PT2 loader without torch headers.
int load_pt2(const std::string& ptpath, Graph& g,
             const std::vector<std::vector<int64_t> >& input_shapes,
             const std::vector<std::string>& input_types);

} // namespace pnnx

#endif // PNNX_LOAD_PT2_H
