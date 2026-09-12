// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_LOAD_PT2_H
#define PNNX_LOAD_PT2_H

#include "ir.h"

namespace pnnx {

bool is_pt2_archive(const std::string& path);
int load_pt2(const std::string& pt2path, Graph& g);

} // namespace pnnx

#endif // PNNX_LOAD_PT2_H
