// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/script.h>

namespace pnnx {

void normalize_module_dtype(torch::jit::Module& mod);

} // namespace pnnx
