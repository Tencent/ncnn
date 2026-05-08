// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/script.h>

namespace pnnx {

void convert_half_to_float(torch::jit::Module& mod);

} // namespace pnnx
