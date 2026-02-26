// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_SAVE_NCNN_H
#define PNNX_SAVE_NCNN_H

#include "ir.h"

namespace pnnx {

int save_ncnn(const Graph& g, const std::string& parampath, const std::string& binpath, const std::string& pypath, const std::vector<std::vector<int64_t> >& input_shapes, int fp16);

} // namespace pnnx

#endif // PNNX_SAVE_NCNN_H
