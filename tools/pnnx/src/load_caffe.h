// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_LOAD_CAFFE_H
#define PNNX_LOAD_CAFFE_H

#include "ir.h"

namespace pnnx {

int load_caffe(const std::string& caffeprotopath, Graph& g,
               const std::vector<std::vector<int64_t> >& input_shapes,
               const std::vector<std::string>& input_types);

} // namespace pnnx

#endif // PNNX_LOAD_CAFFE_H
