// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_PASS_CAFFE_H
#define PNNX_PASS_CAFFE_H

#include "ir.h"

namespace pnnx {

int pass_caffe(Graph& graph);
int check_caffe_operators(const Graph& graph);

} // namespace pnnx

#endif // PNNX_PASS_CAFFE_H
