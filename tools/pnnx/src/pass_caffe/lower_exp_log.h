// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "ir.h"

namespace pnnx {

namespace caffe2pnnx {

int prepare_exp_log(Operator* op);
void lower_exp_log(Graph& graph);

} // namespace caffe2pnnx

} // namespace pnnx
