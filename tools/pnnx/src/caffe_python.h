// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_CAFFE_PYTHON_H
#define PNNX_CAFFE_PYTHON_H

#include <stdio.h>
#include "ir.h"

namespace pnnx {

void export_caffe_python(const Graph& graph, FILE* pyfp);

} // namespace pnnx

#endif // PNNX_CAFFE_PYTHON_H
