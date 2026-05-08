// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "ncnn_dialect.h"

#include <mlir/IR/Builders.h>

namespace mlir {

namespace ncnn {

NCNNDialect::NCNNDialect(mlir::MLIRContext* context)
    : mlir::Dialect("ncnn", context, TypeID::get<NCNNDialect>())
{
    addOperations<
#define GET_OP_LIST
#include "ncnn_ops.cc.inc"
    >();

    // Support unknown operations because not all NCNN operations are
    // registered.
    allowUnknownOperations();
}

} // namespace ncnn

#define GET_OP_CLASSES
#include "ncnn_ops.cc.inc"

} // namespace mlir
