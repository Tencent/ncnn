// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_DIALECT_H
#define NCNN_DIALECT_H

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Pass/Pass.h>

namespace mlir {

namespace ncnn {

class NCNNDialect : public mlir::Dialect
{
public:
    NCNNDialect(mlir::MLIRContext* context);

    static StringRef getDialectNamespace()
    {
        return "ncnn";
    }
};

std::unique_ptr<OperationPass<FuncOp> > createNCNNOptimizePass();

} // namespace ncnn

#define GET_OP_CLASSES
#include "ncnn_ops.h.inc"

} // namespace mlir

#endif // NCNN_DIALECT_H
