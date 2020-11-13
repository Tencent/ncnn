// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef NCNN_DIALECT_H
#define NCNN_DIALECT_H

#include <mlir/IR/Dialect.h>
#include <mlir/IR/Function.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

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

} // namespace ncnn

#define GET_OP_CLASSES
#include "ncnn_ops.h.inc"

} // namespace mlir

#endif // NCNN_DIALECT_H
