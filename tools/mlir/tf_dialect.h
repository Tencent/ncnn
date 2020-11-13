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

#ifndef TF_DIALECT_H
#define TF_DIALECT_H

#include <mlir/Dialect/Traits.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/DerivedAttributeOpInterface.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "tf_traits.h"

namespace mlir {

namespace TF {

#include "tf_op_interfaces.h.inc"

class TensorFlowDialect : public mlir::Dialect
{
public:
    TensorFlowDialect(mlir::MLIRContext* context);

    static StringRef getDialectNamespace()
    {
        return "tf";
    }

    Attribute parseAttribute(DialectAsmParser& parser, Type type) const override;

    // Parse a type registered to this dialect.
    Type parseType(DialectAsmParser& parser) const override;

    // Parses resource type with potential subtypes.
    Type ParseResourceType(DialectAsmParser& parser, Location loc) const;

    // Parse and print variant type. It may have subtypes inferred using shape
    // inference.
    Type ParseVariantType(DialectAsmParser& parser, Location loc) const;

    // Registered hook to materialize a constant operation from a given attribute
    // value with the desired resultant type.
    Operation* materializeConstant(OpBuilder& builder, Attribute value, Type type, Location loc) override;
};

} // namespace TF

} // namespace mlir

#define GET_OP_CLASSES
#include "tf_all_ops.h.inc"

#endif // TF_DIALECT_H
