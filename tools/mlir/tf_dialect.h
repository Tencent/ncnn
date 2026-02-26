// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TF_DIALECT_H
#define TF_DIALECT_H

#include <mlir/Dialect/Traits.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/DerivedAttributeOpInterface.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "tf_traits.h"

namespace mlir {

namespace TF {

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
    Type ParseResourceType(DialectAsmParser& parser) const;

    // Parse and print variant type. It may have subtypes inferred using shape
    // inference.
    Type ParseVariantType(DialectAsmParser& parser) const;

    // Registered hook to materialize a constant operation from a given attribute
    // value with the desired resultant type.
    Operation* materializeConstant(OpBuilder& builder, Attribute value, Type type, Location loc) override;
};

} // namespace TF

} // namespace mlir

#define GET_OP_CLASSES
#include "tf_all_ops.h.inc"

#endif // TF_DIALECT_H
