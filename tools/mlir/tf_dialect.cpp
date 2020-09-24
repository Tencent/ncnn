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

#include "tf_dialect.h"

#include <mlir/Dialect/Traits.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/DerivedAttributeOpInterface.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Parser.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/InliningUtils.h>

#include "tf_attributes.h"
#include "tf_side_effects.h"
#include "tf_traits.h"

namespace mlir {

static LogicalResult Verify(...)
{
    return success();
}
static LogicalResult VerifyPartitionedCall(...)
{
    return success();
}
static LogicalResult VerifyStridedSliceBase(...)
{
    return success();
}
static LogicalResult VerifyUnsortedSegmentReduction(...)
{
    return success();
}

namespace TF {

TensorFlowDialect::TensorFlowDialect(MLIRContext* context)
    : Dialect(/*name=*/"tf", context, TypeID::get<TensorFlowDialect>())
{
    addOperations<
#define GET_OP_LIST
#include "tf_all_ops.cc.inc"
    >();
    addTypes<
#define HANDLE_TF_TYPE(tftype, enumerant, name)      tftype##Type,
#define HANDLE_LAST_TF_TYPE(tftype, enumerant, name) tftype##Type
#include "tf_types.def"
    >();
    //   addInterfaces<TFInlinerInterface, TFDecodeAttributesInterface,
    //                 TFConstantFoldInterface>();
    addAttributes<ShapeAttr, FuncAttr>();

    // Support unknown operations because not all TensorFlow operations are
    // registered.
    allowUnknownOperations();

    //   for (const auto &hook : *TensorFlowDialect::additional_operation_hooks_) {
    //     hook(*this);
    //   }
}

namespace {

ShapeAttr ParseShapeAttr(MLIRContext* context, StringRef spec, Location loc)
{
    auto emit_error = [&, spec]() {
        emitError(loc, "invalid TensorFlow shape attribute: ") << spec;
        return nullptr;
    };

    if (!spec.consume_front("shape<")) return emit_error();

    if (spec.consume_front("*>"))
        return mlir::TF::ShapeAttr::get(context, llvm::None);

    SmallVector<int64_t, 4> shape;
    while (!spec.consume_front(">"))
    {
        int64_t dim;

        if (spec.consume_front("?"))
            dim = -1;
        else if (spec.consumeInteger(10, dim) || dim < 0)
            return emit_error();

        spec.consume_front("x");

        shape.push_back(dim);
    }

    return mlir::TF::ShapeAttr::get(context, llvm::makeArrayRef(shape));
}

// Parses a #tf.func attribute of the following format:
//
//   #tf.func<@symbol, {attr = "value"}>
//
// where the first element is a SymbolRefAttr and the second element is a
// DictionaryAttr.
FuncAttr ParseFuncAttr(MLIRContext* context, StringRef spec, Location loc)
{
    auto emit_error = [&, spec]() {
        emitError(loc, "invalid TensorFlow func attribute: ") << spec;
        return nullptr;
    };

    if (!spec.consume_front("func<")) return emit_error();

    size_t func_name_num_read = 0;
    Attribute func_name_attr = mlir::parseAttribute(spec, context, func_name_num_read);
    if (!func_name_attr || !func_name_attr.isa<SymbolRefAttr>())
        return emit_error();
    spec = spec.drop_front(func_name_num_read);

    if (!spec.consume_front(", ")) return emit_error();

    size_t func_attrs_num_read = 0;
    Attribute func_attrs_attr = mlir::parseAttribute(spec, context, func_attrs_num_read);
    if (!func_attrs_attr || !func_attrs_attr.isa<DictionaryAttr>())
        return emit_error();
    spec = spec.drop_front(func_attrs_num_read);

    if (!spec.consume_front(">")) return emit_error();

    return mlir::TF::FuncAttr::get(context, func_name_attr.cast<SymbolRefAttr>(),
                                   func_attrs_attr.cast<DictionaryAttr>());
}

} // namespace

Attribute TensorFlowDialect::parseAttribute(DialectAsmParser& parser,
        Type type) const
{
    auto spec = parser.getFullSymbolSpec();
    Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());

    if (spec.startswith("shape")) return ParseShapeAttr(getContext(), spec, loc);

    if (spec.startswith("func")) return ParseFuncAttr(getContext(), spec, loc);

    return (emitError(loc, "unknown TensorFlow attribute: " + spec), nullptr);
}

// Parses a type registered to this dialect.
Type TensorFlowDialect::parseType(DialectAsmParser& parser) const
{
    StringRef data;
    if (parser.parseKeyword(&data)) return Type();

    Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());

#define HANDLE_TF_TYPE(tftype, enumerant, name) \
    if (data == name) return tftype##Type::get(getContext());
// Custom TensorFlow types are handled separately at the end as they do partial
// match.
#define HANDLE_CUSTOM_TF_TYPE(tftype, enumerant, name)
// NOLINTNEXTLINE
#include "tf_types.def"

    if (data.startswith("resource")) return ParseResourceType(parser, loc);
    if (data.startswith("variant")) return ParseVariantType(parser, loc);
    return (emitError(loc, "unknown TensorFlow type: " + data), nullptr);
}

namespace {
template<typename TypeWithSubtype>
Type ParseTypeWithSubtype(MLIRContext* context, DialectAsmParser& parser,
                          Location loc)
{
    // Default type without inferred subtypes.
    if (failed(parser.parseOptionalLess())) return TypeWithSubtype::get(context);

    // Most types with subtypes have only one subtype.
    SmallVector<TensorType, 1> subtypes;
    do
    {
        TensorType tensor_ty;
        if (parser.parseType(tensor_ty)) return Type();
        subtypes.push_back(tensor_ty);
    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseGreater()) return Type();
    return TypeWithSubtype::getChecked(subtypes, context, loc);
}
} // anonymous namespace

Type TensorFlowDialect::ParseResourceType(DialectAsmParser& parser,
        Location loc) const
{
    return ParseTypeWithSubtype<ResourceType>(getContext(), parser, loc);
}

Type TensorFlowDialect::ParseVariantType(DialectAsmParser& parser,
        Location loc) const
{
    return ParseTypeWithSubtype<VariantType>(getContext(), parser, loc);
}

Operation* TensorFlowDialect::materializeConstant(OpBuilder& builder,
        Attribute value, Type type,
        Location loc)
{
    return builder.create<ConstOp>(loc, type, value);
}

// Builds a constant op with the specified attribute `value`. The result
// op's type is deduced from `value`; if `value` is of scalar type,
// wraps it up with a tensor type of empty shape.
// TODO(jpienaar): This one differs from the autogenerated one as it takes an
// attribute but always creates an ElementsAttr internally.
void ConstOp::build(OpBuilder& builder, OperationState& result,
                    Attribute value)
{
    ShapedType type;
    if (auto elem_attr = value.dyn_cast<ElementsAttr>())
    {
        return ConstOp::build(builder, result, elem_attr);
    }
    else if (value.isa<BoolAttr, FloatAttr, IntegerAttr>())
    {
        // All TensorFlow types must be tensor types. In the build() method,
        // we want to provide more flexibility by allowing attributes of scalar
        // types. But we need to wrap it up with ElementsAttr to construct
        // valid TensorFlow constants.
        type = RankedTensorType::get(/*shape=*/ {}, value.getType());
        return ConstOp::build(builder, result, DenseElementsAttr::get(type, value));
    }
    // TODO(jpienaar): support other TensorFlow specific types.
    llvm_unreachable("unsupported attribute type for building tf.Const");
}

void ConstOp::build(OpBuilder& builder, OperationState& result, Type type,
                    Attribute value)
{
    // Handle the case where the type and value are already tensors.
    if (type.isa<TensorType>() && value.isa<ElementsAttr>())
    {
        result.addTypes(type);
        result.addAttribute("value", value);
        return;
    }

    // Otherwise, default to the attribute builder.
    ConstOp::build(builder, result, value);
    assert(type == result.types[0] && "type mismatch in construction");
}

LogicalResult ConstOp::inferReturnTypes(
    MLIRContext* context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes)
{
    auto value = attributes.get("value");
    if (!value) return emitOptionalError(location, "missing attribute 'value'");
    if (auto elem_attr = value.dyn_cast<ElementsAttr>())
    {
        inferredReturnTypes.assign({elem_attr.getType()});
        return success();
    }
    return emitOptionalError(location,
                             "attribute 'value' failed to satisfy constraint: "
                             "constant vector/tensor");
}

Region& WhileRegionOp::getLoopBody()
{
    return body();
}

bool WhileRegionOp::isDefinedOutsideOfLoop(Value value)
{
    // If the Op defining the value exists and the defining op is outside the
    // scope of this WhileRegion, then we can infer that its defined outside.
    // The defining Op is outside the scope of this WhileRegion if this
    // WhileRegionOp is not an ancestor of the defining op in the parent chain.
    Operation* def_op = value.getDefiningOp();
    return def_op && !getOperation()->isAncestor(def_op);
}

LogicalResult WhileRegionOp::moveOutOfLoop(
    llvm::ArrayRef<mlir::Operation*> ops)
{
    // Move the hoisted value to just before the while.
    Operation* while_op = this->getOperation();
    for (auto op : ops) op->moveBefore(while_op);
    return success();
}

} // namespace TF

#define GET_OP_CLASSES
#include "tf_all_ops.cc.inc"

} // namespace mlir
