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

#include <stdio.h>

#include <map>
#include <set>

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/MathExtras.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/Traits.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Module.h>
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
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Parser.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/InliningUtils.h>

#include "tf_attributes.h"
#include "tf_traits.h"

namespace mlir {

static LogicalResult Verify(...) { return success(); }
static LogicalResult VerifyPartitionedCall(...) { return success(); }
static LogicalResult VerifyStridedSliceBase(...) { return success(); }
static LogicalResult VerifyUnsortedSegmentReduction(...) { return success(); }

namespace TF {

#include "tf_op_interfaces.h.inc"

class TensorFlowDialect : public mlir::Dialect
{
public:
    TensorFlowDialect(mlir::MLIRContext *context);

    Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;

    // Parse a type registered to this dialect.
    Type parseType(DialectAsmParser &parser) const override;

    // Parses resource type with potential subtypes.
    Type ParseResourceType(DialectAsmParser &parser, Location loc) const;

    // Parse and print variant type. It may have subtypes inferred using shape
    // inference.
    Type ParseVariantType(DialectAsmParser &parser, Location loc) const;

    // Registered hook to materialize a constant operation from a given attribute
    // value with the desired resultant type.
    Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                    Location loc) override;
};

#define GET_OP_CLASSES
#include "tf_ops.h.inc"

namespace {
struct TFInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  // Defines the legality of inlining TF operations.
  bool isLegalToInline(Operation *, Region *,
                       BlockAndValueMapping &) const final {
    // TODO(riverriddle) For now, enable inlining all operations. This isn't
    // correct in the face of operations that cannot be duplicated, but this
    // requires more intricate side-effect modeling.
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  // Attempts to materialize a conversion for a type mismatch between a call
  // from this dialect, and a callable region. This method should generate an
  // operation that takes 'input' as the only operand, and produces a single
  // result of 'resultType'. If a conversion can not be generated, nullptr
  // should be returned.
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type result_type,
                                       Location conversion_loc) const final {
    if (!result_type.isa<TensorType>() || !input.getType().isa<TensorType>())
      return nullptr;
    return builder.create<TF::CastOp>(conversion_loc, result_type, input,
                                      /*truncate=*/builder.getBoolAttr(false));
  }
};
}  // end anonymous namespace

TensorFlowDialect::TensorFlowDialect(mlir::MLIRContext *context)
    : mlir::Dialect("tf", context)
{
    addOperations<
#define GET_OP_LIST
#include "tf_ops.cpp.inc"
        >();

    addTypes<
#define HANDLE_TF_TYPE(tftype, enumerant, name) tftype##Type,
#define HANDLE_LAST_TF_TYPE(tftype, enumerant, name) tftype##Type
#include "tf_types.def"
        >();
    addInterfaces<TFInlinerInterface>();
    addAttributes<ShapeAttr, FuncAttr>();

    // Support unknown operations because not all TensorFlow operations are
    // registered.
    allowUnknownOperations();
}

ShapeAttr ParseShapeAttr(MLIRContext *context, StringRef spec, Location loc) {
  auto emit_error = [&, spec]() {
    emitError(loc, "invalid TensorFlow shape attribute: ") << spec;
    return nullptr;
  };

  if (!spec.consume_front("shape<")) return emit_error();

  if (spec.consume_front("*>"))
    return mlir::TF::ShapeAttr::get(context, llvm::None);

  SmallVector<int64_t, 4> shape;
  while (!spec.consume_front(">")) {
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
FuncAttr ParseFuncAttr(MLIRContext *context, StringRef spec, Location loc) {
  auto emit_error = [&, spec]() {
    emitError(loc, "invalid TensorFlow func attribute: ") << spec;
    return nullptr;
  };

  if (!spec.consume_front("func<")) return emit_error();

  size_t func_name_num_read = 0;
  Attribute func_name_attr =
      mlir::parseAttribute(spec, context, func_name_num_read);
  if (!func_name_attr || !func_name_attr.isa<SymbolRefAttr>())
    return emit_error();
  spec = spec.drop_front(func_name_num_read);

  if (!spec.consume_front(", ")) return emit_error();

  size_t func_attrs_num_read = 0;
  Attribute func_attrs_attr =
      mlir::parseAttribute(spec, context, func_attrs_num_read);
  if (!func_attrs_attr || !func_attrs_attr.isa<DictionaryAttr>())
    return emit_error();
  spec = spec.drop_front(func_attrs_num_read);

  if (!spec.consume_front(">")) return emit_error();

  return mlir::TF::FuncAttr::get(context, func_name_attr.cast<SymbolRefAttr>(),
                                 func_attrs_attr.cast<DictionaryAttr>());
}

Attribute TensorFlowDialect::parseAttribute(DialectAsmParser &parser,
                                            Type type) const {
  auto spec = parser.getFullSymbolSpec();
  Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());

  if (spec.startswith("shape")) return ParseShapeAttr(getContext(), spec, loc);

  if (spec.startswith("func")) return ParseFuncAttr(getContext(), spec, loc);

  return (emitError(loc, "unknown TensorFlow attribute: " + spec), nullptr);
}

// Parses a type registered to this dialect.
Type TensorFlowDialect::parseType(DialectAsmParser &parser) const {
  StringRef data;
  if (parser.parseKeyword(&data)) return Type();

  Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());
  auto typeKind = llvm::StringSwitch<unsigned>(data)
#define HANDLE_TF_TYPE(tftype, enumerant, name) \
  .Case(name, TensorFlowTypes::enumerant)
// Custom TensorFlow types are handled separately at the end as they do partial
// match.
#define HANDLE_CUSTOM_TF_TYPE(tftype, enumerant, name)
// NOLINTNEXTLINE
#include "tf_types.def"
                      .StartsWith("resource", TensorFlowTypes::RESOURCE)
                      .StartsWith("variant", TensorFlowTypes::VARIANT)
                      .Default(0);
  switch (typeKind) {
    default:
      return (emitError(loc, "unknown TensorFlow type: " + data), nullptr);

#define HANDLE_TF_TYPE(tftype, enumerant, name) \
  case TensorFlowTypes::enumerant:              \
    return tftype##Type::get(getContext());
#define HANDLE_CUSTOM_TF_TYPE(tftype, enumerant, name)
// NOLINTNEXTLINE
#include "tf_types.def"
    case TensorFlowTypes::RESOURCE:
      return ParseResourceType(parser, loc);
    case TensorFlowTypes::VARIANT:
      return ParseVariantType(parser, loc);
  }
}

namespace {
template <typename TypeWithSubtype>
Type ParseTypeWithSubtype(MLIRContext *context, DialectAsmParser &parser,
                          Location loc) {
  // Default type without inferred subtypes.
  if (failed(parser.parseOptionalLess())) return TypeWithSubtype::get(context);

  // Most types with subtypes have only one subtype.
  SmallVector<TensorType, 1> subtypes;
  do {
    TensorType tensor_ty;
    if (parser.parseType(tensor_ty)) return Type();
    subtypes.push_back(tensor_ty);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseGreater()) return Type();
  return TypeWithSubtype::getChecked(subtypes, context, loc);
}

}  // anonymous namespace

Type TensorFlowDialect::ParseResourceType(DialectAsmParser &parser,
                                          Location loc) const {
  return ParseTypeWithSubtype<ResourceType>(getContext(), parser, loc);
}

Type TensorFlowDialect::ParseVariantType(DialectAsmParser &parser,
                                         Location loc) const {
  return ParseTypeWithSubtype<VariantType>(getContext(), parser, loc);
}

Operation *TensorFlowDialect::materializeConstant(OpBuilder &builder,
                                                  Attribute value, Type type,
                                                  Location loc) {
  return builder.create<ConstOp>(loc, type, value);
}

#define GET_OP_CLASSES
#include "tf_ops.cpp.inc"

// Builds a constant op with the specified attribute `value`. The result
// op's type is deduced from `value`; if `value` is of scalar type,
// wraps it up with a tensor type of empty shape.
// TODO(jpienaar): This one differs from the autogenerated one as it takes an
// attribute but always creates an ElementsAttr internally.
void ConstOp::build(OpBuilder &builder, OperationState &result,
                    Attribute value) {
  ShapedType type;
  if (auto elem_attr = value.dyn_cast<ElementsAttr>()) {
    return ConstOp::build(builder, result, elem_attr);
  } else if (value.isa<BoolAttr>() || value.isa<FloatAttr>() ||
             value.isa<IntegerAttr>()) {
    // All TensorFlow types must be tensor types. In the build() method,
    // we want to provide more flexibility by allowing attributes of scalar
    // types. But we need to wrap it up with ElementsAttr to construct
    // valid TensorFlow constants.
    type = RankedTensorType::get(/*shape=*/{}, value.getType());
    return ConstOp::build(builder, result, DenseElementsAttr::get(type, value));
  }
  // TODO(jpienaar): support other TensorFlow specific types.
  llvm_unreachable("unsupported attribute type for building tf.Const");
}

void ConstOp::build(OpBuilder &builder, OperationState &result, Type type,
                    Attribute value) {
  // Handle the case where the type and value are already tensors.
  if (type.isa<TensorType>() && value.isa<ElementsAttr>()) {
    result.addTypes(type);
    result.addAttribute("value", value);
    return;
  }

  // Otherwise, default to the attribute builder.
  ConstOp::build(builder, result, value);
  assert(type == result.types[0] && "type mismatch in construction");
}

LogicalResult ConstOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto value = attributes.get("value");
  if (!value) return emitOptionalError(location, "missing attribute 'value'");
  if (auto elem_attr = value.dyn_cast<ElementsAttr>()) {
    inferredReturnTypes.assign({elem_attr.getType()});
    return success();
  }
  return emitOptionalError(location,
                           "attribute 'value' failed to satisfy constraint: "
                           "constant vector/tensor");
}

}

}

static std::string get_mlir_value_uniq_id(const mlir::Value& value)
{
    if (value.getLoc().isa<mlir::FileLineColLoc>())
    {
        mlir::FileLineColLoc floc = value.getLoc().cast<mlir::FileLineColLoc>();

        return floc.getFilename().str() + ":" + std::to_string(floc.getLine()) + ":" + std::to_string(floc.getColumn());
    }

    fprintf(stderr, "unhandled get_mlir_value_uniq_id\n");
    return std::string();
}

static std::vector<int> get_operation_attr_ai(const mlir::Operation& _operation, const char* key)
{
    mlir::Operation& operation = const_cast<mlir::Operation&>(_operation);

    mlir::Attribute attribute = operation.getAttr(key);

    std::vector<int> v;

    if (attribute.isa<mlir::ArrayAttr>())
    {
        mlir::ArrayAttr a = attribute.cast<mlir::ArrayAttr>();

        const int array_size = a.getValue().size();

        v.resize(array_size);
        for (int j=0; j<array_size; j++)
        {
            if (a[j].isa<mlir::IntegerAttr>())
            {
                int64_t ii = a[j].cast<mlir::IntegerAttr>().getInt();
                v[j] = std::max(std::min(ii, (int64_t)INT_MAX), (int64_t)INT_MIN);
            }
        }
    }

    return v;
}

static std::vector<float> get_operation_attr_af(const mlir::Operation& _operation, const char* key)
{
    mlir::Operation& operation = const_cast<mlir::Operation&>(_operation);

    mlir::Attribute attribute = operation.getAttr(key);

    std::vector<float> v;

    if (attribute.isa<mlir::ArrayAttr>())
    {
        mlir::ArrayAttr a = attribute.cast<mlir::ArrayAttr>();

        const int array_size = a.getValue().size();

        v.resize(array_size);
        for (int j=0; j<array_size; j++)
        {
            if (a[j].isa<mlir::FloatAttr>())
            {
                double ff = a[j].cast<mlir::FloatAttr>().getValueAsDouble();
                v[j] = ff;
            }
        }
    }

    return v;
}

int main(int argc, char** argv)
{
    const char* mlirpath = argv[1];
    const char* ncnn_prototxt = argc >= 4 ? argv[2] : "ncnn.param";
    const char* ncnn_modelbin = argc >= 4 ? argv[3] : "ncnn.bin";

    mlir::registerDialect<mlir::StandardOpsDialect>();
    mlir::registerDialect<mlir::TF::TensorFlowDialect>();

    mlir::MLIRContext context;
    mlir::OwningModuleRef m = mlir::parseSourceFile(mlirpath, &context);

//     m->dump();

    mlir::FuncOp main_fn = m->lookupSymbol<mlir::FuncOp>("main");

    auto& bb = main_fn.getBlocks().front();

//     bb.dump();

    FILE* pp = fopen(ncnn_prototxt, "wb");
    FILE* bp = fopen(ncnn_modelbin, "wb");

    // node reference
    std::map<std::string, int> node_reference;

    // weight node and weight reshape node
    std::map<std::string, mlir::Attribute> weights;

    // weight node before BinaryOp
    std::map<std::string, mlir::Attribute> binaryop_weights;


    fprintf(pp, "7767517\n");

    const mlir::Block::OpListType& operations = bb.getOperations();

    int node_count = operations.size();

    // global definition line
    // [layer count] [blob count]
    std::set<std::string> blob_names;
    for (const mlir::Operation& _operation : operations)
    {
        mlir::Operation& operation = const_cast<mlir::Operation&>(_operation);

        std::string op = operation.getName().getStringRef().str();

        int num_input = (int)operation.getNumOperands();
        int num_output = (int)operation.getNumResults();

        if (op == "tf.Const")
        {
            // weight
            std::string output_name = get_mlir_value_uniq_id(operation.getResult(0));
            weights[output_name] = operation.getAttr("value");
            continue;
        }
        else
        {
            bool isBinaryOp = false;
            // TODO add more binaryop
            if (op == "tf.BiasAdd")
            {
                isBinaryOp = true;
            }

            if (isBinaryOp)
            {
                // check weights
                for (int j=0; j<num_input; j++)
                {
                    std::string input_name = get_mlir_value_uniq_id(operation.getOperand(j));

                    std::map<std::string, mlir::Attribute>::iterator it = weights.find(input_name);
                    if (it != weights.end())
                    {
                        // binary op with weight, insert MemoryData layer and const blob
                        binaryop_weights[input_name] = it->second;
                        weights.erase(it);
                    }
                }
            }
        }

        for (int j=0; j<num_input; j++)
        {
            std::string input_name = get_mlir_value_uniq_id(operation.getOperand(j));

            // check weight
            if (weights.find(input_name) != weights.end())
            {
                continue;
            }

            blob_names.insert(input_name);

            if (node_reference.find(input_name) == node_reference.end())
            {
                node_reference[input_name] = 1;
            }
            else
            {
                node_reference[input_name] = node_reference[input_name] + 1;
            }
        }

        for (int j=0; j<num_output; j++)
        {
            std::string output_name = get_mlir_value_uniq_id(operation.getResult(j));

            blob_names.insert(output_name);
        }

//         layer_count ++;
//         blob_count += num_output;
    }

    // remove node_reference entry with reference equals to one
    int splitncnn_blob_count = 0;
    std::map<std::string, int>::iterator it = node_reference.begin();
    while (it != node_reference.end())
    {
        if (it->second == 1)
        {
            node_reference.erase(it++);
        }
        else
        {
            splitncnn_blob_count += it->second;
//             fprintf(stderr, "%s %d\n", it->first.c_str(), it->second);
            ++it;
        }
    }

    fprintf(pp, "%lu %lu\n", node_count + node_reference.size() - weights.size(), blob_names.size() + splitncnn_blob_count);

    int internal_split = 0;

    // model op
    int g_opid = 0;

    for (const mlir::Operation& _operation : operations)
    {
        mlir::Operation& operation = const_cast<mlir::Operation&>(_operation);

        std::string op = operation.getName().getStringRef().str();

        int opid = g_opid++;

        int num_input = (int)operation.getNumOperands();
        int num_output = (int)operation.getNumResults();

        for (int i=0; i<num_input; i++)
        {
            std::string input_name = get_mlir_value_uniq_id(operation.getOperand(i));

            // check weight
            if (weights.find(input_name) != weights.end())
            {
                num_input--;
            }
        }

        if (op == "std.return")
        {
            fprintf(pp, "%-16s", "Noop");
        }
        else if (op == "tf.BiasAdd")
        {
            fprintf(pp, "%-16s", "BinaryOp");
        }
        else if (op == "tf.Const")
        {
            // check weight before BinaryOp
            std::string output_name = get_mlir_value_uniq_id(operation.getResult(0));
            if (binaryop_weights.find(output_name) != binaryop_weights.end())
            {
                fprintf(pp, "%-16s", "MemoryData");
            }
            else
            {
                continue;
            }
        }
        else if (op == "tf.Conv2D")
        {
            fprintf(pp, "%-16s", "Convolution");
        }
        else if (op == "tf.Identity")
        {
            fprintf(pp, "%-16s", "Noop");
        }
        else if (op == "tf.MatMul")
        {
            fprintf(pp, "%-16s", "InnerProduct");
        }
        else if (op == "tf.Placeholder")
        {
            fprintf(pp, "%-16s", "Input");
        }
        else if (op == "tf.Relu")
        {
            fprintf(pp, "%-16s", "ReLU");
        }
        else if (op == "tf.Reshape")
        {
            fprintf(pp, "%-16s", "Reshape");
        }
        else if (op == "tf.Softmax")
        {
            fprintf(pp, "%-16s", "Softmax");
        }
        else
        {
            fprintf(pp, "%-16s", op.c_str());
        }

        fprintf(pp, " op_%d %d %d", opid, num_input, num_output);

        for (int i=0; i<num_input; i++)
        {
            std::string input_name = get_mlir_value_uniq_id(operation.getOperand(i));

            // check weight
            if (weights.find(input_name) != weights.end())
            {
                continue;
            }

            if (node_reference.find(input_name) != node_reference.end())
            {
                int refidx = node_reference[input_name] - 1;
                node_reference[input_name] = refidx;

                char splitsuffix[256];
                sprintf(splitsuffix, "_splitncnn_%d", refidx);
                input_name = input_name + splitsuffix;
            }

            fprintf(pp, " %s", input_name.c_str());
        }

        for (int i=0; i<num_output; i++)
        {
            std::string output_name = get_mlir_value_uniq_id(operation.getResult(i));
            fprintf(pp, " %s", output_name.c_str());
        }


        if (op == "std.return")
        {
        }
        else if (op == "tf.Const")
        {
            // check weight before BinaryOp
            std::string output_name = get_mlir_value_uniq_id(operation.getResult(0));
            if (binaryop_weights.find(output_name) != binaryop_weights.end())
            {
                const mlir::Attribute& M = binaryop_weights[output_name];

                llvm::ArrayRef<int64_t> shape = M.getType().cast<mlir::RankedTensorType>().getShape();

                // c wc hwc
                if (shape.size() == 1)
                {
                    fprintf(pp, " 0=%d", (int)shape[0]);
                }
                else if (shape.size() == 2)
                {
                    fprintf(pp, " 0=%d", (int)shape[1]);
                    fprintf(pp, " 1=%d", (int)shape[0]);
                }
                else if (shape.size() == 3)
                {
                    fprintf(pp, " 0=%d", (int)shape[1]);
                    fprintf(pp, " 1=%d", (int)shape[0]);
                    fprintf(pp, " 2=%d", (int)shape[2]);
                }

                std::vector<float> v;
                if (M.isa<mlir::DenseFPElementsAttr>())
                {
                    mlir::DenseFPElementsAttr afp = M.cast<mlir::DenseFPElementsAttr>();

                    for (auto ff : afp.getFloatValues())
                    {
                        v.push_back(ff.convertToFloat());
                    }
                }

                if (shape.size() != 3)
                {
                    fwrite(v.data(), sizeof(float), v.size(), bp);
                }
                else
                {
                    int w = (int)shape[1];
                    int h = (int)shape[0];
                    int c = (int)shape[2];

                    float tmp;
                    // h-w-c to c-h-w
                    for (int p=0; p<c; p++)
                    {
                        for (int i=0; i<h; i++)
                        {
                            for (int j=0; j<w; j++)
                            {
                                tmp = v[i*w*c + j*c + p];
                                fwrite(&tmp, sizeof(float), 1, bp);
                            }
                        }
                    }
                }
            }
        }
        else if (op == "tf.Conv2D")
        {
            std::string weight_name = get_mlir_value_uniq_id(operation.getOperand(1));
            const mlir::Attribute& W = weights[weight_name];

            llvm::ArrayRef<int64_t> shape = W.getType().cast<mlir::RankedTensorType>().getShape();

//             assert(shape.size() == 4)

            // kh-kw-inch-outch
            int kernel_size_h = shape[0];
            int kernel_size_w = shape[1];
            int num_input = shape[2];
            int num_output = shape[3];
            int weight_data_size = shape[0] * shape[1] * shape[2] * shape[3];

            fprintf(pp, " 0=%d", num_output);
            fprintf(pp, " 1=%d", kernel_size_w);
            fprintf(pp, " 11=%d", kernel_size_h);
            fprintf(pp, " 6=%d", weight_data_size);

            std::vector<int> dilations = get_operation_attr_ai(operation, "dilations");
            std::vector<int> strides = get_operation_attr_ai(operation, "strides");

            if (dilations.size() == 4) {
                fprintf(pp, " 2=%d", dilations[2]);
                fprintf(pp, " 12=%d", dilations[1]);
            }

            if (strides.size() == 4) {
                fprintf(pp, " 3=%d", strides[2]);
                fprintf(pp, " 13=%d", strides[1]);
            }

            std::vector<float> v;
            v.reserve(weight_data_size);
            if (W.isa<mlir::DenseFPElementsAttr>())
            {
                mlir::DenseFPElementsAttr afp = W.cast<mlir::DenseFPElementsAttr>();

                for (auto ff : afp.getFloatValues())
                {
                    v.push_back(ff.convertToFloat());
                }
            }

            // reorder h-w-i-o to o-i-h-w
            {
                int quantize_tag = 0;
                fwrite(&quantize_tag, sizeof(int), 1, bp);

                float tmp;
                for (int p=0; p<num_output; p++)
                {
                    for (int q=0; q<num_input; q++)
                    {
                        for (int i=0; i<kernel_size_h; i++)
                        {
                            for (int j=0; j<kernel_size_w; j++)
                            {
                                tmp = v[i*kernel_size_w*num_input*num_output + j*num_input*num_output + q*num_output + p];
                                fwrite(&tmp, sizeof(float), 1, bp);
                            }
                        }
                    }
                }
            }
        }
        else if (op == "tf.Identity")
        {
        }
        else if (op == "tf.MatMul")
        {
            std::string weight_name = get_mlir_value_uniq_id(operation.getOperand(1));
            const mlir::Attribute& W = weights[weight_name];

            llvm::ArrayRef<int64_t> shape = W.getType().cast<mlir::RankedTensorType>().getShape();

//             assert(shape.size() == 2)

            // inch-outch
            int num_input = shape[0];
            int num_output = shape[1];
            int weight_data_size = shape[0] * shape[1];

            fprintf(pp, " 0=%d", num_output);
            fprintf(pp, " 2=%d", weight_data_size);

            std::vector<float> v;
            v.reserve(weight_data_size);
            if (W.isa<mlir::DenseFPElementsAttr>())
            {
                mlir::DenseFPElementsAttr afp = W.cast<mlir::DenseFPElementsAttr>();

                for (auto ff : afp.getFloatValues())
                {
                    v.push_back(ff.convertToFloat());
                }
            }

            // reorder i-o to o-i
            {
                int quantize_tag = 0;
                fwrite(&quantize_tag, sizeof(int), 1, bp);

                float tmp;
                for (int p=0; p<num_output; p++)
                {
                    for (int q=0; q<num_input; q++)
                    {
                        tmp = v[q*num_output + p];
                        fwrite(&tmp, sizeof(float), 1, bp);
                    }
                }
            }
        }
        else if (op == "tf.Placeholder")
        {
        }
        else if (op == "tf.Relu")
        {
        }
        else if (op == "tf.Reshape")
        {
            std::string weight_name = get_mlir_value_uniq_id(operation.getOperand(1));
            const mlir::Attribute& S = weights[weight_name];

            std::vector<int> v;
            if (S.isa<mlir::DenseIntElementsAttr>())
            {
                mlir::DenseIntElementsAttr ai = S.cast<mlir::DenseIntElementsAttr>();

                for (auto ii : ai.getIntValues())
                {
                    v.push_back(ii.getSExtValue());
                }
            }

            int size = v.size();

            // n h w c
            // n h w
            // n w
            if (size == 4)
            {
                fprintf(pp, " 0=%d 1=%d 2=%d", v[2], v[1], v[3]);
            }
            if (size == 3)
            {
                fprintf(pp, " 0=%d 1=%d 2=-233", v[2], v[1]);
            }
            if (size == 2)
            {
                fprintf(pp, " 0=%d 1=-233 2=-233", v[1]);
            }

        }
        else if (op == "tf.Softmax")
        {
        }

#if 0
        for (const mlir::NamedAttribute& attr : operation.getAttrs())
        {
            const mlir::Identifier& identifier = attr.first;
            const mlir::Attribute& attribute = attr.second;

            fprintf(pp, " %s=", identifier.c_str());

            if (attribute.isa<mlir::AffineMapAttr>())
            {
                fprintf(pp, "AffineMap");
            }
            if (attribute.isa<mlir::ArrayAttr>())
            {
//                 fprintf(pp, "Array");
                mlir::ArrayAttr a = attribute.cast<mlir::ArrayAttr>();
                int array_size = a.getValue().size();
                for (int t=0; t<array_size; t++)
                {
                    if (a[t].isa<mlir::IntegerAttr>())
                    {
                        int64_t ii = a[t].cast<mlir::IntegerAttr>().getInt();
                        fprintf(pp, "%lld,", ii);
                    }
                }
            }
            if (attribute.isa<mlir::BoolAttr>())
            {
//                 fprintf(pp, "Bool");
                mlir::BoolAttr a = attribute.cast<mlir::BoolAttr>();
                fprintf(pp, "%d", a.getValue() ? 1 : 0);
            }
            if (attribute.isa<mlir::DictionaryAttr>())
            {
                fprintf(pp, "Dictionary");
            }
            if (attribute.isa<mlir::FloatAttr>())
            {
                fprintf(pp, "Float");
            }
            if (attribute.isa<mlir::IntegerAttr>())
            {
                fprintf(pp, "Integer");
            }
            if (attribute.isa<mlir::IntegerSetAttr>())
            {
                fprintf(pp, "IntegerSet");
            }
            if (attribute.isa<mlir::OpaqueAttr>())
            {
                fprintf(pp, "Opaque");
            }
            if (attribute.isa<mlir::StringAttr>())
            {
//                 fprintf(pp, "String");
                mlir::StringAttr s = attribute.cast<mlir::StringAttr>();
                fprintf(pp, "%s", s.getValue().empty() ? "" : s.getValue().data());
            }
            if (attribute.isa<mlir::SymbolRefAttr>())
            {
                fprintf(pp, "SymbolRef");
            }
            if (attribute.isa<mlir::FlatSymbolRefAttr>())
            {
                fprintf(pp, "FlatSymbolRef");
            }
            if (attribute.isa<mlir::TypeAttr>())
            {
                fprintf(pp, "Type");
            }
            if (attribute.isa<mlir::UnitAttr>())
            {
                fprintf(pp, "Unit");
            }
            if (attribute.isa<mlir::ElementsAttr>())
            {
                fprintf(pp, "Elements");
            }
            if (attribute.isa<mlir::DenseElementsAttr>())
            {
                fprintf(pp, "DenseElements");
            }
            if (attribute.isa<mlir::DenseFPElementsAttr>())
            {
                fprintf(pp, "DenseFPElements");
            }
            if (attribute.isa<mlir::DenseIntElementsAttr>())
            {
                fprintf(pp, "DenseIntElements");
            }
            if (attribute.isa<mlir::OpaqueElementsAttr>())
            {
                fprintf(pp, "OpaqueElements");
            }
            if (attribute.isa<mlir::SparseElementsAttr>())
            {
                fprintf(pp, "SparseElements");
            }
            if (attribute.isa<mlir::SplatElementsAttr>())
            {
                fprintf(pp, "SplatElements");
            }

        }
#endif

        fprintf(pp, "\n");

        for (int j=0; j<num_output; j++)
        {
            std::string output_name = get_mlir_value_uniq_id(operation.getResult(j));
            if (node_reference.find(output_name) != node_reference.end())
            {
                int refcount = node_reference[output_name];
                if (refcount > 1)
                {
                    char splitname[256];
                    sprintf(splitname, "splitncnn_%d", internal_split);
                    fprintf(pp, "%-16s %-24s %d %d", "Split", splitname, 1, refcount);

                    fprintf(pp, " %s", output_name.c_str());

                    for (int k=0; k<refcount; k++)
                    {
                        fprintf(pp, " %s_splitncnn_%d", output_name.c_str(), k);
                    }
                    fprintf(pp, "\n");

                    internal_split++;
                }
            }
        }
    }

    fclose(pp);
    fclose(bp);

    return 0;
}
