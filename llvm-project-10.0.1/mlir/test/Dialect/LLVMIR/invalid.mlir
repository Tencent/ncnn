// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// expected-error@+1{{llvm.noalias argument attribute of non boolean type}}
func @invalid_noalias(%arg0: !llvm.i32 {llvm.noalias = 3}) {
  "llvm.return"() : () -> ()
}

////////////////////////////////////////////////////////////////////////////////

// Check that parser errors are properly produced and do not crash the compiler.

// -----

func @icmp_non_string(%arg0 : !llvm.i32, %arg1 : !llvm<"i16">) {
  // expected-error@+1 {{expected 'predicate' attribute of string type}}
  llvm.icmp 42 %arg0, %arg0 : !llvm.i32
  return
}

// -----

func @icmp_wrong_string(%arg0 : !llvm.i32, %arg1 : !llvm<"i16">) {
  // expected-error@+1 {{'foo' is an incorrect value of the 'predicate' attribute}}
  llvm.icmp "foo" %arg0, %arg0 : !llvm.i32
  return
}

// -----

func @alloca_missing_input_result_type(%size : !llvm.i64) {
  // expected-error@+1 {{expected trailing function type with one argument and one result}}
  llvm.alloca %size x !llvm.i32 : () -> ()
}

// -----

func @alloca_missing_input_type() {
  // expected-error@+1 {{expected trailing function type with one argument and one result}}
  llvm.alloca %size x !llvm.i32 : () -> (!llvm<"i32*">)
}

// -----

func @alloca_mising_result_type() {
  // expected-error@+1 {{expected trailing function type with one argument and one result}}
  llvm.alloca %size x !llvm.i32 : (!llvm.i64) -> ()
}

// -----

func @alloca_non_function_type() {
  // expected-error@+1 {{expected trailing function type with one argument and one result}}
  llvm.alloca %size x !llvm.i32 : !llvm<"i32*">
}

// -----

func @alloca_nonpositive_alignment(%size : !llvm.i64) {
  // expected-error@+1 {{expected positive alignment}}
  llvm.alloca %size x !llvm.i32 {alignment = -1} : (!llvm.i64) -> (!llvm<"i32*">)
}

// -----

func @gep_missing_input_result_type(%pos : !llvm.i64, %base : !llvm<"float*">) {
  // expected-error@+1 {{expected trailing function type with at least one argument and one result}}
  llvm.getelementptr %base[%pos] : () -> ()
}

// -----

func @gep_missing_input_type(%pos : !llvm.i64, %base : !llvm<"float*">) {
  // expected-error@+1 {{expected trailing function type with at least one argument and one result}}
  llvm.getelementptr %base[%pos] : () -> (!llvm<"float*">)
}

// -----

func @gep_missing_result_type(%pos : !llvm.i64, %base : !llvm<"float*">) {
  // expected-error@+1 {{expected trailing function type with at least one argument and one result}}
  llvm.getelementptr %base[%pos] : (!llvm<"float *">, !llvm.i64) -> ()
}

// -----

func @gep_non_function_type(%pos : !llvm.i64, %base : !llvm<"float*">) {
  // expected-error@+1 {{expected trailing function type with at least one argument and one result}}
  llvm.getelementptr %base[%pos] : !llvm<"float*">
}

// -----

func @load_non_llvm_type(%foo : memref<f32>) {
  // expected-error@+1 {{expected LLVM IR dialect type}}
  llvm.load %foo : memref<f32>
}

// -----

func @load_non_ptr_type(%foo : !llvm.float) {
  // expected-error@+1 {{expected LLVM pointer type}}
  llvm.load %foo : !llvm.float
}

// -----

func @store_non_llvm_type(%foo : memref<f32>, %bar : !llvm.float) {
  // expected-error@+1 {{expected LLVM IR dialect type}}
  llvm.store %bar, %foo : memref<f32>
}

// -----

func @store_non_ptr_type(%foo : !llvm.float, %bar : !llvm.float) {
  // expected-error@+1 {{expected LLVM pointer type}}
  llvm.store %bar, %foo : !llvm.float
}

// -----

func @call_non_function_type(%callee : !llvm<"i8(i8)">, %arg : !llvm<"i8">) {
  // expected-error@+1 {{expected function type}}
  llvm.call %callee(%arg) : !llvm<"i8(i8)">
}

// -----

func @call_too_many_results(%callee : () -> (i32,i32)) {
  // expected-error@+1 {{expected function with 0 or 1 result}}
  llvm.call %callee() : () -> (i32, i32)
}

// -----

func @call_non_llvm_result(%callee : () -> (i32)) {
  // expected-error@+1 {{expected result to have LLVM type}}
  llvm.call %callee() : () -> (i32)
}

// -----

func @call_non_llvm_input(%callee : (i32) -> (), %arg : i32) {
  // expected-error@+1 {{expected LLVM types as inputs}}
  llvm.call %callee(%arg) : (i32) -> ()
}

// -----

func @insertvalue_non_llvm_type(%a : i32, %b : i32) {
  // expected-error@+1 {{expected LLVM IR Dialect type}}
  llvm.insertvalue %a, %b[0] : i32
}

// -----

func @insertvalue_non_array_position() {
  // Note the double-type, otherwise attribute parsing consumes the trailing
  // type of the op as the (wrong) attribute type.
  // expected-error@+1 {{expected an array attribute}}
  llvm.insertvalue %a, %b 0 : i32 : !llvm<"{i32}">
}

// -----

func @insertvalue_non_integer_position() {
  // expected-error@+1 {{expected an array of integer literals}}
  llvm.insertvalue %a, %b[0.0] : !llvm<"{i32}">
}

// -----

func @insertvalue_struct_out_of_bounds() {
  // expected-error@+1 {{position out of bounds}}
  llvm.insertvalue %a, %b[1] : !llvm<"{i32}">
}

// -----

func @insertvalue_array_out_of_bounds() {
  // expected-error@+1 {{position out of bounds}}
  llvm.insertvalue %a, %b[1] : !llvm<"[1 x i32]">
}

// -----

func @insertvalue_wrong_nesting() {
  // expected-error@+1 {{expected wrapped LLVM IR structure/array type}}
  llvm.insertvalue %a, %b[0,0] : !llvm<"{i32}">
}

// -----

func @extractvalue_non_llvm_type(%a : i32, %b : i32) {
  // expected-error@+1 {{expected LLVM IR Dialect type}}
  llvm.extractvalue %b[0] : i32
}

// -----

func @extractvalue_non_array_position() {
  // Note the double-type, otherwise attribute parsing consumes the trailing
  // type of the op as the (wrong) attribute type.
  // expected-error@+1 {{expected an array attribute}}
  llvm.extractvalue %b 0 : i32 : !llvm<"{i32}">
}

// -----

func @extractvalue_non_integer_position() {
  // expected-error@+1 {{expected an array of integer literals}}
  llvm.extractvalue %b[0.0] : !llvm<"{i32}">
}

// -----

func @extractvalue_struct_out_of_bounds() {
  // expected-error@+1 {{position out of bounds}}
  llvm.extractvalue %b[1] : !llvm<"{i32}">
}

// -----

func @extractvalue_array_out_of_bounds() {
  // expected-error@+1 {{position out of bounds}}
  llvm.extractvalue %b[1] : !llvm<"[1 x i32]">
}

// -----

func @extractvalue_wrong_nesting() {
  // expected-error@+1 {{expected wrapped LLVM IR structure/array type}}
  llvm.extractvalue %b[0,0] : !llvm<"{i32}">
}

// -----

// CHECK-LABEL: @invalid_vector_type_1
func @invalid_vector_type_1(%arg0: !llvm<"<4 x float>">, %arg1: !llvm.i32, %arg2: !llvm.float) {
  // expected-error@+1 {{expected LLVM IR dialect vector type for operand #1}}
  %0 = llvm.extractelement %arg2[%arg1 : !llvm.i32] : !llvm.float
}

// -----

// CHECK-LABEL: @invalid_vector_type_2
func @invalid_vector_type_2(%arg0: !llvm<"<4 x float>">, %arg1: !llvm.i32, %arg2: !llvm.float) {
  // expected-error@+1 {{expected LLVM IR dialect vector type for operand #1}}
  %0 = llvm.insertelement %arg2, %arg2[%arg1 : !llvm.i32] : !llvm.float
}

// -----

// CHECK-LABEL: @invalid_vector_type_3
func @invalid_vector_type_3(%arg0: !llvm<"<4 x float>">, %arg1: !llvm.i32, %arg2: !llvm.float) {
  // expected-error@+1 {{expected LLVM IR dialect vector type for operand #1}}
  %0 = llvm.shufflevector %arg2, %arg2 [0 : i32, 0 : i32, 0 : i32, 0 : i32, 7 : i32] : !llvm.float, !llvm.float
}

// -----

func @null_non_llvm_type() {
  // expected-error@+1 {{expected LLVM IR pointer type}}
  llvm.mlir.null : !llvm.i32
}

// -----

// CHECK-LABEL: @nvvm_invalid_shfl_pred_1
func @nvvm_invalid_shfl_pred_1(%arg0 : !llvm.i32, %arg1 : !llvm.i32, %arg2 : !llvm.i32, %arg3 : !llvm.i32) {
  // expected-error@+1 {{expected return type !llvm<"{ ?, i1 }">}}
  %0 = nvvm.shfl.sync.bfly %arg0, %arg3, %arg1, %arg2 {return_value_and_is_valid} : !llvm.i32
}

// -----

// CHECK-LABEL: @nvvm_invalid_shfl_pred_2
func @nvvm_invalid_shfl_pred_2(%arg0 : !llvm.i32, %arg1 : !llvm.i32, %arg2 : !llvm.i32, %arg3 : !llvm.i32) {
  // expected-error@+1 {{expected return type !llvm<"{ ?, i1 }">}}
  %0 = nvvm.shfl.sync.bfly %arg0, %arg3, %arg1, %arg2 {return_value_and_is_valid} : !llvm<"{ i32 }">
}

// -----

// CHECK-LABEL: @nvvm_invalid_shfl_pred_3
func @nvvm_invalid_shfl_pred_3(%arg0 : !llvm.i32, %arg1 : !llvm.i32, %arg2 : !llvm.i32, %arg3 : !llvm.i32) {
  // expected-error@+1 {{expected return type !llvm<"{ ?, i1 }">}}
  %0 = nvvm.shfl.sync.bfly %arg0, %arg3, %arg1, %arg2 {return_value_and_is_valid} : !llvm<"{ i32, i32 }">
}

// -----

// CHECK-LABEL: @nvvm_invalid_mma_0
func @nvvm_invalid_mma_0(%a0 : !llvm.half, %a1 : !llvm<"<2 x half>">,
                         %b0 : !llvm<"<2 x half>">, %b1 : !llvm<"<2 x half>">,
                         %c0 : !llvm.float, %c1 : !llvm.float, %c2 : !llvm.float, %c3 : !llvm.float,
                         %c4 : !llvm.float, %c5 : !llvm.float, %c6 : !llvm.float, %c7 : !llvm.float) {
  // expected-error@+1 {{expected operands to be 4 <halfx2>s followed by either 4 <halfx2>s or 8 floats}}
  %0 = nvvm.mma.sync %a0, %a1, %b0, %b1, %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7 {alayout="row", blayout="row"} : (!llvm.half, !llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float) -> !llvm<"{ float, float, float, float, float, float, float, float }">
  llvm.return %0 : !llvm<"{ float, float, float, float, float, float, float, float }">
}

// -----

// CHECK-LABEL: @nvvm_invalid_mma_1
func @nvvm_invalid_mma_1(%a0 : !llvm<"<2 x half>">, %a1 : !llvm<"<2 x half>">,
                         %b0 : !llvm<"<2 x half>">, %b1 : !llvm<"<2 x half>">,
                         %c0 : !llvm.float, %c1 : !llvm.float, %c2 : !llvm.float, %c3 : !llvm.float,
                         %c4 : !llvm.float, %c5 : !llvm.float, %c6 : !llvm.float, %c7 : !llvm.float) {
  // expected-error@+1 {{expected result type to be a struct of either 4 <halfx2>s or 8 floats}}
  %0 = nvvm.mma.sync %a0, %a1, %b0, %b1, %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7 {alayout="row", blayout="row"} : (!llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float) -> !llvm<"{ float, float, float, float, float, float, float, half }">
  llvm.return %0 : !llvm<"{ float, float, float, float, float, float, float, half }">
}

// -----

// CHECK-LABEL: @nvvm_invalid_mma_2
func @nvvm_invalid_mma_2(%a0 : !llvm<"<2 x half>">, %a1 : !llvm<"<2 x half>">,
                         %b0 : !llvm<"<2 x half>">, %b1 : !llvm<"<2 x half>">,
                         %c0 : !llvm.float, %c1 : !llvm.float, %c2 : !llvm.float, %c3 : !llvm.float,
                         %c4 : !llvm.float, %c5 : !llvm.float, %c6 : !llvm.float, %c7 : !llvm.float) {
  // expected-error@+1 {{alayout and blayout attributes must be set to either "row" or "col"}}
  %0 = nvvm.mma.sync %a0, %a1, %b0, %b1, %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7 : (!llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float) -> !llvm<"{ float, float, float, float, float, float, float, float }">
  llvm.return %0 : !llvm<"{ float, float, float, float, float, float, float, float }">
}

// -----

// CHECK-LABEL: @nvvm_invalid_mma_3
func @nvvm_invalid_mma_3(%a0 : !llvm<"<2 x half>">, %a1 : !llvm<"<2 x half>">,
                         %b0 : !llvm<"<2 x half>">, %b1 : !llvm<"<2 x half>">,
                         %c0 : !llvm<"<2 x half>">, %c1 : !llvm<"<2 x half>">,
                         %c2 : !llvm<"<2 x half>">, %c3 : !llvm<"<2 x half>">) {
  // expected-error@+1 {{unimplemented mma.sync variant}}
  %0 = nvvm.mma.sync %a0, %a1, %b0, %b1, %c0, %c1, %c2, %c3 {alayout="row", blayout="row"} : (!llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm<"<2 x half>">) -> !llvm<"{ float, float, float, float, float, float, float, float }">
  llvm.return %0 : !llvm<"{ float, float, float, float, float, float, float, float }">
}

// -----

// CHECK-LABEL: @nvvm_invalid_mma_4
func @nvvm_invalid_mma_4(%a0 : !llvm<"<2 x half>">, %a1 : !llvm<"<2 x half>">,
                         %b0 : !llvm<"<2 x half>">, %b1 : !llvm<"<2 x half>">,
                         %c0 : !llvm.float, %c1 : !llvm.float, %c2 : !llvm.float, %c3 : !llvm.float,
                         %c4 : !llvm.float, %c5 : !llvm.float, %c6 : !llvm.float, %c7 : !llvm.float) {
  // expected-error@+1 {{unimplemented mma.sync variant}}
  %0 = nvvm.mma.sync %a0, %a1, %b0, %b1, %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7 {alayout="row", blayout="row"} : (!llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float) -> !llvm<"{<2 x half>, <2 x half>, <2 x half>, <2 x half>}">
  llvm.return %0 : !llvm<"{<2 x half>, <2 x half>, <2 x half>, <2 x half>}">
}

// -----

// CHECK-LABEL: @nvvm_invalid_mma_5
func @nvvm_invalid_mma_5(%a0 : !llvm<"<2 x half>">, %a1 : !llvm<"<2 x half>">,
                         %b0 : !llvm<"<2 x half>">, %b1 : !llvm<"<2 x half>">,
                         %c0 : !llvm.float, %c1 : !llvm.float, %c2 : !llvm.float, %c3 : !llvm.float,
                         %c4 : !llvm.float, %c5 : !llvm.float, %c6 : !llvm.float, %c7 : !llvm.float) {
  // expected-error@+1 {{unimplemented mma.sync variant}}
  %0 = nvvm.mma.sync %a0, %a1, %b0, %b1, %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7 {alayout="col", blayout="row"} : (!llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float) -> !llvm<"{ float, float, float, float, float, float, float, float }">
  llvm.return %0 : !llvm<"{ float, float, float, float, float, float, float, float }">
}

// -----

// CHECK-LABEL: @nvvm_invalid_mma_6
func @nvvm_invalid_mma_6(%a0 : !llvm<"<2 x half>">, %a1 : !llvm<"<2 x half>">,
                         %b0 : !llvm<"<2 x half>">, %b1 : !llvm<"<2 x half>">,
                         %c0 : !llvm.float, %c1 : !llvm.float, %c2 : !llvm.float, %c3 : !llvm.float,
                         %c4 : !llvm.float, %c5 : !llvm.float, %c6 : !llvm.float, %c7 : !llvm.float) {
  // expected-error@+1 {{expected the type to be the full list of input and output}}
  %0 = nvvm.mma.sync %a0, %a1, %b0, %b1, %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7 {alayout="col", blayout="row"} : !llvm<"{ float, float, float, float, float, float, float, float }">
  llvm.return %0 : !llvm<"{ float, float, float, float, float, float, float, float }">
}

// -----

// CHECK-LABEL: @nvvm_invalid_mma_7
func @nvvm_invalid_mma_7(%a0 : !llvm<"<2 x half>">, %a1 : !llvm<"<2 x half>">,
                         %b0 : !llvm<"<2 x half>">, %b1 : !llvm<"<2 x half>">,
                         %c0 : !llvm.float, %c1 : !llvm.float, %c2 : !llvm.float, %c3 : !llvm.float,
                         %c4 : !llvm.float, %c5 : !llvm.float, %c6 : !llvm.float, %c7 : !llvm.float) {
  // expected-error@+1 {{expected single result}}
  %0 = nvvm.mma.sync %a0, %a1, %b0, %b1, %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7 {alayout="col", blayout="row"} : (!llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float) -> (!llvm<"{ float, float, float, float, float, float, float, float }">, !llvm.i32)
  llvm.return %0 : (!llvm<"{ float, float, float, float, float, float, float, float }">, !llvm.i32)
}
