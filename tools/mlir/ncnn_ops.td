// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_OPS_TD
#define NCNN_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "tf_op_base.td"

def NCNN_Dialect : Dialect {
  let name = "ncnn";
  let cppNamespace = "ncnn";
}

//===----------------------------------------------------------------------===//
// NCNN op definitions
//===----------------------------------------------------------------------===//

class NCNN_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<NCNN_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// NCNN operations
//===----------------------------------------------------------------------===//

def NCNN_KerasConv2DOp : NCNN_Op<"KerasConv2D", [NoSideEffect]> {

  let arguments = (ins
    F32Tensor:$x,
    F32Tensor:$weight,
    F32Tensor:$bias,

    I64ArrayAttr:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID", "EXPLICIT"]>:$padding,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$explicit_paddings,
    DefaultValuedAttr<I64ArrayAttr, "{1, 1, 1, 1}">:$dilations
  );

  let results = (outs
    F32Tensor:$y
  );
}

def NCNN_KerasDenseOp : NCNN_Op<"KerasDense", [NoSideEffect]> {

  let arguments = (ins
    F32Tensor:$x,
    F32Tensor:$weight,
    F32Tensor:$bias
  );

  let results = (outs
    F32Tensor:$y
  );
}

def NCNN_KerasBatchNormOp : NCNN_Op<"KerasBatchNorm", [NoSideEffect]> {

  let arguments = (ins
    F32Tensor:$x,
    F32Tensor:$gamma,
    F32Tensor:$bias
  );

  let results = (outs
    F32Tensor:$y
  );
}

def NCNN_BinaryOpOp : NCNN_Op<"BinaryOp", [NoSideEffect]> {

  let arguments = (ins
    F32Tensor:$x,
    I32Attr:$op_type,
    I32Attr:$with_scalar,
    F32Attr:$b
  );

  let results = (outs
    F32Tensor:$y
  );
}

def NCNN_InstanceNormOp : NCNN_Op<"InstanceNorm", [NoSideEffect]> {

  let arguments = (ins
    F32Tensor:$x,
    F32Attr:$epsilon
  );

  let results = (outs
    F32Tensor:$y
  );
}

def NCNN_InstanceNormAffineOp : NCNN_Op<"InstanceNormAffine", [NoSideEffect]> {

  let arguments = (ins
    F32Tensor:$x,
    F32Tensor:$gamma,
    F32Tensor:$beta,
    F32Attr:$epsilon
  );

  let results = (outs
    F32Tensor:$y
  );
}

def NCNN_SwishOp : NCNN_Op<"Swish", [NoSideEffect]> {

  let arguments = (ins
    F32Tensor:$x
  );

  let results = (outs
    F32Tensor:$y
  );
}

#endif // NCNN_OPS_TD
