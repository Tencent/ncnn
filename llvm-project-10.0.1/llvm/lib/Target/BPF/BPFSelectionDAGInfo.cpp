//===-- BPFSelectionDAGInfo.cpp - BPF SelectionDAG Info -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the BPFSelectionDAGInfo class.
//
//===----------------------------------------------------------------------===//

#include "BPFTargetMachine.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/DerivedTypes.h"
using namespace llvm;

#define DEBUG_TYPE "bpf-selectiondag-info"

SDValue BPFSelectionDAGInfo::EmitTargetCodeForMemcpy(
    SelectionDAG &DAG, const SDLoc &dl, SDValue Chain, SDValue Dst, SDValue Src,
    SDValue Size, unsigned Align, bool isVolatile, bool AlwaysInline,
    MachinePointerInfo DstPtrInfo, MachinePointerInfo SrcPtrInfo) const {
  // Requires the copy size to be a constant.
  ConstantSDNode *ConstantSize = dyn_cast<ConstantSDNode>(Size);
  if (!ConstantSize)
    return SDValue();

  unsigned CopyLen = ConstantSize->getZExtValue();
  unsigned StoresNumEstimate = alignTo(CopyLen, Align) >> Log2_32(Align);
  // Impose the same copy length limit as MaxStoresPerMemcpy.
  if (StoresNumEstimate > getCommonMaxStoresPerMemFunc())
    return SDValue();

  SDVTList VTs = DAG.getVTList(MVT::Other, MVT::Glue);

  Dst = DAG.getNode(BPFISD::MEMCPY, dl, VTs, Chain, Dst, Src,
                    DAG.getConstant(CopyLen, dl, MVT::i64),
                    DAG.getConstant(Align, dl, MVT::i64));

  return Dst.getValue(0);
}
