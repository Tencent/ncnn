//===-- PPCMachineFunctionInfo.cpp - Private data used for PowerPC --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PPCMachineFunctionInfo.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/MCContext.h"

using namespace llvm;

void PPCFunctionInfo::anchor() {}

MCSymbol *PPCFunctionInfo::getPICOffsetSymbol() const {
  const DataLayout &DL = MF.getDataLayout();
  return MF.getContext().getOrCreateSymbol(Twine(DL.getPrivateGlobalPrefix()) +
                                           Twine(MF.getFunctionNumber()) +
                                           "$poff");
}

MCSymbol *PPCFunctionInfo::getGlobalEPSymbol() const {
  const DataLayout &DL = MF.getDataLayout();
  return MF.getContext().getOrCreateSymbol(Twine(DL.getPrivateGlobalPrefix()) +
                                           "func_gep" +
                                           Twine(MF.getFunctionNumber()));
}

MCSymbol *PPCFunctionInfo::getLocalEPSymbol() const {
  const DataLayout &DL = MF.getDataLayout();
  return MF.getContext().getOrCreateSymbol(Twine(DL.getPrivateGlobalPrefix()) +
                                           "func_lep" +
                                           Twine(MF.getFunctionNumber()));
}

MCSymbol *PPCFunctionInfo::getTOCOffsetSymbol() const {
  const DataLayout &DL = MF.getDataLayout();
  return MF.getContext().getOrCreateSymbol(Twine(DL.getPrivateGlobalPrefix()) +
                                           "func_toc" +
                                           Twine(MF.getFunctionNumber()));
}

bool PPCFunctionInfo::isLiveInSExt(unsigned VReg) const {
  for (const std::pair<unsigned, ISD::ArgFlagsTy> &LiveIn : LiveInAttrs)
    if (LiveIn.first == VReg)
      return LiveIn.second.isSExt();
  return false;
}

bool PPCFunctionInfo::isLiveInZExt(unsigned VReg) const {
  for (const std::pair<unsigned, ISD::ArgFlagsTy> &LiveIn : LiveInAttrs)
    if (LiveIn.first == VReg)
      return LiveIn.second.isZExt();
  return false;
}
