//===-- WebAssemblyOptimizeReturned.cpp - Optimize "returned" attributes --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Optimize calls with "returned" attributes for WebAssembly.
///
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-optimize-returned"

namespace {
class OptimizeReturned final : public FunctionPass,
                               public InstVisitor<OptimizeReturned> {
  StringRef getPassName() const override {
    return "WebAssembly Optimize Returned";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    FunctionPass::getAnalysisUsage(AU);
  }

  bool runOnFunction(Function &F) override;

  DominatorTree *DT = nullptr;

public:
  static char ID;
  OptimizeReturned() : FunctionPass(ID) {}

  void visitCallSite(CallSite CS);
};
} // End anonymous namespace

char OptimizeReturned::ID = 0;
INITIALIZE_PASS(OptimizeReturned, DEBUG_TYPE,
                "Optimize calls with \"returned\" attributes for WebAssembly",
                false, false)

FunctionPass *llvm::createWebAssemblyOptimizeReturned() {
  return new OptimizeReturned();
}

void OptimizeReturned::visitCallSite(CallSite CS) {
  for (unsigned I = 0, E = CS.getNumArgOperands(); I < E; ++I)
    if (CS.paramHasAttr(I, Attribute::Returned)) {
      Instruction *Inst = CS.getInstruction();
      Value *Arg = CS.getArgOperand(I);
      // Ignore constants, globals, undef, etc.
      if (isa<Constant>(Arg))
        continue;
      // Like replaceDominatedUsesWith but using Instruction/Use dominance.
      Arg->replaceUsesWithIf(Inst,
                             [&](Use &U) { return DT->dominates(Inst, U); });
    }
}

bool OptimizeReturned::runOnFunction(Function &F) {
  LLVM_DEBUG(dbgs() << "********** Optimize returned Attributes **********\n"
                       "********** Function: "
                    << F.getName() << '\n');

  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  visit(F);
  return true;
}
