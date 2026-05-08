// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "tf_dialect.h"
#include "ncnn_dialect.h"

using namespace mlir;

namespace mlir {

namespace ncnn {

#include "ncnn_rewriter.inc"

class NCNNOptimizePass : public PassWrapper<NCNNOptimizePass, FunctionPass>
{
public:
    void runOnFunction();
};

void NCNNOptimizePass::runOnFunction()
{
    mlir::OwningRewritePatternList patterns;
    mlir::ncnn::populateWithGenerated(&getContext(), patterns);

    (void)mlir::applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
}

std::unique_ptr<OperationPass<FuncOp> > createNCNNOptimizePass()
{
    return std::make_unique<NCNNOptimizePass>();
}

static PassRegistration<NCNNOptimizePass> pass("ncnn-optimize", "ncnn optimization");

} // namespace ncnn

} // namespace mlir
