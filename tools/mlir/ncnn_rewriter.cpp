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
