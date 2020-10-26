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

#include <mlir/IR/Matchers.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>

#include "tf_dialect.h"
#include "ncnn_dialect.h"

using namespace mlir;

#include "ncnn_rewriter.inc"

namespace mlir {

namespace ncnn {

void BinaryOpOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context)
{
    results.insert<FuseBinaryOpPattern0>(context);
    results.insert<FuseBinaryOpPattern1>(context);
}

void KerasConv2DOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context)
{
    results.insert<FuseKerasConv2DOpPattern>(context);
}

void KerasDenseOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context)
{
    results.insert<FuseKerasDenseOpPattern>(context);
}

void KerasBatchNormOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context)
{
    results.insert<FuseKerasBatchNormOpPattern>(context);
}

void InstanceNormOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context)
{
    results.insert<FuseInstanceNormPattern0>(context);
    results.insert<FuseInstanceNormPattern1>(context);
}

void InstanceNormAffineOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context)
{
    results.insert<FuseInstanceNormAffinePattern>(context);
}

} // namespace ncnn

} // namespace mlir
