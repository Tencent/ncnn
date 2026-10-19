// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "onnx-ml.pb.h"

namespace pnnx {

namespace onnx2pnnx {

// Lower static SequenceConstruct/SplitToSequence regions only when every consumer
// is a constant-index SequenceAt or ConcatFromSequence. Constant tensor data and
// external indices/sizes are unsupported. Disable splitting before shape inference.
// Return whether the graph was changed.
bool eliminate_sequences(onnx::ModelProto& model, bool allow_split = true);

} // namespace onnx2pnnx

} // namespace pnnx
