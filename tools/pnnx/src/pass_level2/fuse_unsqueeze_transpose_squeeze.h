// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_PASS_LEVEL2_FUSE_UNSQUEEZE_TRANSPOSE_SQUEEZE_H
#define PNNX_PASS_LEVEL2_FUSE_UNSQUEEZE_TRANSPOSE_SQUEEZE_H

namespace pnnx {

class Graph;

void fuse_unsqueeze_transpose_squeeze(Graph& g);

} // namespace pnnx

#endif // PNNX_PASS_LEVEL2_FUSE_UNSQUEEZE_TRANSPOSE_SQUEEZE_H
