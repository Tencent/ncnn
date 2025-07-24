// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "mat.h"

namespace ncnn {

// count how many blobs are referenced inside expression
NCNN_EXPORT int count_expression_blobs(const std::string& expr);

// resolve reshape shape from expression and input blobs
// resolve slice indices(starts, ends) from expression and input blobs
// see docs/developer-guide/expression.md
// return 0 if success
NCNN_EXPORT int eval_list_expression(const std::string& expr, const std::vector<Mat>& blobs, std::vector<int>& outlist);

} // namespace ncnn
