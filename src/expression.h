// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2025 THL A29 Limited, a Tencent company. All rights reserved.
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
