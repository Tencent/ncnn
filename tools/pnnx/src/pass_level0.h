// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef PNNX_PASS_LEVEL0_H
#define PNNX_PASS_LEVEL0_H

#include <torch/script.h>
#include "ir.h"

namespace pnnx {

void pass_level0(const torch::jit::Module& mod, std::shared_ptr<torch::jit::Graph>& g, const std::vector<at::Tensor>& input_tensors, const std::vector<at::Tensor>& input_tensors2, const std::vector<std::string>& module_operators, const std::string& ptpath, std::map<std::string, Attribute>& foldable_constants);

} // namespace pnnx

#endif // PNNX_PASS_LEVEL0_H
