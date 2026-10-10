// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level0.h"

#include "pass_level0/constant_unpooling.h"
#include "pass_level0/flatten_input.h"
#include "pass_level0/inline_block.h"
#include "pass_level0/reset_device.h"
#include "pass_level0/shape_inference.h"

namespace pnnx {

void pass_level0(const torch::jit::Module& mod, std::shared_ptr<torch::jit::Graph>& g, const std::vector<at::Tensor>& input_tensors, const std::vector<at::Tensor>& input_tensors2, const std::vector<std::string>& module_operators, const std::string& ptpath, const std::string& device, std::set<std::string>& foldable_constants, const std::string& foldable_constants_zippath)
{
    inline_block(g, module_operators);

    constant_unpooling(g);

    reset_device(g, device);

    flatten_input(g);

    if (!input_tensors.empty())
    {
        shape_inference(mod, g, input_tensors, input_tensors2, module_operators, ptpath, device, foldable_constants, foldable_constants_zippath);
    }
}

} // namespace pnnx
