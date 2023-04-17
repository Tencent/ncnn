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

#include "inline_block.h"
#include "../pass_level1.h"

#include <set>

#include <torch/csrc/jit/passes/quantization/helper.h>
#include <torch/csrc/api/include/torch/version.h>

namespace pnnx {

static void inlineCallTo(torch::jit::Node* to_replace, torch::jit::Function* callee)
{
    torch::jit::WithInsertPoint guard(to_replace);

    std::unordered_map<torch::jit::Value*, torch::jit::Value*> value_map;
#if TORCH_VERSION_MAJOR >= 2 || (TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 11)
    std::vector<torch::jit::Value*> new_outputs = torch::jit::insertGraph(*to_replace->owningGraph(), *(toGraphFunction(*callee).graph()), to_replace->inputs(), value_map);
#else
    std::vector<torch::jit::Value*> new_outputs = torch::jit::insertGraph(*to_replace->owningGraph(), *(callee->graph()), to_replace->inputs(), value_map);
#endif

    const auto& old_outputs = to_replace->outputs();
    for (size_t i = 0; i < old_outputs.size(); ++i)
    {
        new_outputs[i]->copyMetadata(old_outputs[i]);

        old_outputs[i]->replaceAllUsesWith(new_outputs[i]);
    }

    to_replace->destroy();
}

static void inlineCalls(torch::jit::Block* block, const std::vector<std::string>& module_operators, std::set<std::string>& inlined_modules, bool inside_module_op = false)
{
    for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;)
    {
        torch::jit::Node* n = *it++;
        if (n->kind() == c10::prim::CallFunction)
        {
            auto function_constant = n->input(0)->node();
            auto fun_type = function_constant->output()->type()->expect<torch::jit::FunctionType>();
            if (!fun_type->function()->isGraphFunction())
                continue;

#if TORCH_VERSION_MAJOR >= 2 || (TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 11)
            inlineCalls(toGraphFunction(*(fun_type->function())).graph()->block(), module_operators, inlined_modules, inside_module_op);
#else
            inlineCalls(fun_type->function()->graph()->block(), module_operators, inlined_modules, inside_module_op);
#endif

            n->removeInput(0);

            fprintf(stderr, "inline function %s\n", fun_type->function()->name().c_str());

            pnnx::inlineCallTo(n, fun_type->function());
        }
        else if (n->kind() == c10::prim::CallMethod)
        {
            auto class_type = n->input(0)->type()->cast<torch::jit::ClassType>();
            if (!class_type)
                continue;

            const std::string& function_name = n->s(torch::jit::attr::name);
            torch::jit::Function& function = class_type->getMethod(function_name);
            if (!function.isGraphFunction())
                continue;

            std::string class_type_str = torch::jit::removeTorchMangle(class_type->str());

            std::string class_type_str_no_torch_prefix = class_type_str.substr(10);

            if (!inside_module_op)
            {
                if (std::find(module_operators.begin(), module_operators.end(), class_type_str_no_torch_prefix) != module_operators.end())
                {
#if TORCH_VERSION_MAJOR >= 2 || (TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 11)
                    inlineCalls(toGraphFunction(function).graph()->block(), module_operators, inlined_modules, true);
#else
                    inlineCalls(function.graph()->block(), module_operators, inlined_modules, true);
#endif

                    continue;
                }

                bool skip_inline = false;
                for (const auto& ow : get_global_pnnx_fuse_module_passes())
                {
                    if (class_type_str == ow->match_type_str())
                    {
                        skip_inline = true;
                        break;
                    }
                }

                if (skip_inline)
                    continue;
            }

#if TORCH_VERSION_MAJOR >= 2 || (TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 11)
            inlineCalls(toGraphFunction(function).graph()->block(), module_operators, inlined_modules, inside_module_op);
#else
            inlineCalls(function.graph()->block(), module_operators, inlined_modules, inside_module_op);
#endif

            inlined_modules.insert(class_type_str_no_torch_prefix);

            //             fprintf(stderr, "inline %s\n", class_type_str_no_torch_prefix.c_str());
            //             fprintf(stderr, "inline method %s   %s   %s\n", function.name().c_str(), class_type->str().c_str(), n->input(0)->node()->s(torch::jit::attr::name).c_str());

            pnnx::inlineCallTo(n, &function);
        }
        else
        {
            for (auto b : n->blocks())
            {
                inlineCalls(b, module_operators, inlined_modules, inside_module_op);
            }
        }
    }
}

void inline_block(std::shared_ptr<torch::jit::Graph>& graph, const std::vector<std::string>& module_operators)
{
    std::set<std::string> inlined_modules;

    inlineCalls(graph->block(), module_operators, inlined_modules);

    for (const auto& x : inlined_modules)
    {
        if (x == "torch.nn.modules.container.Sequential")
            continue;

        fprintf(stderr, "inline module = %s\n", x.c_str());
    }
}

} // namespace pnnx
