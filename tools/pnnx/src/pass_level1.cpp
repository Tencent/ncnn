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

#include <torch/csrc/jit/passes/quantization/helper.h>
#include <torch/csrc/api/include/torch/version.h>

#include "pass_level1.h"

namespace pnnx {

FuseModulePass::~FuseModulePass()
{
}

void FuseModulePass::write(Operator* /*op*/, const std::shared_ptr<torch::jit::Graph>& /*graph*/) const
{
}

void FuseModulePass::write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph, const torch::jit::Module& /*mod*/) const
{
    write(op, graph);
}

static std::vector<const FuseModulePass*> g_global_pnnx_fuse_module_passes;

const std::vector<const FuseModulePass*>& get_global_pnnx_fuse_module_passes()
{
    return g_global_pnnx_fuse_module_passes;
}

FuseModulePassRegister::FuseModulePassRegister(const FuseModulePass* _pass)
    : pass(_pass)
{
    g_global_pnnx_fuse_module_passes.push_back(pass);
}

FuseModulePassRegister::~FuseModulePassRegister()
{
    delete pass;
}

void pass_level1(const torch::jit::Module& mod, const std::shared_ptr<torch::jit::Graph>& g, Graph& pg)
{
    for (int i = 1; i < (int)g->inputs().size(); i++)
    {
        const auto& in = g->inputs()[i];

        char name[32];
        sprintf(name, "pnnx_input_%d", i - 1);

        Operator* op = pg.new_operator("pnnx.Input", name);
        Operand* r = pg.new_operand(in);
        r->producer = op;
        op->outputs.push_back(r);
    }

    std::map<std::string, std::string> class_type_to_names;
    int pnnx_unknown_index = 0;

    for (const auto& n : g->block()->nodes())
    {
        if (n->kind() == c10::prim::GetAttr)
        {
            // pass
            std::string name = n->s(torch::jit::attr::name);
            //             std::string name = n->debugName();

            auto class_type = n->output(0)->type()->cast<torch::jit::ClassType>();

            if (class_type)
            {
                std::string class_type_str = class_type->str();
                class_type_to_names[class_type_str] = name;
                //             class_type_to_names[class_type_str] = class_type_str + "." + name;
            }
            else
            {
                // Tensor from some class
                //                 Operator* op = pg.new_operator(n->kind().toDisplayString(), name);
                Operator* op = pg.new_operator("pnnx.Attribute", name);

                for (int i = 0; i < (int)n->outputs().size(); i++)
                {
                    const auto& on = n->output(i);
                    Operand* r = pg.new_operand(on);
                    r->producer = op;
                    op->outputs.push_back(r);
                }

                std::deque<std::string> module_names; // = split(n->input(0)->node()->s(torch::jit::attr::name), '.');
                {
                    auto np = n->input(0)->node();
                    while (np->hasAttribute(torch::jit::attr::name))
                    {
                        module_names.push_front(np->s(torch::jit::attr::name));
                        np = np->input(0)->node();
                    }
                }

                std::string wrapped_name;
                auto sub_mod = mod;
                for (auto module_name : module_names)
                {
                    if (wrapped_name.size() > 0)
                        wrapped_name = wrapped_name + "." + module_name;
                    else
                        wrapped_name = module_name;
                    sub_mod = sub_mod.attr(module_name).toModule();
                }

                if (wrapped_name.empty())
                {
                    // top-level module
                    wrapped_name = name;
                }

                op->name = wrapped_name;

                // op->params["this"] = n->input(i)

                // sub_mod.dump(true, true, true);

                op->attrs[name] = sub_mod.attr(name).toTensor();
            }
        }
        else if (n->kind() == c10::prim::Constant) // || n->kind() == c10::prim::ListConstruct)
        {
            char name[32];
            sprintf(name, "pnnx_%d", pnnx_unknown_index++);

            Operator* op = pg.new_operator(n->kind().toDisplayString(), name);

            for (int i = 0; i < (int)n->inputs().size(); i++)
            {
                const auto& in = n->input(i);
                Operand* r = pg.get_operand(in->debugName());
                r->consumers.push_back(op);
                op->inputs.push_back(r);
            }

            for (int i = 0; i < (int)n->outputs().size(); i++)
            {
                const auto& on = n->output(i);
                Operand* r = pg.new_operand(on);
                r->producer = op;
                op->outputs.push_back(r);
            }

            op->params["value"] = n;

            if (op->params["value"].type == 8)
            {
                op->type = "pnnx.Attribute";

                op->params.erase("value");

                op->attrs[name] = n->t(torch::jit::attr::value);
            }
        }
        else if (n->kind() == c10::prim::CallMethod)
        {
            auto class_type = n->input(0)->type()->cast<torch::jit::ClassType>();
            //             const std::string& name = n->s(torch::jit::attr::name);

            //             fprintf(stderr, "call %s\n", class_type->str().c_str());

            std::string name = class_type_to_names[class_type->str()];

            std::string class_type_str = torch::jit::removeTorchMangle(class_type->str());

            std::string optypename = class_type_str;
            for (const auto& ow : get_global_pnnx_fuse_module_passes())
            {
                if (class_type_str != ow->match_type_str())
                    continue;

                optypename = ow->type_str();
                break;
            }

            if (optypename == class_type_str)
            {
                optypename = class_type_str.substr(10);
            }

            Operator* op = pg.new_operator(optypename, name);

            for (int i = 1; i < (int)n->inputs().size(); i++)
            {
                const auto& in = n->input(i);
                Operand* r = pg.get_operand(in->debugName());
                r->consumers.push_back(op);
                op->inputs.push_back(r);
            }

            for (int i = 0; i < (int)n->outputs().size(); i++)
            {
                const auto& on = n->output(i);
                Operand* r = pg.new_operand(on);
                r->producer = op;
                op->outputs.push_back(r);
            }

            for (const auto& ow : get_global_pnnx_fuse_module_passes())
            {
                if (class_type_str != ow->match_type_str())
                    continue;

                auto class_type = n->input(0)->type()->cast<torch::jit::ClassType>();
                torch::jit::Function& function = class_type->getMethod(n->s(torch::jit::attr::name));

                std::deque<std::string> module_names; // = split(n->input(0)->node()->s(torch::jit::attr::name), '.');
                {
                    auto np = n->input(0)->node();
                    while (np->hasAttribute(torch::jit::attr::name))
                    {
                        module_names.push_front(np->s(torch::jit::attr::name));
                        np = np->input(0)->node();
                    }
                }

                std::string wrapped_name;
                auto sub_mod = mod;
                for (auto module_name : module_names)
                {
                    if (wrapped_name.size() > 0)
                        wrapped_name = wrapped_name + "." + module_name;
                    else
                        wrapped_name = module_name;
                    sub_mod = sub_mod.attr(module_name).toModule();
                }

                op->name = wrapped_name;

#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 11
                ow->write(op, toGraphFunction(function).graph(), sub_mod);
#else
                ow->write(op, function.graph(), sub_mod);
#endif

                break;
            }
        }
        // else if (n->kind() == c10::prim::CallFunction)
        // {
        //     fprintf(stderr, "function %s", n->kind().toDisplayString());
        //
        //     AT_ASSERT(cur->input(0)->node()->kind() == c10::prim::Constant);
        //     auto function_constant = cur->input(0)->node();
        //     auto fun_type = function_constant->output()->type()->expect<torch::jit::FunctionType>();
        //     if (!fun_type->function()->isGraphFunction())
        //     {
        //         continue;
        //     }
        //     cur->removeInput(0);
        //
        //     fprintf(stderr, "inline function %s\n", fun_type->function()->name().c_str());
        //
        //     GRAPH_UPDATE("Inlining function '", fun_type->function()->name(), "' to ", *cur);
        //     GRAPH_UPDATE("Function body: ", *fun_type->function()->optimized_graph());
        //     inlineCallTo(cur, fun_type->function(), false);
        //     break;
        // }
        else
        {
            char name[32];
            sprintf(name, "pnnx_%d", pnnx_unknown_index++);

            Operator* op = pg.new_operator(n->kind().toDisplayString(), name);

            for (int i = 0; i < (int)n->inputs().size(); i++)
            {
                const auto& in = n->input(i);
                Operand* r = pg.get_operand(in->debugName());
                r->consumers.push_back(op);
                op->inputs.push_back(r);
            }

            for (int i = 0; i < (int)n->outputs().size(); i++)
            {
                const auto& on = n->output(i);
                Operand* r = pg.new_operand(on);
                r->producer = op;
                op->outputs.push_back(r);
            }
        }
    }

    for (int i = 0; i < (int)g->outputs().size(); i++)
    {
        const auto& in = g->outputs()[i];

        char name[32];
        sprintf(name, "pnnx_output_%d", i);
        Operator* op = pg.new_operator("pnnx.Output", name);
        Operand* r = pg.get_operand(in->debugName());
        r->consumers.push_back(op);
        op->inputs.push_back(r);
    }
}

} // namespace pnnx
