// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

#include <torch/script.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/quantization/helper.h>

namespace pnnx {

std::string TorchNodeProxy::kind() const
{
    return node->kind().toDisplayString();
}

bool TorchNodeProxy::hasNamedInput(const std::string& name) const
{
    return node->hasNamedInput(name);
}

const torch::jit::Value* TorchNodeProxy::namedInput(const std::string& name) const
{
    return node->namedInput(name);
}

int TorchNodeProxy::input_count() const
{
    return node->inputs().size();
}

const torch::jit::Value* TorchNodeProxy::input(int i) const
{
    return node->input(i);
}

int TorchNodeProxy::output_count() const
{
    return node->outputs().size();
}

const torch::jit::Value* TorchNodeProxy::output(int i) const
{
    return node->output(i);
}

bool TorchNodeProxy::is_input_none(int i) const
{
    return node->input(i)->type()->kind() == c10::TypeKind::NoneType;
}

TorchGraphProxy::TorchGraphProxy(const std::shared_ptr<torch::jit::Graph> _graph)
    : graph(_graph)
{
    for (const auto& n : graph->nodes())
    {
        nodes.push_back(n);
    }
}

const TorchNodeProxy* TorchGraphProxy::find_node_by_kind(const std::string& kind) const
{
    for (const auto& n : nodes)
    {
        if (n.node->kind().toDisplayString() == kind)
            return &n;
    }

    return 0;
}

const TorchNodeProxy* TorchGraphProxy::find_producer_node_by_value(const torch::jit::Value* value) const
{
    for (const auto& n : nodes)
    {
        if (n.node == value->node())
            return &n;
    }

    fprintf(stderr, "TorchGraphProxy find_producer_node_by_value failed\n");
    return 0;
}

int TorchGraphProxy::input_count() const
{
    return std::as_const(*graph).inputs().size();
}

const torch::jit::Value* TorchGraphProxy::input(int i) const
{
    return std::as_const(*graph).inputs()[i];
}

int TorchGraphProxy::output_count() const
{
    return std::as_const(*graph).outputs().size();
}

const torch::jit::Value* TorchGraphProxy::output(int i) const
{
    return std::as_const(*graph).outputs()[i];
}

void TorchGraphProxy::dump() const
{
    graph->dump();
}

static void specialize_dropout(torch::jit::Block* block, const torch::jit::Value* self, const torch::jit::Value* input, int rank, const torch::jit::Module& mod)
{
    for (auto it = block->nodes().begin(); it != block->nodes().end();)
    {
        torch::jit::Node* node = *it++;
        for (auto child : node->blocks())
            specialize_dropout(child, self, input, rank, mod);

        if (node->kind() == c10::prim::GetAttr && node->input(0) == self)
        {
            const std::string& name = node->s(torch::jit::attr::name);
            if (!mod.hasattr(name))
                continue;
            const auto value = mod.attr(name);
            if (!value.isBool() && !value.isInt() && !value.isDouble())
                continue;

            torch::jit::WithInsertPoint guard(node);
            node->output()->replaceAllUsesWith(node->owningGraph()->insertConstant(value));
            node->destroy();
        }
        else if (node->kind() == c10::aten::dim && node->input(0) == input && rank >= 0)
        {
            torch::jit::WithInsertPoint guard(node);
            node->output()->replaceAllUsesWith(node->owningGraph()->insertConstant((int64_t)rank));
            node->destroy();
        }
    }
}

void write_dropout_alias(Operator* op, const TorchGraphProxy& graph, const TorchModuleProxy& mod)
{
    if (op->inputs.size() != 1 || op->outputs.size() != 1)
        return;

    // Resolve the serialized wrapper before fusion loses its Tensor object identity.
    auto g = graph.graph->copy();
    torch::jit::Inline(*g);
    if (g->inputs().size() != 2 || g->outputs().size() != 1)
        return;

    const auto input = g->inputs()[1];
    const int rank = op->inputs[0]->shape.empty() ? -1 : (int)op->inputs[0]->shape.size();
    specialize_dropout(g->block(), g->inputs()[0], input, rank, mod.mod);
    torch::jit::ConstantPropagation(g);

    bool dropout = false;
    bool squeeze = false;
    bool unsqueeze = false;
    bool view = false;
    auto value = g->outputs()[0];
    while (value != input)
    {
        const auto node = value->node();
        const std::string type = node->kind().toDisplayString();
        if (type == "aten::feature_dropout" || type == "aten::feature_dropout_")
        {
            if (dropout || node->inputs().size() != 3)
                return;
            const auto training = torch::jit::toIValue(node->input(2));
            const auto probability = torch::jit::toIValue(node->input(1));
            if (!(training && training->isBool() && !training->toBool())
                    && !(probability && ((probability->isDouble() && probability->toDouble() == 0.) || (probability->isInt() && probability->toInt() == 0))))
                return;
            dropout = true;
        }
        else if (type == "aten::squeeze" || type == "aten::squeeze_")
        {
            if (squeeze || dropout || node->inputs().size() != 2)
                return;
            const auto dim = torch::jit::toIValue(node->input(1));
            if (!dim || !dim->isInt() || dim->toInt() != 0)
                return;
            squeeze = true;
            view |= type == "aten::squeeze";
        }
        else if (type == "aten::unsqueeze" || type == "aten::unsqueeze_")
        {
            if (unsqueeze || !squeeze || !dropout || node->inputs().size() != 2)
                return;
            const auto dim = torch::jit::toIValue(node->input(1));
            if (!dim || !dim->isInt() || dim->toInt() != 0)
                return;
            unsqueeze = true;
            view |= type == "aten::unsqueeze";
        }
        else
        {
            return;
        }
        value = node->input(0);
    }
    if (squeeze != unsqueeze)
        return;

    op->type = view ? "aten::alias" : "nn.Dropout";
}

class TorchTensorProxyPrivate
{
public:
    at::Tensor t;
};

TorchTensorProxy::TorchTensorProxy(const at::Tensor& _t)
    : d(new TorchTensorProxyPrivate)
{
    d->t = _t;
}

TorchTensorProxy::~TorchTensorProxy()
{
    delete d;
}

const at::Tensor& TorchTensorProxy::t() const
{
    return d->t;
}

int TorchTensorProxy::size(size_t i) const
{
    return d->t.size(i);
}

TorchModuleProxy::TorchModuleProxy(const torch::jit::Module& _mod)
    : mod(_mod)
{
    const std::vector<c10::ClassAttribute>& attributes = mod._ivalue()->type()->getAttributes();
    for (size_t i = 0; i < attributes.size(); i++)
    {
        const std::string& name = attributes[i].getName();
        const c10::IValue& ivalue = mod._ivalue()->getSlot(i);

        if (name.empty())
            continue;

        if (ivalue.isTensor())
            attrs.emplace(name, ivalue.toTensor());

        if (ivalue.isModule())
        {
            const torch::jit::Module submod = ivalue.toModule();

            const std::vector<c10::ClassAttribute>& sub_attributes = submod._ivalue()->type()->getAttributes();
            for (size_t j = 0; j < sub_attributes.size(); j++)
            {
                const std::string& sub_name = sub_attributes[j].getName();
                const c10::IValue& sub_ivalue = submod._ivalue()->getSlot(j);

                if (sub_name.empty())
                    continue;

                if (sub_ivalue.isTensor())
                    attrs.emplace(name + "." + sub_name, sub_ivalue.toTensor());
            }
        }
    }
}

bool TorchModuleProxy::hasattr(const std::string& name) const
{
    return attrs.find(name) != attrs.end();
}

const TorchTensorProxy& TorchModuleProxy::attr(const std::string& name) const
{
    return attrs.at(name);
}

FuseModulePass::~FuseModulePass()
{
}

void FuseModulePass::write(Operator* /*op*/, const TorchGraphProxy& /*graph*/) const
{
}

void FuseModulePass::write(Operator* op, const TorchGraphProxy& graph, const TorchModuleProxy& /*mod*/) const
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

} // namespace pnnx
