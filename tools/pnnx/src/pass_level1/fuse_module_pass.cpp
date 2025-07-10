// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

#include <torch/script.h>
#include <torch/csrc/jit/api/module.h>
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
