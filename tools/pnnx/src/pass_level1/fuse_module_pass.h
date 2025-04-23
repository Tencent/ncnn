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

#ifndef PNNX_FUSE_MODULE_PASS_H
#define PNNX_FUSE_MODULE_PASS_H

#include "ir.h"

#include <memory>
#include <unordered_map>
namespace torch {
namespace jit {
struct Graph;
struct Module;
struct Node;
struct Value;
} // namespace jit
} // namespace torch

namespace pnnx {

class TorchNodeProxy
{
public:
    TorchNodeProxy(const torch::jit::Node* _node) : node(_node) {}

    std::string kind() const;

    bool hasNamedInput(const std::string& name) const;
    const torch::jit::Value* namedInput(const std::string& name) const;

    std::vector<const torch::jit::Value*> inputs() const;
    std::vector<const torch::jit::Value*> outputs() const;

    const torch::jit::Value* input(int i) const;
    const torch::jit::Value* output(int i) const;

public:
    const torch::jit::Node* node;
};

class TorchGraphProxy
{
public:
    TorchGraphProxy(const std::shared_ptr<torch::jit::Graph> _graph);

    // bool has_node(const std::string& name) const;
    const TorchNodeProxy* find_node_by_kind(const std::string& kind) const;

    const TorchNodeProxy* find_producer_node_by_value(const torch::jit::Value* value) const;

    std::vector<const torch::jit::Value*> inputs() const;
    std::vector<const torch::jit::Value*> outputs() const;

    const torch::jit::Value* input(int i) const;
    const torch::jit::Value* output(int i) const;

    void dump() const;

public:
    const std::shared_ptr<torch::jit::Graph> graph;

public:
    std::vector<TorchNodeProxy> nodes;
};

class TorchTensorProxyPrivate;
class TorchTensorProxy
{
public:
    TorchTensorProxy(const at::Tensor& _t);

    const at::Tensor& t() const;

    int size(size_t i) const;

private:
    std::unique_ptr<TorchTensorProxyPrivate> d;
};

class TorchModuleProxy
{
public:
    TorchModuleProxy(const torch::jit::Module& _mod);

    bool hasattr(const std::string& name) const;
    const TorchTensorProxy& attr(const std::string& name) const;

public:
    const torch::jit::Module& mod;

private:
    std::unordered_map<std::string, TorchTensorProxy> attrs;
};


class FuseModulePass
{
public:
    virtual ~FuseModulePass();

    virtual const char* match_type_str() const = 0;

    virtual const char* type_str() const = 0;

    virtual void write(Operator* op, const TorchGraphProxy& graph) const;

    virtual void write(Operator* op, const TorchGraphProxy& graph, const TorchModuleProxy& mod) const;

    virtual void write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph) const;

    virtual void write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph, const torch::jit::Module& mod) const;
};

class FuseModulePassRegister
{
public:
    FuseModulePassRegister(const FuseModulePass* pass);
    ~FuseModulePassRegister();
    const FuseModulePass* pass;
};

const std::vector<const FuseModulePass*>& get_global_pnnx_fuse_module_passes();

#define REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(CLASS) \
    static FuseModulePassRegister g_global_pnnx_fusemodulepass_##CLASS##_register(new CLASS);

} // namespace pnnx

#endif // PNNX_FUSE_MODULE_PASS_H
