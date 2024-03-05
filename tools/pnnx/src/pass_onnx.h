// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef PNNX_PASS_ONNX_H
#define PNNX_PASS_ONNX_H

#include <string>
#include <vector>
#include <unordered_map>

namespace onnx {
class AttributeProto;
class FunctionProto;
class ModelProto;
class NodeProto;
class TensorProto;
class ValueInfoProto;
} // namespace onnx

namespace pnnx {

class Operator;
class Graph;

namespace onnx2pnnx {

class OnnxAttributeProxy
{
public:
    OnnxAttributeProxy(const onnx::AttributeProto& _attr)
        : attr(_attr)
    {
    }

    operator float() const
    {
        return value_f();
    }
    operator int64_t() const
    {
        return value_i();
    }
    operator std::string() const
    {
        return value_s();
    }
    operator std::vector<float>() const
    {
        return value_fs();
    }
    operator std::vector<int64_t>() const
    {
        return value_is();
    }
    operator std::vector<std::string>() const
    {
        return value_ss();
    }

    float value_f() const;
    int64_t value_i() const;
    std::string value_s() const;
    std::vector<float> value_fs() const;
    std::vector<int64_t> value_is() const;
    std::vector<std::string> value_ss() const;

public:
    const onnx::AttributeProto& attr;
};

class OnnxNodeProxy
{
public:
    OnnxNodeProxy(const onnx::NodeProto& _node);

    bool has_attribute(const std::string& name) const;
    const OnnxAttributeProxy attribute(const std::string& name) const;

public:
    const onnx::NodeProto& node;

protected:
    std::unordered_map<std::string, int> attributes;
};

class OnnxFunctionProxy
{
public:
    OnnxFunctionProxy(const onnx::ModelProto& _model, const onnx::NodeProto& _caller, const onnx::FunctionProto& _function);

    bool has_typed_node(const std::string& type) const;
    bool has_named_node(const std::string& name) const;
    const OnnxNodeProxy typed_node(const std::string& type) const;
    const OnnxNodeProxy named_node(const std::string& name) const;

    const OnnxNodeProxy find_producer(const std::string& name) const;

    bool has_initializer(const std::string& name) const;
    const onnx::TensorProto& initializer(const std::string& name) const;

public:
    const onnx::ModelProto& model;
    const onnx::NodeProto& caller;
    const onnx::FunctionProto& function;

protected:
    std::unordered_map<std::string, int> typed_nodes;
    std::unordered_map<std::string, int> named_nodes;
    std::unordered_map<std::string, int> initializers;
};

class OnnxModelProxy
{
public:
    OnnxModelProxy(const onnx::ModelProto& _model);

    bool has_node(const std::string& name) const;
    const OnnxNodeProxy node(const std::string& name) const;

    bool has_function(const std::string& name, const std::string& caller) const;
    const OnnxFunctionProxy function(const std::string& name, const std::string& caller) const;

    bool has_valueinfo(const std::string& name) const;
    const onnx::ValueInfoProto& valueinfo(const std::string& name) const;

    bool has_initializer(const std::string& name) const;
    const onnx::TensorProto& initializer(const std::string& name) const;

public:
    const onnx::ModelProto& model;

protected:
    std::unordered_map<std::string, int> nodes;
    std::unordered_map<std::string, int> functions;
    std::unordered_map<std::string, int> valueinfos;
    std::unordered_map<std::string, int> initializers;
};

class FuseFunctionPass
{
public:
    virtual ~FuseFunctionPass();

    virtual const char* match_type_str() const = 0;

    virtual const char* type_str() const = 0;

    virtual void write(Operator* op, const OnnxFunctionProxy& function) const;
};

class FuseFunctionPassRegister
{
public:
    FuseFunctionPassRegister(const FuseFunctionPass* pass);
    ~FuseFunctionPassRegister();
    const FuseFunctionPass* pass;
};

const std::vector<const FuseFunctionPass*>& get_global_pnnx_fuse_function_passes();

#define REGISTER_GLOBAL_PNNX_FUSE_FUNCTION_PASS(CLASS) \
    static FuseFunctionPassRegister g_global_pnnx_fusefunctionpass_##CLASS##_register(new CLASS);

} // namespace onnx2pnnx

void pass_onnx(const onnx::ModelProto& model, Graph& pnnx_graph);

} // namespace pnnx

#endif // PNNX_PASS_ONNX_H
