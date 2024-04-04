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

#include "pass_onnx.h"

#include "onnx.pb.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include <fstream>

#include "ir.h"

namespace pnnx {

namespace onnx2pnnx {

static float get_tensor_f(const onnx::TensorProto& tensor)
{
    int64_t numel = 1;
    for (int k = 0; k < tensor.dims_size(); k++)
    {
        numel *= tensor.dims(k);
    }

    if (numel != 1)
    {
        fprintf(stderr, "get_tensor_f numel %ld\n", numel);
    }

    if (tensor.data_type() == onnx::TensorProto::FLOAT)
    {
        if (tensor.has_raw_data())
        {
            // assert tensor.raw_data().size() == 4
            return ((float*)tensor.raw_data().data())[0];
        }

        // assert tensor.float_data().size() == 1
        return tensor.float_data().at(0);
    }

    // fatal error
    fprintf(stderr, "get_tensor_f failed\n");
    return 0.f;
}

static std::vector<float> get_tensor_af(const onnx::TensorProto& tensor)
{
    if (tensor.dims_size() != 1)
    {
        fprintf(stderr, "get_tensor_af dims_size %d\n", (int)tensor.dims_size());
    }

    const int64_t numel = tensor.dims(0);

    if (tensor.data_type() == onnx::TensorProto::FLOAT)
    {
        const float* p = tensor.has_raw_data() ? (float*)tensor.raw_data().data() : tensor.float_data().data();
        std::vector<float> af(numel);
        memcpy(af.data(), p, sizeof(float) * numel);
        return af;
    }

    // fatal error
    fprintf(stderr, "get_tensor_af failed\n");
    return std::vector<float>();
}

static int64_t get_tensor_i(const onnx::TensorProto& tensor)
{
    int64_t numel = 1;
    for (int k = 0; k < tensor.dims_size(); k++)
    {
        numel *= tensor.dims(k);
    }

    if (numel != 1)
    {
        fprintf(stderr, "get_tensor_i numel %ld\n", numel);
    }

    if (tensor.data_type() == onnx::TensorProto::INT32)
    {
        if (tensor.has_raw_data())
        {
            // assert tensor.raw_data().size() == 4
            return ((int*)tensor.raw_data().data())[0];
        }

        // assert tensor.int32_data().size() == 1
        return tensor.int32_data().at(0);
    }

    if (tensor.data_type() == onnx::TensorProto::INT64)
    {
        if (tensor.has_raw_data())
        {
            // assert tensor.raw_data().size() == 8
            return ((int64_t*)tensor.raw_data().data())[0];
        }

        // assert tensor.int64_data().size() == 1
        return tensor.int64_data().at(0);
    }

    // fatal error
    fprintf(stderr, "get_tensor_i failed\n");
    return 0;
}

static std::vector<int64_t> get_tensor_ai(const onnx::TensorProto& tensor)
{
    if (tensor.dims_size() != 1)
    {
        fprintf(stderr, "get_tensor_af dims_size %d\n", (int)tensor.dims_size());
    }

    const int64_t numel = tensor.dims(0);

    if (tensor.data_type() == onnx::TensorProto::INT32)
    {
        const int* p = tensor.has_raw_data() ? (int*)tensor.raw_data().data() : tensor.int32_data().data();
        std::vector<int64_t> ai(numel);
        for (int i = 0; i < numel; i++)
            ai[i] = p[i];
        return ai;
    }

    if (tensor.data_type() == onnx::TensorProto::INT64)
    {
        const int64_t* p = tensor.has_raw_data() ? (int64_t*)tensor.raw_data().data() : tensor.int64_data().data();
        std::vector<int64_t> ai(numel);
        memcpy(ai.data(), p, sizeof(int64_t) * numel);
        return ai;
    }

    // fatal error
    fprintf(stderr, "get_tensor_ai failed\n");
    return std::vector<int64_t>();
}

float OnnxAttributeProxy::value_f() const
{
    if (attr.type() == onnx::AttributeProto::FLOAT)
    {
        return attr.f();
    }

    if (attr.type() == onnx::AttributeProto::TENSOR)
    {
        return get_tensor_f(attr.t());
    }

    fprintf(stderr, "OnnxAttributeProxy value_f failed\n");
    return 0.f;
}

int64_t OnnxAttributeProxy::value_i() const
{
    if (attr.type() == onnx::AttributeProto::INT)
    {
        return attr.i();
    }

    if (attr.type() == onnx::AttributeProto::TENSOR)
    {
        return get_tensor_i(attr.t());
    }

    fprintf(stderr, "OnnxAttributeProxy value_i failed\n");
    return 0;
}

std::string OnnxAttributeProxy::value_s() const
{
    if (attr.type() != onnx::AttributeProto::STRING)
        fprintf(stderr, "OnnxAttributeProxy value_s failed\n");

    return attr.s();
}

std::vector<float> OnnxAttributeProxy::value_fs() const
{
    if (attr.type() == onnx::AttributeProto::FLOATS)
    {
        const int size = attr.floats().size();
        std::vector<float> fs(size);
        for (int i = 0; i < size; i++)
        {
            fs[i] = attr.floats().at(i);
        }
        return fs;
    }

    if (attr.type() == onnx::AttributeProto::TENSOR)
    {
        return get_tensor_af(attr.t());
    }

    fprintf(stderr, "OnnxAttributeProxy value_fs failed\n");
    return std::vector<float>();
}

std::vector<int64_t> OnnxAttributeProxy::value_is() const
{
    if (attr.type() == onnx::AttributeProto::INTS)
    {
        const int size = attr.ints().size();
        std::vector<int64_t> is(size);
        for (int i = 0; i < size; i++)
        {
            is[i] = attr.ints().at(i);
        }
        return is;
    }

    if (attr.type() == onnx::AttributeProto::TENSOR)
    {
        return get_tensor_ai(attr.t());
    }

    fprintf(stderr, "OnnxAttributeProxy value_is failed\n");
    return std::vector<int64_t>();
}

std::vector<std::string> OnnxAttributeProxy::value_ss() const
{
    if (attr.type() != onnx::AttributeProto::STRINGS)
        fprintf(stderr, "OnnxAttributeProxy value_ss failed\n");

    const int size = attr.strings().size();
    std::vector<std::string> ss(size);
    for (int i = 0; i < size; i++)
    {
        ss[i] = attr.strings().at(i);
    }
    return ss;
}

OnnxNodeProxy::OnnxNodeProxy(const onnx::NodeProto& _node)
    : node(_node)
{
    // extract attribute info
    for (int i = 0; i < node.attribute_size(); i++)
    {
        const std::string& name = node.attribute(i).name();
        attributes.insert(std::make_pair(name, i));
    }
}

bool OnnxNodeProxy::has_attribute(const std::string& name) const
{
    return attributes.count(name);
}

const OnnxAttributeProxy OnnxNodeProxy::attribute(const std::string& name) const
{
    int attribute_index = attributes.at(name);
    return node.attribute(attribute_index);
}

OnnxFunctionProxy::OnnxFunctionProxy(const onnx::ModelProto& _model, const onnx::NodeProto& _caller, const onnx::FunctionProto& _function)
    : model(_model), caller(_caller), function(_function)
{
    for (int i = 0; i < function.node_size(); i++)
    {
        const std::string& name = function.node(i).name();
        named_nodes.insert(std::make_pair(name, i));

        const std::string& type = function.node(i).op_type();
        typed_nodes.insert(std::make_pair(type, i));
    }

    for (int i = 0; i < caller.input_size(); i++)
    {
        const std::string& function_argument = caller.input(i);

        int initializer_index = -1;
        for (int j = 0; j < model.graph().initializer_size(); j++)
        {
            if (model.graph().initializer(j).name() == function_argument)
            {
                initializer_index = j;
                break;
            }
        }

        const std::string& function_parameter = function.input(i);
        initializers.insert(std::make_pair(function_parameter, initializer_index));
    }
}

bool OnnxFunctionProxy::has_typed_node(const std::string& type) const
{
    return typed_nodes.count(type);
}

bool OnnxFunctionProxy::has_named_node(const std::string& name) const
{
    return named_nodes.count(name);
}

const OnnxNodeProxy OnnxFunctionProxy::typed_node(const std::string& type) const
{
    int node_index = typed_nodes.at(type);
    return function.node(node_index);
}

const OnnxNodeProxy OnnxFunctionProxy::named_node(const std::string& name) const
{
    int node_index = named_nodes.at(name);
    return function.node(node_index);
}

const OnnxNodeProxy OnnxFunctionProxy::find_producer(const std::string& name) const
{
    // find Constant node which produces name
    for (int i = 0; i < function.node_size(); i++)
    {
        const onnx::NodeProto& node = function.node(i);
        for (int j = 0; j < node.output_size(); j++)
        {
            if (node.output(j) == name)
            {
                return node;
            }
        }
    }

    // should never reach here
    return function.node(0);
}

bool OnnxFunctionProxy::has_initializer(const std::string& name) const
{
    return initializers.count(name);
}

const onnx::TensorProto& OnnxFunctionProxy::initializer(const std::string& name) const
{
    int initializer_index = initializers.at(name);
    return model.graph().initializer(initializer_index);
}

OnnxModelProxy::OnnxModelProxy(const onnx::ModelProto& _model)
    : model(_model)
{
    for (int i = 0; i < model.graph().node_size(); i++)
    {
        const std::string& name = model.graph().node(i).name();
        nodes.insert(std::make_pair(name, i));

        for (int j = 0; j < model.functions_size(); j++)
        {
            const std::string& function_name = model.functions(j).name();
            if (function_name == model.graph().node(i).op_type())
            {
                functions.insert(std::make_pair(function_name + name, j));
            }
        }
    }

    for (int i = 0; i < model.graph().input_size(); i++)
    {
        const std::string& name = model.graph().input(i).name();
        valueinfos.insert(std::make_pair(name, -1));
    }
    for (int i = 0; i < model.graph().output_size(); i++)
    {
        const std::string& name = model.graph().output(i).name();
        valueinfos.insert(std::make_pair(name, -2));
    }

    for (int i = 0; i < model.graph().value_info_size(); i++)
    {
        const std::string& name = model.graph().value_info(i).name();
        valueinfos.insert(std::make_pair(name, i));
    }

    for (int i = 0; i < model.graph().initializer_size(); i++)
    {
        const std::string& name = model.graph().initializer(i).name();
        initializers.insert(std::make_pair(name, i));
    }
}

bool OnnxModelProxy::has_node(const std::string& name) const
{
    return nodes.count(name);
}

const OnnxNodeProxy OnnxModelProxy::node(const std::string& name) const
{
    int node_index = nodes.at(name);
    return model.graph().node(node_index);
}

bool OnnxModelProxy::has_function(const std::string& name, const std::string& caller) const
{
    return functions.count(name + caller);
}

const OnnxFunctionProxy OnnxModelProxy::function(const std::string& name, const std::string& caller) const
{
    int function_index = functions.at(name + caller);
    return OnnxFunctionProxy(model, node(caller).node, model.functions(function_index));
}

bool OnnxModelProxy::has_valueinfo(const std::string& name) const
{
    return valueinfos.count(name);
}

const onnx::ValueInfoProto& OnnxModelProxy::valueinfo(const std::string& name) const
{
    int valueinfo_index = valueinfos.at(name);
    if (valueinfo_index == -1)
    {
        for (int i = 0; i < model.graph().input_size(); i++)
        {
            if (model.graph().input(i).name() == name)
                return model.graph().input(i);
        }
    }
    if (valueinfo_index == -2)
    {
        for (int i = 0; i < model.graph().output_size(); i++)
        {
            if (model.graph().output(i).name() == name)
                return model.graph().output(i);
        }
    }

    return model.graph().value_info(valueinfo_index);
}

bool OnnxModelProxy::has_initializer(const std::string& name) const
{
    return initializers.count(name);
}

const onnx::TensorProto& OnnxModelProxy::initializer(const std::string& name) const
{
    int initializer_index = initializers.at(name);
    return model.graph().initializer(initializer_index);
}

FuseFunctionPass::~FuseFunctionPass()
{
}

void FuseFunctionPass::write(Operator* /*op*/, const OnnxFunctionProxy& /*function*/) const
{
}

static std::vector<const FuseFunctionPass*> g_global_pnnx_fuse_function_passes;

const std::vector<const FuseFunctionPass*>& get_global_pnnx_fuse_function_passes()
{
    return g_global_pnnx_fuse_function_passes;
}

FuseFunctionPassRegister::FuseFunctionPassRegister(const FuseFunctionPass* _pass)
    : pass(_pass)
{
    g_global_pnnx_fuse_function_passes.push_back(pass);
}

FuseFunctionPassRegister::~FuseFunctionPassRegister()
{
    delete pass;
}

} // namespace onnx2pnnx

static bool string_starts_with(const std::string& s, const std::string& s2)
{
    return strncmp(s.c_str(), s2.c_str(), s2.size()) == 0;
}

static void fuse_list_unpack(Graph& graph)
{
    // prim::Constant + aten::getitem ...  ->  prim::ListUnpack

    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "aten::getitem")
                continue;

            Operand* op_in = op->inputs[0];

            const int item_count = (int)op_in->consumers.size();

            std::vector<Operator*> getitem_ops(item_count);

            Operator* cur = op;

            bool full_getitem = true;
            for (Operator* op2 : op_in->consumers)
            {
                if (op2->type != "aten::getitem")
                {
                    fprintf(stderr, "unbalanced getitem\n");
                    full_getitem = false;
                    break;
                }

                int gi = op2->inputs[1]->producer->params.at("value").i;
                getitem_ops[gi] = op2;

                if (std::find(graph.ops.begin(), graph.ops.end(), op2) < std::find(graph.ops.begin(), graph.ops.end(), cur))
                    cur = op2;
            }

            if (!full_getitem)
                continue;

            matched = true;

            // delete all getitem ops and replace with ListUnpack
            Operator* op_list_unpack = graph.new_operator_before("prim::ListUnpack", op->name, cur);

            op_list_unpack->inputs.push_back(op_in);
            for (auto op2 : getitem_ops)
            {
                op_in->remove_consumer(op2);
            }
            op_in->consumers.push_back(op_list_unpack);

            op_list_unpack->outputs.resize(getitem_ops.size());
            for (size_t j = 0; j < getitem_ops.size(); j++)
            {
                op_list_unpack->outputs[j] = getitem_ops[j]->outputs[0];
                getitem_ops[j]->outputs[0]->producer = op_list_unpack;
            }

            for (auto op2 : getitem_ops)
            {
                op2->inputs[1]->remove_consumer(op2);

                graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op2));
                delete op2;
            }

            break;
        }

        if (!matched)
            break;
    }
}

void pass_onnx(const onnx::ModelProto& model, Graph& pnnx_graph)
{
    onnx2pnnx::OnnxModelProxy modelproxy(model);

    const onnx::GraphProto& graph = model.graph();

    for (int i = 0; i < graph.input_size(); i++)
    {
        const std::string& output = graph.input(i).name();

        Operator* op = pnnx_graph.new_operator("pnnx.Input", output);

        const onnx::ValueInfoProto& value = modelproxy.valueinfo(output);

        Operand* op_out = pnnx_graph.new_operand(value);

        op_out->producer = op;
        op->outputs.push_back(op_out);
    }

    for (int i = 0; i < graph.node_size(); i++)
    {
        const onnx::NodeProto& node = graph.node(i);

        const std::string& op_type = node.op_type();

        std::string sim_op_type;

        if (node.domain().empty())
        {
            // native onnx op
            sim_op_type = op_type;

            if (op_type == "SequenceConstruct")
            {
                sim_op_type = "prim::ListConstruct";
            }

            if (op_type == "Slice")
            {
                sim_op_type = "aten::slice";
            }

            if (op_type == "Transpose")
            {
                sim_op_type = "aten::permute";
            }
        }
        else if (string_starts_with(op_type, "aten_"))
        {
            // aten_view
            sim_op_type = std::string("aten::") + op_type.substr(5);
        }
        else if (string_starts_with(op_type, "_aten_"))
        {
            // _aten_roll_shift_and_dim_onnx
            sim_op_type = std::string("aten::") + op_type.substr(6);
        }
        else if (string_starts_with(op_type, "prims_"))
        {
            // prims_convert_element_type
            sim_op_type = std::string("prim::") + op_type.substr(6);
        }
        else if (string_starts_with(op_type, "nn_"))
        {
            // torch_nn_modules_conv_Conv2d                 _conv1_1
            sim_op_type = op_type;
            // nn_Conv2d_i -> nn.Conv2d
            sim_op_type[2] = '.';
            if (sim_op_type.find_first_of('_') != std::string::npos)
                sim_op_type = sim_op_type.substr(0, sim_op_type.find_first_of('_'));
        }
        else
        {
            // custom function
            sim_op_type = std::string("custom_op.") + op_type;
        }

        // fprintf(fp, "%-24s %-8s", sim_op_type.c_str(), node.name().c_str());

        Operator* op = pnnx_graph.new_operator(sim_op_type, node.name());

        // bool is_function = modelproxy.has_function(node.op_type(), node.name());

        bool is_function_op = string_starts_with(sim_op_type, "nn.") || string_starts_with(sim_op_type, "custom_op.");

        bool is_aten_op = string_starts_with(sim_op_type, "aten::");

        bool is_prim_op = string_starts_with(sim_op_type, "prim::");

        for (int j = 0; j < node.input_size(); j++)
        {
            const std::string& input = node.input(j);

            if (modelproxy.has_initializer(input))
            {
                // skip function weight
                if (is_function_op)
                    continue;

                const onnx::TensorProto& tensor = modelproxy.initializer(input);

                int64_t numel = 1;
                for (int k = 0; k < tensor.dims_size(); k++)
                {
                    numel *= tensor.dims(k);
                }

                if (numel == 1)
                {
                    Operator* op_const = pnnx_graph.new_operator_before("prim::Constant", input, op);

                    Operand* op_const_out = pnnx_graph.new_operand(input);

                    op_const_out->producer = op_const;
                    op_const->outputs.push_back(op_const_out);

                    if (tensor.data_type() == onnx::TensorProto::INT32)
                    {
                        if (tensor.has_raw_data())
                        {
                            // assert tensor.raw_data().size() == 4
                            op_const->params["value"] = ((int*)tensor.raw_data().data())[0];
                        }
                        else
                        {
                            // assert tensor.int32_data().size() == 1
                            op_const->params["value"] = tensor.int32_data().at(0);
                        }
                    }
                    else if (tensor.data_type() == onnx::TensorProto::INT64)
                    {
                        int64_t i64;
                        if (tensor.has_raw_data())
                        {
                            // assert tensor.raw_data().size() == 8
                            i64 = ((int64_t*)tensor.raw_data().data())[0];
                        }
                        else
                        {
                            // assert tensor.int64_data().size() == 1
                            i64 = tensor.int64_data().at(0);
                        }
                        if (i64 == std::numeric_limits<int64_t>::max()) i64 = INT_MAX;
                        if (i64 == std::numeric_limits<int64_t>::min()) i64 = INT_MIN;
                        op_const->params["value"] = (int)i64;
                    }
                    else if (tensor.data_type() == onnx::TensorProto::FLOAT)
                    {
                        if (tensor.has_raw_data())
                        {
                            // assert tensor.raw_data().size() == 4
                            op_const->params["value"] = ((float*)tensor.raw_data().data())[0];
                        }
                        else
                        {
                            // assert tensor.float_data().size() == 1
                            op_const->params["value"] = tensor.float_data().at(0);
                        }
                    }
                    else if (tensor.data_type() == onnx::TensorProto::BOOL)
                    {
                        if (tensor.has_raw_data())
                        {
                            // assert tensor.raw_data().size() == 2
                            op_const->params["value"] = ((uint16_t*)tensor.raw_data().data())[0] ? true : false;
                        }
                        else
                        {
                            // assert tensor.int32_data().size() == 1
                            op_const->params["value"] = tensor.int32_data().at(0) ? true : false;
                        }
                    }
                    else
                    {
                        fprintf(stderr, "unknown constant scalar type %d\n", (int)tensor.data_type());
                    }
                }
                else if (is_aten_op && tensor.dims_size() == 1 && (tensor.data_type() == onnx::TensorProto::INT32 || tensor.data_type() == onnx::TensorProto::INT64))
                {
                    // create list expression
                    Operator* op_const = pnnx_graph.new_operator_before("pnnx.Expression", input, op);

                    Operand* op_const_out = pnnx_graph.new_operand(input);

                    op_const_out->producer = op_const;
                    op_const->outputs.push_back(op_const_out);

                    const int list_size = tensor.dims(0);
                    if (tensor.data_type() == onnx::TensorProto::INT32)
                    {
                        std::vector<int> ai(list_size);
                        if (tensor.has_raw_data())
                        {
                            // assert tensor.raw_data().size() == 4 * list_size
                            memcpy((void*)ai.data(), (int*)tensor.raw_data().data(), sizeof(int) * list_size);
                        }
                        else
                        {
                            // assert tensor.int32_data().size() == list_size
                            memcpy((void*)ai.data(), tensor.int32_data().data(), sizeof(int) * list_size);
                        }
                        std::string expr = "[";
                        for (int k = 0; k < (int)ai.size(); k++)
                        {
                            expr += std::to_string(ai[k]);
                            if (k != (int)ai.size() - 1)
                                expr += ",";
                        }
                        expr += "]";
                        op_const->params["expr"] = expr;
                    }
                    else if (tensor.data_type() == onnx::TensorProto::INT64)
                    {
                        std::vector<int64_t> ai(list_size);
                        if (tensor.has_raw_data())
                        {
                            // assert tensor.raw_data().size() == 8 * list_size
                            memcpy((void*)ai.data(), (int64_t*)tensor.raw_data().data(), sizeof(int64_t) * list_size);
                        }
                        else
                        {
                            // assert tensor.int64_data().size() == list_size
                            memcpy((void*)ai.data(), tensor.int64_data().data(), sizeof(int64_t) * list_size);
                        }
                        std::string expr = "[";
                        for (int k = 0; k < (int)ai.size(); k++)
                        {
                            int64_t i64 = ai[k];
                            if (i64 == std::numeric_limits<int64_t>::max()) i64 = INT_MAX;
                            if (i64 == std::numeric_limits<int64_t>::min()) i64 = INT_MIN;
                            expr += std::to_string(i64);
                            if (k != (int)ai.size() - 1)
                                expr += ",";
                        }
                        expr += "]";
                        op_const->params["expr"] = expr;
                    }
                }
                else
                {
                    // create constant for functions
                    Operator* op_const = pnnx_graph.new_operator_before("pnnx.Attribute", input, op);

                    Operand* op_const_out = pnnx_graph.new_operand(tensor);

                    op_const_out->producer = op_const;
                    op_const->outputs.push_back(op_const_out);

                    op_const->attrs["data"] = tensor;
                }
            }

            Operand* op_in = pnnx_graph.get_operand(input);

            op_in->consumers.push_back(op);
            op->inputs.push_back(op_in);
        }

        for (int j = 0; j < node.output_size(); j++)
        {
            const std::string& output = node.output(j);

            Operand* op_out = 0;

            if (modelproxy.has_valueinfo(output))
            {
                const onnx::ValueInfoProto& value = modelproxy.valueinfo(output);
                op_out = pnnx_graph.new_operand(value);
            }
            else
            {
                op_out = pnnx_graph.new_operand(output);
            }

            op_out->producer = op;
            op->outputs.push_back(op_out);
        }

        if (is_function_op)
        {
            const onnx2pnnx::OnnxFunctionProxy function = modelproxy.function(node.op_type(), node.name());

            for (const auto& ow : onnx2pnnx::get_global_pnnx_fuse_function_passes())
            {
                if (sim_op_type != ow->match_type_str())
                    continue;

                op->type = ow->type_str();
                ow->write(op, function);

                break;
            }
        }
        else if (is_aten_op)
        {
            // extract attributes
            for (int j = 0; j < node.attribute_size(); j++)
            {
                const onnx::AttributeProto& attr = node.attribute(j);

                op->params[attr.name()] = attr;
            }

            if (op_type == "Slice")
            {
                // data start end dim step -> data dim start end step
                op->inputnames = {"data", "dim", "start", "end", "step"};
                op->inputs = {op->inputs[0], op->inputs[3], op->inputs[1], op->inputs[2], op->inputs[4]};
            }

            if (op_type == "Transpose")
            {
                op->params["dims"] = op->params["perm"];
                op->params.erase("perm");
            }
        }
        else if (is_prim_op)
        {
            // do nothing :)
        }
        else
        {
            // onnx native op, extract attributes
            for (int j = 0; j < node.attribute_size(); j++)
            {
                const onnx::AttributeProto& attr = node.attribute(j);

                op->params[attr.name()] = attr;
            }
        }
    }

    for (int i = 0; i < graph.output_size(); i++)
    {
        const std::string& input = graph.output(i).name();

        Operator* op = pnnx_graph.new_operator("pnnx.Output", input);

        Operand* op_in = pnnx_graph.get_operand(input);

        op_in->consumers.push_back(op);
        op->inputs.push_back(op_in);
    }

    // post process
    fuse_list_unpack(pnnx_graph);
}

} // namespace pnnx
