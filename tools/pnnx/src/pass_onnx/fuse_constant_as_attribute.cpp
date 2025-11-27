// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_constant_as_attribute.h"

#include <sstream>
#include <string>
#include <unordered_set>

#include "dead_code_elimination.h"

namespace pnnx {

namespace onnx2pnnx {

struct constant_as_attribute
{
    const char* op_type;
    int input_index;
    const char* attribute;
};

static constant_as_attribute caas[] = {
    {"Clip", 1, "min"},
    {"Clip", 2, "max"},
    {"Expand", 1, "shape"},
    {"Gather", 1, "indices"},
    {"If", 0, "cond"},
    {"Pad", 1, "pads"},
    {"Pad", 2, "value"},
    {"ReduceL1", 1, "axes"},
    {"ReduceL2", 1, "axes"},
    {"ReduceLogSumExp", 1, "axes"},
    {"ReduceMax", 1, "axes"},
    {"ReduceMean", 1, "axes"},
    {"ReduceMin", 1, "axes"},
    {"ReduceProd", 1, "axes"},
    {"ReduceSum", 1, "axes"},
    {"Reshape", 1, "shape"},
    {"Resize", 1, "roi"},
    {"Resize", 2, "scales"},
    {"Resize", 3, "sizes"},
    {"Slice", 1, "starts"},
    {"Slice", 2, "ends"},
    {"Slice", 3, "axes"},
    {"Slice", 4, "steps"},
    {"Squeeze", 1, "axes"},
    {"Tile", 1, "repeats"},
    {"Unsqueeze", 1, "axes"},
    {"Upsample", 1, "scales"},
};

static const char* get_constant_as_attribute(const std::string& op_type, int input_index)
{
    const int caas_count = sizeof(caas) / sizeof(caas[0]);
    for (int i = 0; i < caas_count; i++)
    {
        if (op_type == caas[i].op_type && input_index == caas[i].input_index)
            return caas[i].attribute;
    }

    return NULL;
}

void fuse_constant_as_attribute(onnx::ModelProto& model)
{
    // collect initializers
    std::unordered_map<std::string, int> initializers;
    {
        const onnx::GraphProto& graph = model.graph();
        for (int i = 0; i < graph.initializer_size(); i++)
        {
            initializers.insert(std::make_pair(graph.initializer(i).name(), i));
        }
    }

    onnx::GraphProto* graph = model.mutable_graph();

    for (int i = 0; i < graph->node_size(); i++)
    {
        onnx::NodeProto* node = graph->mutable_node(i);
        if (!node->domain().empty())
        {
            // native onnx op
            continue;
        }

        const std::string& op_type = node->op_type();

        std::vector<int> fused_input_indexes;
        for (int j = 0; j < node->input_size(); j++)
        {
            const std::string& input = node->input(j);
            if (input.empty())
                continue;

            if (initializers.find(input) == initializers.end())
                continue;

            const char* attr_name = get_constant_as_attribute(op_type, j);
            if (!attr_name)
                continue;

            // fprintf(stderr, "fuse_constant_as_attribute  %s %d -> %s\n", op_type.c_str(), j, attr_name);

            const onnx::TensorProto& tensor = graph->initializer(initializers.at(input));

            int64_t numel = 1;
            for (int k = 0; k < tensor.dims_size(); k++)
            {
                numel *= tensor.dims(k);
            }

            if (numel == 1)
            {
                // int or float scalar
                if (tensor.data_type() == onnx::TensorProto::INT32)
                {
                    int i32;
                    if (tensor.has_raw_data())
                    {
                        // assert tensor.raw_data().size() == 4
                        i32 = ((int*)tensor.raw_data().data())[0];
                    }
                    else
                    {
                        // assert tensor.int32_data().size() == 1
                        i32 = tensor.int32_data().at(0);
                    }

                    onnx::AttributeProto* attr = node->add_attribute();
                    attr->set_name(std::string(attr_name));
                    attr->set_type(onnx::AttributeProto::INT);
                    attr->set_i(i32);
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
                    if (i64 == std::numeric_limits<int64_t>::max() - 1) i64 = INT_MAX - 1;
                    if (i64 == std::numeric_limits<int64_t>::min()) i64 = INT_MIN;
                    if (i64 == std::numeric_limits<int64_t>::min() + 1) i64 = INT_MIN + 1;

                    onnx::AttributeProto* attr = node->add_attribute();
                    attr->set_name(std::string(attr_name));
                    attr->set_type(onnx::AttributeProto::INT);
                    attr->set_i((int)i64);
                }
                else if (tensor.data_type() == onnx::TensorProto::FLOAT)
                {
                    float f32;
                    if (tensor.has_raw_data())
                    {
                        // assert tensor.raw_data().size() == 4
                        f32 = ((float*)tensor.raw_data().data())[0];
                    }
                    else
                    {
                        // assert tensor.float_data().size() == 1
                        f32 = tensor.float_data().at(0);
                    }

                    onnx::AttributeProto* attr = node->add_attribute();
                    attr->set_name(std::string(attr_name));
                    attr->set_type(onnx::AttributeProto::FLOAT);
                    attr->set_f(f32);
                }
                else if (tensor.data_type() == onnx::TensorProto::BOOL)
                {
                    bool bb;
                    if (tensor.has_raw_data())
                    {
                        // assert tensor.raw_data().size() == 2
                        bb = ((uint16_t*)tensor.raw_data().data())[0] ? true : false;
                    }
                    else
                    {
                        // assert tensor.int32_data().size() == 1
                        bb = tensor.int32_data().at(0) ? true : false;
                    }

                    onnx::AttributeProto* attr = node->add_attribute();
                    attr->set_name(std::string(attr_name));
                    attr->set_type(onnx::AttributeProto::INT);
                    attr->set_i(bb ? 1 : 0);
                }
                else
                {
                    fprintf(stderr, "unknown constant scalar type %d\n", (int)tensor.data_type());
                    continue;
                }
            }
            else if (tensor.dims_size() == 1)
            {
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

                    onnx::AttributeProto* attr = node->add_attribute();
                    attr->set_name(std::string(attr_name));
                    attr->set_type(onnx::AttributeProto::INTS);
                    for (auto i32 : ai)
                    {
                        attr->add_ints(i32);
                    }
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

                    onnx::AttributeProto* attr = node->add_attribute();
                    attr->set_name(std::string(attr_name));
                    attr->set_type(onnx::AttributeProto::INTS);
                    for (auto i64 : ai)
                    {
                        if (i64 == std::numeric_limits<int64_t>::max()) i64 = INT_MAX;
                        if (i64 == std::numeric_limits<int64_t>::max() - 1) i64 = INT_MAX - 1;
                        if (i64 == std::numeric_limits<int64_t>::min()) i64 = INT_MIN;
                        if (i64 == std::numeric_limits<int64_t>::min() + 1) i64 = INT_MIN + 1;

                        attr->add_ints((int)i64);
                    }
                }
                else if (tensor.data_type() == onnx::TensorProto::FLOAT)
                {
                    std::vector<float> af(list_size);
                    if (tensor.has_raw_data())
                    {
                        // assert tensor.raw_data().size() == 4 * list_size
                        memcpy((void*)af.data(), (float*)tensor.raw_data().data(), sizeof(float) * list_size);
                    }
                    else
                    {
                        // assert tensor.float_data().size() == list_size
                        memcpy((void*)af.data(), tensor.float_data().data(), sizeof(float) * list_size);
                    }

                    onnx::AttributeProto* attr = node->add_attribute();
                    attr->set_name(std::string(attr_name));
                    attr->set_type(onnx::AttributeProto::FLOATS);
                    for (auto f32 : af)
                    {
                        attr->add_floats(f32);
                    }
                }
                else
                {
                    fprintf(stderr, "unknown constant list type %d\n", (int)tensor.data_type());
                    continue;
                }
            }
            else
            {
                // tensor type, cannot fuse it as scalar attribute
                continue;
            }

            fused_input_indexes.push_back(j);
        }

        // drop inputs
        for (int j = (int)fused_input_indexes.size() - 1; j >= 0; j--)
        {
            const int fused_input_index = fused_input_indexes[j];

            //  ..... fii .......
            const int node_input_size = node->input_size();
            for (int k = fused_input_index; k < node_input_size - 1; k++)
            {
                node->mutable_input()->SwapElements(k, k + 1);
            }

            //  ..... ....... fii
            node->mutable_input()->RemoveLast();
        }
    }

    onnx2pnnx::dead_code_elimination(model);
}

} // namespace onnx2pnnx

} // namespace pnnx
