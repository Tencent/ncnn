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

#include "fold_constants.h"

#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <onnxruntime_c_api.h>

#include "dead_code_elimination.h"

namespace pnnx {

namespace onnx2pnnx {

static size_t sizeof_onnx_datatype(ONNXTensorElementDataType type)
{
    switch (type)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
        return 0;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:
        return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
        return 8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
        return 16;
    default:
        break;
    }

    return 0;
}

static ONNXTensorElementDataType get_onnx_tensor_elem_data_type(const std::string& type)
{
    if (type == "i8") return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    if (type == "u8") return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    if (type == "i16") return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    if (type == "u16") return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
    if (type == "i32") return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    if (type == "u32") return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
    if (type == "i64") return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    if (type == "u64") return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    if (type == "f16") return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    if (type == "f32") return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    if (type == "f64") return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    if (type == "bf16") return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    if (type == "c64") return ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64;
    if (type == "c128") return ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128;

    // unknown
    fprintf(stderr, "unsupported tensor elem data type %s\n", type.c_str());
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}

static void onnx_tensor_fill_random(void* ort_val_data, const std::vector<int64_t>& shape, ONNXTensorElementDataType datatype)
{
    if (shape.empty())
        return;

    int64_t n = 1;
    for (size_t i = 0; i < shape.size(); i++)
    {
        n *= shape[i];
    }

    if (datatype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    {
        float* p = (float*)ort_val_data;
        for (int64_t i = 0; i < n; i++)
        {
            p[i] = 0.1f;
        }
    }

    if (datatype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
    {
        int64_t* p = (int64_t*)ort_val_data;
        for (int64_t i = 0; i < n; i++)
        {
            p[i] = 7;
        }
    }
}

static bool check_outputs_foldable(const onnx::NodeProto& node, const std::unordered_set<std::string>& non_foldable_outputs)
{
    for (int i = 0; i < node.input_size(); i++)
    {
        if (non_foldable_outputs.find(node.input(i)) != non_foldable_outputs.end())
            return false;
    }

    // recurse subgraph
    for (int i = 0; i < node.attribute_size(); i++)
    {
        const onnx::AttributeProto& attr = node.attribute(i);

        if (attr.type() == onnx::AttributeProto::GRAPH)
        {
            const onnx::GraphProto& sg = attr.g();

            for (int j = 0; j < sg.node_size(); j++)
            {
                if (!check_outputs_foldable(sg.node(j), non_foldable_outputs))
                    return false;
            }
        }
        if (attr.type() == onnx::AttributeProto::GRAPHS)
        {
            for (int k = 0; k < attr.graphs().size(); k++)
            {
                const onnx::GraphProto& sg = attr.graphs().at(k);

                for (int j = 0; j < sg.node_size(); j++)
                {
                    if (!check_outputs_foldable(sg.node(j), non_foldable_outputs))
                        return false;
                }
            }
        }
    }

    return true;
}

void fold_constants(onnx::ModelProto& model,
                    const std::vector<std::vector<int64_t> >& input_shapes,
                    const std::vector<std::string>& input_types,
                    const std::vector<std::vector<int64_t> >& input_shapes2,
                    const std::vector<std::string>& input_types2)
{
    bool ignore_aten_size = input_shapes2.empty();

    // collect initializers
    std::unordered_set<std::string> initializers;
    {
        const onnx::GraphProto& graph = model.graph();
        for (int i = 0; i < graph.initializer_size(); i++)
        {
            initializers.insert(graph.initializer(i).name());
        }
    }

    // collect all outputs that have no links with graph inputs
    std::vector<std::string> foldable_constants;
    {
        const onnx::GraphProto& graph = model.graph();

        std::unordered_set<std::string> foldable_outputs;
        std::unordered_set<std::string> non_foldable_outputs;
        for (int i = 0; i < graph.input_size(); i++)
        {
            non_foldable_outputs.insert(graph.input(i).name());
        }

        for (int i = 0; i < graph.node_size(); i++)
        {
            const onnx::NodeProto& node = graph.node(i);

            bool is_outputs_foldable = check_outputs_foldable(node, non_foldable_outputs);

            const std::string& op_type = node.op_type();

            // TODO whitelist for static shape
            // aten::size
            // aten::_shape_as_tensor
            if (op_type == "aten_new_empty"
                    || op_type == "aten_new_full"
                    || op_type == "aten_new_ones"
                    || op_type == "aten_new_zeros"
                    || op_type == "aten_empty_like"
                    || op_type == "aten_full_like"
                    || op_type == "aten_ones_like"
                    || op_type == "aten_zeros_like")
            {
                is_outputs_foldable = ignore_aten_size;
            }

            // TODO whitelist for static shape
            if (op_type == "Shape")
            {
                is_outputs_foldable = ignore_aten_size;
            }

            // TODO whitelist for static type
            if (op_type == "CastLike")
            {
                is_outputs_foldable = non_foldable_outputs.find(node.input(0)) == non_foldable_outputs.end();
            }

            if (!is_outputs_foldable)
            {
                for (int j = 0; j < node.input_size(); j++)
                {
                    if (non_foldable_outputs.find(node.input(j)) == non_foldable_outputs.end())
                    {
                        // some input/output may have empty name, it causes trouble, skip it
                        if (node.input(j).empty())
                            continue;

                        foldable_outputs.insert(node.input(j));
                        // fprintf(stderr, "foldable_outputs %s\n", node.input(j).c_str());
                    }
                }

                for (int j = 0; j < node.output_size(); j++)
                {
                    non_foldable_outputs.insert(node.output(j));
                }
            }
        }

        // skip initializers
        for (const std::string& x : foldable_outputs)
        {
            if (initializers.find(x) == initializers.end())
            {
                foldable_constants.push_back(x);
            }
        }
    }

    if (foldable_constants.empty())
        return;

    onnx::GraphProto* graph = model.mutable_graph();

    // save original outputs
    std::vector<std::string> orig_outputs;
    {
        for (int i = 0; i < graph->output_size(); i++)
        {
            orig_outputs.push_back(graph->output(i).name());
        }
    }

    // add foldable outputs to onnx output
    {
        graph->clear_output();

        for (size_t i = 0; i < foldable_constants.size(); i++)
        {
            graph->add_output()->set_name(foldable_constants[i]);
        }
    }

    // generate temp onnx graph
    std::string tmp_onnx_data;
    {
        std::stringstream tmp_onnx_data_ss;
        if (!model.SerializeToOstream(&tmp_onnx_data_ss))
        {
            fprintf(stderr, "write onnx failed\n");
            return;
        }

        tmp_onnx_data = tmp_onnx_data_ss.str();
    }

    // onnxrt inference
    {
        const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

        OrtStatus* ort_status = 0;

        OrtEnv* ort_env = 0;
        ort_status = ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "pnnx", &ort_env);
        if (ort_status)
        {
            fprintf(stderr, "ort CreateEnv failed %s\n", ort_api->GetErrorMessage(ort_status));
        }

        OrtSessionOptions* ort_session_opt = 0;
        ort_status = ort_api->CreateSessionOptions(&ort_session_opt);
        if (ort_status)
        {
            fprintf(stderr, "ort CreateSessionOptions failed %s\n", ort_api->GetErrorMessage(ort_status));
        }

        ort_status = ort_api->SetSessionGraphOptimizationLevel(ort_session_opt, ORT_DISABLE_ALL);
        if (ort_status)
        {
            fprintf(stderr, "ort SetSessionGraphOptimizationLevel failed %s\n", ort_api->GetErrorMessage(ort_status));
        }

        // ort_status = ort_api->SetIntraOpNumThreads(ort_session_opt, 4);
        // if (ort_status)
        // {
        //     fprintf(stderr, "ort SetIntraOpNumThreads failed %s\n", ort_api->GetErrorMessage(ort_status));
        // }
        //
        // ort_status = ort_api->SetInterOpNumThreads(ort_session_opt, 4);
        // if (ort_status)
        // {
        //     fprintf(stderr, "ort SetInterOpNumThreads failed %s\n", ort_api->GetErrorMessage(ort_status));
        // }

        OrtSession* ort_session = 0;
        ort_status = ort_api->CreateSessionFromArray(ort_env, (const void*)tmp_onnx_data.data(), tmp_onnx_data.size(), ort_session_opt, &ort_session);
        if (ort_status)
        {
            fprintf(stderr, "ort CreateSession failed %s\n", ort_api->GetErrorMessage(ort_status));
        }

        OrtRunOptions* ort_run_opt = 0;
        ort_status = ort_api->CreateRunOptions(&ort_run_opt);
        if (ort_status)
        {
            fprintf(stderr, "ort CreateRunOptions failed %s\n", ort_api->GetErrorMessage(ort_status));
        }

        OrtAllocator* ort_allocator = 0;
        ort_status = ort_api->GetAllocatorWithDefaultOptions(&ort_allocator);
        if (ort_status)
        {
            fprintf(stderr, "ort GetAllocatorWithDefaultOptions failed %s\n", ort_api->GetErrorMessage(ort_status));
        }

        std::vector<const char*> input_names;
        std::vector<OrtValue*> inputs;
        for (int i = 0; i < graph->input_size(); i++)
        {
            const onnx::ValueInfoProto& value = graph->input(i);

            std::vector<int64_t> shape;
            ONNXTensorElementDataType datatype;
            if (!input_shapes.empty())
            {
                shape = input_shapes[i];
                datatype = get_onnx_tensor_elem_data_type(input_types[i]);
            }
            else
            {
                const onnx::TensorShapeProto& tsp = value.type().tensor_type().shape();
                for (int k = 0; k < tsp.dim_size(); k++)
                {
                    shape.push_back(tsp.dim(k).dim_value());
                }

                datatype = (ONNXTensorElementDataType)value.type().tensor_type().elem_type();
            }

            OrtValue* ort_val = 0;
            ort_status = ort_api->CreateTensorAsOrtValue(ort_allocator, (const int64_t*)shape.data(), shape.size(), datatype, &ort_val);
            if (ort_status)
            {
                fprintf(stderr, "ort CreateTensorAsOrtValue failed %s\n", ort_api->GetErrorMessage(ort_status));
            }

            void* ort_val_data = 0;
            ort_status = ort_api->GetTensorMutableData(ort_val, &ort_val_data);
            if (ort_status)
            {
                fprintf(stderr, "ort GetTensorMutableData failed %s\n", ort_api->GetErrorMessage(ort_status));
            }
            onnx_tensor_fill_random(ort_val_data, shape, datatype);

            input_names.push_back(value.name().c_str());
            inputs.push_back(ort_val);
        }

        std::vector<const char*> output_names;
        std::vector<OrtValue*> outputs;
        for (size_t i = 0; i < foldable_constants.size(); i++)
        {
            output_names.push_back(foldable_constants[i].c_str());
            outputs.push_back(0);
        }

        ort_status = ort_api->Run(ort_session, ort_run_opt,
                                  input_names.data(), inputs.data(), input_names.size(),
                                  output_names.data(), output_names.size(), outputs.data());
        if (ort_status)
        {
            fprintf(stderr, "ort Run failed %s\n", ort_api->GetErrorMessage(ort_status));
        }

        // TODO get output data

        // graph->clear_output();

        for (size_t i = 0; i < output_names.size(); i++)
        {
            OrtTensorTypeAndShapeInfo* info = 0;
            ort_status = ort_api->GetTensorTypeAndShape(outputs[i], &info);
            if (ort_status)
            {
                fprintf(stderr, "ort GetTensorTypeAndShape failed %s\n", ort_api->GetErrorMessage(ort_status));
            }

            ONNXTensorElementDataType datatype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            ort_status = ort_api->GetTensorElementType(info, &datatype);
            if (ort_status)
            {
                fprintf(stderr, "ort GetTensorElementType failed %s\n", ort_api->GetErrorMessage(ort_status));
            }

            size_t out_dims = 0;
            ort_status = ort_api->GetDimensionsCount(info, &out_dims);
            if (ort_status)
            {
                fprintf(stderr, "ort GetDimensionsCount failed %s\n", ort_api->GetErrorMessage(ort_status));
            }

            // fprintf(stderr, "   out_dims = %lu\n", out_dims);

            std::vector<int64_t> out_shape;
            out_shape.resize(out_dims);
            ort_status = ort_api->GetDimensions(info, out_shape.data(), out_dims);
            if (ort_status)
            {
                fprintf(stderr, "ort GetDimensions failed %s\n", ort_api->GetErrorMessage(ort_status));
            }

            void* tensor_data = 0;
            ort_status = ort_api->GetTensorMutableData(outputs[i], &tensor_data);
            if (ort_status)
            {
                fprintf(stderr, "ort GetTensorMutableData failed %s\n", ort_api->GetErrorMessage(ort_status));
            }

            size_t elemcount = 0;
            ort_status = ort_api->GetTensorShapeElementCount(info, &elemcount);
            if (ort_status)
            {
                fprintf(stderr, "ort GetTensorShapeElementCount failed %s\n", ort_api->GetErrorMessage(ort_status));
            }

            // fprintf(stderr, "%16s = ", output_names[i]);
            // for (size_t j = 0; j < out_dims; j++)
            // {
            //     fprintf(stderr, "%lu ", out_shape[j]);
            // }
            // fprintf(stderr, "\n");

            // unlink any node that has this output
            {
                for (int j = 0; j < graph->node_size(); j++)
                {
                    const onnx::NodeProto& node = graph->node(j);

                    bool is_producer = false;
                    int producer_node_output_index = -1;
                    for (int k = 0; k < node.output_size(); k++)
                    {
                        if (node.output(k) == output_names[i])
                        {
                            is_producer = true;
                            producer_node_output_index = k;
                            break;
                        }
                    }

                    if (is_producer)
                    {
                        graph->mutable_node(j)->set_output(producer_node_output_index, std::string("pnnx_unlink_") + output_names[i]);
                        break;
                    }
                }
            }

            // create initializer
            {
                onnx::TensorProto* tp = graph->add_initializer();
                tp->set_name(output_names[i]);

                for (size_t j = 0; j < out_dims; j++)
                {
                    tp->add_dims(out_shape[j]);
                }

                tp->set_data_type((int32_t)datatype);

                std::string* data = tp->mutable_raw_data();
                data->resize(sizeof_onnx_datatype(datatype) * elemcount);
                memcpy((void*)data->data(), tensor_data, sizeof_onnx_datatype(datatype) * elemcount);
            }

            ort_api->ReleaseTensorTypeAndShapeInfo(info);
        }

        for (size_t i = 0; i < input_names.size(); i++)
        {
            ort_api->ReleaseValue(inputs[i]);
        }

        for (size_t i = 0; i < output_names.size(); i++)
        {
            ort_api->ReleaseValue(outputs[i]);
        }

        ort_api->ReleaseRunOptions(ort_run_opt);
        ort_api->ReleaseSession(ort_session);
        ort_api->ReleaseSessionOptions(ort_session_opt);
        ort_api->ReleaseEnv(ort_env);
    }

    // restore original outputs
    {
        graph->clear_output();

        for (size_t i = 0; i < orig_outputs.size(); i++)
        {
            graph->add_output()->set_name(orig_outputs[i]);
        }
    }

    onnx2pnnx::dead_code_elimination(model);
}

void fold_constants_dynamic_shape(onnx::ModelProto& model,
                                  const std::vector<std::vector<int64_t> >& input_shapes,
                                  const std::vector<std::string>& input_types)
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

    // collect all outputs that have no links with graph inputs
    std::vector<std::string> foldable_constants;
    {
        const onnx::GraphProto& graph = model.graph();

        std::unordered_set<std::string> foldable_outputs;
        std::unordered_set<std::string> non_foldable_outputs;
        for (int i = 0; i < graph.input_size(); i++)
        {
            non_foldable_outputs.insert(graph.input(i).name());
        }

        for (int i = 0; i < graph.node_size(); i++)
        {
            const onnx::NodeProto& node = graph.node(i);

            bool is_outputs_foldable = check_outputs_foldable(node, non_foldable_outputs);

            const std::string& op_type = node.op_type();

            if (op_type == "Slice")
            {
                // match Shape + Slice pattern for partial static shape
                const std::string& input = node.input(0);
                bool is_producer_shape = false;
                int producer_node_output_index = -1;
                if (!input.empty() && initializers.find(input) == initializers.end())
                {
                    for (int j = 0; j < i; j++)
                    {
                        const onnx::NodeProto& node0 = graph.node(j);

                        for (int k = 0; k < node0.output_size(); k++)
                        {
                            if (node0.output(k) == input)
                            {
                                producer_node_output_index = j;
                                break;
                            }
                        }

                        if (producer_node_output_index != -1)
                        {
                            const onnx::NodeProto& node0 = graph.node(producer_node_output_index);
                            is_producer_shape = node0.op_type() == "Shape";
                            break;
                        }
                    }
                }

                if (is_producer_shape)
                {
                    // get shape info
                    const onnx::NodeProto& node0 = graph.node(producer_node_output_index);

                    int value_info_index = -1;
                    bool value_is_graph_input = false;
                    for (int j = 0; j < graph.input_size(); j++)
                    {
                        if (graph.input(j).name() == node0.input(0))
                        {
                            value_info_index = j;
                            value_is_graph_input = true;
                            break;
                        }
                    }
                    if (value_info_index == -1)
                    {
                        for (int j = 0; j < graph.value_info_size(); j++)
                        {
                            if (graph.value_info(j).name() == node0.input(0))
                            {
                                value_info_index = j;
                                break;
                            }
                        }
                    }

                    std::vector<int> shape;
                    if (value_info_index != -1)
                    {
                        const onnx::ValueInfoProto& value = value_is_graph_input ? graph.input(value_info_index) : graph.value_info(value_info_index);
                        const onnx::TensorShapeProto& tsp = value.type().tensor_type().shape();
                        shape.resize(tsp.dim_size());
                        for (int j = 0; j < tsp.dim_size(); j++)
                        {
                            if (tsp.dim(j).has_dim_value())
                            {
                                shape[j] = tsp.dim(j).dim_value();
                            }
                            else
                            {
                                shape[j] = -1;
                            }
                        }
                    }

                    if (!shape.empty())
                    {
                        std::vector<int> slice_args;
                        for (int j = 1; j < node.input_size(); j++)
                        {
                            if (initializers.find(node.input(j)) != initializers.end())
                            {
                                const onnx::TensorProto& tensor = graph.initializer(initializers.at(node.input(j)));
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
                                slice_args.push_back((int)i64);
                            }
                        }

                        // check slice dim=0 step=1
                        if (slice_args.size() == 2 || (slice_args.size() == 3 && slice_args[2] == 0) || (slice_args.size() == 4 && slice_args[2] == 0 && slice_args[3] == 1))
                        {
                            int start = slice_args[0];
                            int end = slice_args[1];

                            is_outputs_foldable = true;
                            for (int j = start; j < end; j++)
                            {
                                if (shape[j] == -1)
                                {
                                    is_outputs_foldable = false;
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            if (op_type == "Gather")
            {
                // match Shape + Gather pattern for partial static shape
                const std::string& input = node.input(0);
                bool is_producer_shape = false;
                int producer_node_output_index = -1;
                if (!input.empty() && initializers.find(input) == initializers.end())
                {
                    for (int j = 0; j < i; j++)
                    {
                        const onnx::NodeProto& node0 = graph.node(j);

                        for (int k = 0; k < node0.output_size(); k++)
                        {
                            if (node0.output(k) == input)
                            {
                                producer_node_output_index = j;
                                break;
                            }
                        }

                        if (producer_node_output_index != -1)
                        {
                            const onnx::NodeProto& node0 = graph.node(producer_node_output_index);
                            is_producer_shape = node0.op_type() == "Shape";
                            break;
                        }
                    }
                }

                if (is_producer_shape)
                {
                    // get shape info
                    const onnx::NodeProto& node0 = graph.node(producer_node_output_index);

                    int value_info_index = -1;
                    bool value_is_graph_input = false;
                    for (int j = 0; j < graph.input_size(); j++)
                    {
                        if (graph.input(j).name() == node0.input(0))
                        {
                            value_info_index = j;
                            value_is_graph_input = true;
                            break;
                        }
                    }
                    if (value_info_index == -1)
                    {
                        for (int j = 0; j < graph.value_info_size(); j++)
                        {
                            if (graph.value_info(j).name() == node0.input(0))
                            {
                                value_info_index = j;
                                break;
                            }
                        }
                    }

                    std::vector<int> shape;
                    if (value_info_index != -1)
                    {
                        const onnx::ValueInfoProto& value = value_is_graph_input ? graph.input(value_info_index) : graph.value_info(value_info_index);
                        const onnx::TensorShapeProto& tsp = value.type().tensor_type().shape();
                        shape.resize(tsp.dim_size());
                        for (int j = 0; j < tsp.dim_size(); j++)
                        {
                            if (tsp.dim(j).has_dim_value())
                            {
                                shape[j] = tsp.dim(j).dim_value();
                            }
                            else
                            {
                                shape[j] = -1;
                            }
                        }
                    }

                    if (!shape.empty())
                    {
                        if (initializers.find(node.input(1)) != initializers.end())
                        {
                            const onnx::TensorProto& tensor = graph.initializer(initializers.at(node.input(1)));
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
                            int gather_indices = (int)i64;

                            is_outputs_foldable = true;

                            if (shape[gather_indices] == -1)
                            {
                                is_outputs_foldable = false;
                            }
                        }
                    }
                }
            }

            if (!is_outputs_foldable)
            {
                for (int j = 0; j < node.input_size(); j++)
                {
                    if (non_foldable_outputs.find(node.input(j)) == non_foldable_outputs.end())
                    {
                        // some input/output may have empty name, it causes trouble, skip it
                        if (node.input(j).empty())
                            continue;

                        foldable_outputs.insert(node.input(j));
                    }
                }

                for (int j = 0; j < node.output_size(); j++)
                {
                    non_foldable_outputs.insert(node.output(j));
                }
            }
        }

        // skip initializers
        for (const std::string& x : foldable_outputs)
        {
            if (initializers.find(x) == initializers.end())
            {
                foldable_constants.push_back(x);
            }
        }
    }

    if (foldable_constants.empty())
        return;

    onnx::GraphProto* graph = model.mutable_graph();

    // save original outputs
    std::vector<std::string> orig_outputs;
    {
        for (int i = 0; i < graph->output_size(); i++)
        {
            orig_outputs.push_back(graph->output(i).name());
        }
    }

    // add foldable outputs to onnx output
    {
        graph->clear_output();

        for (size_t i = 0; i < foldable_constants.size(); i++)
        {
            graph->add_output()->set_name(foldable_constants[i]);
        }
    }

    // generate temp onnx graph
    std::string tmp_onnx_data;
    {
        std::stringstream tmp_onnx_data_ss;
        if (!model.SerializeToOstream(&tmp_onnx_data_ss))
        {
            fprintf(stderr, "write onnx failed\n");
            return;
        }

        tmp_onnx_data = tmp_onnx_data_ss.str();
    }

    // onnxrt inference
    {
        const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

        OrtStatus* ort_status = 0;

        OrtEnv* ort_env = 0;
        ort_status = ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "pnnx", &ort_env);
        if (ort_status)
        {
            fprintf(stderr, "ort CreateEnv failed %s\n", ort_api->GetErrorMessage(ort_status));
        }

        OrtSessionOptions* ort_session_opt = 0;
        ort_status = ort_api->CreateSessionOptions(&ort_session_opt);
        if (ort_status)
        {
            fprintf(stderr, "ort CreateSessionOptions failed %s\n", ort_api->GetErrorMessage(ort_status));
        }

        ort_status = ort_api->SetSessionGraphOptimizationLevel(ort_session_opt, ORT_DISABLE_ALL);
        if (ort_status)
        {
            fprintf(stderr, "ort SetSessionGraphOptimizationLevel failed %s\n", ort_api->GetErrorMessage(ort_status));
        }

        // ort_status = ort_api->SetIntraOpNumThreads(ort_session_opt, 4);
        // if (ort_status)
        // {
        //     fprintf(stderr, "ort SetIntraOpNumThreads failed %s\n", ort_api->GetErrorMessage(ort_status));
        // }
        //
        // ort_status = ort_api->SetInterOpNumThreads(ort_session_opt, 4);
        // if (ort_status)
        // {
        //     fprintf(stderr, "ort SetInterOpNumThreads failed %s\n", ort_api->GetErrorMessage(ort_status));
        // }

        OrtSession* ort_session = 0;
        ort_status = ort_api->CreateSessionFromArray(ort_env, (const void*)tmp_onnx_data.data(), tmp_onnx_data.size(), ort_session_opt, &ort_session);
        if (ort_status)
        {
            fprintf(stderr, "ort CreateSession failed %s\n", ort_api->GetErrorMessage(ort_status));
        }

        OrtRunOptions* ort_run_opt = 0;
        ort_status = ort_api->CreateRunOptions(&ort_run_opt);
        if (ort_status)
        {
            fprintf(stderr, "ort CreateRunOptions failed %s\n", ort_api->GetErrorMessage(ort_status));
        }

        OrtAllocator* ort_allocator = 0;
        ort_status = ort_api->GetAllocatorWithDefaultOptions(&ort_allocator);
        if (ort_status)
        {
            fprintf(stderr, "ort GetAllocatorWithDefaultOptions failed %s\n", ort_api->GetErrorMessage(ort_status));
        }

        std::vector<const char*> input_names;
        std::vector<OrtValue*> inputs;
        for (int i = 0; i < graph->input_size(); i++)
        {
            const onnx::ValueInfoProto& value = graph->input(i);

            std::vector<int64_t> shape;
            ONNXTensorElementDataType datatype;
            if (!input_shapes.empty())
            {
                shape = input_shapes[i];
                datatype = get_onnx_tensor_elem_data_type(input_types[i]);
            }
            else
            {
                const onnx::TensorShapeProto& tsp = value.type().tensor_type().shape();
                for (int k = 0; k < tsp.dim_size(); k++)
                {
                    shape.push_back(tsp.dim(k).dim_value());
                }

                datatype = (ONNXTensorElementDataType)value.type().tensor_type().elem_type();
            }

            OrtValue* ort_val = 0;
            ort_status = ort_api->CreateTensorAsOrtValue(ort_allocator, (const int64_t*)shape.data(), shape.size(), datatype, &ort_val);
            if (ort_status)
            {
                fprintf(stderr, "ort CreateTensorAsOrtValue failed %s\n", ort_api->GetErrorMessage(ort_status));
            }

            void* ort_val_data = 0;
            ort_status = ort_api->GetTensorMutableData(ort_val, &ort_val_data);
            if (ort_status)
            {
                fprintf(stderr, "ort GetTensorMutableData failed %s\n", ort_api->GetErrorMessage(ort_status));
            }
            onnx_tensor_fill_random(ort_val_data, shape, datatype);

            input_names.push_back(value.name().c_str());
            inputs.push_back(ort_val);
        }

        std::vector<const char*> output_names;
        std::vector<OrtValue*> outputs;
        for (size_t i = 0; i < foldable_constants.size(); i++)
        {
            output_names.push_back(foldable_constants[i].c_str());
            outputs.push_back(0);
        }

        ort_status = ort_api->Run(ort_session, ort_run_opt,
                                  input_names.data(), inputs.data(), input_names.size(),
                                  output_names.data(), output_names.size(), outputs.data());
        if (ort_status)
        {
            fprintf(stderr, "ort Run failed %s\n", ort_api->GetErrorMessage(ort_status));
        }

        // TODO get output data

        // graph->clear_output();

        for (size_t i = 0; i < output_names.size(); i++)
        {
            OrtTensorTypeAndShapeInfo* info = 0;
            ort_status = ort_api->GetTensorTypeAndShape(outputs[i], &info);
            if (ort_status)
            {
                fprintf(stderr, "ort GetTensorTypeAndShape failed %s\n", ort_api->GetErrorMessage(ort_status));
            }

            ONNXTensorElementDataType datatype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            ort_status = ort_api->GetTensorElementType(info, &datatype);
            if (ort_status)
            {
                fprintf(stderr, "ort GetTensorElementType failed %s\n", ort_api->GetErrorMessage(ort_status));
            }

            size_t out_dims = 0;
            ort_status = ort_api->GetDimensionsCount(info, &out_dims);
            if (ort_status)
            {
                fprintf(stderr, "ort GetDimensionsCount failed %s\n", ort_api->GetErrorMessage(ort_status));
            }

            // fprintf(stderr, "   out_dims = %lu\n", out_dims);

            std::vector<int64_t> out_shape;
            out_shape.resize(out_dims);
            ort_status = ort_api->GetDimensions(info, out_shape.data(), out_dims);
            if (ort_status)
            {
                fprintf(stderr, "ort GetDimensions failed %s\n", ort_api->GetErrorMessage(ort_status));
            }

            void* tensor_data = 0;
            ort_status = ort_api->GetTensorMutableData(outputs[i], &tensor_data);
            if (ort_status)
            {
                fprintf(stderr, "ort GetTensorMutableData failed %s\n", ort_api->GetErrorMessage(ort_status));
            }

            size_t elemcount = 0;
            ort_status = ort_api->GetTensorShapeElementCount(info, &elemcount);
            if (ort_status)
            {
                fprintf(stderr, "ort GetTensorShapeElementCount failed %s\n", ort_api->GetErrorMessage(ort_status));
            }

            // fprintf(stderr, "%16s = ", output_names[i]);
            // for (size_t j = 0; j < out_dims; j++)
            // {
            //     fprintf(stderr, "%lu ", out_shape[j]);
            // }
            // fprintf(stderr, "\n");

            // unlink any node that has this output
            {
                for (int j = 0; j < graph->node_size(); j++)
                {
                    const onnx::NodeProto& node = graph->node(j);

                    bool is_producer = false;
                    int producer_node_output_index = -1;
                    for (int k = 0; k < node.output_size(); k++)
                    {
                        if (node.output(k) == output_names[i])
                        {
                            is_producer = true;
                            producer_node_output_index = k;
                            break;
                        }
                    }

                    if (is_producer)
                    {
                        graph->mutable_node(j)->set_output(producer_node_output_index, std::string("pnnx_unlink_") + output_names[i]);
                        break;
                    }
                }
            }

            // create initializer
            {
                onnx::TensorProto* tp = graph->add_initializer();
                tp->set_name(output_names[i]);

                for (size_t j = 0; j < out_dims; j++)
                {
                    tp->add_dims(out_shape[j]);
                }

                tp->set_data_type((int32_t)datatype);

                std::string* data = tp->mutable_raw_data();
                data->resize(sizeof_onnx_datatype(datatype) * elemcount);
                memcpy((void*)data->data(), tensor_data, sizeof_onnx_datatype(datatype) * elemcount);
            }

            ort_api->ReleaseTensorTypeAndShapeInfo(info);
        }

        for (size_t i = 0; i < input_names.size(); i++)
        {
            ort_api->ReleaseValue(inputs[i]);
        }

        for (size_t i = 0; i < output_names.size(); i++)
        {
            ort_api->ReleaseValue(outputs[i]);
        }

        ort_api->ReleaseRunOptions(ort_run_opt);
        ort_api->ReleaseSession(ort_session);
        ort_api->ReleaseSessionOptions(ort_session_opt);
        ort_api->ReleaseEnv(ort_env);
    }

    // restore original outputs
    {
        graph->clear_output();

        for (size_t i = 0; i < orig_outputs.size(); i++)
        {
            graph->add_output()->set_name(orig_outputs[i]);
        }
    }

    onnx2pnnx::dead_code_elimination(model);
}

} // namespace onnx2pnnx

} // namespace pnnx
