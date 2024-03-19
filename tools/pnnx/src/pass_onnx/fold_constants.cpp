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
#include <unordered_set>

#include <onnxruntime_c_api.h>

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

void fold_constants(onnx::ModelProto& model)
{
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

            const std::string& op_type = node.op_type();

            bool is_outputs_foldable = true;
            for (int j = 0; j < node.input_size(); j++)
            {
                if (non_foldable_outputs.find(node.input(j)) != non_foldable_outputs.end())
                {
                    is_outputs_foldable = false;
                    break;
                }
            }

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
                is_outputs_foldable = true;
            }

            // TODO whitelist for static shape
            if (op_type == "Shape")
            {
                is_outputs_foldable = true;
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
            const onnx::TensorShapeProto& tsp = value.type().tensor_type().shape();
            for (int k = 0; k < tsp.dim_size(); k++)
            {
                // TODO has_dim_value ?
                shape.push_back(tsp.dim(k).dim_value());
            }

            ONNXTensorElementDataType datatype = (ONNXTensorElementDataType)value.type().tensor_type().elem_type();

            OrtValue* ort_val = 0;
            ort_status = ort_api->CreateTensorAsOrtValue(ort_allocator, (const int64_t*)shape.data(), shape.size(), datatype, &ort_val);
            if (ort_status)
            {
                fprintf(stderr, "ort CreateTensorAsOrtValue failed %s\n", ort_api->GetErrorMessage(ort_status));
            }

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
}

} // namespace onnx2pnnx

} // namespace pnnx
