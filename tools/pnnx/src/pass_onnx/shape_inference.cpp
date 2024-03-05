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

#include "shape_inference.h"

#include <sstream>
#include <string>
#include <vector>

#include <onnxruntime_c_api.h>

namespace pnnx {

namespace onnx2pnnx {

static bool string_starts_with(const std::string& s, const std::string& s2)
{
    return strncmp(s.c_str(), s2.c_str(), s2.size()) == 0;
}

void shape_inference(onnx::ModelProto& model)
{
    onnx::GraphProto* graph = model.mutable_graph();

    // save original outputs
    std::vector<std::string> orig_outputs;
    {
        for (int i = 0; i < graph->output_size(); i++)
        {
            orig_outputs.push_back(graph->output(i).name());
        }
    }

    // collect intermediates
    std::vector<std::string> intermediates;
    {
        for (int i = 0; i < graph->node_size(); i++)
        {
            const onnx::NodeProto& node = graph->node(i);

            const std::string& op_type = node.op_type();

            // blacklist some glues
            if (op_type == "Constant")
                continue;

            // TODO fuse cat
            if (op_type == "SequenceConstruct")
                continue;

            // TODO fuse chunk/tensor_split
            if (op_type == "aten_split")
                continue;

            if (node.domain().empty() || string_starts_with(op_type, "nn_") || string_starts_with(op_type, "aten_") || string_starts_with(op_type, "_aten_"))
            {
                for (int j = 0; j < node.output_size(); j++)
                {
                    // some input/output may have empty name, it causes trouble, skip it
                    if (node.output(j).empty())
                        continue;

                    intermediates.push_back(node.output(j));
                }
            }
        }
    }

    // add intermediates to onnx output
    {
        graph->clear_output();

        for (size_t i = 0; i < intermediates.size(); i++)
        {
            graph->add_output()->set_name(intermediates[i]);
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
        for (size_t i = 0; i < intermediates.size(); i++)
        {
            output_names.push_back(intermediates[i].c_str());
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

        graph->clear_output();

        for (size_t i = 0; i < output_names.size(); i++)
        {
            OrtTypeInfo* type_info = 0;
            ort_status = ort_api->GetTypeInfo(outputs[i], &type_info);
            if (ort_status)
            {
                fprintf(stderr, "ort GetTypeInfo failed %s\n", ort_api->GetErrorMessage(ort_status));
            }

            ONNXType type = ONNX_TYPE_UNKNOWN;
            if (type_info)
            {
                ort_status = ort_api->GetOnnxTypeFromTypeInfo(type_info, &type);
                if (ort_status)
                {
                    fprintf(stderr, "ort GetOnnxTypeFromTypeInfo failed %s\n", ort_api->GetErrorMessage(ort_status));
                }
            }

            if (type == ONNX_TYPE_TENSOR)
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

                // fprintf(stderr, "%16s = ", output_names[i]);
                // for (size_t j = 0; j < out_dims; j++)
                // {
                //     fprintf(stderr, "%lu ", out_shape[j]);
                // }
                // fprintf(stderr, "\n");

                // assign value info
                {
                    onnx::ValueInfoProto* value = 0;

                    // maybe output
                    for (size_t j = 0; j < orig_outputs.size(); j++)
                    {
                        if (orig_outputs[j] == output_names[i])
                        {
                            value = graph->add_output();
                            value->set_name(output_names[i]);
                            break;
                        }
                    }
                    if (!value)
                    {
                        for (int j = 0; j < graph->value_info_size(); j++)
                        {
                            if (graph->mutable_value_info(j)->name() == output_names[i])
                            {
                                value = graph->mutable_value_info(j);
                                break;
                            }
                        }
                        if (!value)
                        {
                            value = graph->add_value_info();
                            value->set_name(output_names[i]);
                        }
                    }

                    // fprintf(stderr, "assign value info %s\n", value->name().c_str());

                    value->mutable_type()->mutable_tensor_type()->set_elem_type((int32_t)datatype);

                    onnx::TensorShapeProto* tsp = value->mutable_type()->mutable_tensor_type()->mutable_shape();

                    tsp->clear_dim();
                    for (size_t j = 0; j < out_dims; j++)
                    {
                        tsp->add_dim()->set_dim_value(out_shape[j]);
                    }
                }

                ort_api->ReleaseTensorTypeAndShapeInfo(info);
            }

            if (type_info)
            {
                ort_api->ReleaseTypeInfo(type_info);
            }
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
}

} // namespace onnx2pnnx

} // namespace pnnx
