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

#include "eliminate_noop.h"

#include <sstream>
#include <string>
#include <unordered_set>

#include <onnxruntime_c_api.h>

namespace pnnx {

namespace onnx2pnnx {

void eliminate_noop(onnx::ModelProto& model)
{
    onnx::GraphProto* graph = model.mutable_graph();

    for (int i = 0; i < graph->node_size(); i++)
    {
        const onnx::NodeProto& node = graph->node(i);
        const std::string& op_type = node.op_type();

        if (op_type == "Identity" || op_type == "aten_copy")
        {
            const std::string& input_name = node.input(0);
            const std::string& output_name = node.output(0);

            for (int j = i + 1; j < graph->node_size(); j++)
            {
                onnx::NodeProto* node2 = graph->mutable_node(j);

                for (int k = 0; k < node2->input_size(); k++)
                {
                    if (node2->input(k) == output_name)
                    {
                        node2->set_input(k, input_name);
                    }
                }
            }
        }
    }
}

} // namespace onnx2pnnx

} // namespace pnnx
