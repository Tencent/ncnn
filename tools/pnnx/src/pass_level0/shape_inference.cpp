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

#include "shape_inference.h"

namespace pnnx {

void shape_inference(const torch::jit::Module& mod, std::shared_ptr<torch::jit::Graph>& graph, const std::vector<at::Tensor>& input_tensors)
{
    // collect all intermediate output tensors
    std::vector<torch::jit::Value*> values;
    for (const auto& n : graph->nodes())
    {
        for (const auto& on : n->outputs())
        {
            auto tensor_type = on->type()->cast<torch::jit::TensorType>();
            if (!tensor_type)
                continue;

            values.push_back(on);
        }
    }

    // set new graph output
    auto old_output = graph->outputs()[0];

    torch::jit::Node* new_return_node = graph->createTuple(at::ArrayRef(values));

    graph->appendNode(new_return_node);

    graph->eraseOutput(0);
    graph->registerOutput(new_return_node->outputs()[0]);

    // inference for all tensors
    std::vector<torch::jit::IValue> inputs;
    for (size_t i = 0; i < input_tensors.size(); i++)
    {
        const at::Tensor& it = input_tensors[i];

        inputs.push_back(it);
        graph->inputs()[1 + i]->setType(c10::TensorType::create(it));
    }

    auto outputs = mod.copy().forward(inputs).toTuple();

    // assign shape info
    int index = 0;
    for (auto e : outputs->elements())
    {
        values[index]->setType(c10::TensorType::create(e.toTensor()));

        index++;
    }

    // restore old graph output
    graph->eraseOutput(0);
    graph->registerOutput(old_output);

    new_return_node->removeAllInputs();
    new_return_node->destroy();
}

} // namespace pnnx
