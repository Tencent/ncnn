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

static bool value_link_input(const torch::jit::Value* v, const std::vector<torch::jit::Value*>& inputs)
{
    for (auto x : inputs)
    {
        if (v == x)
            return true;
    }

    for (size_t i = 0; i < v->node()->inputs().size(); i++)
    {
        bool link = value_link_input(v->node()->inputs()[i], inputs);
        if (link)
            return true;
    }

    return false;
}

static bool value_link_output(const torch::jit::Value* v, const std::vector<torch::jit::Value*>& outputs)
{
    for (auto x : outputs)
    {
        if (v == x)
            return true;
    }

    for (size_t i = 0; i < v->uses().size(); i++)
    {
        auto node = v->uses()[i].user;
        for (auto x : node->outputs())
        {
            bool link = value_link_output(x, outputs);
            if (link)
                return true;
        }
    }

    return false;
}

void shape_inference(const torch::jit::Module& mod, std::shared_ptr<torch::jit::Graph>& graph, const std::vector<at::Tensor>& input_tensors, const std::vector<at::Tensor>& input_tensors2, std::map<std::string, Attribute>& foldable_constants)
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

    // collect graph inputs outputs
    std::vector<torch::jit::Value*> g_inputs;
    for (size_t i = 1; i < graph->inputs().size(); i++)
    {
        g_inputs.push_back(graph->inputs()[i]);
    }
    std::vector<torch::jit::Value*> g_outputs;
    for (size_t i = 0; i < graph->outputs().size(); i++)
    {
        g_outputs.push_back(graph->outputs()[i]);
    }

    // set new graph output
    auto old_output = graph->outputs()[0];

    torch::jit::Node* new_return_node = graph->createTuple(at::ArrayRef<torch::jit::Value*>(values));

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

    std::map<torch::jit::Value*, at::Tensor> output_tensors;

    if (input_tensors2.empty())
    {
        // assign shape info
        int index = 0;
        for (auto e : outputs->elements())
        {
            auto v = values[index];
            v->setType(c10::TensorType::create(e.toTensor()));

            // check if value that does not depend on inputs
            if (!value_link_input(v, g_inputs) && value_link_output(v, g_outputs))
            {
                output_tensors[v] = e.toTensor();
            }

            index++;
        }
    }
    else
    {
        std::vector<torch::jit::IValue> inputs2;
        for (size_t i = 0; i < input_tensors2.size(); i++)
        {
            const at::Tensor& it = input_tensors2[i];

            inputs2.push_back(it);
            graph->inputs()[1 + i]->setType(c10::TensorType::create(it));
        }

        auto outputs2 = mod.copy().forward(inputs2).toTuple();

        fprintf(stderr, "assign dynamic shape info\n");

        // assign dynamic shape info
        for (size_t i = 0; i < input_tensors.size(); i++)
        {
            auto type1 = c10::TensorType::create(input_tensors[i]);
            auto type2 = c10::TensorType::create(input_tensors2[i]);

            std::vector<c10::ShapeSymbol> sizes1 = type1->symbolic_sizes().sizes().value();
            std::vector<c10::ShapeSymbol> sizes2 = type2->symbolic_sizes().sizes().value();

            for (size_t i = 0; i < sizes1.size(); i++)
            {
                if (sizes1[i] == sizes2[i])
                    continue;

                sizes1[i] = c10::ShapeSymbol::fromStaticSize(-1);
            }

            auto finaltype = type1->withSymbolicShapes(c10::SymbolicShape(sizes1));

            graph->inputs()[1 + i]->setType(finaltype);
        }

        int index = 0;
        for (auto e : outputs->elements())
        {
            auto type1 = c10::TensorType::create(e.toTensor());
            auto type2 = c10::TensorType::create(outputs2->elements()[index].toTensor());

            std::vector<c10::ShapeSymbol> sizes1 = type1->symbolic_sizes().sizes().value();
            std::vector<c10::ShapeSymbol> sizes2 = type2->symbolic_sizes().sizes().value();

            for (size_t i = 0; i < sizes1.size(); i++)
            {
                if (sizes1[i] == sizes2[i])
                    continue;

                sizes1[i] = c10::ShapeSymbol::fromStaticSize(-1);
            }

            auto finaltype = type1->withSymbolicShapes(c10::SymbolicShape(sizes1));

            auto v = values[index];
            v->setType(finaltype);

            // check if value that does not depend on inputs
            if (!value_link_input(v, g_inputs) && value_link_output(v, g_outputs))
            {
                output_tensors[v] = e.toTensor();
            }

            index++;
        }
    }

    for (auto xx : output_tensors)
    {
        auto v = xx.first;
        auto tensor = xx.second;

        bool link_to_output = false;
        for (size_t i = 0; i < v->uses().size(); i++)
        {
            auto node = v->uses()[i].user;
            for (auto x : node->outputs())
            {
                if (output_tensors.find(x) == output_tensors.end() && x != new_return_node->outputs()[0])
                {
                    link_to_output = true;
                    break;
                }
            }
        }

        const int ndim = (int)tensor.dim();
        if (link_to_output && ndim > 0)
        {
            foldable_constants[v->debugName()] = Attribute(tensor);
        }
    }

    // restore old graph output
    graph->eraseOutput(0);
    graph->registerOutput(old_output);

    new_return_node->removeAllInputs();
    new_return_node->destroy();
}

} // namespace pnnx
