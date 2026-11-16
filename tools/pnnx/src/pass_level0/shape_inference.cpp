// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "shape_inference.h"
#include <unordered_set>

#include "storezip.h"
#include "utils.h"
#include "pass_level0/constant_unpooling.h"
#include "pass_level0/normalize_module_dtype.h"
#include "pass_level0/flatten_input.h"
#include "pass_level0/inline_block.h"
#include "pass_level0/normalize_dtype.h"
#include "pass_level0/reset_device.h"
#include "pass_level0/shape_inference.h"

namespace pnnx {

static bool is_inplace_op(const std::string& optype)
{
    return torch_mutation_kind(optype) != TorchMutationNone;
}

static bool is_static_shape_foldable(const std::string& optype)
{
    return optype == "aten::size"
           || optype == "aten::new_empty"
           || optype == "aten::new_full"
           || optype == "aten::new_ones"
           || optype == "aten::new_zeros"
           || optype == "aten::empty_like"
           || optype == "aten::full_like"
           || optype == "aten::ones_like"
           || optype == "aten::zeros_like"
           || optype == "aten::_shape_as_tensor";
}

static void build_value_link_input_map(const torch::jit::Node* node, const std::unordered_map<std::string, torch::jit::Value*>& value_alias_map,
                                       std::unordered_map<std::string, int>& value_link_input_map, bool ignore_aten_size)
{
    std::string optype = node->kind().toDisplayString();

    if (ignore_aten_size && is_static_shape_foldable(optype))
    {
        return;
    }

    for (size_t i = 0; i < node->outputs().size(); i++)
    {
        auto out2 = node->outputs()[i];

        std::string os = out2->debugName();

        if (!os.empty() && value_link_input_map.find(os) != value_link_input_map.end())
            continue;

        auto tensor_type = out2->type()->cast<torch::jit::TensorType>();
        if (tensor_type)
        {
            value_link_input_map[os] = 1;
        }

        for (size_t j = 0; j < out2->uses().size(); j++)
        {
            auto node2 = out2->uses()[j].user;

            build_value_link_input_map(node2, value_alias_map, value_link_input_map, true);
        }
    }

    if (is_inplace_op(optype))
    {
        // infect input0 and its alias
        while (1)
        {
            auto in2 = torch_alias_input(node);

            std::string is = in2->debugName();

            if (is.empty())
                break;

            if (value_alias_map.find(is) == value_alias_map.end())
                break;

            auto in3 = value_alias_map.at(is);

            auto tensor_type = in3->type()->cast<torch::jit::TensorType>();
            if (!tensor_type)
                break;

            is = in3->debugName();

            if (value_link_input_map.find(is) != value_link_input_map.end())
                break;

            for (size_t j = 0; j < in3->uses().size(); j++)
            {
                auto node2 = in3->uses()[j].user;

                build_value_link_input_map(node2, value_alias_map, value_link_input_map, true);
            }

            break;
        }
    }
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

        std::string op_type = node->kind().toDisplayString();
        if (is_inplace_op(op_type))
        {
            // optimize me: track other inplace op inputs
            return true;
        }
    }

    return false;
}

int shape_inference(std::shared_ptr<torch::jit::Graph>& graph, const std::vector<at::Tensor>& input_tensors, const std::vector<at::Tensor>& input_tensors2,
                    const std::vector<std::string>& module_operators, const std::string& ptpath, const std::string& device, std::set<std::string>& foldable_constants, const std::string& foldable_constants_zippath)
{
    // collect all intermediate output tensors
    std::vector<std::unordered_set<std::string> > more_value_names;
    std::vector<std::vector<torch::jit::Value*> > more_values;
    {
        std::unordered_set<std::string> value_names;
        std::vector<torch::jit::Value*> values;
        for (const auto& n : graph->nodes())
        {
            for (const auto& v : n->outputs())
            {
                auto tensor_type = v->type()->cast<torch::jit::TensorType>();
                if (!tensor_type)
                    continue;

                value_names.insert(v->debugName());
                values.push_back(v);
            }

            // too many intermediate blobs in one inference results oom
            if (value_names.size() >= 1000)
            {
                more_value_names.push_back(value_names);
                value_names.clear();

                more_values.push_back(values);
                values.clear();
            }
        }

        if (value_names.size() > 0)
        {
            more_value_names.push_back(value_names);
            more_values.push_back(values);
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

    std::vector<torch::jit::IValue> inputs;
    for (size_t i = 0; i < input_tensors.size(); i++)
    {
        const at::Tensor& it = input_tensors[i];
        inputs.push_back(it.is_complex() ? it.toType(c10::ScalarType::ComplexFloat) : it.is_floating_point() ? it.toType(c10::ScalarType::Float) : it);
    }

    std::vector<torch::jit::IValue> inputs2;
    for (size_t i = 0; i < input_tensors2.size(); i++)
    {
        const at::Tensor& it = input_tensors2[i];
        inputs2.push_back(it.is_complex() ? it.toType(c10::ScalarType::ComplexFloat) : it.is_floating_point() ? it.toType(c10::ScalarType::Float) : it);
    }

    // bookkeep foldable tensors
    std::unordered_map<std::string, int> value_link_input_map;
    {
        // build value alias map for inplace op
        std::unordered_map<std::string, torch::jit::Value*> value_alias_map;

        for (const auto& n : graph->block()->nodes())
        {
            if (n->kind() == c10::prim::GetAttr)
                continue;

            if (n->kind() == c10::prim::Constant)
                continue;

            std::string optype = n->kind().toDisplayString();

            // fprintf(stderr, "optype = %s\n", optype.c_str());

            if (!is_inplace_op(optype) && torch_alias_kind(n) == TorchAliasNone)
                continue;

            if (n->inputs().size() == 0)
                continue;

            if (n->outputs().size() == 0)
                continue;

            std::string is = torch_alias_input(n)->debugName();

            if (is.empty())
                continue;

            for (size_t i = 0; i < n->outputs().size(); i++)
            {
                auto out2 = n->output(i);

                auto tensor_type = out2->type()->cast<torch::jit::TensorType>();
                if (!tensor_type)
                    continue;

                std::string os = out2->debugName();

                if (os.empty())
                    continue;

                if (value_alias_map.find(is) == value_alias_map.end())
                {
                    value_alias_map[os] = torch_alias_input(n);
                }
                else
                {
                    value_alias_map[os] = value_alias_map[is];
                }
            }
        }

        // print value_alias_map
        // for (const auto& x : value_alias_map)
        // {
        //     fprintf(stderr, "alias %s -> %s\n", x.first.c_str(), x.second->debugName().c_str());
        // }

        bool ignore_aten_size = input_tensors2.empty();
        for (size_t i = 1; i < graph->inputs().size(); i++)
        {
            auto in0 = graph->inputs()[i];

            for (size_t j = 0; j < in0->uses().size(); j++)
            {
                auto node = in0->uses()[j].user;

                build_value_link_input_map(node, value_alias_map, value_link_input_map, ignore_aten_size);
            }
        }

        // print value_link_input_map
        // for (const auto& x : value_link_input_map)
        // {
        //     fprintf(stderr, "link_input %s %d\n", x.first.c_str(), x.second);
        // }
    }

    StoreZipWriter zip;
    zip.open(foldable_constants_zippath);

    for (size_t p = 0; p < more_value_names.size(); p++)
    {
        std::unordered_set<std::string>& value_names = more_value_names[p];
        std::vector<torch::jit::Value*>& values = more_values[p];

        // auto mod2 = mod.deepcopy();

        torch::jit::Module mod2 = torch::jit::load(ptpath, (device == "gpu") ? c10::kCUDA : c10::kCPU);
        mod2.eval();

        auto method = mod2.find_method("forward");
        if (!method)
        {
            method = mod2.get_methods()[0];
        }

        auto graph2 = method->graph();

        inline_block(graph2, module_operators);

        constant_unpooling(graph2);

        flatten_input(graph2);

        if (normalize_dtype(graph2, mod2, input_tensors) != 0)
            return -1;
        reset_device(graph2, device);
        normalize_module_dtype(mod2);

        std::vector<torch::jit::Value*> values2;
        bool has_metadata_mutation = false;
        for (auto n : graph2->nodes())
        {
            const std::string type = n->kind().toDisplayString();
            if (torch_mutation_kind(type) == TorchMutationMetadata)
                has_metadata_mutation = true;

            for (const auto& v : n->outputs())
            {
                auto tensor_type = v->type()->cast<torch::jit::TensorType>();
                if (!tensor_type)
                    continue;

                if (value_names.find(v->debugName()) != value_names.end())
                {
                    values2.push_back(v);
                    // fprintf(stderr, "%s  ", v->debugName().c_str());
                }
            }
        }
        if (has_metadata_mutation)
        {
            // Keep each value's metadata at its definition, before subsequent
            // in-place shape changes. alias shares data without copying it.
            for (auto& v : values2)
            {
                auto snapshot = graph2->create(c10::Symbol::fromQualString("aten::alias"), {v});
                snapshot->output()->setType(v->type());
                snapshot->insertAfter(v->node());
                v = snapshot->output();
            }
        }
        fprintf(stderr, "\n----------------\n\n");

        // set new graph output
        torch::jit::Node* new_return_node = graph2->createTuple(at::ArrayRef<torch::jit::Value*>(values2));

        graph2->appendNode(new_return_node);

        graph2->eraseOutput(0);
        graph2->registerOutput(new_return_node->outputs()[0]);

        // construct schema for new inputs and outputs
        {
            auto oldfs = method->function().getSchema();

            std::vector<c10::Argument> arguments;
            std::vector<c10::Argument> returns;
            for (size_t i = 0; i < graph2->inputs().size(); i++)
            {
                auto v = graph2->inputs()[i];
                arguments.push_back(c10::Argument(v->debugName(), v->type()));
            }
            for (size_t i = 0; i < graph2->outputs().size(); i++)
            {
                auto v = graph2->outputs()[i];
                returns.push_back(c10::Argument(v->debugName(), v->type()));
            }

            c10::FunctionSchema newfs(oldfs.name(), oldfs.overload_name(), arguments, returns);
            method->function().setSchema(newfs);
        }

        // inference for all tensors
        auto outputs = mod2.copy().get_method(method->name())(inputs).toTuple();

        if (input_tensors2.empty())
        {
            // assign shape info
            for (size_t i = 0; i < values2.size(); i++)
            {
                auto v = values[i];
                auto t = outputs->elements()[i].toTensor();

                v->setType(c10::TensorType::create(t));

                // check if value that does not depend on inputs
                if (value_link_input_map.find(v->debugName()) == value_link_input_map.end() && value_link_output(v, g_outputs))
                {
                    // fprintf(stderr, "foldable_constant %s\n", v->debugName().c_str());
                    foldable_constants.insert(v->debugName());

                    at::Tensor t2 = t.cpu().contiguous();
                    zip.write_file(v->debugName(), (const char*)t2.data_ptr(), t2.nbytes());
                }
            }
        }
        else
        {
            // assign dynamic shape info
            auto outputs2 = mod2.copy().get_method(method->name())(inputs2).toTuple();

            fprintf(stderr, "assign dynamic shape info\n");

            for (size_t i = 0; i < values2.size(); i++)
            {
                auto v = values[i];
                auto t = outputs->elements()[i].toTensor();
                auto t2 = outputs2->elements()[i].toTensor();

                auto type1 = c10::TensorType::create(t);
                auto type2 = c10::TensorType::create(t2);

                std::vector<c10::ShapeSymbol> sizes1 = type1->symbolic_sizes().sizes().value();
                std::vector<c10::ShapeSymbol> sizes2 = type2->symbolic_sizes().sizes().value();

                bool is_shape_static = true;
                for (size_t i = 0; i < sizes1.size(); i++)
                {
                    if (sizes1[i] == sizes2[i])
                        continue;

                    sizes1[i] = c10::ShapeSymbol::fromStaticSize(-1);
                    is_shape_static = false;
                }

                auto finaltype = type1->withSymbolicShapes(c10::SymbolicShape(sizes1));

                v->setType(finaltype);

                // check if value that does not depend on inputs
                if (is_shape_static && value_link_input_map.find(v->debugName()) == value_link_input_map.end() && value_link_output(v, g_outputs))
                {
                    // fprintf(stderr, "foldable_constant %s\n", v->debugName().c_str());
                    foldable_constants.insert(v->debugName());

                    at::Tensor t2 = t.cpu().contiguous();
                    zip.write_file(v->debugName(), (const char*)t2.data_ptr(), t2.nbytes());
                }
            }
        }
    }

    zip.close();

    if (input_tensors2.empty())
    {
        for (size_t i = 0; i < input_tensors.size(); i++)
        {
            auto type = c10::TensorType::create(inputs[i].toTensor());

            graph->inputs()[1 + i]->setType(type);
        }
    }
    else
    {
        for (size_t i = 0; i < input_tensors.size(); i++)
        {
            auto type1 = c10::TensorType::create(inputs[i].toTensor());
            auto type2 = c10::TensorType::create(inputs2[i].toTensor());

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
    }
    return 0;
}

} // namespace pnnx
