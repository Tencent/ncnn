// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "normalize_dtype.h"

#include "torch_alias.h"

#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/shape_analysis.h>

#include <map>
#include <set>

namespace pnnx {

static c10::ScalarType normalized_scalar_type(c10::ScalarType type)
{
    if (c10::isFloatingType(type))
        return c10::ScalarType::Float;
    if (c10::isComplexType(type))
        return c10::ScalarType::ComplexFloat;
    return type;
}

static c10::IValue original_attribute(const torch::jit::Value* v, const torch::jit::Module& mod)
{
    const torch::jit::Node* n = v->node();
    if (n->kind() == c10::prim::Param && v == n->owningGraph()->inputs()[0])
        return mod._ivalue();
    if (n->kind() != c10::prim::GetAttr)
        return c10::IValue();

    c10::IValue parent = original_attribute(n->input(0), mod);
    if (!parent.isObject())
        return c10::IValue();
    return parent.toObject()->getAttr(n->s(c10::attr::name));
}

static c10::ScalarType original_scalar_type(const torch::jit::Value* v, const torch::jit::Module& mod, std::map<const torch::jit::Value*, c10::ScalarType>& types)
{
    if (types.count(v))
        return types.at(v);

    c10::ScalarType type = c10::ScalarType::Undefined;
    types[v] = type;
    if (!v->type()->cast<c10::TensorType>())
        return type;
    const torch::jit::Node* n = v->node();
    const std::string op = n->kind().toDisplayString();
    if (n->kind() == c10::prim::GetAttr)
    {
        c10::IValue attr = original_attribute(v, mod);
        if (attr.isTensor())
            type = attr.toTensor().scalar_type();
    }
    else if (n->kind() == c10::prim::Constant && n->hasAttribute(c10::attr::value) && n->kindOf(c10::attr::value) == torch::jit::AttributeKind::t)
    {
        type = n->t(c10::attr::value).scalar_type();
    }
    else if (op == "aten::type_as" || (op == "aten::to" && n->hasNamedInput("other")))
    {
        type = original_scalar_type(n->input(1), mod, types);
    }
    else if (n->kind().is_aten() && n->hasNamedInput("dtype"))
    {
        auto dtype = torch::jit::toIValue(n->namedInput("dtype"));
        if (dtype && dtype->isInt())
            type = (c10::ScalarType)dtype->toInt();
        else if (dtype && dtype->isNone() && !n->inputs().empty() && n->input(0)->type()->cast<c10::TensorType>())
            type = original_scalar_type(n->input(0), mod, types);
    }
    else if (!n->inputs().empty() && op != "aten::view_as_real" && op != "aten::view_as_complex" && op != "aten::real" && op != "aten::imag"
             && (torch_alias_kind(n) != TorchAliasNone || op == "aten::clone" || torch_mutation_kind(op) == TorchMutationData || torch_mutation_kind(op) == TorchMutationMetadata))
    {
        type = original_scalar_type(torch_alias_input(n), mod, types);
    }
    else if (n->kind() == c10::prim::CallMethod)
    {
        c10::IValue attr = original_attribute(n->input(0), mod);
        if (attr.isObject())
        {
            torch::jit::Module submod(attr.toObject());
            auto method = submod.find_method(n->s(c10::attr::name));
            if (method && method->graph()->outputs().size() == 1)
            {
                auto g = method->graph();
                std::map<const torch::jit::Value*, c10::ScalarType> argument_types;
                for (size_t i = 1; i < n->inputs().size() && i < g->inputs().size(); i++)
                    argument_types[g->inputs()[i]] = original_scalar_type(n->input(i), mod, types);
                type = original_scalar_type(g->outputs()[0], submod, argument_types);
            }
        }
    }
    else if ((op == "aten::linear" || op == "aten::_convolution" || op == "aten::convolution"
              || op == "aten::conv1d" || op == "aten::conv2d" || op == "aten::conv3d"
              || op == "aten::mm" || op == "aten::bmm" || op == "aten::matmul"
              || op == "aten::addmm" || op == "aten::baddbmm")
             && n->inputs().size() >= 2)
    {
        const c10::ScalarType input = original_scalar_type(n->input(0), mod, types);
        const c10::ScalarType weight = original_scalar_type(n->input(1), mod, types);
        if ((c10::isFloatingType(input) || c10::isComplexType(input)) && input == weight)
            type = input;
    }

    if (type == c10::ScalarType::Undefined)
    {
        auto tensor_type = v->type()->cast<c10::TensorType>();
        if (tensor_type && tensor_type->scalarType())
            type = *tensor_type->scalarType();
    }
    types[v] = type;
    return type;
}

static void infer_original_scalar_types(const std::shared_ptr<torch::jit::Graph>& graph, const torch::jit::Module& mod, const std::vector<at::Tensor>& input_tensors,
                                        std::map<const torch::jit::Value*, c10::ScalarType>& types)
{
    // Analyze a copy, so shape propagation cannot insert expands into the model.
    auto g = graph->copy();
    for (size_t i = 0; i < input_tensors.size() && i + 1 < g->inputs().size(); i++)
        g->inputs()[i + 1]->setType(c10::TensorType::create(input_tensors[i]));

    std::map<const torch::jit::Value*, const torch::jit::Value*> originals;
    auto original_node = graph->nodes().begin();
    for (auto n : g->nodes())
    {
        for (size_t i = 0; i < n->outputs().size(); i++)
            originals[n->output(i)] = (*original_node)->output(i);
        ++original_node;
    }

    // Metadata writes invalidate flow-insensitive JIT shape information. Only
    // use its results before the first such write; shared views retain dtype.
    std::vector<torch::jit::Node*> tail;
    for (auto n : g->nodes())
    {
        const TorchMutationKind mutation = torch_mutation_kind(n->kind().toDisplayString());
        if (!tail.empty() || mutation == TorchMutationMetadata || mutation == TorchMutationStorage)
            tail.push_back(n);
    }
    while (!g->outputs().empty())
        g->eraseOutput(0);
    for (auto it = tail.rbegin(); it != tail.rend(); ++it)
    {
        for (auto v : (*it)->outputs())
            originals.erase(v);
        (*it)->destroy();
    }

    for (auto it = g->nodes().begin(); it != g->nodes().end();)
    {
        auto n = *it++;
        if (n->kind() == c10::prim::GetAttr)
        {
            c10::IValue attr = original_attribute(n->output(), mod);
            if (attr.isTensor())
            {
                auto v = g->addInput();
                v->setType(c10::TensorType::create(attr.toTensor()));
                n->output()->replaceAllUsesWith(v);
                originals.erase(n->output());
                n->destroy();
            }
        }
        else if (n->kind() == c10::aten::to || n->kind() == c10::aten::type_as)
        {
            const c10::ScalarType type = original_scalar_type(originals.at(n->output()), mod, types);
            if (type != c10::ScalarType::Undefined)
            {
                // JIT shape propagation does not infer cast outputs. Supplying
                // their declared type also keeps later arithmetic analyzable.
                auto tensor_type = n->input(0)->type()->cast<c10::TensorType>();
                auto v = g->addInput();
                v->setType(tensor_type->withScalarType(type));
                n->output()->replaceAllUsesWith(v);
                originals.erase(n->output());
                n->destroy();
            }
        }
    }
    try
    {
        torch::jit::PropagateInputShapes(g);
    }
    catch (const std::exception&)
    {
        return;
    }

    for (auto it = types.begin(); it != types.end();)
    {
        if (it->second == c10::ScalarType::Undefined)
            it = types.erase(it);
        else
            ++it;
    }
    std::set<const torch::jit::Value*> untrusted;
    for (auto n : g->nodes())
    {
        bool complete_inputs = true;
        for (auto input : n->inputs())
        {
            auto tensor_type = input->type()->cast<c10::TensorType>();
            if (untrusted.count(input) || (tensor_type && !tensor_type->dim()))
                complete_inputs = false;
        }
        for (auto v : n->outputs())
        {
            auto tensor_type = v->type()->cast<c10::TensorType>();
            if (!complete_inputs)
            {
                untrusted.insert(v);
                continue;
            }
            // Legacy JIT shape rules can keep an integer input dtype for unary
            // operations such as exp whose actual result is floating point.
            if (!tensor_type)
                continue;
            if (!tensor_type->scalarType() || !c10::isFloatingType(*tensor_type->scalarType()))
            {
                if (n->kind() != c10::prim::Constant && (!originals.count(v) || original_scalar_type(originals.at(v), mod, types) == c10::ScalarType::Undefined))
                    untrusted.insert(v);
                continue;
            }
            if (originals.count(v) && !types.count(originals.at(v)))
                types[originals.at(v)] = *tensor_type->scalarType();
        }
    }
}

static bool mutable_unknown_cast(const std::shared_ptr<torch::jit::Graph>& graph, const std::set<const torch::jit::Node*>& unknown_casts)
{
    // Opaque module calls may write any alias. Inspect their actual bodies on
    // a copy while retaining the module boundaries used by later fusion.
    auto g = graph->copy();
    std::set<const torch::jit::Node*> copied_casts;
    auto original_node = graph->nodes().begin();
    for (auto n : g->nodes())
    {
        if (unknown_casts.count(*original_node))
            copied_casts.insert(n);
        ++original_node;
    }
    torch::jit::Inline(*g);
    torch::jit::AliasDb aliases(g);
    std::vector<const torch::jit::Node*> casts;
    for (torch::jit::Node* n : g->nodes())
    {
        for (const torch::jit::Node* cast : casts)
        {
            if (aliases.writesToAlias(n, {cast->input(0), cast->output()}))
            {
                fprintf(stderr, "cannot determine original dtype of %s before observable mutation\n", cast->kind().toDisplayString());
                return true;
            }
        }
        if (copied_casts.count(n))
            casts.push_back(n);
    }
    return false;
}

int normalize_dtype(std::shared_ptr<torch::jit::Graph>& graph, const torch::jit::Module& mod, const std::vector<at::Tensor>& input_tensors)
{
    std::map<const torch::jit::Value*, c10::ScalarType> types;
    for (size_t i = 0; i < input_tensors.size() && i + 1 < graph->inputs().size(); i++)
        types[graph->inputs()[i + 1]] = input_tensors[i].scalar_type();

    bool infer_types = false;
    for (auto n : graph->nodes())
    {
        if (n->kind() != c10::aten::to && n->kind() != c10::aten::type_as)
            continue;
        if (original_scalar_type(n->input(0), mod, types) == c10::ScalarType::Undefined
                || original_scalar_type(n->output(), mod, types) == c10::ScalarType::Undefined)
            infer_types = true;
    }
    if (infer_types)
        infer_original_scalar_types(graph, mod, input_tensors, types);

    std::set<const torch::jit::Node*> unknown_casts;
    std::vector<torch::jit::Node*> cast_copies;
    for (torch::jit::Node* n : graph->nodes())
    {
        const std::string op = n->kind().toDisplayString();
        if (op != "aten::to" && op != "aten::type_as")
            continue;

        if (n->hasNamedInput("copy"))
        {
            auto copy = torch::jit::toIValue(n->namedInput("copy"));
            if (copy && copy->isBool() && copy->toBool())
            {
                continue;
            }
        }
        const c10::ScalarType source = original_scalar_type(n->input(0), mod, types);
        const c10::ScalarType target = original_scalar_type(n->output(), mod, types);
        if (source != c10::ScalarType::Undefined && target != c10::ScalarType::Undefined)
        {
            if (source != target && normalized_scalar_type(source) == normalized_scalar_type(target))
                cast_copies.push_back(n);
        }
        else
        {
            // A device-only conversion does not change the scalar type.
            if (op == "aten::to" && n->hasNamedInput("dtype"))
            {
                auto dtype = torch::jit::toIValue(n->namedInput("dtype"));
                if (dtype && dtype->isNone())
                    continue;
            }
            if (target == c10::ScalarType::Undefined || c10::isFloatingType(target) || c10::isComplexType(target))
                unknown_casts.insert(n);
        }
    }
    if (!unknown_casts.empty() && mutable_unknown_cast(graph, unknown_casts))
        return -1;

    for (torch::jit::Node* n : cast_copies)
    {
        torch::jit::WithInsertPoint guard(n);
        if (n->hasNamedInput("copy"))
        {
            const size_t index = *n->schema().argumentIndexWithName("copy");
            n->replaceInput(index, graph->insertConstant(true));
        }
        else
        {
            auto clone = graph->create(c10::Symbol::fromQualString("aten::clone"), {n->input(0), graph->insertConstant(c10::IValue())});
            clone->output()->setType(n->input(0)->type());
            clone->insertBefore(n);
            n->replaceInput(0, clone->output());
        }
    }

    for (torch::jit::Node* n : graph->nodes())
    {
        if (n->kind() == c10::prim::Constant && n->hasAttribute(c10::attr::value) && n->kindOf(c10::attr::value) == torch::jit::AttributeKind::t)
        {
            const at::Tensor t = n->t(c10::attr::value);
            const c10::ScalarType type = normalized_scalar_type(t.scalar_type());
            if (type != t.scalar_type())
            {
                const at::Tensor fp32 = t.toType(type);
                n->t_(c10::attr::value, fp32);
                n->output()->setType(c10::TensorType::create(fp32));
            }
        }
        if (!n->kind().is_aten())
            continue;
        if (n->hasNamedInput("dtype"))
        {
            torch::jit::Node* dtype_node = n->namedInput("dtype")->node();
            if (dtype_node->hasAttribute(c10::attr::value) && dtype_node->kindOf(c10::attr::value) == torch::jit::AttributeKind::i)
            {
                const c10::ScalarType type = (c10::ScalarType)dtype_node->i(c10::attr::value);
                dtype_node->i_(c10::attr::value, (int)normalized_scalar_type(type));
            }
        }
    }
    return 0;
}

} // namespace pnnx
