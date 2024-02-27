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

#include "load_torchscript.h"

#if _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <torch/script.h>
#include <torch/csrc/api/include/torch/version.h>
#ifdef PNNX_TORCHVISION
namespace vision {
int64_t cuda_version();
} // namespace vision
#endif

#include "pass_level0.h"
#include "pass_level1.h"

namespace pnnx {

static int get_at_tensor_type(const at::ScalarType& st)
{
    if (st == c10::ScalarType::Float) return 1;
    if (st == c10::ScalarType::Double) return 2;
    if (st == c10::ScalarType::Half) return 3;
    if (st == c10::ScalarType::Int) return 4;
    if (st == c10::ScalarType::QInt32) return 4;
    if (st == c10::ScalarType::Long) return 5;
    if (st == c10::ScalarType::Short) return 6;
    if (st == c10::ScalarType::Char) return 7;
    if (st == c10::ScalarType::QInt8) return 7;
    if (st == c10::ScalarType::Byte) return 8;
    if (st == c10::ScalarType::QUInt8) return 8;
    if (st == c10::ScalarType::Bool) return 9;
    if (st == c10::ScalarType::ComplexFloat) return 10;
    if (st == c10::ScalarType::ComplexDouble) return 11;
    if (st == c10::ScalarType::ComplexHalf) return 12;
    return 0; // unknown type
}

static size_t type_to_elemsize(int type)
{
    if (type == 1) return 4;
    if (type == 2) return 8;
    if (type == 3) return 2;
    if (type == 4) return 4;
    if (type == 5) return 8;
    if (type == 6) return 2;
    if (type == 7) return 1;
    if (type == 8) return 1;
    if (type == 9) return 1;
    if (type == 10) return 8;
    if (type == 11) return 16;
    if (type == 12) return 4;
    return 0; // null
}

Parameter::Parameter(const torch::jit::Node* value_node)
{
    type = 0;

    if (value_node->kind() == c10::prim::Constant)
    {
        if (value_node->output()->type()->kind() == c10::TypeKind::NoneType)
        {
            type = 0;
            return;
        }

        if (!value_node->hasAttribute(torch::jit::attr::value))
        {
            fprintf(stderr, "no attribute value\n");
            value_node->dump();
            return;
        }

        switch (value_node->output()->type()->kind())
        {
        case c10::TypeKind::NoneType:
        {
            type = 0;
            break;
        }
        case c10::TypeKind::BoolType:
        {
            type = 1;
            b = value_node->i(torch::jit::attr::value);
            break;
        }
        case c10::TypeKind::IntType:
        {
            type = 2;
            int64_t i64 = value_node->i(torch::jit::attr::value);
            if (i64 == std::numeric_limits<int64_t>::max()) i64 = INT_MAX;
            if (i64 == std::numeric_limits<int64_t>::min()) i64 = INT_MIN;
            i = (int)i64;
            break;
        }
        case c10::TypeKind::FloatType:
        {
            type = 3;
            f = (float)value_node->f(torch::jit::attr::value);
            break;
        }
        case c10::TypeKind::StringType:
        {
            type = 4;
            s = value_node->s(torch::jit::attr::value);
            break;
        }
        case c10::TypeKind::DeviceObjType:
        {
            type = 4;
            s = value_node->s(torch::jit::attr::value);
            break;
        }
#if TORCH_VERSION_MAJOR >= 2 || (TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 9)
        case c10::TypeKind::ComplexType:
        {
            type = 10;
            c = std::complex<float>(value_node->c(torch::jit::attr::value));
            break;
        }
#endif
        case c10::TypeKind::TensorType:
        {
            at::Tensor t = value_node->t(torch::jit::attr::value);

            if (t.dim() == 0 && t.numel() == 1)
            {
                if (t.scalar_type() == c10::ScalarType::Long)
                {
                    type = 2;
                    int64_t i64 = t.item<int64_t>();
                    if (i64 == std::numeric_limits<int64_t>::max()) i64 = INT_MAX;
                    if (i64 == std::numeric_limits<int64_t>::min()) i64 = INT_MIN;
                    i = (int)i64;
                }
                else if (t.scalar_type() == c10::ScalarType::Int)
                {
                    type = 2;
                    i = t.item<int>();
                }
                else if (t.scalar_type() == c10::ScalarType::Double)
                {
                    type = 3;
                    f = (float)t.item<double>();
                }
                else if (t.scalar_type() == c10::ScalarType::Float)
                {
                    type = 3;
                    f = t.item<float>();
                }
                else if (t.scalar_type() == c10::ScalarType::ComplexDouble)
                {
                    type = 10;
                    c = std::complex<float>(t.item<c10::complex<double> >());
                }
                else if (t.scalar_type() == c10::ScalarType::ComplexFloat)
                {
                    type = 10;
                    c = std::complex<float>(t.item<c10::complex<float> >());
                }
                else
                {
                    fprintf(stderr, "unknown Parameter value kind %s of TensorType, t.dim = 0\n", value_node->kind().toDisplayString());
                }
            }
            else
            {
                // constant tensor will become pnnx attribute node later
                type = 8;
            }

            break;
        }
        case c10::TypeKind::ListType:
        {
            switch (value_node->output()->type()->containedTypes()[0]->kind())
            {
            case c10::TypeKind::IntType:
            {
                type = 5;
                std::vector<int64_t> i64s = value_node->ival(torch::jit::attr::value).toIntVector();
                for (auto i64 : i64s)
                {
                    if (i64 == std::numeric_limits<int64_t>::max()) i64 = INT_MAX;
                    if (i64 == std::numeric_limits<int64_t>::min()) i64 = INT_MIN;
                    ai.push_back(i64);
                }
                break;
            }
            case c10::TypeKind::FloatType:
            {
                type = 6;
                std::vector<double> fs = value_node->ival(torch::jit::attr::value).toDoubleVector();
                for (auto f : fs)
                {
                    af.push_back((float)f);
                }
                break;
            }
            default:
            {
                fprintf(stderr, "unknown Parameter value list element kind %s\n", c10::typeKindToString(value_node->output()->type()->containedTypes()[0]->kind()));
                break;
            }
            }
            break;
        }
        default:
        {
            fprintf(stderr, "unknown Parameter value kind %s\n", c10::typeKindToString(value_node->output()->type()->kind()));
            break;
        }
        }
    }
    else if (value_node->kind() == c10::prim::ListConstruct)
    {
        switch (value_node->output()->type()->cast<c10::ListType>()->getElementType()->kind())
        {
        case c10::TypeKind::IntType:
        {
            type = 5;
            for (const auto& x : value_node->inputs())
            {
                if (!x->node()->hasAttribute(torch::jit::attr::value))
                {
                    fprintf(stderr, "no attribute value in int list\n");
                    ai.push_back(0);
                    continue;
                }

                ai.push_back((int)x->node()->i(torch::jit::attr::value));
            }
            break;
        }
        case c10::TypeKind::FloatType:
        {
            type = 6;
            for (const auto& x : value_node->inputs())
            {
                if (!x->node()->hasAttribute(torch::jit::attr::value))
                {
                    fprintf(stderr, "no attribute value in float list\n");
                    af.push_back(0.f);
                    continue;
                }

                af.push_back((float)x->node()->f(torch::jit::attr::value));
            }
            break;
        }
        case c10::TypeKind::StringType:
        {
            type = 7;
            for (const auto& x : value_node->inputs())
            {
                if (!x->node()->hasAttribute(torch::jit::attr::value))
                {
                    fprintf(stderr, "no attribute value in string list\n");
                    as.push_back("");
                    continue;
                }

                as.push_back(x->node()->s(torch::jit::attr::value));
            }
            break;
        }
#if TORCH_VERSION_MAJOR >= 2 || (TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 9)
        case c10::TypeKind::ComplexType:
        {
            type = 11;
            for (const auto& x : value_node->inputs())
            {
                if (!x->node()->hasAttribute(torch::jit::attr::value))
                {
                    fprintf(stderr, "no attribute value in complex list\n");
                    ac.push_back(std::complex<float>(0.f, 0.f));
                    continue;
                }

                ac.push_back(std::complex<float>(x->node()->c(torch::jit::attr::value)));
            }
            break;
        }
#endif
        default:
        {
            fprintf(stderr, "unknown Parameter value list element kind %s\n", c10::typeKindToString(value_node->output()->type()->cast<c10::ListType>()->getElementType()->kind()));
            break;
        }
        }
    }
    else
    {
        fprintf(stderr, "unknown Parameter value_node kind %s\n", value_node->kind().toDisplayString());
    }
}

Parameter::Parameter(const torch::jit::Value* value)
    : Parameter(value->node())
{
}

Attribute::Attribute(const at::Tensor& t)
{
    type = get_at_tensor_type(t.scalar_type());

    const int ndim = (int)t.dim();

    if (ndim == 0)
    {
        shape = {1};

        data.resize(type_to_elemsize(type));

        if (t.scalar_type() == c10::ScalarType::Long)
        {
            int64_t i = t.item<int64_t>();
            memcpy((void*)data.data(), (const void*)&i, data.size());
        }
        else if (t.scalar_type() == c10::ScalarType::Int)
        {
            int i = t.item<int>();
            memcpy((void*)data.data(), (const void*)&i, data.size());
        }
        else if (t.scalar_type() == c10::ScalarType::Double)
        {
            double f = t.item<double>();
            memcpy((void*)data.data(), (const void*)&f, data.size());
        }
        else if (t.scalar_type() == c10::ScalarType::Float)
        {
            float f = t.item<float>();
            memcpy((void*)data.data(), (const void*)&f, data.size());
        }
        else
        {
            fprintf(stderr, "unknown Attribute tensor scalar type %d\n", type);
        }

        return;
    }

    shape.resize(ndim);
    for (int i = 0; i < ndim; i++)
        shape[i] = t.size(i);

    if (shape.size() > 0)
    {
        data.resize(elemcount() * type_to_elemsize(type));
        memcpy((void*)data.data(), (const void*)t.cpu().contiguous().data_ptr(), data.size());
    }
}

Operand* Graph::new_operand(const torch::jit::Value* v)
{
    // Operand* r = new Operand;
    // r->name = v->debugName();

    Operand* r = new_operand(v->debugName());

    r->type = -1;

    auto pt = v->type()->cast<c10::TensorType>();
    if (pt)
    {
        if (pt->scalarType().has_value() && pt->dim().has_value())
        {
            r->type = get_at_tensor_type(pt->scalarType().value());
            const int ndim = (int)pt->dim().value();
            r->shape.resize(ndim);
            for (int i = 0; i < ndim; i++)
            {
                if (pt->sizes()[i].has_value())
                    r->shape[i] = (int)pt->sizes()[i].value();
                else
                    r->shape[i] = -1;
            }
        }
    }

    // operands.push_back(r);
    return r;
}

static c10::ScalarType input_type_to_c10_ScalarType(const std::string& t)
{
    if (t == "c64") return torch::kComplexFloat;
    if (t == "c32") return torch::kComplexHalf;
    if (t == "c128") return torch::kComplexDouble;
    if (t == "f32") return torch::kFloat32;
    if (t == "f16") return torch::kFloat16;
    if (t == "f64") return torch::kFloat64;
    if (t == "i32") return torch::kInt32;
    if (t == "i16") return torch::kInt16;
    if (t == "i64") return torch::kInt64;
    if (t == "i8") return torch::kInt8;
    if (t == "u8") return torch::kUInt8;

    fprintf(stderr, "unsupported type %s fallback to f32\n", t.c_str());
    return torch::kFloat32;
}

const torch::jit::Node* find_node_by_kind(const std::shared_ptr<torch::jit::Graph>& graph, const std::string& kind)
{
    for (const auto& n : graph->nodes())
    {
        if (n->kind().toDisplayString() == kind)
            return n;
    }

    return 0;
}

int load_torchscript(const std::string& ptpath, Graph& pnnx_graph,
                     const std::string& device,
                     const std::vector<std::vector<int64_t> >& input_shapes,
                     const std::vector<std::string>& input_types,
                     const std::vector<std::vector<int64_t> >& input_shapes2,
                     const std::vector<std::string>& input_types2,
                     const std::vector<std::string>& customop_modules,
                     const std::vector<std::string>& module_operators,
                     const std::string& foldable_constants_zippath,
                     std::set<std::string>& foldable_constants)
{
#ifdef PNNX_TORCHVISION
    // call some vision api to register vision ops  :P
    (void)vision::cuda_version();
#endif

    for (auto m : customop_modules)
    {
        fprintf(stderr, "load custom module %s\n", m.c_str());
#if _WIN32
        HMODULE handle = LoadLibraryExA(m.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!handle)
        {
            fprintf(stderr, "LoadLibraryExA %s failed %d\n", m.c_str(), GetLastError());
        }
#else
        void* handle = dlopen(m.c_str(), RTLD_LAZY);
        if (!handle)
        {
            fprintf(stderr, "dlopen %s failed %s\n", m.c_str(), dlerror());
        }
#endif
    }

    std::vector<at::Tensor> input_tensors;
    for (size_t i = 0; i < input_shapes.size(); i++)
    {
        const std::vector<int64_t>& shape = input_shapes[i];
        const std::string& type = input_types[i];

        at::Tensor t = torch::ones(shape, input_type_to_c10_ScalarType(type));
        if (device == "gpu")
            t = t.cuda();

        input_tensors.push_back(t);
    }

    std::vector<at::Tensor> input_tensors2;
    for (size_t i = 0; i < input_shapes2.size(); i++)
    {
        const std::vector<int64_t>& shape = input_shapes2[i];
        const std::string& type = input_types2[i];

        at::Tensor t = torch::ones(shape, input_type_to_c10_ScalarType(type));
        if (device == "gpu")
            t = t.cuda();

        input_tensors2.push_back(t);
    }

    torch::jit::Module mod;

    try
    {
        mod = torch::jit::load(ptpath, (device == "gpu") ? c10::kCUDA : c10::kCPU);
    }
    catch (const c10::Error& e)
    {
        fprintf(stderr, "Load torchscript failed: %s\n", e.what());

        fprintf(stderr, "Please export model to torchscript as follows\n");
        fprintf(stderr, "------------------------------------------\n");
        fprintf(stderr, "import torch\n");
        fprintf(stderr, "import torchvision.models as models\n\n");
        fprintf(stderr, "net = models.resnet18(pretrained=True)\n");
        fprintf(stderr, "net = net.eval()\n\n");
        fprintf(stderr, "x = torch.rand(1, 3, 224, 224)\n");
        fprintf(stderr, "mod = torch.jit.trace(net, x)\n");
        fprintf(stderr, "mod.save(\"resnet18.pt\")\n");
        fprintf(stderr, "------------------------------------------\n");

        return -1;
    }

    mod.eval();

    //     mod.dump(true, false, false);
    //     mod.dump(true, true, true);

    auto method = mod.find_method("forward");
    if (!method)
    {
        auto methods = mod.get_methods();
        if (methods.empty())
        {
            fprintf(stderr, "No method in torchscript\n");
            return -1;
        }

        method = methods[0];
        fprintf(stderr, "Use method %s as the entrypoint instead of forward\n", method->name().c_str());
    }

    auto g = method->graph();

    // g->dump();

    fprintf(stderr, "############# pass_level0\n");

    pnnx::pass_level0(mod, g, input_tensors, input_tensors2, module_operators, ptpath, device, foldable_constants, foldable_constants_zippath);

    // g->dump();

    fprintf(stderr, "############# pass_level1\n");

    pnnx::pass_level1(mod, g, module_operators, pnnx_graph);

    return 0;
}

} // namespace pnnx
