/* Tencent is pleased to support the open source community by making ncnn available.
 *
 * Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software distributed
 * under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#if _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#ifdef PNNX_TORCHVISION
// register torchvision ops via including headers
#include <torchvision/vision.h>
#endif

#include <ir.h>
#include <pass_level0.h>
#include <pass_level1.h>
#include <pass_level2.h>
#include <pass_level3.h>
#include <pass_level4.h>
#include <pass_level5.h>

#include "pass_ncnn.h"
#include "save_ncnn.h"

using namespace pnnx;
namespace py = pybind11;

static std::string get_basename(const std::string& path)
{
    std::string base = path.substr(0, path.find_last_of('.'));
    // sanitize -
    std::replace(base.begin(), base.end(), '-', '_');
    return base;
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

static void py_input_to_c_input(std::vector<std::vector<int64_t> >& input_shapes,
                                const py::list& py_input_shapes)
{
    for (auto vec : py_input_shapes)
    {
        std::vector<int64_t> sub_shapes = {};
        for (auto v : vec)
        {
            //            std::cout << v.cast<int64_t>() << "\n";
            sub_shapes.push_back(v.cast<int64_t>());
        }
        input_shapes.push_back(sub_shapes);
    }
}

void pnnx_export(const std::string& ptpath,
                 const py::list& py_input_shapes,
                 const std::vector<std::string>& input_types,
                 const py::list& py_input_shapes2,
                 const std::vector<std::string>& input_types2,
                 const std::string& device,
                 const std::vector<std::string>& customop_modules,
                 const std::vector<std::string>& module_operators,
                 const int64_t optlevel,
                 const std::string pnnxparam,
                 const std::string pnnxbin,
                 const std::string pnnxpy,
                 const std::string pnnxonnx,
                 const std::string ncnnparam,
                 const std::string ncnnbin,
                 const std::string ncnnpy)
{
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

    std::vector<std::vector<int64_t> > input_shapes = {};
    py_input_to_c_input(input_shapes, py_input_shapes);

    std::vector<std::vector<int64_t> > input_shapes2 = {};
    py_input_to_c_input(input_shapes2, py_input_shapes2);

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

        return;
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
            return;
        }

        method = methods[0];
        fprintf(stderr, "Use method %s as the entrypoint instead of forward\n", method->name().c_str());
    }

    auto g = method->graph();

    //     g->dump();

    fprintf(stderr, "############# pass_level0\n");

    std::string ptbase = get_basename(ptpath);
    std::set<std::string> foldable_constants;
    std::string foldable_constants_zippath = ptbase + ".foldable_constants.zip";
    pnnx::pass_level0(mod, g, input_tensors, input_tensors2, module_operators, ptpath, device, foldable_constants, foldable_constants_zippath);

    //     g->dump();

    fprintf(stderr, "############# pass_level1\n");

    pnnx::Graph pnnx_graph;
    pnnx::pass_level1(mod, g, module_operators, pnnx_graph);

    //     g->dump();

    fprintf(stderr, "############# pass_level2\n");

    pnnx::pass_level2(pnnx_graph);

    pnnx_graph.save("debug.param", "debug.bin");

    if (optlevel >= 1)
    {
        fprintf(stderr, "############# pass_level3\n");

        pnnx::pass_level3(pnnx_graph, foldable_constants, foldable_constants_zippath);

        fprintf(stderr, "############# pass_level4\n");

        pnnx::pass_level4(pnnx_graph);
    }

    pnnx_graph.save("debug2.param", "debug2.bin");

    if (optlevel >= 2)
    {
        fprintf(stderr, "############# pass_level5\n");

        pnnx::pass_level5(pnnx_graph, foldable_constants, foldable_constants_zippath);
    }

    // delete foldable_constants_zippath
    remove(foldable_constants_zippath.c_str());

    std::string pnnxparampath = ptbase + ".pnnx.param";
    std::string pnnxbinpath = ptbase + ".pnnx.bin";
    std::string pnnxpypath = ptbase + "_pnnx.py";
    std::string pnnxonnxpath = ptbase + ".pnnx.onnx";
    std::string ncnnparampath = ptbase + ".ncnn.param";
    std::string ncnnbinpath = ptbase + ".ncnn.bin";
    std::string ncnnpypath = ptbase + "_ncnn.py";
    int fp16 = 1;

    if (strcmp(pnnxparam.c_str(), "") != 0)
    {
        pnnxparampath = pnnxparam;
    }
    if (strcmp(pnnxbin.c_str(), "") != 0)
    {
        pnnxbinpath = pnnxbin;
    }
    if (strcmp(pnnxpy.c_str(), "") != 0)
    {
        pnnxpypath = pnnxpy;
    }
    if (strcmp(pnnxonnx.c_str(), "") != 0)
    {
        pnnxonnxpath = pnnxonnx;
    }
    if (strcmp(ncnnparam.c_str(), "") != 0)
    {
        ncnnparampath = ncnnparam;
    }
    if (strcmp(ncnnbin.c_str(), "") != 0)
    {
        ncnnbinpath = ncnnbin;
    }
    if (strcmp(ncnnpy.c_str(), "") != 0)
    {
        ncnnpypath = ncnnpy;
    }

    pnnx_graph.save(pnnxparampath, pnnxbinpath);

    pnnx_graph.python(pnnxpypath, pnnxbinpath);

#if BUILD_PNNX2ONNX
    pnnx::save_onnx(pnnx_graph, pnnxonnxpath.c_str(), fp16);
#else
    fprintf(stderr, "pnnx build without onnx-zero support, skip saving onnx\n");
#endif

    //     if (optlevel >= 2)
    {
        fprintf(stderr, "############# pass_ncnn\n");

        pnnx::pass_ncnn(pnnx_graph, module_operators);

        pnnx::save_ncnn(pnnx_graph, ncnnparampath, ncnnbinpath, ncnnpypath, fp16);
    }
}

PYBIND11_MODULE(pnnx, m)
{
    m.doc() = R"pbdoc(
    pnnx python wrapper
    -----------------------
    .. currentmodule:: pypnnx
    .. autosummary::
       :toctree: _generate
    )pbdoc";

    m.def("pnnx_export", &pnnx_export, "Export pytorch model.");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
