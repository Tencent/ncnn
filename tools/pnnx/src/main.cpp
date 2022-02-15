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

#include <stdio.h>

#if _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <string>
#include <vector>

#include <torch/script.h>

#ifdef PNNX_TORCHVISION
// register torchvision ops via including headers
#include <torchvision/vision.h>
#endif

#include "ir.h"
#include "pass_level0.h"
#include "pass_level1.h"
#include "pass_level2.h"
#include "pass_level3.h"
#include "pass_level4.h"
#include "pass_level5.h"

#include "pass_ncnn.h"

static std::string get_basename(const std::string& path)
{
    return path.substr(0, path.find_last_of('.'));
}

static void parse_string_list(char* s, std::vector<std::string>& list)
{
    list.clear();

    char* pch = strtok(s, ",");
    while (pch != NULL)
    {
        list.push_back(std::string(pch));

        pch = strtok(NULL, ",");
    }
}

static void print_string_list(const std::vector<std::string>& list)
{
    for (size_t i = 0; i < list.size(); i++)
    {
        fprintf(stderr, "%s", list[i].c_str());
        if (i + 1 != list.size())
            fprintf(stderr, ",");
    }
}

static void parse_shape_list(char* s, std::vector<std::vector<int64_t> >& shapes, std::vector<std::string>& types)
{
    shapes.clear();
    types.clear();

    char* pch = strtok(s, "[]");
    while (pch != NULL)
    {
        // assign user data type
        if (!types.empty() && (pch[0] == 'f' || pch[0] == 'i' || pch[0] == 'u'))
        {
            char type[32];
            int nscan = sscanf(pch, "%31[^,]", type);
            if (nscan == 1)
            {
                types[types.size() - 1] = std::string(type);
            }
        }

        // parse a,b,c
        int v;
        int nconsumed = 0;
        int nscan = sscanf(pch, "%d%n", &v, &nconsumed);
        if (nscan == 1)
        {
            // ok we get shape
            pch += nconsumed;

            std::vector<int64_t> s;
            s.push_back(v);

            nscan = sscanf(pch, ",%d%n", &v, &nconsumed);
            while (nscan == 1)
            {
                pch += nconsumed;

                s.push_back(v);

                nscan = sscanf(pch, ",%d%n", &v, &nconsumed);
            }

            // shape end
            shapes.push_back(s);
            types.push_back("f32");
        }

        pch = strtok(NULL, "[]");
    }
}

static void print_shape_list(const std::vector<std::vector<int64_t> >& shapes, const std::vector<std::string>& types)
{
    for (size_t i = 0; i < shapes.size(); i++)
    {
        const std::vector<int64_t>& s = shapes[i];
        const std::string& t = types[i];
        fprintf(stderr, "[");
        for (size_t j = 0; j < s.size(); j++)
        {
            fprintf(stderr, "%ld", s[j]);
            if (j != s.size() - 1)
                fprintf(stderr, ",");
        }
        fprintf(stderr, "]");
        fprintf(stderr, "%s", t.c_str());
        if (i != shapes.size() - 1)
            fprintf(stderr, ",");
    }
}

static c10::ScalarType input_type_to_c10_ScalarType(const std::string& t)
{
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

static void show_usage()
{
    fprintf(stderr, "Usage: pnnx [model.pt] [(key=value)...]\n");
    fprintf(stderr, "  pnnxparam=model.pnnx.param\n");
    fprintf(stderr, "  pnnxbin=model.pnnx.bin\n");
    fprintf(stderr, "  pnnxpy=model_pnnx.py\n");
    fprintf(stderr, "  ncnnparam=model.ncnn.param\n");
    fprintf(stderr, "  ncnnbin=model.ncnn.bin\n");
    fprintf(stderr, "  ncnnpy=model_ncnn.py\n");
    fprintf(stderr, "  optlevel=2\n");
    fprintf(stderr, "  device=cpu/gpu\n");
    fprintf(stderr, "  inputshape=[1,3,224,224],...\n");
    fprintf(stderr, "  inputshape2=[1,3,320,320],...\n");
#if _WIN32
    fprintf(stderr, "  customop=C:\\Users\\nihui\\AppData\\Local\\torch_extensions\\torch_extensions\\Cache\\fused\\fused.dll,...\n");
#else
    fprintf(stderr, "  customop=/home/nihui/.cache/torch_extensions/fused/fused.so,...\n");
#endif
    fprintf(stderr, "  moduleop=models.common.Focus,models.yolo.Detect,...\n");
    fprintf(stderr, "Sample usage: pnnx mobilenet_v2.pt inputshape=[1,3,224,224]\n");
    fprintf(stderr, "              pnnx yolov5s.pt inputshape=[1,3,640,640]f32 inputshape2=[1,3,320,320]f32 device=gpu moduleop=models.common.Focus,models.yolo.Detect\n");
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        show_usage();
        return -1;
    }

    for (int i = 1; i < argc; i++)
    {
        if (argv[i][0] == '-')
        {
            show_usage();
            return -1;
        }
    }

    std::string ptpath = std::string(argv[1]);

    std::string ptbase = get_basename(ptpath);

    std::string pnnxparampath = ptbase + ".pnnx.param";
    std::string pnnxbinpath = ptbase + ".pnnx.bin";
    std::string pnnxpypath = ptbase + "_pnnx.py";
    std::string ncnnparampath = ptbase + ".ncnn.param";
    std::string ncnnbinpath = ptbase + ".ncnn.bin";
    std::string ncnnpypath = ptbase + "_ncnn.py";
    int optlevel = 2;
    std::string device = "cpu";
    std::vector<std::vector<int64_t> > input_shapes;
    std::vector<std::string> input_types;
    std::vector<std::vector<int64_t> > input_shapes2;
    std::vector<std::string> input_types2;
    std::vector<std::string> customop_modules;
    std::vector<std::string> module_operators;

    for (int i = 2; i < argc; i++)
    {
        // key=value
        char* kv = argv[i];

        char* eqs = strchr(kv, '=');
        if (eqs == NULL)
        {
            fprintf(stderr, "unrecognized arg %s\n", kv);
            continue;
        }

        // split k v
        eqs[0] = '\0';
        const char* key = kv;
        char* value = eqs + 1;

        if (strcmp(key, "pnnxparam") == 0)
            pnnxparampath = std::string(value);
        if (strcmp(key, "pnnxbin") == 0)
            pnnxbinpath = std::string(value);
        if (strcmp(key, "pnnxpy") == 0)
            pnnxpypath = std::string(value);
        if (strcmp(key, "ncnnparam") == 0)
            ncnnparampath = std::string(value);
        if (strcmp(key, "ncnnbin") == 0)
            ncnnbinpath = std::string(value);
        if (strcmp(key, "ncnnpy") == 0)
            ncnnpypath = std::string(value);
        if (strcmp(key, "optlevel") == 0)
            optlevel = atoi(value);
        if (strcmp(key, "device") == 0)
            device = value;
        if (strcmp(key, "inputshape") == 0)
            parse_shape_list(value, input_shapes, input_types);
        if (strcmp(key, "inputshape2") == 0)
            parse_shape_list(value, input_shapes2, input_types2);
        if (strcmp(key, "customop") == 0)
            parse_string_list(value, customop_modules);
        if (strcmp(key, "moduleop") == 0)
            parse_string_list(value, module_operators);
    }

    // print options
    {
        fprintf(stderr, "pnnxparam = %s\n", pnnxparampath.c_str());
        fprintf(stderr, "pnnxbin = %s\n", pnnxbinpath.c_str());
        fprintf(stderr, "pnnxpy = %s\n", pnnxpypath.c_str());
        fprintf(stderr, "ncnnparam = %s\n", ncnnparampath.c_str());
        fprintf(stderr, "ncnnbin = %s\n", ncnnbinpath.c_str());
        fprintf(stderr, "ncnnpy = %s\n", ncnnpypath.c_str());
        fprintf(stderr, "optlevel = %d\n", optlevel);
        fprintf(stderr, "device = %s\n", device.c_str());
        fprintf(stderr, "inputshape = ");
        print_shape_list(input_shapes, input_types);
        fprintf(stderr, "\n");
        fprintf(stderr, "inputshape2 = ");
        print_shape_list(input_shapes2, input_types2);
        fprintf(stderr, "\n");
        fprintf(stderr, "customop = ");
        print_string_list(customop_modules);
        fprintf(stderr, "\n");
        fprintf(stderr, "moduleop = ");
        print_string_list(module_operators);
        fprintf(stderr, "\n");
    }

    for (auto m : customop_modules)
    {
        fprintf(stderr, "load custom module %s\n", m.c_str());
#if _WIN32
        HMODULE handle = LoadLibraryExA(m.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!handle)
        {
            fprintf(stderr, "LoadLibraryExA %s failed %s\n", m.c_str(), GetLastError());
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

    torch::jit::Module mod = torch::jit::load(ptpath);

    mod.eval();

    //     mod.dump(true, false, false);
    //     mod.dump(true, true, true);

    auto g = mod.get_method("forward").graph();

    //     g->dump();

    fprintf(stderr, "############# pass_level0\n");

    std::map<std::string, pnnx::Attribute> foldable_constants;
    pnnx::pass_level0(mod, g, input_tensors, input_tensors2, module_operators, ptpath, foldable_constants);

    //     g->dump();

    fprintf(stderr, "############# pass_level1\n");

    pnnx::Graph pnnx_graph;
    pnnx::pass_level1(mod, g, pnnx_graph);

    //     g->dump();

    fprintf(stderr, "############# pass_level2\n");

    pnnx::pass_level2(pnnx_graph);

    pnnx_graph.save("debug.param", "debug.bin");

    if (optlevel >= 1)
    {
        fprintf(stderr, "############# pass_level3\n");

        pnnx::pass_level3(pnnx_graph);

        fprintf(stderr, "############# pass_level4\n");

        pnnx::pass_level4(pnnx_graph);
    }

    pnnx_graph.save("debug2.param", "debug2.bin");

    if (optlevel >= 2)
    {
        fprintf(stderr, "############# pass_level5\n");

        pnnx::pass_level5(pnnx_graph, foldable_constants);
    }

    pnnx_graph.save(pnnxparampath, pnnxbinpath);

    pnnx_graph.python(pnnxpypath, pnnxbinpath);

    //     if (optlevel >= 2)
    {
        fprintf(stderr, "############# pass_ncnn\n");

        pnnx::pass_ncnn(pnnx_graph);

        pnnx_graph.ncnn(ncnnparampath, ncnnbinpath, ncnnpypath);
    }

    //     pnnx::Graph pnnx_graph2;

    //     pnnx_graph2.load("pnnx.param", "pnnx.bin");
    //     pnnx_graph2.save("pnnx2.param", "pnnx2.bin");

    return 0;
}
