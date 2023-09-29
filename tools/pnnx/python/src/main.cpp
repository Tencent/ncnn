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

void pnnx_module_export_with_shapes(torch::jit::Module model,
                                        std::vector<std::vector<int64_t> > input_shapes,
                                        std::vector<std::string> input_types = {"f32"},
                                        std::string device = "cpu")
{
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

    m.def("pnnx_module_export_with_shapes", &pnnx_module_export_with_shapes, "Export pytorch model with shapes.");

    #ifdef VERSION_INFO
        m.attr("__version__") = VERSION_INFO;
    #else
        m.attr("__version__") = "dev";
    #endif
}
