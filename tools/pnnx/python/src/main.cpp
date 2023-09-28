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

int add(int i, int j)
{
    return i + j;
}

PYBIND11_MODULE(pnnx, m)
{
    m.doc() = "pybind11 example plugin";
    m.def("add", &add, "A function which adds two numbers");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
