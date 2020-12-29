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

#ifndef PYBIND11_NCNN_MAT_H
#define PYBIND11_NCNN_MAT_H

#include <string>

#include <pybind11/pybind11.h>

#include <mat.h>

std::string get_mat_format(const ncnn::Mat& m)
{
    std::string format;
    if (m.elemsize == 4)
    {
        format = pybind11::format_descriptor<float>::format();
    }
    if (m.elemsize == 2)
    {
        // see https://docs.python.org/3/library/struct.html#format-characters
        format = "e";
    }
    if (m.elemsize == 1)
    {
        format = pybind11::format_descriptor<int8_t>::format();
    }
    return format;
}

#endif