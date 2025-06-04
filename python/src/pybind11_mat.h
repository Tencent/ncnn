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

namespace py = pybind11;

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

// possible values for format:
// i (int32_t)
// f (float)
// d (double)
// leave it to empty to use get_mat_format
py::buffer_info to_buffer_info(ncnn::Mat& m, const std::string& format = "")
{
    if (m.elemsize != 1 && m.elemsize != 2 && m.elemsize != 4)
    {
        std::ostringstream ss;
        ss << "Convert ncnn.Mat to numpy.ndarray. Support only elemsize 1, 2, 4; but given "
           << m.elemsize;
        py::pybind11_fail(ss.str());
    }
    if (m.elempack != 1)
    {
        std::ostringstream ss;
        ss << "Convert ncnn.Mat to numpy.ndarray. Support only elempack == 1, but "
           "given "
           << m.elempack;
        py::pybind11_fail(ss.str());
    }
    std::string _format(format);
    if (_format.empty())
    {
        _format = get_mat_format(m);
    }
    std::vector<py::ssize_t> shape;
    std::vector<py::ssize_t> strides;
    if (m.dims == 1)
    {
        shape.push_back(m.w);
        strides.push_back(m.elemsize);
    }
    else if (m.dims == 2)
    {
        shape.push_back(m.h);
        shape.push_back(m.w);
        strides.push_back(m.w * m.elemsize);
        strides.push_back(m.elemsize);
    }
    else if (m.dims == 3)
    {
        shape.push_back(m.c);
        shape.push_back(m.h);
        shape.push_back(m.w);
        strides.push_back(m.cstep * m.elemsize);
        strides.push_back(m.w * m.elemsize);
        strides.push_back(m.elemsize);
    }
    else if (m.dims == 4)
    {
        shape.push_back(m.c);
        shape.push_back(m.d);
        shape.push_back(m.h);
        shape.push_back(m.w);
        strides.push_back(m.cstep * m.elemsize);
        strides.push_back(m.w * m.h * m.elemsize);
        strides.push_back(m.w * m.elemsize);
        strides.push_back(m.elemsize);
    }
    return py::buffer_info(m.data,     /* Pointer to buffer */
                           m.elemsize, /* Size of one scalar */
                           _format,    /* Python struct-style format descriptor */
                           m.dims,     /* Number of dimensions */
                           shape,      /* Buffer dimensions */
                           strides     /* Strides (in bytes) for each index */
                          );
}

#endif
