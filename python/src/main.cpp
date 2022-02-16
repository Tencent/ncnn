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
#include <pybind11/functional.h>

#include <cpu.h>
#include <gpu.h>
#include <net.h>
#include <option.h>
#include <blob.h>
#include <paramdict.h>

#include "pybind11_mat.h"
#include "pybind11_datareader.h"
#include "pybind11_allocator.h"
#include "pybind11_modelbin.h"
#include "pybind11_layer.h"
using namespace ncnn;

namespace py = pybind11;

struct LayerFactory
{
    std::string name;
    int index;
    std::function<Layer*()> creator;
    std::function<void(Layer*)> destroyer;
    layer_creator_func creator_func;
    layer_destroyer_func destroyer_func;
};

#define LayerFactoryDeclear(n)                  \
    static ncnn::Layer* LayerCreator##n(void*); \
    static void LayerDestroyer##n(ncnn::Layer*, void*);

LayerFactoryDeclear(0);
LayerFactoryDeclear(1);
LayerFactoryDeclear(2);
LayerFactoryDeclear(3);
LayerFactoryDeclear(4);
LayerFactoryDeclear(5);
LayerFactoryDeclear(6);
LayerFactoryDeclear(7);
LayerFactoryDeclear(8);
LayerFactoryDeclear(9);

std::vector<LayerFactory> g_layer_factroys = {
    {"", -1, nullptr, nullptr, LayerCreator0, LayerDestroyer0},
    {"", -1, nullptr, nullptr, LayerCreator1, LayerDestroyer1},
    {"", -1, nullptr, nullptr, LayerCreator2, LayerDestroyer2},
    {"", -1, nullptr, nullptr, LayerCreator3, LayerDestroyer3},
    {"", -1, nullptr, nullptr, LayerCreator4, LayerDestroyer4},
    {"", -1, nullptr, nullptr, LayerCreator5, LayerDestroyer5},
    {"", -1, nullptr, nullptr, LayerCreator6, LayerDestroyer6},
    {"", -1, nullptr, nullptr, LayerCreator7, LayerDestroyer7},
    {"", -1, nullptr, nullptr, LayerCreator8, LayerDestroyer8},
    {"", -1, nullptr, nullptr, LayerCreator9, LayerDestroyer9},
};
int g_layer_factroy_index = 0;

#define LayerFactoryDefine(n)                                  \
    static ncnn::Layer* LayerCreator##n(void* p)               \
    {                                                          \
        if (g_layer_factroys[n].creator != nullptr)            \
        {                                                      \
            return g_layer_factroys[n].creator();              \
        }                                                      \
        return nullptr;                                        \
    }                                                          \
    static void LayerDestroyer##n(ncnn::Layer* layer, void* p) \
    {                                                          \
        if (g_layer_factroys[n].destroyer)                     \
        {                                                      \
            g_layer_factroys[n].destroyer(layer);              \
        }                                                      \
    }

LayerFactoryDefine(0);
LayerFactoryDefine(1);
LayerFactoryDefine(2);
LayerFactoryDefine(3);
LayerFactoryDefine(4);
LayerFactoryDefine(5);
LayerFactoryDefine(6);
LayerFactoryDefine(7);
LayerFactoryDefine(8);
LayerFactoryDefine(9);

PYBIND11_MODULE(ncnn, m)
{
    auto atexit = py::module_::import("atexit");
    atexit.attr("register")(py::cpp_function([]() {
        for (int i = 0; i < g_layer_factroys.size(); i++)
        {
            g_layer_factroys[i].creator = nullptr;
            g_layer_factroys[i].destroyer = nullptr;
        }
    }));

    py::class_<Allocator, PyAllocator<> >(m, "Allocator");
    py::class_<PoolAllocator, Allocator, PyAllocatorOther<PoolAllocator> >(m, "PoolAllocator")
    .def(py::init<>())
    .def("set_size_compare_ratio", &PoolAllocator::set_size_compare_ratio, py::arg("src"))
    .def("clear", &PoolAllocator::clear)
    .def("fastMalloc", &PoolAllocator::fastMalloc, py::arg("size"))
    .def("fastFree", &PoolAllocator::fastFree, py::arg("ptr"));
    py::class_<UnlockedPoolAllocator, Allocator, PyAllocatorOther<UnlockedPoolAllocator> >(m, "UnlockedPoolAllocator")
    .def(py::init<>())
    .def("set_size_compare_ratio", &UnlockedPoolAllocator::set_size_compare_ratio, py::arg("src"))
    .def("clear", &UnlockedPoolAllocator::clear)
    .def("fastMalloc", &UnlockedPoolAllocator::fastMalloc, py::arg("size"))
    .def("fastFree", &UnlockedPoolAllocator::fastFree, py::arg("ptr"));

    py::class_<DataReader, PyDataReader<> >(m, "DataReader")
    .def(py::init<>())
#if NCNN_STRING
    .def("scan", &DataReader::scan, py::arg("format"), py::arg("p"))
#endif // NCNN_STRING
    .def("read", &DataReader::read, py::arg("buf"), py::arg("size"));
    py::class_<DataReaderFromEmpty, DataReader, PyDataReaderOther<DataReaderFromEmpty> >(m, "DataReaderFromEmpty")
    .def(py::init<>())
#if NCNN_STRING
    .def("scan", &DataReaderFromEmpty::scan, py::arg("format"), py::arg("p"))
#endif // NCNN_STRING
    .def("read", &DataReaderFromEmpty::read, py::arg("buf"), py::arg("size"));

    py::class_<Blob>(m, "Blob")
    .def(py::init<>())
#if NCNN_STRING
    .def_readwrite("name", &Blob::name)
#endif // NCNN_STRING
    .def_readwrite("producer", &Blob::producer)
    .def_readwrite("consumer", &Blob::consumer)
    .def_readwrite("shape", &Blob::shape);

    py::class_<ModelBin, PyModelBin<> >(m, "ModelBin");
    py::class_<ModelBinFromDataReader, ModelBin, PyModelBinOther<ModelBinFromDataReader> >(m, "ModelBinFromDataReader")
    .def(py::init<const DataReader&>(), py::arg("dr"))
    .def("load", &ModelBinFromDataReader::load, py::arg("w"), py::arg("type"));
    py::class_<ModelBinFromMatArray, ModelBin, PyModelBinOther<ModelBinFromMatArray> >(m, "ModelBinFromMatArray")
    .def(py::init<const Mat*>(), py::arg("weights"))
    .def("load", &ModelBinFromMatArray::load, py::arg("w"), py::arg("type"));

    py::class_<ParamDict>(m, "ParamDict")
    .def(py::init<>())
    .def("type", &ParamDict::type, py::arg("id"))
    .def("get", (int (ParamDict::*)(int, int) const) & ParamDict::get, py::arg("id"), py::arg("def"))
    .def("get", (float (ParamDict::*)(int, float) const) & ParamDict::get, py::arg("id"), py::arg("def"))
    .def("get", (Mat(ParamDict::*)(int, const Mat&) const) & ParamDict::get, py::arg("id"), py::arg("def"))
    .def("set", (void (ParamDict::*)(int, int)) & ParamDict::set, py::arg("id"), py::arg("i"))
    .def("set", (void (ParamDict::*)(int, float)) & ParamDict::set, py::arg("id"), py::arg("f"))
    .def("set", (void (ParamDict::*)(int, const Mat&)) & ParamDict::set, py::arg("id"), py::arg("v"));

    py::class_<Option>(m, "Option")
    .def(py::init<>())
    .def_readwrite("lightmode", &Option::lightmode)
    .def_readwrite("num_threads", &Option::num_threads)
    .def_readwrite("blob_allocator", &Option::blob_allocator)
    .def_readwrite("workspace_allocator", &Option::workspace_allocator)
#if NCNN_VULKAN
    .def_readwrite("blob_vkallocator", &Option::blob_vkallocator)
    .def_readwrite("workspace_vkallocator", &Option::workspace_vkallocator)
    .def_readwrite("staging_vkallocator", &Option::staging_vkallocator)
    //.def_readwrite("pipeline_cache", &Option::pipeline_cache)
#endif // NCNN_VULKAN
    .def_readwrite("openmp_blocktime", &Option::openmp_blocktime)
    .def_readwrite("use_winograd_convolution", &Option::use_winograd_convolution)
    .def_readwrite("use_sgemm_convolution", &Option::use_sgemm_convolution)
    .def_readwrite("use_int8_inference", &Option::use_int8_inference)
    .def_readwrite("use_vulkan_compute", &Option::use_vulkan_compute)
    .def_readwrite("use_bf16_storage", &Option::use_bf16_storage)
    .def_readwrite("use_fp16_packed", &Option::use_fp16_packed)
    .def_readwrite("use_fp16_storage", &Option::use_fp16_storage)
    .def_readwrite("use_fp16_arithmetic", &Option::use_fp16_arithmetic)
    .def_readwrite("use_int8_packed", &Option::use_int8_packed)
    .def_readwrite("use_int8_storage", &Option::use_int8_storage)
    .def_readwrite("use_int8_arithmetic", &Option::use_int8_arithmetic)
    .def_readwrite("use_packing_layout", &Option::use_packing_layout)
    .def_readwrite("use_shader_pack8", &Option::use_shader_pack8)
    .def_readwrite("use_subgroup_basic", &Option::use_subgroup_basic)
    .def_readwrite("use_subgroup_vote", &Option::use_subgroup_vote)
    .def_readwrite("use_subgroup_ballot", &Option::use_subgroup_ballot)
    .def_readwrite("use_subgroup_shuffle", &Option::use_subgroup_shuffle)
    .def_readwrite("use_image_storage", &Option::use_image_storage)
    .def_readwrite("use_tensor_storage", &Option::use_tensor_storage);

    py::class_<Mat> mat(m, "Mat", py::buffer_protocol());
    mat.def(py::init<>())
    .def(py::init(
    [](py::tuple shape, size_t elemsize, int elempack, Allocator* allocator) {
        Mat* mat = nullptr;
        switch (shape.size())
        {
        case 1:
            mat = new Mat(shape[0].cast<int>(), elemsize, elempack, allocator);
            break;
        case 2:
            mat = new Mat(shape[0].cast<int>(), shape[1].cast<int>(), elemsize, elempack, allocator);
            break;
        case 3:
            mat = new Mat(shape[0].cast<int>(), shape[1].cast<int>(), shape[2].cast<int>(), elemsize, elempack, allocator);
            break;
        case 4:
            mat = new Mat(shape[0].cast<int>(), shape[1].cast<int>(), shape[2].cast<int>(), shape[3].cast<int>(), elemsize, elempack, allocator);
            break;
        default:
            std::stringstream ss;
            ss << "shape must be 1, 2, 3 or 4 dims, not " << shape.size();
            pybind11::pybind11_fail(ss.str());
        }
        return mat;
    }),
    py::arg("shape"), py::kw_only(),
    py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
    .def(py::init<int, size_t, int, Allocator*>(),
         py::arg("w"), py::kw_only(),
         py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
    .def(py::init<int, int, size_t, int, Allocator*>(),
         py::arg("w"), py::arg("h"), py::kw_only(),
         py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
    .def(py::init<int, int, int, size_t, int, Allocator*>(),
         py::arg("w"), py::arg("h"), py::arg("c"), py::kw_only(),
         py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
    .def(py::init<int, int, int, int, size_t, int, Allocator*>(),
         py::arg("w"), py::arg("h"), py::arg("d"), py::arg("c"), py::kw_only(),
         py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)

    .def(py::init<const Mat&>(), py::arg("m"))

    .def(py::init([](py::buffer const b) {
        py::buffer_info info = b.request();
        if (info.ndim > 4)
        {
            std::stringstream ss;
            ss << "convert numpy.ndarray to ncnn.Mat only dims <=4 support now, but given " << info.ndim;
            pybind11::pybind11_fail(ss.str());
        }

        size_t elemsize = 4u;
        if (info.format == py::format_descriptor<double>::format())
        {
            elemsize = 8u;
        }
        if (info.format == py::format_descriptor<float>::format() || info.format == py::format_descriptor<int>::format())
        {
            elemsize = 4u;
        }
        else if (info.format == "e")
        {
            elemsize = 2u;
        }
        else if (info.format == py::format_descriptor<int8_t>::format() || info.format == py::format_descriptor<uint8_t>::format())
        {
            elemsize = 1u;
        }

        Mat* v = nullptr;
        if (info.ndim == 1)
        {
            v = new Mat((int)info.shape[0], info.ptr, elemsize);
        }
        else if (info.ndim == 2)
        {
            v = new Mat((int)info.shape[1], (int)info.shape[0], info.ptr, elemsize);
        }
        else if (info.ndim == 3)
        {
            v = new Mat((int)info.shape[2], (int)info.shape[1], (int)info.shape[0], info.ptr, elemsize);

            // in ncnn, buffer to construct ncnn::Mat need align to ncnn::alignSize
            // with (w * h * elemsize, 16) / elemsize, but the buffer from numpy not
            // so we set the cstep as numpy's cstep
            v->cstep = (int)info.shape[2] * (int)info.shape[1];
        }
        else if (info.ndim == 4)
        {
            v = new Mat((int)info.shape[3], (int)info.shape[2], (int)info.shape[1], (int)info.shape[0], info.ptr, elemsize);

            // in ncnn, buffer to construct ncnn::Mat need align to ncnn::alignSize
            // with (w * h * d elemsize, 16) / elemsize, but the buffer from numpy not
            // so we set the cstep as numpy's cstep
            v->cstep = (int)info.shape[3] * (int)info.shape[2] * (int)info.shape[1];
        }
        return v;
    }),
    py::arg("array"))
    .def_buffer([](Mat& m) -> py::buffer_info {
        if (m.elemsize != 1 && m.elemsize != 2 && m.elemsize != 4)
        {
            std::stringstream ss;
            ss << "convert ncnn.Mat to numpy.ndarray only elemsize 1, 2, 4 support now, but given " << m.elemsize;
            pybind11::pybind11_fail(ss.str());
        }
        if (m.elempack != 1)
        {
            std::stringstream ss;
            ss << "convert ncnn.Mat to numpy.ndarray only elempack 1 support now, but given " << m.elempack;
            pybind11::pybind11_fail(ss.str());
        }
        std::string format = get_mat_format(m);
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
        return py::buffer_info(
            m.data,     /* Pointer to buffer */
            m.elemsize, /* Size of one scalar */
            format,     /* Python struct-style format descriptor */
            m.dims,     /* Number of dimensions */
            shape,      /* Buffer dimensions */
            strides     /* Strides (in bytes) for each index */
        );
    })
    //.def("fill", (void (Mat::*)(int))(&Mat::fill), py::arg("v"))
    .def("fill", (void (Mat::*)(float))(&Mat::fill), py::arg("v"))
    .def("clone", &Mat::clone, py::arg("allocator") = nullptr)
    .def("clone_from", &Mat::clone_from, py::arg("mat"), py::arg("allocator") = nullptr)
    .def(
        "reshape",
    [](Mat& mat, py::tuple shape, Allocator* allocator) {
        switch (shape.size())
        {
        case 1:
            return mat.reshape(shape[0].cast<int>(), allocator);
        case 2:
            return mat.reshape(shape[0].cast<int>(), shape[1].cast<int>(), allocator);
        case 3:
            return mat.reshape(shape[0].cast<int>(), shape[1].cast<int>(), shape[2].cast<int>(), allocator);
        case 4:
            return mat.reshape(shape[0].cast<int>(), shape[1].cast<int>(), shape[2].cast<int>(), shape[3].cast<int>(), allocator);
        default:
            std::stringstream ss;
            ss << "shape must be 1, 2, 3 or 4 dims, not " << shape.size();
            pybind11::pybind11_fail(ss.str());
        }
        return Mat();
    },
    py::arg("shape") = py::tuple(1), py::arg("allocator") = nullptr)
    .def("reshape", (Mat(Mat::*)(int, Allocator*) const) & Mat::reshape,
         py::arg("w"), py::kw_only(), py::arg("allocator") = nullptr)
    .def("reshape", (Mat(Mat::*)(int, int, Allocator*) const) & Mat::reshape,
         py::arg("w"), py::arg("h"), py::kw_only(), py::arg("allocator") = nullptr)
    .def("reshape", (Mat(Mat::*)(int, int, int, Allocator*) const) & Mat::reshape,
         py::arg("w"), py::arg("h"), py::arg("c"), py::kw_only(), py::arg("allocator") = nullptr)
    .def("reshape", (Mat(Mat::*)(int, int, int, int, Allocator*) const) & Mat::reshape,
         py::arg("w"), py::arg("h"), py::arg("d"), py::arg("c"), py::kw_only(), py::arg("allocator") = nullptr)

    .def(
        "create",
    [](Mat& mat, py::tuple shape, size_t elemsize, int elempack, Allocator* allocator) {
        switch (shape.size())
        {
        case 1:
            return mat.create(shape[0].cast<int>(), elemsize, elempack, allocator);
        case 2:
            return mat.create(shape[0].cast<int>(), shape[1].cast<int>(), elemsize, elempack, allocator);
        case 3:
            return mat.create(shape[0].cast<int>(), shape[1].cast<int>(), shape[2].cast<int>(), elemsize, elempack, allocator);
        case 4:
            return mat.create(shape[0].cast<int>(), shape[1].cast<int>(), shape[2].cast<int>(), shape[3].cast<int>(), elemsize, elempack, allocator);
        default:
            std::stringstream ss;
            ss << "shape must be 1, 2, 3 or 4 dims, not " << shape.size();
            pybind11::pybind11_fail(ss.str());
        }
        return;
    },
    py::arg("shape"), py::kw_only(),
    py::arg("elemsize") = 4, py::arg("elempack") = 1,
    py::arg("allocator") = nullptr)
    .def("create", (void (Mat::*)(int, size_t, int, Allocator*)) & Mat::create,
         py::arg("w"), py::kw_only(),
         py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
    .def("create", (void (Mat::*)(int, int, size_t, int, Allocator*)) & Mat::create,
         py::arg("w"), py::arg("h"), py::kw_only(),
         py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
    .def("create", (void (Mat::*)(int, int, int, size_t, int, Allocator*)) & Mat::create,
         py::arg("w"), py::arg("h"), py::arg("c"), py::kw_only(),
         py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
    .def("create", (void (Mat::*)(int, int, int, int, size_t, int, Allocator*)) & Mat::create,
         py::arg("w"), py::arg("h"), py::arg("d"), py::arg("c"), py::kw_only(),
         py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
    .def("create_like", (void (Mat::*)(const Mat&, Allocator*)) & Mat::create_like,
         py::arg("m"), py::arg("allocator") = nullptr)
    .def("addref", &Mat::addref)
    .def("release", &Mat::release)
    .def("empty", &Mat::empty)
    .def("total", &Mat::total)
    .def("elembits", &Mat::elembits)
    .def("shape", &Mat::shape)
    .def("channel", (Mat(Mat::*)(int)) & Mat::channel, py::arg("c"))
    //.def("channel", (const Mat (Mat::*)(int) const) & Mat::channel, py::arg("c"))
    .def("depth", (Mat(Mat::*)(int)) & Mat::depth, py::arg("z"))
    //.def("depth", (const Mat (Mat::*)(int) const) & Mat::depth, py::arg("z"))
    .def(
        "row",
    [](Mat& m, int y) {
        if (m.elempack != 1)
        {
            std::stringstream ss;
            ss << "get ncnn.Mat row only elempack 1 support now, but given " << m.elempack;
            pybind11::pybind11_fail(ss.str());
        }

        switch (m.elemsize)
        {
        case 1:
            return py::memoryview::from_buffer(m.row<int8_t>(y), {m.w}, {sizeof(int8_t)});
        //case 2:
        //    return py::memoryview::from_buffer(m.row<short>(y), {m.w}, {sizeof(short)});
        case 4:
            return py::memoryview::from_buffer(m.row<float>(y), {m.w}, {sizeof(float)});
        default:
            std::stringstream ss;
            ss << "ncnn.Mat row elemsize " << m.elemsize << "not support now";
            pybind11::pybind11_fail(ss.str());
        }
        return py::memoryview::from_buffer(m.row<float>(y), {m.w}, {sizeof(float)});
    },
    py::arg("y"))
    .def("channel_range", (Mat(Mat::*)(int, int)) & Mat::channel_range, py::arg("c"), py::arg("channels"))
    //.def("channel_range", (const Mat (Mat::*)(int, int) const) & Mat::channel_range, py::arg("c"), py::arg("channels"))
    .def("depth_range", (Mat(Mat::*)(int, int)) & Mat::depth_range, py::arg("z"), py::arg("depths"))
    //.def("depth_range", (const Mat (Mat::*)(int, int) const) & Mat::depth_range, py::arg("z"), py::arg("depths"))
    .def("row_range", (Mat(Mat::*)(int, int)) & Mat::row_range, py::arg("y"), py::arg("rows"))
    //.def("row_range", (const Mat (Mat::*)(int, int) const) & Mat::row_range, py::arg("y"), py::arg("rows"))
    .def("range", (Mat(Mat::*)(int, int)) & Mat::range, py::arg("x"), py::arg("n"))
    //.def("range", (const Mat (Mat::*)(int, int) const) & Mat::range, py::arg("x"), py::arg("n"))
    .def(
    "__getitem__", [](const Mat& m, size_t i) {
        return m[i];
    },
    py::arg("i"))
    .def(
    "__setitem__", [](Mat& m, size_t i, float v) {
        m[i] = v;
    },
    py::arg("i"), py::arg("v"))
    .def("__len__", [](Mat& m) {
        return m.w;
    })

    //convenient construct from pixel data
    .def_static(
    "from_pixels", [](py::buffer const b, int type, int w, int h, Allocator* allocator) {
        return Mat::from_pixels((const unsigned char*)b.request().ptr, type, w, h, allocator);
    },
    py::arg("array"), py::arg("type"), py::arg("w"), py::arg("h"), py::arg("allocator") = nullptr)
    .def_static(
    "from_pixels", [](py::buffer const b, int type, int w, int h, int stride, Allocator* allocator) {
        return Mat::from_pixels((const unsigned char*)b.request().ptr, type, w, h, stride, allocator);
    },
    py::arg("array"), py::arg("type"), py::arg("w"), py::arg("h"), py::arg("stride"), py::arg("allocator") = nullptr)
    .def_static(
    "from_pixels_resize", [](py::buffer const b, int type, int w, int h, int target_width, int target_height, Allocator* allocator) {
        return Mat::from_pixels_resize((const unsigned char*)b.request().ptr,
                                       type, w, h, target_width, target_height, allocator);
    },
    py::arg("array"), py::arg("type"), py::arg("w"), py::arg("h"), py::arg("target_width"), py::arg("target_height"), py::arg("allocator") = nullptr)
    .def_static(
    "from_pixels_resize", [](py::buffer const b, int type, int w, int h, int stride, int target_width, int target_height, Allocator* allocator) {
        return Mat::from_pixels_resize((const unsigned char*)b.request().ptr,
                                       type, w, h, stride, target_width, target_height, allocator);
    },
    py::arg("array"), py::arg("type"), py::arg("w"), py::arg("h"), py::arg("stride"), py::arg("target_width"), py::arg("target_height"), py::arg("allocator") = nullptr)
    .def_static(
    "from_pixels_roi", [](py::buffer const b, int type, int w, int h, int roix, int roiy, int roiw, int roih, Allocator* allocator) {
        return Mat::from_pixels_roi((const unsigned char*)b.request().ptr,
                                    type, w, h, roix, roiy, roiw, roih, allocator);
    },
    py::arg("array"), py::arg("type"), py::arg("w"), py::arg("h"), py::arg("roix"), py::arg("roiy"), py::arg("roiw"), py::arg("roih"), py::arg("allocator") = nullptr)
    .def_static(
    "from_pixels_roi", [](py::buffer const b, int type, int w, int h, int stride, int roix, int roiy, int roiw, int roih, Allocator* allocator) {
        return Mat::from_pixels_roi((const unsigned char*)b.request().ptr,
                                    type, w, h, stride, roix, roiy, roiw, roih, allocator);
    },
    py::arg("array"), py::arg("type"), py::arg("w"), py::arg("h"), py::arg("stride"), py::arg("roix"), py::arg("roiy"), py::arg("roiw"), py::arg("roih"), py::arg("allocator") = nullptr)
    .def_static(
    "from_pixels_roi_resize", [](py::buffer const b, int type, int w, int h, int roix, int roiy, int roiw, int roih, int target_width, int target_height, Allocator* allocator) {
        return Mat::from_pixels_roi_resize((const unsigned char*)b.request().ptr,
                                           type, w, h, roix, roiy, roiw, roih, target_width, target_height, allocator);
    },
    py::arg("array"), py::arg("type"), py::arg("w"), py::arg("h"), py::arg("roix"), py::arg("roiy"), py::arg("roiw"), py::arg("roih"), py::arg("target_width"), py::arg("target_height"), py::arg("allocator") = nullptr)
    .def_static(
    "from_pixels_roi_resize", [](py::buffer const b, int type, int w, int h, int stride, int roix, int roiy, int roiw, int roih, int target_width, int target_height, Allocator* allocator) {
        return Mat::from_pixels_roi_resize((const unsigned char*)b.request().ptr,
                                           type, w, h, stride, roix, roiy, roiw, roih, target_width, target_height, allocator);
    },
    py::arg("array"), py::arg("type"), py::arg("w"), py::arg("h"), py::arg("stride"), py::arg("roix"), py::arg("roiy"), py::arg("roiw"), py::arg("roih"), py::arg("target_width"), py::arg("target_height"), py::arg("allocator") = nullptr)
    .def(
    "substract_mean_normalize", [](Mat& mat, std::vector<float>& mean, std::vector<float>& norm) {
        return mat.substract_mean_normalize(mean.size() > 0 ? &mean[0] : 0, norm.size() > 0 ? &norm[0] : 0);
    },
    py::arg("mean"), py::arg("norm"))
    .def_readwrite("refcount", &Mat::refcount)
    .def_readwrite("elemsize", &Mat::elemsize)
    .def_readwrite("elempack", &Mat::elempack)
    .def_readwrite("allocator", &Mat::allocator)
    .def_readwrite("dims", &Mat::dims)
    .def_readwrite("w", &Mat::w)
    .def_readwrite("h", &Mat::h)
    .def_readwrite("d", &Mat::d)
    .def_readwrite("c", &Mat::c)
    .def_readwrite("cstep", &Mat::cstep)
    .def("__repr__", [](const Mat& m) {
        std::stringstream ss;
        ss << "<ncnn.Mat w=" << m.w << " h=" << m.h << " d=" << m.d << " c=" << m.c << " dims=" << m.dims
           << " cstep=" << m.cstep << " elemsize=" << m.elemsize << " elempack=" << m.elempack << "\n\t"
           << "refcount=" << (m.refcount ? *m.refcount : 0) << " data=0x" << static_cast<const void*>(m.data)
           << " allocator=0x" << static_cast<const void*>(m.allocator) << ">\n";

        const int max_count = m.dims == 1 ? 10 : 6;
        if (m.dims == 1)
        {
            ss << "[";
            bool dot_printed_w = false;

            if (m.elemsize == 1)
            {
                const int8_t* row = m.row<int8_t>(0);
                for (int i = 0; i < m.w; i++)
                {
                    if (i < max_count / 2 || i >= m.w - max_count / 2)
                    {
                        if (i > 0)
                        {
                            ss << ", ";
                        }
                        ss << static_cast<int>(row[i]);
                    }
                    else if (!dot_printed_w)
                    {
                        dot_printed_w = true;
                        ss << ", ...";
                    }
                }
            }
            if (m.elemsize == 4)
            {
                const float* row = m.row<float>(0);
                for (int i = 0; i < m.w; i++)
                {
                    if (i < max_count / 2 || i >= m.w - max_count / 2)
                    {
                        if (i > 0)
                        {
                            ss << ", ";
                        }
                        ss << row[i];
                    }
                    else if (!dot_printed_w)
                    {
                        dot_printed_w = true;
                        ss << ", ...";
                    }
                }
            }
            ss << "]";
        }
        else if (m.dims == 2)
        {
            bool dot_printed_h = false;
            ss << "[";
            for (int j = 0; j < m.h; j++)
            {
                bool dot_printed_w = false;
                if (j < max_count / 2 || j >= m.h - max_count / 2)
                {
                    ss << "[";
                    if (m.elemsize == 1)
                    {
                        const int8_t* row = m.row<int8_t>(j);
                        for (int i = 0; i < m.w; i++)
                        {
                            if (i < max_count / 2 || i >= m.w - max_count / 2)
                            {
                                if (i > 0)
                                {
                                    ss << ", ";
                                }
                                ss << static_cast<int>(row[i]);
                            }
                            else if (!dot_printed_w)
                            {
                                dot_printed_w = true;
                                ss << ", ...";
                            }
                        }
                    }
                    if (m.elemsize == 4)
                    {
                        const float* row = m.row<float>(j);
                        for (int i = 0; i < m.w; i++)
                        {
                            if (i < max_count / 2 || i >= m.w - max_count / 2)
                            {
                                if (i > 0)
                                {
                                    ss << ", ";
                                }
                                ss << row[i];
                            }
                            else if (!dot_printed_w)
                            {
                                dot_printed_w = true;
                                ss << ", ...";
                            }
                        }
                    }
                    ss << "]";
                    if (j < m.h - 1)
                    {
                        ss << "\n";
                    }
                }
                else if (!dot_printed_h)
                {
                    dot_printed_h = true;
                    ss << "...\n";
                }
            }
            ss << "]\n";
        }
        else if (m.dims == 3)
        {
            bool dot_printed_c = false;
            ss << "[";
            for (int k = 0; k < m.c; k++)
            {
                bool dot_printed_h = false;
                if (k < max_count / 2 || k >= m.c - max_count / 2)
                {
                    Mat channel = m.channel(k);
                    if (k > 0)
                    {
                        ss << " ";
                    }
                    ss << "[";
                    for (int j = 0; j < channel.h; j++)
                    {
                        bool dot_printed_w = false;
                        if (j < max_count / 2 || j >= channel.h - max_count / 2)
                        {
                            if (j > 0)
                            {
                                ss << "  ";
                            }
                            ss << "[";
                            if (m.elemsize == 1)
                            {
                                const int8_t* row = channel.row<int8_t>(j);
                                for (int i = 0; i < channel.w; i++)
                                {
                                    if (i < max_count / 2 || i >= channel.w - max_count / 2)
                                    {
                                        if (i > 0)
                                        {
                                            ss << ", ";
                                        }
                                        ss << static_cast<int>(row[i]);
                                    }
                                    else if (!dot_printed_w)
                                    {
                                        dot_printed_w = true;
                                        ss << ", ...";
                                    }
                                }
                            }
                            if (m.elemsize == 4)
                            {
                                const float* row = channel.row<float>(j);
                                for (int i = 0; i < m.w; i++)
                                {
                                    if (i < max_count / 2 || i >= m.w - max_count / 2)
                                    {
                                        if (i > 0)
                                        {
                                            ss << ", ";
                                        }
                                        ss << row[i];
                                    }
                                    else if (!dot_printed_w)
                                    {
                                        dot_printed_w = true;
                                        ss << ", ...";
                                    }
                                }
                            }
                            ss << "]";
                            if (j < channel.h - 1)
                            {
                                ss << "\n";
                            }
                        }
                        else if (!dot_printed_h)
                        {
                            dot_printed_h = true;
                            ss << "  ...\n";
                        }
                    } // for j
                    ss << "]";
                    if (k < m.c - 1)
                    {
                        ss << "\n\n";
                    }
                }
                else if (!dot_printed_c)
                {
                    dot_printed_c = true;
                    ss << " ...\n";
                }
            } // for k
            ss << "]\n";
        }
        else if (m.dims == 4)
        {
            bool dot_printed_c = false;
            ss << "[";
            for (int k = 0; k < m.c; k++)
            {
                bool dot_printed_d = false;
                if (k < max_count / 2 || k >= m.c - max_count / 2)
                {
                    Mat channel = m.channel(k);
                    if (k > 0)
                    {
                        ss << " ";
                    }
                    ss << "[";
                    for (int z = 0; z < channel.d; z++)
                    {
                        bool dot_printed_h = false;
                        if (z < max_count / 2 || z >= channel.d - max_count / 2)
                        {
                            if (z > 0)
                            {
                                ss << "  ";
                            }
                            ss << "[";
                            for (int j = 0; j < channel.h; j++)
                            {
                                bool dot_printed_w = false;
                                if (j < max_count / 2 || j >= channel.h - max_count / 2)
                                {
                                    if (j > 0)
                                    {
                                        ss << "  ";
                                    }
                                    ss << "[";
                                    if (m.elemsize == 1)
                                    {
                                        const int8_t* row = channel.depth(z).row<int8_t>(j);
                                        for (int i = 0; i < channel.w; i++)
                                        {
                                            if (i < max_count / 2 || i >= channel.w - max_count / 2)
                                            {
                                                if (i > 0)
                                                {
                                                    ss << ", ";
                                                }
                                                ss << static_cast<int>(row[i]);
                                            }
                                            else if (!dot_printed_w)
                                            {
                                                dot_printed_w = true;
                                                ss << ", ...";
                                            }
                                        }
                                    }
                                    if (m.elemsize == 4)
                                    {
                                        const float* row = channel.depth(z).row<float>(j);
                                        for (int i = 0; i < m.w; i++)
                                        {
                                            if (i < max_count / 2 || i >= m.w - max_count / 2)
                                            {
                                                if (i > 0)
                                                {
                                                    ss << ", ";
                                                }
                                                ss << row[i];
                                            }
                                            else if (!dot_printed_w)
                                            {
                                                dot_printed_w = true;
                                                ss << ", ...";
                                            }
                                        }
                                    }
                                    ss << "]";
                                    if (j < channel.h - 1)
                                    {
                                        ss << "\n";
                                    }
                                }
                                else if (!dot_printed_h)
                                {
                                    dot_printed_h = true;
                                    ss << "  ...\n";
                                }
                            } // for j
                            ss << "]";
                            if (z < channel.d - 1)
                            {
                                ss << "\n";
                            }
                        }
                        else if (!dot_printed_d)
                        {
                            dot_printed_d = true;
                            ss << " ...\n";
                        }
                    } // for z
                    ss << "]";
                    if (k < m.c - 1)
                    {
                        ss << "\n\n";
                    }
                }
                else if (!dot_printed_c)
                {
                    dot_printed_c = true;
                    ss << " ...\n";
                }
            } // for k
            ss << "]\n";
        }
        return ss.str();
    });

    py::enum_<ncnn::Mat::PixelType>(mat, "PixelType")
    .value("PIXEL_CONVERT_SHIFT", ncnn::Mat::PixelType::PIXEL_CONVERT_SHIFT)
    .value("PIXEL_FORMAT_MASK", ncnn::Mat::PixelType::PIXEL_FORMAT_MASK)
    .value("PIXEL_CONVERT_MASK", ncnn::Mat::PixelType::PIXEL_CONVERT_MASK)

    .value("PIXEL_RGB", ncnn::Mat::PixelType::PIXEL_RGB)
    .value("PIXEL_BGR", ncnn::Mat::PixelType::PIXEL_BGR)
    .value("PIXEL_GRAY", ncnn::Mat::PixelType::PIXEL_GRAY)
    .value("PIXEL_RGBA", ncnn::Mat::PixelType::PIXEL_RGBA)
    .value("PIXEL_BGRA", ncnn::Mat::PixelType::PIXEL_BGRA)

    .value("PIXEL_RGB2BGR", ncnn::Mat::PixelType::PIXEL_RGB2BGR)
    .value("PIXEL_RGB2GRAY", ncnn::Mat::PixelType::PIXEL_RGB2GRAY)
    .value("PIXEL_RGB2RGBA", ncnn::Mat::PixelType::PIXEL_RGB2RGBA)
    .value("PIXEL_RGB2BGRA", ncnn::Mat::PixelType::PIXEL_RGB2BGRA)

    .value("PIXEL_BGR2RGB", ncnn::Mat::PixelType::PIXEL_BGR2RGB)
    .value("PIXEL_BGR2GRAY", ncnn::Mat::PixelType::PIXEL_BGR2GRAY)
    .value("PIXEL_BGR2RGBA", ncnn::Mat::PixelType::PIXEL_BGR2RGBA)
    .value("PIXEL_BGR2BGRA", ncnn::Mat::PixelType::PIXEL_BGR2BGRA)

    .value("PIXEL_GRAY2RGB", ncnn::Mat::PixelType::PIXEL_GRAY2RGB)
    .value("PIXEL_GRAY2BGR", ncnn::Mat::PixelType::PIXEL_GRAY2BGR)
    .value("PIXEL_GRAY2RGBA", ncnn::Mat::PixelType::PIXEL_GRAY2RGBA)
    .value("PIXEL_GRAY2BGRA", ncnn::Mat::PixelType::PIXEL_GRAY2BGRA)

    .value("PIXEL_RGBA2RGB", ncnn::Mat::PixelType::PIXEL_RGBA2RGB)
    .value("PIXEL_RGBA2BGR", ncnn::Mat::PixelType::PIXEL_RGBA2BGR)
    .value("PIXEL_RGBA2GRAY", ncnn::Mat::PixelType::PIXEL_RGBA2GRAY)
    .value("PIXEL_RGBA2BGRA", ncnn::Mat::PixelType::PIXEL_RGBA2BGRA)

    .value("PIXEL_BGRA2RGB", ncnn::Mat::PixelType::PIXEL_BGRA2RGB)
    .value("PIXEL_BGRA2BGR", ncnn::Mat::PixelType::PIXEL_BGRA2BGR)
    .value("PIXEL_BGRA2GRAY", ncnn::Mat::PixelType::PIXEL_BGRA2GRAY)
    .value("PIXEL_BGRA2RGBA", ncnn::Mat::PixelType::PIXEL_BGRA2RGBA);

    py::class_<Extractor>(m, "Extractor")
    .def("__enter__", [](Extractor& ex) -> Extractor& { return ex; })
    .def("__exit__", [](Extractor& ex, pybind11::args) {
        ex.clear();
    })
    .def("clear", &Extractor::clear)
    .def("set_light_mode", &Extractor::set_light_mode, py::arg("enable"))
    .def("set_num_threads", &Extractor::set_num_threads, py::arg("num_threads"))
    .def("set_blob_allocator", &Extractor::set_blob_allocator, py::arg("allocator"))
    .def("set_workspace_allocator", &Extractor::set_workspace_allocator, py::arg("allocator"))
#if NCNN_STRING
    .def("input", (int (Extractor::*)(const char*, const Mat&)) & Extractor::input, py::arg("blob_name"), py::arg("in"))
    .def("extract", (int (Extractor::*)(const char*, Mat&, int)) & Extractor::extract, py::arg("blob_name"), py::arg("feat"), py::arg("type") = 0)
    .def(
    "extract", [](Extractor& ex, const char* blob_name, int type) {
        ncnn::Mat feat;
        int ret = ex.extract(blob_name, feat, type);
        return py::make_tuple(ret, feat.clone());
    },
    py::arg("blob_name"), py::arg("type") = 0)
#endif
    .def("input", (int (Extractor::*)(int, const Mat&)) & Extractor::input)
    .def("extract", (int (Extractor::*)(int, Mat&, int)) & Extractor::extract, py::arg("blob_index"), py::arg("feat"), py::arg("type") = 0)
    .def(
    "extract", [](Extractor& ex, int blob_index, int type) {
        ncnn::Mat feat;
        int ret = ex.extract(blob_index, feat, type);
        return py::make_tuple(ret, feat.clone());
    },
    py::arg("blob_index"), py::arg("type") = 0);

    py::class_<Layer, PyLayer>(m, "Layer")
    .def(py::init<>())
    .def("load_param", &Layer::load_param, py::arg("pd"))
    .def("load_model", &Layer::load_model, py::arg("mb"))
    .def("create_pipeline", &Layer::create_pipeline, py::arg("opt"))
    .def("destroy_pipeline", &Layer::destroy_pipeline, py::arg("opt"))
    .def_readwrite("one_blob_only", &Layer::one_blob_only)
    .def_readwrite("support_inplace", &Layer::support_inplace)
    .def_readwrite("support_vulkan", &Layer::support_vulkan)
    .def_readwrite("support_packing", &Layer::support_packing)
    .def_readwrite("support_bf16_storage", &Layer::support_bf16_storage)
    .def_readwrite("support_fp16_storage", &Layer::support_fp16_storage)
    .def_readwrite("support_image_storage", &Layer::support_image_storage)
    .def("forward", (int (Layer::*)(const std::vector<Mat>&, std::vector<Mat>&, const Option&) const) & Layer::forward,
         py::arg("bottom_blobs"), py::arg("top_blobs"), py::arg("opt"))
    .def("forward", (int (Layer::*)(const Mat&, Mat&, const Option&) const) & Layer::forward,
         py::arg("bottom_blob"), py::arg("top_blob"), py::arg("opt"))
    .def("forward_inplace", (int (Layer::*)(std::vector<Mat>&, const Option&) const) & Layer::forward_inplace,
         py::arg("bottom_top_blobs"), py::arg("opt"))
    .def("forward_inplace", (int (Layer::*)(Mat&, const Option&) const) & Layer::forward_inplace,
         py::arg("bottom_top_blob"), py::arg("opt"))
    .def_readwrite("typeindex", &Layer::typeindex)
#if NCNN_STRING
    .def_readwrite("type", &Layer::type)
    .def_readwrite("name", &Layer::name)
#endif // NCNN_STRING
    .def_readwrite("bottoms", &Layer::bottoms)
    .def_readwrite("tops", &Layer::tops)
    .def_readwrite("bottom_shapes", &Layer::bottom_shapes)
    .def_readwrite("top_shapes", &Layer::top_shapes);

    py::class_<Net>(m, "Net")
    .def(py::init<>())
    .def_readwrite("opt", &Net::opt)
    .def("__enter__", [](Net& net) -> Net& { return net; })
    .def("__exit__", [](Net& net, pybind11::args) {
        net.clear();
    })

#if NCNN_VULKAN
    .def("set_vulkan_device", (void (Net::*)(int)) & Net::set_vulkan_device, py::arg("device_index"))
    .def("set_vulkan_device", (void (Net::*)(const VulkanDevice*)) & Net::set_vulkan_device, py::arg("vkdev"))
    .def("vulkan_device", &Net::vulkan_device, py::return_value_policy::reference_internal)
#endif // NCNN_VULKAN

#if NCNN_STRING
    .def(
    "register_custom_layer", [](Net& net, const char* type, const std::function<ncnn::Layer*()>& creator, const std::function<void(ncnn::Layer*)>& destroyer) {
        if (g_layer_factroy_index == g_layer_factroys.size())
        {
            std::stringstream ss;
            ss << "python version only support " << g_layer_factroys.size() << " custom layers now";
            pybind11::pybind11_fail(ss.str());
        }
        LayerFactory& lf = g_layer_factroys[g_layer_factroy_index++];
        lf.name = type;
        lf.creator = creator;
        lf.destroyer = destroyer;
        return net.register_custom_layer(lf.name.c_str(), lf.creator_func, lf.destroyer_func);
    },
    py::arg("type"), py::arg("creator"), py::arg("destroyer"))
#endif //NCNN_STRING
    .def(
    "register_custom_layer", [](Net& net, int index, const std::function<ncnn::Layer*()>& creator, const std::function<void(ncnn::Layer*)>& destroyer) {
        if (g_layer_factroy_index == g_layer_factroys.size())
        {
            std::stringstream ss;
            ss << "python version only support " << g_layer_factroys.size() << " custom layers now";
            pybind11::pybind11_fail(ss.str());
        }
        LayerFactory& lf = g_layer_factroys[g_layer_factroy_index++];
        lf.index = index;
        lf.creator = creator;
        lf.destroyer = destroyer;
        return net.register_custom_layer(index, lf.creator_func, lf.destroyer_func);
    },
    py::arg("index"), py::arg("creator"), py::arg("destroyer"))
#if NCNN_STRING
    .def("load_param", (int (Net::*)(const DataReader&)) & Net::load_param, py::arg("dr"))
#endif // NCNN_STRING
    .def("load_param_bin", (int (Net::*)(const DataReader&)) & Net::load_param_bin, py::arg("dr"))
    .def("load_model", (int (Net::*)(const DataReader&)) & Net::load_model, py::arg("dr"))

#if NCNN_STDIO
#if NCNN_STRING
    .def("load_param", (int (Net::*)(const char*)) & Net::load_param, py::arg("protopath"))
#endif // NCNN_STRING
    .def("load_param_bin", (int (Net::*)(const char*)) & Net::load_param_bin, py::arg("protopath"))
    .def("load_model", (int (Net::*)(const char*)) & Net::load_model, py::arg("modelpath"))
#endif // NCNN_STDIO

    .def("clear", &Net::clear)
    .def("create_extractor", &Net::create_extractor, py::keep_alive<0, 1>()) //net should be kept alive until retuned ex is freed by gc

    .def("input_indexes", &Net::input_indexes, py::return_value_policy::reference)
    .def("input_indexes", &Net::output_indexes, py::return_value_policy::reference)
#if NCNN_STRING
    .def("input_names", &Net::input_names, py::return_value_policy::reference)
    .def("output_names", &Net::output_names, py::return_value_policy::reference)
#endif // NCNN_STRING

    .def("blobs", &Net::blobs, py::return_value_policy::reference_internal)
    .def("layers", &Net::layers, py::return_value_policy::reference_internal);

    py::enum_<ncnn::BorderType>(m, "BorderType")
    .value("BORDER_CONSTANT", ncnn::BorderType::BORDER_CONSTANT)
    .value("BORDER_REPLICATE", ncnn::BorderType::BORDER_REPLICATE);

    m.def("cpu_support_arm_neon", &cpu_support_arm_neon);
    m.def("cpu_support_arm_vfpv4", &cpu_support_arm_vfpv4);
    m.def("cpu_support_arm_asimdhp", &cpu_support_arm_asimdhp);
    m.def("cpu_support_x86_avx2", &cpu_support_x86_avx2);
    m.def("cpu_support_x86_avx", &cpu_support_x86_avx);
    m.def("get_cpu_count", &get_cpu_count);
    m.def("get_little_cpu_count", &get_little_cpu_count);
    m.def("get_big_cpu_count", &get_big_cpu_count);
    m.def("get_cpu_powersave", &get_cpu_powersave);
    m.def("set_cpu_powersave", &set_cpu_powersave, py::arg("powersave"));
    m.def("get_omp_num_threads", &get_omp_num_threads);
    m.def("set_omp_num_threads", &set_omp_num_threads, py::arg("num_threads"));
    m.def("get_omp_dynamic", &get_omp_dynamic);
    m.def("set_omp_dynamic", &set_omp_dynamic, py::arg("dynamic"));
    m.def("get_omp_thread_num", &get_omp_thread_num);
    m.def("get_kmp_blocktime", &get_kmp_blocktime);
    m.def("set_kmp_blocktime", &set_kmp_blocktime, py::arg("time_ms"));

    m.def("copy_make_border", &copy_make_border,
          py::arg("src"), py::arg("dst"),
          py::arg("top"), py::arg("bottom"), py::arg("left"), py::arg("right"),
          py::arg("type"), py::arg("v"), py::arg("opt") = Option());
    m.def(
        "copy_make_border",
    [](const Mat& src, int top, int bottom, int left, int right, int type, float v, const Option& opt) {
        Mat dst;
        copy_make_border(src, dst, top, bottom, left, right, type, v, opt);
        return dst;
    },
    py::arg("src"),
    py::arg("top"), py::arg("bottom"), py::arg("left"), py::arg("right"),
    py::arg("type"), py::arg("v"), py::arg("opt") = Option());

    m.def("copy_make_border_3d", &copy_make_border_3d,
          py::arg("src"), py::arg("dst"),
          py::arg("top"), py::arg("bottom"), py::arg("left"), py::arg("right"), py::arg("front"), py::arg("behind"),
          py::arg("type"), py::arg("v"), py::arg("opt") = Option());
    m.def(
        "copy_make_border_3d",
    [](const Mat& src, int top, int bottom, int left, int right, int front, int behind, int type, float v, const Option& opt) {
        Mat dst;
        copy_make_border_3d(src, dst, top, bottom, left, right, front, behind, type, v, opt);
        return dst;
    },
    py::arg("src"),
    py::arg("top"), py::arg("bottom"), py::arg("left"), py::arg("right"), py::arg("front"), py::arg("behind"),
    py::arg("type"), py::arg("v"), py::arg("opt") = Option());

    m.def("copy_cut_border", &copy_cut_border,
          py::arg("src"), py::arg("dst"),
          py::arg("top"), py::arg("bottom"), py::arg("left"), py::arg("right"),
          py::arg("opt") = Option());
    m.def(
        "copy_cut_border",
    [](const Mat& src, int top, int bottom, int left, int right, const Option& opt) {
        Mat dst;
        copy_cut_border(src, dst, top, bottom, left, right, opt);
        return dst;
    },
    py::arg("src"),
    py::arg("top"), py::arg("bottom"), py::arg("left"), py::arg("right"),
    py::arg("opt") = Option());

    m.def("resize_nearest", &resize_nearest,
          py::arg("src"), py::arg("dst"),
          py::arg("w"), py::arg("h"),
          py::arg("opt") = Option());
    m.def(
        "resize_nearest",
    [](const Mat& src, int w, int h, const Option& opt) {
        Mat dst;
        resize_nearest(src, dst, w, h);
        return dst;
    },
    py::arg("src"),
    py::arg("w"), py::arg("h"),
    py::arg("opt") = Option());

    m.def("resize_bilinear", &resize_bilinear,
          py::arg("src"), py::arg("dst"),
          py::arg("w"), py::arg("h"),
          py::arg("opt") = Option());
    m.def(
        "resize_bilinear",
    [](const Mat& src, int w, int h, const Option& opt) {
        Mat dst;
        resize_bilinear(src, dst, w, h, opt);
        return dst;
    },
    py::arg("src"),
    py::arg("w"), py::arg("h"),
    py::arg("opt") = Option());

    m.def("resize_bicubic", &resize_bicubic,
          py::arg("src"), py::arg("dst"),
          py::arg("w"), py::arg("h"),
          py::arg("opt") = Option());
    m.def(
        "resize_bicubic",
    [](const Mat& src, int w, int h, const Option& opt) {
        Mat dst;
        resize_bicubic(src, dst, w, h, opt);
        return dst;
    },
    py::arg("src"),
    py::arg("w"), py::arg("h"),
    py::arg("opt") = Option());

    m.def("convert_packing", &convert_packing,
          py::arg("src"), py::arg("dst"),
          py::arg("elempack"),
          py::arg("opt") = Option());
    m.def(
        "convert_packing",
    [](const Mat& src, int elempack, const Option& opt) {
        Mat dst;
        convert_packing(src, dst, elempack, opt);
        return dst;
    },
    py::arg("src"),
    py::arg("elempack"),
    py::arg("opt") = Option());

    m.def("flatten", &flatten,
          py::arg("src"), py::arg("dst"),
          py::arg("opt") = Option());
    m.def(
        "flatten",
    [](const Mat& src, const Option& opt) {
        Mat dst;
        flatten(src, dst, opt);
        return dst;
    },
    py::arg("src"),
    py::arg("opt") = Option());

    m.def("cast_float32_to_float16", &cast_float32_to_float16,
          py::arg("src"), py::arg("dst"),
          py::arg("opt") = Option());
    m.def(
        "cast_float32_to_float16",
    [](const Mat& src, const Option& opt) {
        Mat dst;
        cast_float32_to_float16(src, dst, opt);
        return dst;
    },
    py::arg("src"),
    py::arg("opt") = Option());

    m.def("cast_float16_to_float32", &cast_float16_to_float32,
          py::arg("src"), py::arg("dst"),
          py::arg("opt") = Option());
    m.def(
        "cast_float16_to_float32",
    [](const Mat& src, const Option& opt) {
        Mat dst;
        cast_float16_to_float32(src, dst, opt);
        return dst;
    },
    py::arg("src"),
    py::arg("opt") = Option());

    m.def("cast_int8_to_float32", &cast_int8_to_float32,
          py::arg("src"), py::arg("dst"),
          py::arg("opt") = Option());
    m.def(
        "cast_int8_to_float32",
    [](const Mat& src, const Option& opt) {
        Mat dst;
        cast_int8_to_float32(src, dst, opt);
        return dst;
    },
    py::arg("src"),
    py::arg("opt") = Option());

    m.def("cast_float32_to_bfloat16", &cast_float32_to_bfloat16,
          py::arg("src"), py::arg("dst"),
          py::arg("opt") = Option());
    m.def(
        "cast_float32_to_bfloat16",
    [](const Mat& src, const Option& opt) {
        Mat dst;
        cast_float32_to_bfloat16(src, dst, opt);
        return dst;
    },
    py::arg("src"),
    py::arg("opt") = Option());

    m.def("cast_bfloat16_to_float32", &cast_bfloat16_to_float32,
          py::arg("src"), py::arg("dst"),
          py::arg("opt") = Option());
    m.def(
        "cast_bfloat16_to_float32",
    [](const Mat& src, const Option& opt) {
        Mat dst;
        cast_bfloat16_to_float32(src, dst, opt);
        return dst;
    },
    py::arg("src"),
    py::arg("opt") = Option());

    m.def("quantize_to_int8", &quantize_to_int8,
          py::arg("src"), py::arg("dst"),
          py::arg("scale_data"),
          py::arg("opt") = Option());
    m.def(
        "quantize_to_int8",
    [](const Mat& src, const Mat& scale_data, const Option& opt) {
        Mat dst;
        quantize_to_int8(src, dst, scale_data, opt);
        return dst;
    },
    py::arg("src"),
    py::arg("scale_data"),
    py::arg("opt") = Option());

#if NCNN_STRING
    m.def("layer_to_index", &layer_to_index, py::arg("type"));
    m.def(
        "create_layer",
    [](const char* type) {
        return static_cast<Layer*>(create_layer(type));
    },
    py::arg("type"));
    m.def(
        "create_layer",
    [](int index) {
        return static_cast<Layer*>(create_layer(index));
    },
    py::arg("index"));
#endif //NCNN_STRING

#if NCNN_VULKAN
    m.def("create_gpu_instance", &create_gpu_instance);
    m.def("destroy_gpu_instance", &destroy_gpu_instance);
    m.def("get_gpu_count", &get_gpu_count);
    m.def("get_default_gpu_index", &get_default_gpu_index);
    m.def("get_gpu_info", &get_gpu_info, py::arg("device_index") = 0, py::return_value_policy::reference);
    m.def("get_gpu_device", &get_gpu_device, py::arg("device_index") = 0, py::return_value_policy::reference);

    py::class_<VkBufferMemory>(m, "VkBufferMemory")
    .def_readwrite("offset", &VkBufferMemory::offset)
    .def_readwrite("capacity", &VkBufferMemory::capacity)
    .def_readwrite("refcount", &VkBufferMemory::refcount);

    py::class_<VkImageMemory>(m, "VkImageMemory")
    .def_readwrite("width", &VkImageMemory::width)
    .def_readwrite("height", &VkImageMemory::height)
    .def_readwrite("depth", &VkImageMemory::depth)
    .def_readwrite("refcount", &VkImageMemory::refcount);

    py::class_<VkAllocator, PyVkAllocator<> >(m, "VkAllocator")
    .def_readonly("vkdev", &VkAllocator::vkdev)
    .def_readwrite("buffer_memory_type_index", &VkAllocator::buffer_memory_type_index)
    .def_readwrite("image_memory_type_index", &VkAllocator::image_memory_type_index)
    .def_readwrite("mappable", &VkAllocator::mappable)
    .def_readwrite("coherent", &VkAllocator::coherent);

    py::class_<VkBlobAllocator, VkAllocator, PyVkAllocatorOther<VkBlobAllocator> >(m, "VkBlobAllocator")
    .def(py::init<const VulkanDevice*>())
    .def("clear", &VkBlobAllocator::clear)
    .def("fastMalloc", (VkBufferMemory * (VkBlobAllocator::*)(size_t size)) & VkBlobAllocator::fastMalloc, py::return_value_policy::reference_internal)
    .def("fastFree", (void (VkBlobAllocator::*)(VkBufferMemory * ptr)) & VkBlobAllocator::fastFree)
    .def("fastMalloc", (VkImageMemory * (VkBlobAllocator::*)(int, int, int, size_t, int)) & VkBlobAllocator::fastMalloc, py::return_value_policy::reference_internal)
    .def("fastFree", (void (VkBlobAllocator::*)(VkImageMemory * ptr)) & VkBlobAllocator::fastFree);

    py::class_<VkWeightAllocator, VkAllocator, PyVkAllocatorOther<VkWeightAllocator> >(m, "VkWeightAllocator")
    .def(py::init<const VulkanDevice*>())
    .def("clear", &VkWeightAllocator::clear)
    .def("fastMalloc", (VkBufferMemory * (VkWeightAllocator::*)(size_t size)) & VkWeightAllocator::fastMalloc, py::return_value_policy::reference_internal)
    .def("fastFree", (void (VkWeightAllocator::*)(VkBufferMemory * ptr)) & VkWeightAllocator::fastFree)
    .def("fastMalloc", (VkImageMemory * (VkWeightAllocator::*)(int, int, int, size_t, int)) & VkWeightAllocator::fastMalloc, py::return_value_policy::reference_internal)
    .def("fastFree", (void (VkWeightAllocator::*)(VkImageMemory * ptr)) & VkWeightAllocator::fastFree);

    py::class_<VkStagingAllocator, VkAllocator, PyVkAllocatorOther<VkStagingAllocator> >(m, "VkStagingAllocator")
    .def(py::init<const VulkanDevice*>())
    .def("set_size_compare_ratio", &VkStagingAllocator::set_size_compare_ratio)
    .def("clear", &VkStagingAllocator::clear)
    .def("fastMalloc", (VkBufferMemory * (VkStagingAllocator::*)(size_t size)) & VkStagingAllocator::fastMalloc, py::return_value_policy::reference_internal)
    .def("fastFree", (void (VkStagingAllocator::*)(VkBufferMemory * ptr)) & VkStagingAllocator::fastFree)
    .def("fastMalloc", (VkImageMemory * (VkStagingAllocator::*)(int, int, int, size_t, int)) & VkStagingAllocator::fastMalloc, py::return_value_policy::reference_internal)
    .def("fastFree", (void (VkStagingAllocator::*)(VkImageMemory * ptr)) & VkStagingAllocator::fastFree);

    py::class_<VkWeightStagingAllocator, VkAllocator, PyVkAllocatorOther<VkWeightStagingAllocator> >(m, "VkWeightStagingAllocator")
    .def(py::init<const VulkanDevice*>())
    .def("fastMalloc", (VkBufferMemory * (VkWeightStagingAllocator::*)(size_t size)) & VkWeightStagingAllocator::fastMalloc, py::return_value_policy::reference_internal)
    .def("fastFree", (void (VkWeightStagingAllocator::*)(VkBufferMemory * ptr)) & VkWeightStagingAllocator::fastFree)
    .def("fastMalloc", (VkImageMemory * (VkWeightStagingAllocator::*)(int, int, int, size_t, int)) & VkWeightStagingAllocator::fastMalloc, py::return_value_policy::reference_internal)
    .def("fastFree", (void (VkWeightStagingAllocator::*)(VkImageMemory * ptr)) & VkWeightStagingAllocator::fastFree);

    py::class_<GpuInfo>(m, "GpuInfo")
    .def(py::init<>())
    .def("api_version", &GpuInfo::api_version)
    .def("driver_version", &GpuInfo::driver_version)
    .def("vendor_id", &GpuInfo::vendor_id)
    .def("device_id", &GpuInfo::device_id)
    .def("pipeline_cache_uuid", [](GpuInfo& gpuinfo) {
        return py::memoryview::from_buffer(gpuinfo.pipeline_cache_uuid(), {VK_UUID_SIZE}, {sizeof(uint8_t) * VK_UUID_SIZE});
    })
    .def("type", &GpuInfo::type);

    py::class_<VulkanDevice>(m, "VulkanDevice")
    .def(py::init<int>(), py::arg("device_index") = 0)
    .def(
    "info", [](VulkanDevice& dev) {
        return &dev.info;
    },
    py::return_value_policy::reference_internal);
#endif // NCNN_VULKAN

    m.doc() = R"pbdoc(
        ncnn python wrapper
        -----------------------
        .. currentmodule:: pyncnn
        .. autosummary::
           :toctree: _generate
    )pbdoc";

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
