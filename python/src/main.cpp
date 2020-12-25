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

#define LayerFactoryDeclear(n) \
static ncnn::Layer* LayerCreator##n();  \
static void LayerDestroyer##n(ncnn::Layer*);

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

#define LayerFactoryDefine(n) \
static ncnn::Layer* LayerCreator##n()   \
{   \
    return g_layer_factroys[n].creator();   \
}   \
static void LayerDestroyer##n(ncnn::Layer* layer)     \
{                                         \
    g_layer_factroys[n].destroyer(layer); \
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
    py::class_<Allocator, PyAllocator<> >(m, "Allocator");
    py::class_<PoolAllocator, Allocator, PyAllocatorOther<PoolAllocator> >(m, "PoolAllocator")
        .def(py::init<>())
        .def("set_size_compare_ratio", &PoolAllocator::set_size_compare_ratio)
        .def("clear", &PoolAllocator::clear)
        .def("fastMalloc", &PoolAllocator::fastMalloc)
        .def("fastFree", &PoolAllocator::fastFree);
    py::class_<UnlockedPoolAllocator, Allocator, PyAllocatorOther<UnlockedPoolAllocator> >(m, "UnlockedPoolAllocator")
        .def(py::init<>())
        .def("set_size_compare_ratio", &UnlockedPoolAllocator::set_size_compare_ratio)
        .def("clear", &UnlockedPoolAllocator::clear)
        .def("fastMalloc", &UnlockedPoolAllocator::fastMalloc)
        .def("fastFree", &UnlockedPoolAllocator::fastFree);

    py::class_<DataReader, PyDataReader<> >(m, "DataReader")
        .def(py::init<>())
        .def("scan", &DataReader::scan)
        .def("read", &DataReader::read);
    py::class_<DataReaderFromEmpty, DataReader, PyDataReaderOther<DataReaderFromEmpty> >(m, "DataReaderFromEmpty")
        .def(py::init<>())
        .def("scan", &DataReaderFromEmpty::scan)
        .def("read", &DataReaderFromEmpty::read);

    py::class_<Blob>(m, "Blob")
        .def(py::init<>())
#if NCNN_STRING
        .def_readwrite("name", &Blob::name)
#endif // NCNN_STRING
        .def_readwrite("producer", &Blob::producer)
        .def_readwrite("consumers", &Blob::consumers)
        .def_readwrite("shape", &Blob::shape);

    py::class_<ModelBin, PyModelBin<> >(m, "ModelBin");
    py::class_<ModelBinFromDataReader, ModelBin, PyModelBinOther<ModelBinFromDataReader> >(m, "ModelBinFromDataReader")
        .def(py::init<const DataReader&>())
        .def("load", &ModelBinFromDataReader::load);
    py::class_<ModelBinFromMatArray, ModelBin, PyModelBinOther<ModelBinFromMatArray> >(m, "ModelBinFromMatArray")
        .def(py::init<const Mat*>())
        .def("load", &ModelBinFromMatArray::load);

    py::class_<ParamDict>(m, "ParamDict")
        .def(py::init<>())
        .def("get", (int (ParamDict::*)(int, int) const) & ParamDict::get)
        .def("get", (float (ParamDict::*)(int, float) const) & ParamDict::get)
        .def("get", (Mat(ParamDict::*)(int, const Mat&) const) & ParamDict::get)
        .def("set", (void (ParamDict::*)(int, int)) & ParamDict::set)
        .def("set", (void (ParamDict::*)(int, float)) & ParamDict::set)
        .def("set", (void (ParamDict::*)(int, const Mat&)) & ParamDict::set);

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
#endif // NCNN_VULKAN
        .def_readwrite("use_winograd_convolution", &Option::openmp_blocktime)
        .def_readwrite("use_winograd_convolution", &Option::use_winograd_convolution)
        .def_readwrite("use_sgemm_convolution", &Option::use_sgemm_convolution)
        .def_readwrite("use_int8_inference", &Option::use_int8_inference)
        .def_readwrite("use_vulkan_compute", &Option::use_vulkan_compute)
        .def_readwrite("use_fp16_packed", &Option::use_fp16_packed)
        .def_readwrite("use_fp16_storage", &Option::use_fp16_storage)
        .def_readwrite("use_fp16_arithmetic", &Option::use_fp16_arithmetic)
        .def_readwrite("use_int8_storage", &Option::use_int8_storage)
        .def_readwrite("use_int8_arithmetic", &Option::use_int8_arithmetic)
        .def_readwrite("use_packing_layout", &Option::use_packing_layout)
        .def_readwrite("use_shader_pack8", &Option::use_shader_pack8)
        .def_readwrite("use_shader_pack8", &Option::use_image_storage)
        .def_readwrite("use_subgroup_basic", &Option::use_subgroup_basic)
        .def_readwrite("use_subgroup_vote", &Option::use_subgroup_vote)
        .def_readwrite("use_subgroup_ballot", &Option::use_subgroup_ballot)
        .def_readwrite("use_subgroup_shuffle", &Option::use_subgroup_shuffle)
        .def_readwrite("use_image_storage", &Option::use_image_storage)
        .def_readwrite("use_bf16_storage", &Option::use_bf16_storage)
        .def_readwrite("use_bf16_storage", &Option::use_weight_fp16_storage);

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
                         mat = new Mat(shape[0].cast<int>(), shape[1].cast<int>(), shape[2].cast<int>(),
                                       elemsize, elempack, allocator);
                         break;
                     default:
                         std::stringstream ss;
                         ss << "shape must be 1, 2 or 3 dims, not " << shape.size();
                         throw pybind11::value_error(ss.str());
                     }
                     return mat;
                 }),
             py::arg("shape") = py::tuple(1), py::arg("elemsize") = 4,
             py::arg("elempack") = 1, py::arg("allocator") = nullptr)
        .def(py::init<int, size_t, int, Allocator*>(),
             py::arg("w") = 1,
             py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
        .def(py::init<int, int, size_t, int, Allocator*>(),
             py::arg("w") = 1, py::arg("h") = 1,
             py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
        .def(py::init<int, int, int, size_t, int, Allocator*>(),
             py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1,
             py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)

        .def(py::init<const Mat&>())

        .def(py::init(
                 [](py::tuple shape, void* data, size_t elemsize, int elempack, Allocator* allocator) {
                     Mat* mat = nullptr;
                     switch (shape.size())
                     {
                     case 1:
                         mat = new Mat(shape[0].cast<int>(), data, elemsize, elempack, allocator);
                         break;
                     case 2:
                         mat = new Mat(shape[0].cast<int>(), shape[1].cast<int>(), data, elemsize, elempack, allocator);
                         break;
                     case 3:
                         mat = new Mat(shape[0].cast<int>(), shape[1].cast<int>(), shape[2].cast<int>(), data,
                                       elemsize, elempack, allocator);
                         break;
                     default:
                         std::stringstream ss;
                         ss << "shape must be 1, 2 or 3 dims, not " << shape.size();
                         throw pybind11::value_error(ss.str());
                     }
                     return mat;
                 }),
             py::arg("shape") = py::tuple(1), py::arg("data") = nullptr,
             py::arg("elemsize") = 4, py::arg("elempack") = 1,
             py::arg("allocator") = nullptr)
        .def(py::init<int, void*, size_t, int, Allocator*>(),
             py::arg("w") = 1, py::arg("data") = nullptr,
             py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
        .def(py::init<int, int, void*, size_t, int, Allocator*>(),
             py::arg("w") = 1, py::arg("h") = 1, py::arg("data") = nullptr,
             py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
        .def(py::init<int, int, int, void*, size_t, int, Allocator*>(),
             py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1, py::arg("data") = nullptr,
             py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
        .def(py::init([](py::buffer const b) {
            py::buffer_info info = b.request();
            if (info.ndim > 3)
            {
                throw std::runtime_error("Incompatible buffer dims");
            }

            //printf("numpy dtype = %s\n", info.format.c_str());
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
            }
            return v;
        }))
        .def_buffer([](Mat& m) -> py::buffer_info {
            std::string format = get_mat_format(m);
            std::vector<ssize_t> shape;
            std::vector<ssize_t> strides;
            //todo strides not correct
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
            return py::buffer_info(
                m.data,     /* Pointer to buffer */
                m.elemsize, /* Size of one scalar */
                format,     /* Python struct-style format descriptor */
                m.dims,     /* Number of dimensions */
                shape,      /* Buffer dimensions */
                strides     /* Strides (in bytes) for each index */
            );
        })
        //todo assign
        //.def(py::self=py::self)
        .def("fill", (void (Mat::*)(int))(&Mat::fill))
        .def("fill", (void (Mat::*)(float))(&Mat::fill))
        .def("clone", (Mat(Mat::*)(Allocator*)) & Mat::clone, py::arg("allocator") = nullptr)
        .def("clone_from", &Mat::clone_from)
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
                default:
                    std::stringstream ss;
                    ss << "shape must be 1, 2 or 3 dims, not " << shape.size();
                    throw pybind11::value_error(ss.str());
                }
                return Mat();
            },
            py::arg("shape") = py::tuple(1), py::arg("allocator") = nullptr)
        .def("reshape", (Mat(Mat::*)(int, Allocator*) const) & Mat::reshape,
             py::arg("w") = 1, py::arg("allocator") = nullptr)
        .def("reshape", (Mat(Mat::*)(int, int, Allocator*) const) & Mat::reshape,
             py::arg("w") = 1, py::arg("h") = 1, py::arg("allocator") = nullptr)
        .def("reshape", (Mat(Mat::*)(int, int, int, Allocator*) const) & Mat::reshape,
             py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1, py::arg("allocator") = nullptr)

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
                    return mat.create(shape[0].cast<int>(), shape[1].cast<int>(), shape[2].cast<int>(),
                                      elemsize, elempack, allocator);
                default:
                    std::stringstream ss;
                    ss << "shape must be 1, 2 or 3 dims, not " << shape.size();
                    throw pybind11::value_error(ss.str());
                }
                return;
            },
            py::arg("shape") = py::tuple(1),
            py::arg("elemsize") = 4, py::arg("elempack") = 1,
            py::arg("allocator") = nullptr)
        .def("create", (void (Mat::*)(int, size_t, int, Allocator*)) & Mat::create,
             py::arg("w") = 1,
             py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
        .def("create", (void (Mat::*)(int, int, size_t, int, Allocator*)) & Mat::create,
             py::arg("w") = 1, py::arg("h") = 1,
             py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
        .def("create", (void (Mat::*)(int, int, int, size_t, int, Allocator*)) & Mat::create,
             py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1,
             py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
        .def("create_like", (void (Mat::*)(const Mat&, Allocator*)) & Mat::create_like,
             py::arg("m") = nullptr, py::arg("allocator") = nullptr)
#if NCNN_VULKAN
        .def("create_like", (void (Mat::*)(const VkMat&, Allocator*)) & Mat::create_like,
             py::arg("m") = nullptr, py::arg("allocator") = nullptr)
        .def("create_like", (void (Mat::*)(const VkImageMat&, Allocator*)) & Mat::create_like,
             py::arg("im") = nullptr, py::arg("allocator") = nullptr)
#endif // NCNN_VULKAN
        .def("addref", &Mat::addref)
        .def("release", &Mat::release)
        .def("empty", &Mat::empty)
        .def("total", &Mat::total)
        .def("channel", (Mat(Mat::*)(int)) & Mat::channel)
        .def("channel", (const Mat (Mat::*)(int) const) & Mat::channel)
        .def("row", [](Mat& m, int y) {
            if (m.elemsize != 4)
            {
                throw std::runtime_error("only float/int32 type mat.row support now");
            }
            return py::array_t<float>(m.w, m.row(y));
        })
        .def("channel_range", (Mat(Mat::*)(int, int)) & Mat::channel_range)
        .def("channel_range", (const Mat (Mat::*)(int, int) const) & Mat::channel_range)
        .def("row_range", (Mat(Mat::*)(int, int)) & Mat::row_range)
        .def("row_range", (const Mat (Mat::*)(int, int) const) & Mat::row_range)
        .def("range", (Mat(Mat::*)(int, int)) & Mat::range)
        .def("range", (const Mat (Mat::*)(int, int) const) & Mat::range)
        .def("__getitem__", [](const Mat& m, size_t i) {
            return m[i];
        })
        .def("__setitem__", [](Mat& m, size_t i, float v) {
            m[i] = v;
        })
        //convenient construct from pixel data
        .def_static("from_pixels", [](py::buffer const b, int type, int w, int h) {
            return Mat::from_pixels((const unsigned char*)b.request().ptr, type, w, h);
        })
        .def_static("from_pixels", [](py::buffer const b, int type, int w, int h, Allocator* allocator) {
            return Mat::from_pixels((const unsigned char*)b.request().ptr, type, w, h, allocator);
        })
        .def_static("from_pixels", [](py::buffer const b, int type, int w, int h, int stride) {
            return Mat::from_pixels((const unsigned char*)b.request().ptr, type, w, h, stride);
        })
        .def_static("from_pixels", [](py::buffer const b, int type, int w, int h, int stride, Allocator* allocator) {
            return Mat::from_pixels((const unsigned char*)b.request().ptr, type, w, h, stride, allocator);
        })
        .def_static("from_pixels_resize", [](py::buffer const b, int type, int w, int h, int target_width, int target_height) {
            return Mat::from_pixels_resize((const unsigned char*)b.request().ptr, type, w, h, target_width, target_height);
        })
        .def_static("from_pixels_resize", [](py::buffer const b, int type, int w, int h, int target_width, int target_height, Allocator* allocator) {
            return Mat::from_pixels_resize((const unsigned char*)b.request().ptr, type, w, h, target_width, target_height, allocator);
        })
        .def_static("from_pixels_resize", [](py::buffer const b, int type, int w, int h, int stride, int target_width, int target_height) {
            return Mat::from_pixels_resize((const unsigned char*)b.request().ptr, type, w, h, stride, target_width, target_height);
        })
        .def_static("from_pixels_resize", [](py::buffer const b, int type, int w, int h, int stride, int target_width, int target_height, Allocator* allocator) {
            return Mat::from_pixels_resize((const unsigned char*)b.request().ptr, type, w, h, stride, target_width, target_height, allocator);
        })
        .def("to_pixels", (void (Mat::*)(unsigned char*, int) const) & Mat::to_pixels)
        .def("to_pixels", (void (Mat::*)(unsigned char*, int, int) const) & Mat::to_pixels)
        .def("to_pixels_resize", (void (Mat::*)(unsigned char*, int, int, int) const) & Mat::to_pixels_resize)
        .def("to_pixels_resize", (void (Mat::*)(unsigned char*, int, int, int, int) const) & Mat::to_pixels_resize)
        .def("substract_mean_normalize", [](Mat& mat, std::vector<float>& mean, std::vector<float>& norm) {
            return mat.substract_mean_normalize(mean.size() > 0 ? &mean[0] : 0, norm.size() > 0 ? &norm[0] : 0);
        })
        .def("from_float16", &Mat::from_float16)
        .def_readwrite("data", &Mat::data)
        .def_readwrite("refcount", &Mat::refcount)
        .def_readwrite("elemsize", &Mat::elemsize)
        .def_readwrite("elempack", &Mat::elempack)
        .def_readwrite("allocator", &Mat::allocator)
        .def_readwrite("dims", &Mat::dims)
        .def_readwrite("w", &Mat::w)
        .def_readwrite("h", &Mat::h)
        .def_readwrite("c", &Mat::c)
        .def_readwrite("cstep", &Mat::cstep)
        .def("__repr__", [](const Mat& m) {
            char buf[256] = {0};
            sprintf(buf, "<ncnn.Mat w=%d h=%d c=%d dims=%d cstep=%zd elemsize=%zd elempack=%d\n\trefcount=%d data=0x%p allocator=0x%p>",
                    m.w, m.h, m.c, m.dims, m.cstep, m.elemsize, m.elempack, m.refcount ? *m.refcount : 0, m.data, m.allocator);
            return std::string(buf);
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

#if NCNN_VULKAN
    py::class_<VkMat>(m, "VkMat")
        .def(py::init<>())
        .def(py::init(
                 [](py::tuple shape, size_t elemsize, int elempack, VkAllocator* allocator) {
                     VkMat* mat = nullptr;
                     switch (shape.size())
                     {
                     case 1:
                         mat = new VkMat(shape[0].cast<int>(), elemsize, elempack, allocator);
                         break;
                     case 2:
                         mat = new VkMat(shape[0].cast<int>(), shape[1].cast<int>(), elemsize, elempack, allocator);
                         break;
                     case 3:
                         mat = new VkMat(shape[0].cast<int>(), shape[1].cast<int>(), shape[2].cast<int>(),
                                         elemsize, elempack, allocator);
                         break;
                     default:
                         std::stringstream ss;
                         ss << "shape must be 1, 2 or 3 dims, not " << shape.size();
                         throw pybind11::value_error(ss.str());
                     }
                     return mat;
                 }),
             py::arg("shape") = py::tuple(1), py::arg("elemsize") = 4,
             py::arg("elempack") = 1, py::arg("allocator") = nullptr)
        .def(py::init<int, size_t, int, VkAllocator*>(),
             py::arg("w") = 1,
             py::arg("elemsize") = 4, py::arg("elempack") = 1,
             py::arg("allocator") = nullptr)
        .def(py::init<int, int, size_t, int, VkAllocator*>(),
             py::arg("w") = 1, py::arg("h") = 1,
             py::arg("elemsize") = 4, py::arg("elempack") = 1,
             py::arg("allocator") = nullptr)
        .def(py::init<int, int, int, size_t, int, VkAllocator*>(),
             py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1,
             py::arg("elemsize") = 4, py::arg("elempack") = 1,
             py::arg("allocator") = nullptr)

        .def(py::init<const VkMat&>())

        .def(py::init(
                 [](py::tuple shape, VkBufferMemory* data, size_t elemsize, int elempack, VkAllocator* allocator) {
                     VkMat* mat = nullptr;
                     switch (shape.size())
                     {
                     case 1:
                         mat = new VkMat(shape[0].cast<int>(), data, elemsize, elempack, allocator);
                         break;
                     case 2:
                         mat = new VkMat(shape[0].cast<int>(), shape[1].cast<int>(), data, elemsize, elempack, allocator);
                         break;
                     case 3:
                         mat = new VkMat(shape[0].cast<int>(), shape[1].cast<int>(), shape[2].cast<int>(), data,
                                         elemsize, elempack, allocator);
                         break;
                     default:
                         std::stringstream ss;
                         ss << "shape must be 1, 2 or 3 dims, not " << shape.size();
                         throw pybind11::value_error(ss.str());
                     }
                     return mat;
                 }),
             py::arg("shape") = py::tuple(1), py::arg("data") = nullptr,
             py::arg("elemsize") = 4, py::arg("elempack") = 1,
             py::arg("allocator") = nullptr)
        .def(py::init<int, VkBufferMemory*, size_t, int, VkAllocator*>(),
             py::arg("w") = 1, py::arg("data") = nullptr,
             py::arg("elemsize") = 4, py::arg("elempack") = 1,
             py::arg("allocator") = nullptr)
        .def(py::init<int, int, VkBufferMemory*, size_t, int, VkAllocator*>(),
             py::arg("w") = 1, py::arg("h") = 1, py::arg("data") = nullptr,
             py::arg("elemsize") = 4, py::arg("elempack") = 1,
             py::arg("allocator") = nullptr)
        .def(py::init<int, int, int, VkBufferMemory*, size_t, int, VkAllocator*>(),
             py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1, py::arg("data") = nullptr,
             py::arg("elemsize") = 4, py::arg("elempack") = 1,
             py::arg("allocator") = nullptr)

        .def(
            "create",
            [](VkMat& mat, py::tuple shape, size_t elemsize, int elempack, VkAllocator* allocator) {
                switch (shape.size())
                {
                case 1:
                    return mat.create(shape[0].cast<int>(), elemsize, elempack, allocator);
                case 2:
                    return mat.create(shape[0].cast<int>(), shape[1].cast<int>(), elemsize, elempack, allocator);
                case 3:
                    return mat.create(shape[0].cast<int>(), shape[1].cast<int>(), shape[2].cast<int>(),
                                      elemsize, elempack, allocator);
                default:
                    std::stringstream ss;
                    ss << "shape must be 1, 2 or 3 dims, not " << shape.size();
                    throw pybind11::value_error(ss.str());
                }
                return;
            },
            py::arg("shape") = py::tuple(1),
            py::arg("elemsize") = 4, py::arg("elempack") = 1,
            py::arg("allocator") = nullptr)
        .def("create", (void (VkMat::*)(int, size_t, int, VkAllocator*)) & VkMat::create,
             py::arg("w") = 1,
             py::arg("elemsize") = 4, py::arg("elempack") = 1,
             py::arg("allocator") = nullptr)
        .def("create", (void (VkMat::*)(int, int, size_t, int, VkAllocator*)) & VkMat::create,
             py::arg("w") = 1, py::arg("h") = 1,
             py::arg("elemsize") = 4, py::arg("elempack") = 1,
             py::arg("allocator") = nullptr)
        .def("create", (void (VkMat::*)(int, int, int, size_t, int, VkAllocator*)) & VkMat::create,
             py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1,
             py::arg("elemsize") = 4, py::arg("elempack") = 1,
             py::arg("allocator") = nullptr)

        .def("create_like", (void (VkMat::*)(const Mat&, VkAllocator*)) & VkMat::create_like)
        .def("create_like", (void (VkMat::*)(const VkMat&, VkAllocator*)) & VkMat::create_like)
        .def("mapped", &VkMat::mapped)
        .def("mapped_ptr", &VkMat::mapped_ptr)
        .def("addref", &VkMat::addref)
        .def("release", &VkMat::release)
        .def("empty", &VkMat::empty)
        .def("total", &VkMat::total)
        .def("elembits", &VkMat::elembits)
        .def("shape", &VkMat::shape)
        //.def("buffer", &VkMat::buffer)
        .def("buffer_offset", &VkMat::buffer_offset)
        .def("buffer_capacity", &VkMat::buffer_capacity)
        .def_readwrite("data", &VkMat::data)
        .def_readwrite("refcount", &VkMat::refcount)
        .def_readwrite("elemsize", &VkMat::elemsize)
        .def_readwrite("elempack", &VkMat::elempack)
        .def_readwrite("allocator", &VkMat::allocator)
        .def_readwrite("dims", &VkMat::dims)
        .def_readwrite("w", &VkMat::w)
        .def_readwrite("h", &VkMat::h)
        .def_readwrite("c", &VkMat::c)
        .def_readwrite("cstep", &VkMat::cstep);

    py::class_<VkImageMat>(m, "VkImageMat")
        .def(py::init<>())
        .def(py::init(
                 [](py::tuple shape, size_t elemsize, int elempack, VkAllocator* allocator) {
                     VkImageMat* mat = nullptr;
                     switch (shape.size())
                     {
                     case 1:
                         mat = new VkImageMat(shape[0].cast<int>(), elemsize, elempack, allocator);
                         break;
                     case 2:
                         mat = new VkImageMat(shape[0].cast<int>(), shape[1].cast<int>(), elemsize, elempack, allocator);
                         break;
                     case 3:
                         mat = new VkImageMat(shape[0].cast<int>(), shape[1].cast<int>(), shape[2].cast<int>(),
                                              elemsize, elempack, allocator);
                         break;
                     default:
                         std::stringstream ss;
                         ss << "shape must be 1, 2 or 3 dims, not " << shape.size();
                         throw pybind11::value_error(ss.str());
                     }
                     return mat;
                 }),
             py::arg("shape") = py::tuple(1), py::arg("elemsize") = 4,
             py::arg("elempack") = 1, py::arg("allocator") = nullptr)
        .def(py::init<int, size_t, int, VkAllocator*>(),
             py::arg("w") = 1,
             py::arg("elemsize") = 4, py::arg("elempack") = 1,
             py::arg("allocator") = nullptr)
        .def(py::init<int, int, size_t, int, VkAllocator*>(),
             py::arg("w") = 1, py::arg("h") = 1,
             py::arg("elemsize") = 4, py::arg("elempack") = 1,
             py::arg("allocator") = nullptr)
        .def(py::init<int, int, int, size_t, int, VkAllocator*>(),
             py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1,
             py::arg("elemsize") = 4, py::arg("elempack") = 1,
             py::arg("allocator") = nullptr)

        .def(py::init<const VkImageMat&>())

        .def(py::init(
                 [](py::tuple shape, VkImageMemory* data, size_t elemsize, int elempack, VkAllocator* allocator) {
                     VkImageMat* mat = nullptr;
                     switch (shape.size())
                     {
                     case 1:
                         mat = new VkImageMat(shape[0].cast<int>(), data, elemsize, elempack, allocator);
                         break;
                     case 2:
                         mat = new VkImageMat(shape[0].cast<int>(), shape[1].cast<int>(), data, elemsize, elempack, allocator);
                         break;
                     case 3:
                         mat = new VkImageMat(shape[0].cast<int>(), shape[1].cast<int>(), shape[2].cast<int>(), data,
                                              elemsize, elempack, allocator);
                         break;
                     default:
                         std::stringstream ss;
                         ss << "shape must be 1, 2 or 3 dims, not " << shape.size();
                         throw pybind11::value_error(ss.str());
                     }
                     return mat;
                 }),
             py::arg("shape") = py::tuple(1), py::arg("data") = nullptr,
             py::arg("elemsize") = 4, py::arg("elempack") = 1,
             py::arg("allocator") = nullptr)
        .def(py::init<int, VkImageMemory*, size_t, int, VkAllocator*>(),
             py::arg("w") = 1, py::arg("data") = nullptr,
             py::arg("elemsize") = 4, py::arg("elempack") = 1,
             py::arg("allocator") = nullptr)
        .def(py::init<int, int, VkImageMemory*, size_t, int, VkAllocator*>(),
             py::arg("w") = 1, py::arg("h") = 1, py::arg("data") = nullptr,
             py::arg("elemsize") = 4, py::arg("elempack") = 1,
             py::arg("allocator") = nullptr)
        .def(py::init<int, int, int, VkImageMemory*, size_t, int, VkAllocator*>(),
             py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1, py::arg("data") = nullptr,
             py::arg("elemsize") = 4, py::arg("elempack") = 1,
             py::arg("allocator") = nullptr)

        .def(
            "create",
            [](VkImageMat& mat, py::tuple shape, size_t elemsize, int elempack, VkAllocator* allocator) {
                switch (shape.size())
                {
                case 1:
                    return mat.create(shape[0].cast<int>(), elemsize, elempack, allocator);
                case 2:
                    return mat.create(shape[0].cast<int>(), shape[1].cast<int>(), elemsize, elempack, allocator);
                case 3:
                    return mat.create(shape[0].cast<int>(), shape[1].cast<int>(), shape[2].cast<int>(),
                                      elemsize, elempack, allocator);
                default:
                    std::stringstream ss;
                    ss << "shape must be 1, 2 or 3 dims, not " << shape.size();
                    throw pybind11::value_error(ss.str());
                }
                return;
            },
            py::arg("shape") = py::tuple(1),
            py::arg("elemsize") = 4, py::arg("elempack") = 1,
            py::arg("allocator") = nullptr)
        .def("create", (void (VkImageMat::*)(int, size_t, int, VkAllocator*)) & VkImageMat::create,
             py::arg("w") = 1,
             py::arg("elemsize") = 4, py::arg("elempack") = 1,
             py::arg("allocator") = nullptr)
        .def("create", (void (VkImageMat::*)(int, int, size_t, int, VkAllocator*)) & VkImageMat::create,
             py::arg("w") = 1, py::arg("h") = 1,
             py::arg("elemsize") = 4, py::arg("elempack") = 1,
             py::arg("allocator") = nullptr)
        .def("create", (void (VkImageMat::*)(int, int, int, size_t, int, VkAllocator*)) & VkImageMat::create,
             py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1,
             py::arg("elemsize") = 4, py::arg("elempack") = 1,
             py::arg("allocator") = nullptr)

        .def("create_like", (void (VkImageMat::*)(const Mat&, VkAllocator*)) & VkImageMat::create_like)
        .def("create_like", (void (VkImageMat::*)(const VkMat&, VkAllocator*)) & VkImageMat::create_like)
        .def("create_like", (void (VkImageMat::*)(const VkImageMat&, VkAllocator*)) & VkImageMat::create_like)

        .def("mapped", &VkImageMat::mapped)
        //.def_readwrite("mapped_ptr", &VkImageMat::mapped_ptr)
        .def("addref", &VkImageMat::addref)
        .def("release", &VkImageMat::release)
        .def("empty", &VkImageMat::empty)
        .def("total", &VkImageMat::total)
        .def("elembits", &VkImageMat::elembits)
        .def("shape", &VkImageMat::shape)
        //.def("image", &VkImageMat::image)
        //.def("imageview", &VkImageMat::imageview)
        .def_readwrite("data", &VkImageMat::data)
        .def_readwrite("refcount", &VkImageMat::refcount)
        .def_readwrite("elemsize", &VkImageMat::elemsize)
        .def_readwrite("elempack", &VkImageMat::elempack)
        .def_readwrite("allocator", &VkImageMat::allocator)
        .def_readwrite("dims", &VkImageMat::dims)
        .def_readwrite("w", &VkImageMat::w)
        .def_readwrite("h", &VkImageMat::h)
        .def_readwrite("c", &VkImageMat::c);
#endif //NCNN_VULKAN

    py::class_<Extractor>(m, "Extractor")
        .def("set_light_mode", &Extractor::set_light_mode)
        .def("set_num_threads", &Extractor::set_num_threads)
        .def("set_blob_allocator", &Extractor::set_blob_allocator)
        .def("set_workspace_allocator", &Extractor::set_workspace_allocator)
#if NCNN_VULKAN
        .def("set_vulkan_compute", &Extractor::set_vulkan_compute)
        .def("set_blob_vkallocator", &Extractor::set_blob_vkallocator)
        .def("set_workspace_vkallocator", &Extractor::set_workspace_vkallocator)
        .def("set_staging_vkallocator", &Extractor::set_staging_vkallocator)
#endif // NCNN_VULKAN
#if NCNN_STRING
        .def("input", (int (Extractor::*)(const char*, const Mat&)) & Extractor::input)
        .def("extract", (int (Extractor::*)(const char*, Mat&, int)) & Extractor::extract,
             "get result by blob name", py::arg("blob_name"), py::arg("feat"), py::arg("type") = 0)
        .def(
            "extract", [](Extractor& ex, const char* blob_name, int type) {
                ncnn::Mat feat;
                int ret = ex.extract(blob_name, feat, type);
                return py::make_tuple(ret, feat);
            },
            "get result by blob name", py::arg("blob_name"), py::arg("type") = 0)
#endif
        .def("input", (int (Extractor::*)(int, const Mat&)) & Extractor::input)
        .def("extract", (int (Extractor::*)(int, Mat&, int)) & Extractor::extract, "get result by blob index", py::arg("blob_index"), py::arg("feat"), py::arg("type") = 0)
        .def(
            "extract", [](Extractor& ex, int blob_index, int type) {
                ncnn::Mat feat;
                int ret = ex.extract(blob_index, feat, type);
                return py::make_tuple(ret, feat);
            },
            "get result by blob index", py::arg("blob_index"), py::arg("type") = 0)
#if NCNN_VULKAN
        .def("input", (int (Extractor::*)(const char*, const VkMat&)) & Extractor::input)
        .def("extract", (int (Extractor::*)(const char*, VkMat&, VkCompute&)) & Extractor::extract)
#if NCNN_STRING
#endif // NCNN_STRING
        .def("input", (int (Extractor::*)(int, const VkMat&)) & Extractor::input)
        .def("extract", (int (Extractor::*)(int, VkMat&, VkCompute&)) & Extractor::extract)
#endif // NCNN_VULKAN
        ;

    py::class_<Layer, PyLayer>(m, "Layer")
        .def(py::init<>())
        .def("load_param", &Layer::load_param)
        .def("load_model", &Layer::load_model)
        .def("create_pipeline", &Layer::create_pipeline)
        .def("destroy_pipeline", &Layer::destroy_pipeline)
        .def_readwrite("one_blob_only", &Layer::one_blob_only)
        .def_readwrite("support_inplace", &Layer::support_inplace)
        .def_readwrite("support_vulkan", &Layer::support_vulkan)
        .def_readwrite("support_packing", &Layer::support_packing)
        .def_readwrite("support_bf16_storage", &Layer::support_bf16_storage)
        .def_readwrite("support_fp16_storage", &Layer::support_fp16_storage)
        .def_readwrite("support_image_storage", &Layer::support_image_storage)
        .def_readwrite("use_int8_inference", &Layer::use_int8_inference)
        .def_readwrite("support_weight_fp16_storage", &Layer::support_weight_fp16_storage)
        .def("forward", (int (Layer::*)(const std::vector<Mat>&, std::vector<Mat>&, const Option&) const) & Layer::forward)
        .def("forward", (int (Layer::*)(const Mat&, Mat&, const Option&) const) & Layer::forward)
        //.def("forward_inplace", ( int( Layer::* )( std::vector<Mat>&, const Option& ) const )&Layer::forward_inplace)
        .def("forward_inplace", (int (Layer::*)(Mat&, const Option&) const) & Layer::forward_inplace)
#if NCNN_VULKAN
        .def("upload_model", &Layer::upload_model)
        .def("forward", (int (Layer::*)(const std::vector<VkMat>&, std::vector<VkMat>&, VkCompute&, const Option&) const) & Layer::forward)
        .def("forward", (int (Layer::*)(const VkMat&, VkMat&, VkCompute& cmd, const Option&) const) & Layer::forward)
        .def("forward_inplace", (int (Layer::*)(std::vector<VkMat>&, VkCompute&, const Option&) const) & Layer::forward_inplace)
        .def("forward_inplace", (int (Layer::*)(VkMat&, VkCompute&, const Option&) const) & Layer::forward_inplace)
#endif // NCNN_VULKAN
        .def_readwrite("typeindex", &Layer::typeindex)
        .def_readwrite("type", &Layer::type)
        .def_readwrite("name", &Layer::name)
        .def_readwrite("bottoms", &Layer::bottoms)
        .def_readwrite("tops", &Layer::tops)
        .def_readwrite("bottom_shapes", &Layer::bottom_shapes)
        .def_readwrite("top_shapes", &Layer::top_shapes);

    py::class_<Net>(m, "Net")
        .def(py::init<>())
        .def_readwrite("opt", &Net::opt)
#if NCNN_VULKAN
        .def("set_vulkan_device", (void (Net::*)(int)) & Net::set_vulkan_device)
        .def("set_vulkan_device", (void (Net::*)(const VulkanDevice*)) & Net::set_vulkan_device)
        .def("vulkan_device", &Net::vulkan_device)
#endif // NCNN_VULKAN
#if NCNN_STRING
        .def("register_custom_layer", [](Net& net, const char* type, 
            const std::function<ncnn::Layer*()>& creator, 
            const std::function<void(ncnn::Layer*)>& destroyer) {
            if (g_layer_factroy_index == g_layer_factroys.size())
            {
                std::stringstream ss;
                ss << "python version only support " << g_layer_factroys.size() << " custom layers now";
                throw pybind11::value_error(ss.str());
            }
            LayerFactory& lf = g_layer_factroys[g_layer_factroy_index++];
            lf.name = type;
            lf.creator = creator;
            lf.destroyer = destroyer;
            return net.register_custom_layer(lf.name.c_str(), lf.creator_func, lf.destroyer_func);
        })
#endif //NCNN_STRING
        .def("register_custom_layer", [](Net& net, int index, const std::function<ncnn::Layer*()>& creator, const std::function<void(ncnn::Layer*)>& destroyer) {
            if (g_layer_factroy_index == g_layer_factroys.size())
            {
                std::stringstream ss;
                ss << "python version only support " << g_layer_factroys.size() << " custom layers now";
                throw pybind11::value_error(ss.str());
            }
            LayerFactory& lf = g_layer_factroys[g_layer_factroy_index++];
            lf.index = index;
            lf.creator = creator;
            lf.destroyer = destroyer;
            return net.register_custom_layer(index, lf.creator_func, lf.destroyer_func);
        })
#if NCNN_STRING
        .def("load_param", (int (Net::*)(const DataReader&)) & Net::load_param)
#endif // NCNN_STRING
        .def("load_param_bin", (int (Net::*)(const DataReader&)) & Net::load_param_bin)
        .def("load_model", (int (Net::*)(const DataReader&)) & Net::load_model)

#if NCNN_STDIO
#if NCNN_STRING
        .def("load_param", (int (Net::*)(const char*)) & Net::load_param)
        .def("load_param_mem", (int (Net::*)(const char*)) & Net::load_param_mem)
#endif // NCNN_STRING
        .def("load_param_bin", (int (Net::*)(const char*)) & Net::load_param_bin)
        .def("load_model", (int (Net::*)(const char*)) & Net::load_model)
#endif // NCNN_STDIO

        //todo load from memory
        //.def("load_param", (int (Net::*)(const unsigned char*))(&Net::load_param))
        //.def("load_model", (int (Net::*)(const unsigned char*))(&Net::load_model))

        .def("clear", &Net::clear)
        .def("create_extractor", &Net::create_extractor);

    py::enum_<ncnn::BorderType>(m, "BorderType")
        .value("BORDER_CONSTANT", ncnn::BorderType::BORDER_CONSTANT)
        .value("BORDER_REPLICATE", ncnn::BorderType::BORDER_REPLICATE);

    m.def("cpu_support_arm_neon", &cpu_support_arm_neon);
    m.def("cpu_support_arm_vfpv4", &cpu_support_arm_vfpv4);
    m.def("cpu_support_arm_asimdhp", &cpu_support_arm_asimdhp);
    m.def("get_cpu_count", &get_cpu_count);
    m.def("get_cpu_powersave", &get_cpu_powersave);
    m.def("set_cpu_powersave", &set_cpu_powersave);
    m.def("get_omp_num_threads", &get_omp_num_threads);
    m.def("set_omp_num_threads", &set_omp_num_threads);
    m.def("get_omp_dynamic", &get_omp_dynamic);
    m.def("set_omp_dynamic", &set_omp_dynamic);
    m.def("build_with_gpu", []() {
#if NCNN_VULKAN
        return true;
#else
            return false;
#endif
    });
#if NCNN_PIXEL
    m.def("yuv420sp2rgb", [](py::buffer const yuv420sp, int w, int h, py::buffer rgb) {
        return yuv420sp2rgb((unsigned char*)yuv420sp.request().ptr, w, h, (unsigned char*)rgb.request().ptr);
    });
    m.def("resize_bilinear_c1", [](py::buffer const src, int srcw, int srch, py::buffer dst, int w, int h) {
        return resize_bilinear_c1((unsigned char*)src.request().ptr, srcw, srch, (unsigned char*)dst.request().ptr, w, h);
    });
    m.def("resize_bilinear_c2", [](py::buffer const src, int srcw, int srch, py::buffer dst, int w, int h) {
        return resize_bilinear_c2((unsigned char*)src.request().ptr, srcw, srch, (unsigned char*)dst.request().ptr, w, h);
    });
    m.def("resize_bilinear_c3", [](py::buffer const src, int srcw, int srch, py::buffer dst, int w, int h) {
        return resize_bilinear_c3((unsigned char*)src.request().ptr, srcw, srch, (unsigned char*)dst.request().ptr, w, h);
    });
    m.def("resize_bilinear_c4", [](py::buffer const src, int srcw, int srch, py::buffer dst, int w, int h) {
        return resize_bilinear_c4((unsigned char*)src.request().ptr, srcw, srch, (unsigned char*)dst.request().ptr, w, h);
    });
    m.def("resize_bilinear_c1", [](py::buffer const src, int srcw, int srch, int srcstride, py::buffer dst, int w, int h, int stride) {
        return resize_bilinear_c1((unsigned char*)src.request().ptr, srcw, srch, srcstride, (unsigned char*)dst.request().ptr, w, h, stride);
    });
    m.def("resize_bilinear_c2", [](py::buffer const src, int srcw, int srch, int srcstride, py::buffer dst, int w, int h, int stride) {
        return resize_bilinear_c2((unsigned char*)src.request().ptr, srcw, srch, srcstride, (unsigned char*)dst.request().ptr, w, h, stride);
    });
    m.def("resize_bilinear_c3", [](py::buffer const src, int srcw, int srch, int srcstride, py::buffer dst, int w, int h, int stride) {
        return resize_bilinear_c3((unsigned char*)src.request().ptr, srcw, srch, srcstride, (unsigned char*)dst.request().ptr, w, h, stride);
    });
    m.def("resize_bilinear_c4", [](py::buffer const src, int srcw, int srch, int srcstride, py::buffer dst, int w, int h, int stride) {
        return resize_bilinear_c4((unsigned char*)src.request().ptr, srcw, srch, srcstride, (unsigned char*)dst.request().ptr, w, h, stride);
    });
    m.def("resize_bilinear_yuv420sp", [](py::buffer const src, int srcw, int srch, py::buffer dst, int w, int h) {
        return resize_bilinear_yuv420sp((unsigned char*)src.request().ptr, srcw, srch, (unsigned char*)dst.request().ptr, w, h);
    });
#endif // NCNN_PIXEL
#if NCNN_PIXEL_ROTATE
    m.def("kanna_rotate_c1", [](py::buffer const src, int srcw, int srch, py::buffer dst, int w, int h, int type) {
        return kanna_rotate_c1((unsigned char*)src.request().ptr, srcw, srch, (unsigned char*)dst.request().ptr, w, h, type);
    });
    m.def("kanna_rotate_c2", [](py::buffer const src, int srcw, int srch, py::buffer dst, int w, int h, int type) {
        return kanna_rotate_c2((unsigned char*)src.request().ptr, srcw, srch, (unsigned char*)dst.request().ptr, w, h, type);
    });
    m.def("kanna_rotate_c3", [](py::buffer const src, int srcw, int srch, py::buffer dst, int w, int h, int type) {
        return kanna_rotate_c3((unsigned char*)src.request().ptr, srcw, srch, (unsigned char*)dst.request().ptr, w, h, type);
    });
    m.def("kanna_rotate_c4", [](py::buffer const src, int srcw, int srch, py::buffer dst, int w, int h, int type) {
        return kanna_rotate_c4((unsigned char*)src.request().ptr, srcw, srch, (unsigned char*)dst.request().ptr, w, h, type);
    });
    m.def("kanna_rotate_c1", [](py::buffer const src, int srcw, int srch, int srcstride, py::buffer dst, int w, int h, int stride, int type) {
        return kanna_rotate_c1((unsigned char*)src.request().ptr, srcw, srch, srcstride, (unsigned char*)dst.request().ptr, w, h, stride, type);
    });
    m.def("kanna_rotate_c2", [](py::buffer const src, int srcw, int srch, int srcstride, py::buffer dst, int w, int h, int stride, int type) {
        return kanna_rotate_c2((unsigned char*)src.request().ptr, srcw, srch, srcstride, (unsigned char*)dst.request().ptr, w, h, stride, type);
    });
    m.def("kanna_rotate_c3", [](py::buffer const src, int srcw, int srch, int srcstride, py::buffer dst, int w, int h, int stride, int type) {
        return kanna_rotate_c3((unsigned char*)src.request().ptr, srcw, srch, srcstride, (unsigned char*)dst.request().ptr, w, h, stride, type);
    });
    m.def("kanna_rotate_c4", [](py::buffer const src, int srcw, int srch, int srcstride, py::buffer dst, int w, int h, int stride, int type) {
        return kanna_rotate_c4((unsigned char*)src.request().ptr, srcw, srch, srcstride, (unsigned char*)dst.request().ptr, w, h, stride, type);
    });
    m.def("kanna_rotate_yuv420sp", [](py::buffer const src, int srcw, int srch, py::buffer dst, int w, int h, int type) {
        return kanna_rotate_yuv420sp((unsigned char*)src.request().ptr, srcw, srch, (unsigned char*)dst.request().ptr, w, h, type);
    });
#endif // NCNN_PIXEL_ROTATE
    m.def("copy_make_border", [](const Mat& src, Mat& dst, int top, int bottom, int left, int right, int type, float v) {
        return copy_make_border(src, dst, top, bottom, left, right, type, v);
    });
    m.def("copy_make_border", [](const Mat& src, int top, int bottom, int left, int right, int type, float v) {
        Mat dst;
        copy_make_border(src, dst, top, bottom, left, right, type, v);
        return dst;
    });
    m.def("copy_make_border", [](const Mat& src, Mat& dst, int top, int bottom, int left, int right, int type, float v, const Option& opt) {
        return copy_make_border(src, dst, top, bottom, left, right, type, v, opt);
    });
    m.def("copy_make_border", [](const Mat& src, int top, int bottom, int left, int right, int type, float v, const Option& opt) {
        Mat dst;
        copy_make_border(src, dst, top, bottom, left, right, type, v, opt);
        return dst;
    });
    m.def("copy_cut_border", [](const Mat& src, Mat& dst, int top, int bottom, int left, int right) {
        return copy_cut_border(src, dst, top, bottom, left, right);
    });
    m.def("copy_cut_border", [](const Mat& src, int top, int bottom, int left, int right) {
        Mat dst;
        copy_cut_border(src, dst, top, bottom, left, right);
        return dst;
    });
    m.def("copy_cut_border", [](const Mat& src, Mat& dst, int top, int bottom, int left, int right, const Option& opt) {
        return copy_cut_border(src, dst, top, bottom, left, right, opt);
    });
    m.def("copy_cut_border", [](const Mat& src, int top, int bottom, int left, int right, const Option& opt) {
        Mat dst;
        copy_cut_border(src, dst, top, bottom, left, right, opt);
        return dst;
    });
    m.def("resize_bilinear", [](const Mat& src, Mat& dst, int w, int h) {
        return resize_bilinear(src, dst, w, h);
    });
    m.def("resize_bilinear", [](const Mat& src, int w, int h) {
        Mat dst;
        resize_bilinear(src, dst, w, h);
        return dst;
    });
    m.def("resize_bilinear", [](const Mat& src, Mat& dst, int w, int h, const Option& opt) {
        return resize_bilinear(src, dst, w, h, opt);
    });
    m.def("resize_bilinear", [](const Mat& src, int w, int h, const Option& opt) {
        Mat dst;
        resize_bilinear(src, dst, w, h, opt);
        return dst;
    });
    m.def("resize_bicubic", [](const Mat& src, Mat& dst, int w, int h) {
        return resize_bicubic(src, dst, w, h);
    });
    m.def("resize_bicubic", [](const Mat& src, int w, int h) {
        Mat dst;
        resize_bicubic(src, dst, w, h);
        return dst;
    });
    m.def("resize_bicubic", [](const Mat& src, Mat& dst, int w, int h, const Option& opt) {
        return resize_bicubic(src, dst, w, h, opt);
    });
    m.def("resize_bicubic", [](const Mat& src, int w, int h, const Option& opt) {
        Mat dst;
        resize_bicubic(src, dst, w, h, opt);
        return dst;
    });
    m.def("convert_packing", [](const Mat& src, Mat& dst, int elempack) {
        return convert_packing(src, dst, elempack);
    });
    m.def("convert_packing", [](const Mat& src, int elempack) {
        Mat dst;
        convert_packing(src, dst, elempack);
        return dst;
    });
    m.def("convert_packing", [](const Mat& src, Mat& dst, int elempack, const Option& opt) {
        return convert_packing(src, dst, elempack, opt);
    });
    m.def("convert_packing", [](const Mat& src, int elempack, const Option& opt) {
        Mat dst;
        convert_packing(src, dst, elempack, opt);
        return dst;
    });
    m.def("cast_float32_to_float16", [](const Mat& src, Mat& dst) {
        return cast_float32_to_float16(src, dst);
    });
    m.def("cast_float32_to_float16", [](const Mat& src) {
        Mat dst;
        cast_float32_to_float16(src, dst);
        return dst;
    });
    m.def("cast_float32_to_float16", [](const Mat& src, Mat& dst, const Option& opt) {
        return cast_float32_to_float16(src, dst, opt);
    });
    m.def("cast_float32_to_float16", [](const Mat& src, const Option& opt) {
        Mat dst;
        cast_float32_to_float16(src, dst, opt);
        return dst;
    });
    m.def("cast_float16_to_float32", [](const Mat& src, Mat& dst) {
        return cast_float16_to_float32(src, dst);
    });
    m.def("cast_float16_to_float32", [](const Mat& src) {
        Mat dst;
        cast_float16_to_float32(src, dst);
        return dst;
    });
    m.def("cast_float16_to_float32", [](const Mat& src, Mat& dst, const Option& opt) {
        return cast_float16_to_float32(src, dst, opt);
    });
    m.def("cast_float16_to_float32", [](const Mat& src, const Option& opt) {
        Mat dst;
        cast_float16_to_float32(src, dst, opt);
        return dst;
    });
    m.def("cast_int8_to_float32", [](const Mat& src, Mat& dst) {
        return cast_int8_to_float32(src, dst);
    });
    m.def("cast_int8_to_float32", [](const Mat& src) {
        Mat dst;
        cast_int8_to_float32(src, dst);
        return dst;
    });
    m.def("cast_int8_to_float32", [](const Mat& src, Mat& dst, const Option& opt) {
        return cast_int8_to_float32(src, dst, opt);
    });
    m.def("cast_int8_to_float32", [](const Mat& src, const Option& opt) {
        Mat dst;
        cast_int8_to_float32(src, dst, opt);
        return dst;
    });
    m.def("quantize_float32_to_int8", [](const Mat& src, Mat& dst, float scale) {
        return quantize_float32_to_int8(src, dst, scale);
    });
    m.def("quantize_float32_to_int8", [](const Mat& src, float scale) {
        Mat dst;
        quantize_float32_to_int8(src, dst, scale);
        return dst;
    });
    m.def("quantize_float32_to_int8", [](const Mat& src, Mat& dst, float scale, const Option& opt) {
        return quantize_float32_to_int8(src, dst, scale, opt);
    });
    m.def("quantize_float32_to_int8", [](const Mat& src, float scale, const Option& opt) {
        Mat dst;
        quantize_float32_to_int8(src, dst, scale, opt);
        return dst;
    });
    m.def("dequantize_int32_to_float32", [](Mat& m, float scale, py::buffer bias, int bias_data_size) {
        return dequantize_int32_to_float32(m, scale, (float*)bias.request().ptr, bias_data_size);
    });
    m.def("dequantize_int32_to_float32", [](Mat& m, float scale, py::buffer bias, int bias_data_size, const Option& opt) {
        return dequantize_int32_to_float32(m, scale, (float*)bias.request().ptr, bias_data_size, opt);
    });
    m.def("requantize_int8_to_int8", [](const Mat& src, Mat& dst, float scale_in, float scale_out, py::buffer bias, int bias_data_size, int fusion_relu) {
        return requantize_int8_to_int8(src, dst, scale_in, scale_out, (float*)bias.request().ptr, bias_data_size, fusion_relu);
    });
    m.def("requantize_int8_to_int8", [](const Mat& src, float scale_in, float scale_out, py::buffer bias, int bias_data_size, int fusion_relu) {
        Mat dst;
        requantize_int8_to_int8(src, dst, scale_in, scale_out, (float*)bias.request().ptr, bias_data_size, fusion_relu);
        return dst;
    });
    m.def("requantize_int8_to_int8", [](const Mat& src, Mat& dst, float scale_in, float scale_out, py::buffer bias, int bias_data_size, int fusion_relu, const Option& opt) {
        return requantize_int8_to_int8(src, dst, scale_in, scale_out, (float*)bias.request().ptr, bias_data_size, fusion_relu, opt);
    });
    m.def("requantize_int8_to_int8", [](const Mat& src, float scale_in, float scale_out, py::buffer bias, int bias_data_size, int fusion_relu, const Option& opt) {
        Mat dst;
        requantize_int8_to_int8(src, dst, scale_in, scale_out, (float*)bias.request().ptr, bias_data_size, fusion_relu, opt);
        return dst;
    });
#if NCNN_STRING
    m.def("layer_to_index", &layer_to_index);
    m.def("create_layer", [](const char* type) {
        return static_cast<Layer*>(create_layer(type));
    });
    m.def("create_layer", [](int index) {
        return static_cast<Layer*>(create_layer(index));
    });
    m.def("unregister_custom_layer", [](const char* type) {
        for (int i = 0; i < g_layer_factroys.size(); i++)
        {
            if (g_layer_factroys[i].name == type)
            {
                g_layer_factroys[i].name = "";
                g_layer_factroys[i].creator = nullptr;
                g_layer_factroys[i].destroyer = nullptr;
            }
            return true;
        }
        return false;
    });
#endif //NCNN_STRING
    m.def("unregister_custom_layer", [](int index) {
        for (int i = 0; i < g_layer_factroys.size(); i++)
        {
            if (g_layer_factroys[i].index == index)
            {
                g_layer_factroys[i].index = -1;
                g_layer_factroys[i].creator = nullptr;
                g_layer_factroys[i].destroyer = nullptr;
            }
            return true;
        }
        return false;
    });

#if NCNN_VULKAN
    m.def("create_gpu_instance", &create_gpu_instance);
    m.def("destroy_gpu_instance", &destroy_gpu_instance);
    m.def("get_gpu_count", &get_gpu_count);
    m.def("get_default_gpu_index", &get_default_gpu_index);
    m.def("get_gpu_info", &get_gpu_info, py::arg("device_index") = 0);
    m.def("get_gpu_device", &get_gpu_device, py::arg("device_index") = 0);

    py::class_<VkBufferMemory>(m, "VkBufferMemory")
        .def(py::init<>())
        //.def_readwrite("buffer", &VkBufferMemory::buffer)
        .def_readwrite("offset", &VkBufferMemory::offset)
        .def_readwrite("capacity", &VkBufferMemory::capacity)
        //.def_readwrite("memory", &VkBufferMemory::memory)
        .def_readwrite("mapped_ptr", &VkBufferMemory::mapped_ptr)
        .def_readwrite("refcount", &VkBufferMemory::refcount);

    py::class_<VkImageMemory>(m, "VkImageMemory")
        .def(py::init<>())
        //.def_readwrite("image", &VkImageMemory::image)
        //.def_readwrite("imageview", &VkImageMemory::imageview)
        .def_readwrite("width", &VkImageMemory::width)
        .def_readwrite("height", &VkImageMemory::height)
        .def_readwrite("depth", &VkImageMemory::depth)
        .def_readwrite("format", &VkImageMemory::format)
        //.def_readwrite("memory", &VkImageMemory::memory)
        .def_readwrite("mapped_ptr", &VkImageMemory::mapped_ptr)
        .def_readwrite("bind_offset", &VkImageMemory::bind_offset)
        .def_readwrite("bind_capacity", &VkImageMemory::bind_capacity)

        .def_readwrite("refcount", &VkImageMemory::refcount);

    py::class_<VkAllocator, PyVkAllocator<> >(m, "VkAllocator")
        .def_readwrite("vkdev", &VkAllocator::vkdev)
        .def_readwrite("buffer_memory_type_index", &VkAllocator::buffer_memory_type_index)
        .def_readwrite("image_memory_type_index", &VkAllocator::image_memory_type_index)
        .def_readwrite("mappable", &VkAllocator::mappable)
        .def_readwrite("coherent", &VkAllocator::coherent);

    py::class_<VkBlobAllocator, VkAllocator, PyVkAllocatorOther<VkBlobAllocator> >(m, "VkBlobAllocator")
        .def(py::init<const VulkanDevice*>())
        .def("clear", &VkBlobAllocator::clear)
        .def("fastMalloc", (VkBufferMemory * (VkBlobAllocator::*)(size_t size)) & VkBlobAllocator::fastMalloc)
        .def("fastFree", (void (VkBlobAllocator::*)(VkBufferMemory * ptr)) & VkBlobAllocator::fastFree)
        .def("fastMalloc", (VkImageMemory * (VkBlobAllocator::*)(int, int, int, size_t, int)) & VkBlobAllocator::fastMalloc)
        .def("fastFree", (void (VkBlobAllocator::*)(VkImageMemory * ptr)) & VkBlobAllocator::fastFree);

    py::class_<VkWeightAllocator, VkAllocator, PyVkAllocatorOther<VkWeightAllocator> >(m, "VkWeightAllocator")
        .def(py::init<const VulkanDevice*>())
        .def("clear", &VkWeightAllocator::clear)
        .def("fastMalloc", (VkBufferMemory * (VkWeightAllocator::*)(size_t size)) & VkWeightAllocator::fastMalloc)
        .def("fastFree", (void (VkWeightAllocator::*)(VkBufferMemory * ptr)) & VkWeightAllocator::fastFree)
        .def("fastMalloc", (VkImageMemory * (VkWeightAllocator::*)(int, int, int, size_t, int)) & VkWeightAllocator::fastMalloc)
        .def("fastFree", (void (VkWeightAllocator::*)(VkImageMemory * ptr)) & VkWeightAllocator::fastFree);

    py::class_<VkStagingAllocator, VkAllocator, PyVkAllocatorOther<VkStagingAllocator> >(m, "VkStagingAllocator")
        .def(py::init<const VulkanDevice*>())
        .def("set_size_compare_ratio", &VkStagingAllocator::set_size_compare_ratio)
        .def("clear", &VkStagingAllocator::clear)
        .def("fastMalloc", (VkBufferMemory * (VkStagingAllocator::*)(size_t size)) & VkStagingAllocator::fastMalloc)
        .def("fastFree", (void (VkStagingAllocator::*)(VkBufferMemory * ptr)) & VkStagingAllocator::fastFree)
        .def("fastMalloc", (VkImageMemory * (VkStagingAllocator::*)(int, int, int, size_t, int)) & VkStagingAllocator::fastMalloc)
        .def("fastFree", (void (VkStagingAllocator::*)(VkImageMemory * ptr)) & VkStagingAllocator::fastFree);

    py::class_<VkWeightStagingAllocator, VkAllocator, PyVkAllocatorOther<VkWeightStagingAllocator> >(m, "VkWeightStagingAllocator")
        .def(py::init<const VulkanDevice*>())
        .def("fastMalloc", (VkBufferMemory * (VkWeightStagingAllocator::*)(size_t size)) & VkWeightStagingAllocator::fastMalloc)
        .def("fastFree", (void (VkWeightStagingAllocator::*)(VkBufferMemory * ptr)) & VkWeightStagingAllocator::fastFree)
        .def("fastMalloc", (VkImageMemory * (VkWeightStagingAllocator::*)(int, int, int, size_t, int)) & VkWeightStagingAllocator::fastMalloc)
        .def("fastFree", (void (VkWeightStagingAllocator::*)(VkImageMemory * ptr)) & VkWeightStagingAllocator::fastFree);

    py::class_<GpuInfo>(m, "GpuInfo")
        .def(py::init<>())
        //.def_readwrite("physical_device", &GpuInfo::physical_device)

        //.def_readwrite("physicalDeviceMemoryProperties", &GpuInfo::physicalDeviceMemoryProperties)

        .def_readwrite("api_version", &GpuInfo::api_version)
        .def_readwrite("driver_version", &GpuInfo::driver_version)
        .def_readwrite("vendor_id", &GpuInfo::vendor_id)
        .def_readwrite("device_id", &GpuInfo::device_id)
        .def_readwrite("device_name", &GpuInfo::device_name)
        //.def_readwrite("pipeline_cache_uuid", &GpuInfo::pipeline_cache_uuid)

        .def_readwrite("type", &GpuInfo::type)

        .def_readwrite("max_shared_memory_size", &GpuInfo::max_shared_memory_size)
        //.def_readwrite("max_workgroup_count", &GpuInfo::max_workgroup_count)
        .def_readwrite("max_workgroup_invocations", &GpuInfo::max_workgroup_invocations)
        //.def_readwrite("max_workgroup_size", &GpuInfo::max_workgroup_size)
        .def_readwrite("memory_map_alignment", &GpuInfo::memory_map_alignment)
        .def_readwrite("buffer_offset_alignment", &GpuInfo::buffer_offset_alignment)
        .def_readwrite("non_coherent_atom_size", &GpuInfo::non_coherent_atom_size)
        .def_readwrite("buffer_image_granularity", &GpuInfo::buffer_image_granularity)
        .def_readwrite("max_image_dimension_1d", &GpuInfo::max_image_dimension_1d)
        .def_readwrite("max_image_dimension_2d", &GpuInfo::max_image_dimension_2d)
        .def_readwrite("max_image_dimension_3d", &GpuInfo::max_image_dimension_3d)
        .def_readwrite("timestamp_period", &GpuInfo::timestamp_period)

        .def_readwrite("compute_queue_family_index", &GpuInfo::compute_queue_family_index)
        .def_readwrite("graphics_queue_family_index", &GpuInfo::graphics_queue_family_index)
        .def_readwrite("transfer_queue_family_index", &GpuInfo::transfer_queue_family_index)

        .def_readwrite("compute_queue_count", &GpuInfo::compute_queue_count)
        .def_readwrite("graphics_queue_count", &GpuInfo::graphics_queue_count)
        .def_readwrite("transfer_queue_count", &GpuInfo::transfer_queue_count)

        .def_readwrite("unified_compute_transfer_queue", &GpuInfo::unified_compute_transfer_queue)

        .def_readwrite("subgroup_size", &GpuInfo::subgroup_size)
        .def_readwrite("support_subgroup_basic", &GpuInfo::support_subgroup_basic)
        .def_readwrite("support_subgroup_vote", &GpuInfo::support_subgroup_vote)
        .def_readwrite("support_subgroup_ballot", &GpuInfo::support_subgroup_ballot)
        .def_readwrite("support_subgroup_shuffle", &GpuInfo::support_subgroup_shuffle)

        .def_readwrite("bug_storage_buffer_no_l1", &GpuInfo::bug_storage_buffer_no_l1)
        .def_readwrite("bug_corrupted_online_pipeline_cache", &GpuInfo::bug_corrupted_online_pipeline_cache)

        .def_readwrite("bug_implicit_fp16_arithmetic", &GpuInfo::bug_implicit_fp16_arithmetic)

        .def_readwrite("support_fp16_packed", &GpuInfo::support_fp16_packed)
        .def_readwrite("support_fp16_storage", &GpuInfo::support_fp16_storage)
        .def_readwrite("support_fp16_arithmetic", &GpuInfo::support_fp16_arithmetic)
        .def_readwrite("support_int8_storage", &GpuInfo::support_int8_storage)
        .def_readwrite("support_int8_arithmetic", &GpuInfo::support_int8_arithmetic)

        .def_readwrite("support_ycbcr_conversion", &GpuInfo::support_ycbcr_conversion)

        .def_readwrite("support_VK_KHR_8bit_storage", &GpuInfo::support_VK_KHR_8bit_storage)
        .def_readwrite("support_VK_KHR_16bit_storage", &GpuInfo::support_VK_KHR_16bit_storage)
        .def_readwrite("support_VK_KHR_bind_memory2", &GpuInfo::support_VK_KHR_bind_memory2)
        .def_readwrite("support_VK_KHR_dedicated_allocation", &GpuInfo::support_VK_KHR_dedicated_allocation)
        .def_readwrite("support_VK_KHR_descriptor_update_template", &GpuInfo::support_VK_KHR_descriptor_update_template)
        .def_readwrite("support_VK_KHR_external_memory", &GpuInfo::support_VK_KHR_external_memory)
        .def_readwrite("support_VK_KHR_get_memory_requirements2", &GpuInfo::support_VK_KHR_get_memory_requirements2)
        .def_readwrite("support_VK_KHR_maintenance1", &GpuInfo::support_VK_KHR_maintenance1)
        .def_readwrite("support_VK_KHR_push_descriptor", &GpuInfo::support_VK_KHR_push_descriptor)
        .def_readwrite("support_VK_KHR_sampler_ycbcr_conversion", &GpuInfo::support_VK_KHR_sampler_ycbcr_conversion)
        .def_readwrite("support_VK_KHR_shader_float16_int8", &GpuInfo::support_VK_KHR_shader_float16_int8)
        .def_readwrite("support_VK_KHR_shader_float_controls", &GpuInfo::support_VK_KHR_shader_float_controls)
        .def_readwrite("support_VK_KHR_storage_buffer_storage_class", &GpuInfo::support_VK_KHR_storage_buffer_storage_class)
        .def_readwrite("support_VK_KHR_swapchain", &GpuInfo::support_VK_KHR_swapchain)
        .def_readwrite("support_VK_EXT_memory_budget", &GpuInfo::support_VK_EXT_memory_budget)
        .def_readwrite("support_VK_EXT_queue_family_foreign", &GpuInfo::support_VK_EXT_queue_family_foreign);

    py::class_<VulkanDevice>(m, "VulkanDevice")
        .def(py::init<int>(), py::arg("device_index") = 0)
        //.def_readonly("info", &VulkanDevice::info)
        //.def("get_shader_module", &VulkanDevice::get_shader_module)
        //.def("create_shader_module", &VulkanDevice::create_shader_module)
        //.def("compile_shader_module", &VulkanDevice::compile_shader_module)
        //.def("compile_shader_module", &VulkanDevice::compile_shader_module)
        .def("find_memory_index", &VulkanDevice::find_memory_index)
        .def("is_mappable", &VulkanDevice::is_mappable)
        .def("is_coherent", &VulkanDevice::is_coherent)
        //.def("acquire_queue", &VulkanDevice::acquire_queue)
        //.def("reclaim_queue", &VulkanDevice::reclaim_queue)
        .def("acquire_blob_allocator", &VulkanDevice::acquire_blob_allocator)
        .def("reclaim_blob_allocator", &VulkanDevice::reclaim_blob_allocator)
        .def("acquire_staging_allocator", &VulkanDevice::acquire_staging_allocator)
        .def("reclaim_staging_allocator", &VulkanDevice::reclaim_staging_allocator)
        //todo not compelete
        ;

    //py::class_<VkCompute>(m, "VkCompute")
    //    .def(py::init<const VulkanDevice*>())
    //    .def("record_upload", &VkCompute::record_upload)
    //    .def("record_download", &VkCompute::record_download)
    //    .def("record_clone", &VkCompute::record_clone)
    //    .def("record_copy_region", &VkCompute::record_copy_region)
    //    .def("record_copy_regions", &VkCompute::record_copy_regions)
    //    .def("record_pipeline", &VkCompute::record_pipeline)
    //    .def("record_download", &VkCompute::record_download)

    //#if NCNN_BENCHMARK
    //        .def("record_write_timestamp", &VkCompute::record_write_timestamp)
    //#endif // NCNN_BENCHMARK
    //
    //        .def("record_queue_transfer_acquire", &VkCompute::record_queue_transfer_acquire)
    //        .def("submit_and_wait", &VkCompute::submit_and_wait)
    //        .def("reset", &VkCompute::reset)
    //
    //#if NCNN_BENCHMARK
    //        .def("create_query_pool", &VkCompute::create_query_pool)
    //        .def("get_query_pool_results", &VkCompute::get_query_pool_results)
    //#endif // NCNN_BENCHMARK
    //        ;

    //py::class_<VkTransfer>(m, "VkTransfer")
    //    .def(py::init<const VulkanDevice*>())
    //    .def("record_upload", &VkTransfer::record_upload)
    //    .def("submit_and_wait", &VkTransfer::submit_and_wait);

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