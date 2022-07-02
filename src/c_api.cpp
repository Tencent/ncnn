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

#include "platform.h"

#if NCNN_C_API

#include "c_api.h"

#include <stdlib.h>

#include "allocator.h"
#include "blob.h"
#include "datareader.h"
#include "layer.h"
#include "modelbin.h"
#include "net.h"
#include "option.h"
#include "paramdict.h"

using ncnn::Allocator;
using ncnn::Blob;
using ncnn::DataReader;
using ncnn::Extractor;
using ncnn::Layer;
using ncnn::Mat;
using ncnn::ModelBin;
using ncnn::Net;
using ncnn::Option;
using ncnn::ParamDict;

#ifdef __cplusplus
extern "C" {
#endif

const char* ncnn_version()
{
    return NCNN_VERSION_STRING;
}

/* allocator api */
class Allocator_c_api : public ncnn::Allocator
{
public:
    Allocator_c_api(ncnn_allocator_ext_t _ext): ncnn::Allocator()
    {
        ext = _ext;
    }

    virtual void* fastMalloc(size_t size)
    {
        return ext.fast_malloc(ext.pdata, size);
    }

    virtual void fastFree(void* ptr)
    {
        return ext.fast_free(ext.pdata, ptr);
    }
public:
    ncnn_allocator_ext_t ext;
};

ncnn_allocator_t ncnn_allocator_create(ncnn_allocator_ext_t ext)
{
    return new Allocator_c_api(ext);
}

ncnn_allocator_t ncnn_allocator_create_pool_allocator()
{
    return new ncnn::PoolAllocator();
}

ncnn_allocator_t ncnn_allocator_create_unlocked_pool_allocator()
{
    return new ncnn::UnlockedPoolAllocator();
}

void* ncnn_allocator_fast_malloc(ncnn_allocator_t allocator, size_t size)
{
    return allocator->fastMalloc(size);
}

void ncnn_allocator_fast_free(ncnn_allocator_t allocator, void* ptr)
{
    allocator->fastFree(ptr);
}

void ncnn_allocator_destroy(ncnn_allocator_t allocator)
{
    delete allocator;
}

/* option api */
ncnn_option_t ncnn_option_create()
{
    return Option();
}

/* mat api */
ncnn_mat_t ncnn_mat_create()
{
    return Mat();
}

ncnn_mat_t ncnn_mat_create_1d(int w, ncnn_allocator_t allocator)
{
    return Mat(w, (size_t)4u, allocator);
}

ncnn_mat_t ncnn_mat_create_2d(int w, int h, ncnn_allocator_t allocator)
{
    return Mat(w, h, (size_t)4u, allocator);
}

ncnn_mat_t ncnn_mat_create_3d(int w, int h, int c, ncnn_allocator_t allocator)
{
    return Mat(w, h, c, (size_t)4u, allocator);
}

ncnn_mat_t ncnn_mat_create_4d(int w, int h, int d, int c, ncnn_allocator_t allocator)
{
    return Mat(w, h, d, c, (size_t)4u, allocator);
}

ncnn_mat_t ncnn_mat_create_external_1d(int w, void* data, ncnn_allocator_t allocator)
{
    return Mat(w, data, (size_t)4u, allocator);
}

ncnn_mat_t ncnn_mat_create_external_2d(int w, int h, void* data, ncnn_allocator_t allocator)
{
    return Mat(w, h, data, (size_t)4u, allocator);
}

ncnn_mat_t ncnn_mat_create_external_3d(int w, int h, int c, void* data, ncnn_allocator_t allocator)
{
    return Mat(w, h, c, data, (size_t)4u, allocator);
}

ncnn_mat_t ncnn_mat_create_external_4d(int w, int h, int d, int c, void* data, ncnn_allocator_t allocator)
{
    return Mat(w, h, d, c, data, (size_t)4u, allocator);
}

ncnn_mat_t ncnn_mat_create_1d_elem(int w, size_t elemsize, int elempack, ncnn_allocator_t allocator)
{
    return Mat(w, elemsize, elempack, allocator);
}

ncnn_mat_t ncnn_mat_create_2d_elem(int w, int h, size_t elemsize, int elempack, ncnn_allocator_t allocator)
{
    return Mat(w, h, elemsize, elempack, allocator);
}

ncnn_mat_t ncnn_mat_create_3d_elem(int w, int h, int c, size_t elemsize, int elempack, ncnn_allocator_t allocator)
{
    return Mat(w, h, c, elemsize, elempack, allocator);
}

ncnn_mat_t ncnn_mat_create_4d_elem(int w, int h, int d, int c, size_t elemsize, int elempack, ncnn_allocator_t allocator)
{
    return Mat(w, h, d, c, elemsize, elempack, allocator);
}

ncnn_mat_t ncnn_mat_create_external_1d_elem(int w, void* data, size_t elemsize, int elempack, ncnn_allocator_t allocator)
{
    return Mat(w, data, elemsize, elempack, allocator);
}

ncnn_mat_t ncnn_mat_create_external_2d_elem(int w, int h, void* data, size_t elemsize, int elempack, ncnn_allocator_t allocator)
{
    return Mat(w, h, data, elemsize, elempack, allocator);
}

ncnn_mat_t ncnn_mat_create_external_3d_elem(int w, int h, int c, void* data, size_t elemsize, int elempack, ncnn_allocator_t allocator)
{
    return Mat(w, h, c, data, elemsize, elempack, allocator);
}

ncnn_mat_t ncnn_mat_create_external_4d_elem(int w, int h, int d, int c, void* data, size_t elemsize, int elempack, ncnn_allocator_t allocator)
{
    return Mat(w, h, d, c, data, elemsize, elempack, allocator);
}

void ncnn_mat_destroy(ncnn_mat_t mat) {
    //calling its destructor
}

void ncnn_mat_fill_float(ncnn_mat_t mat, float v)
{
    mat.fill(v);
}

ncnn_mat_t ncnn_mat_clone(const ncnn_mat_t mat, ncnn_allocator_t allocator)
{
    return mat.clone(allocator);
}

ncnn_mat_t ncnn_mat_reshape_1d(const ncnn_mat_t mat, int w, ncnn_allocator_t allocator)
{
    return mat.reshape(w, allocator);
}

ncnn_mat_t ncnn_mat_reshape_2d(const ncnn_mat_t mat, int w, int h, ncnn_allocator_t allocator)
{
    return mat.reshape(w, h, allocator);
}

ncnn_mat_t ncnn_mat_reshape_3d(const ncnn_mat_t mat, int w, int h, int c, ncnn_allocator_t allocator)
{
    return mat.reshape(w, h, c, allocator);
}

ncnn_mat_t ncnn_mat_reshape_4d(const ncnn_mat_t mat, int w, int h, int d, int c, ncnn_allocator_t allocator)
{
    return mat.reshape(w, h, d, c, allocator);
}

size_t ncnn_mat_get_elemsize(const ncnn_mat_t mat)
{
    return ((const Mat*)mat)->elemsize;
}

#if NCNN_PIXEL

/* mat pixel api */
ncnn_mat_t ncnn_mat_from_pixels(const unsigned char* pixels, int type, int w, int h, int stride, ncnn_allocator_t allocator)
{
    return (ncnn_mat_t)(Mat(Mat::from_pixels(pixels, type, w, h, stride, allocator)));
}

ncnn_mat_t ncnn_mat_from_pixels_resize(const unsigned char* pixels, int type, int w, int h, int stride, int target_width, int target_height, ncnn_allocator_t allocator)
{
    return (ncnn_mat_t)(Mat(Mat::from_pixels_resize(pixels, type, w, h, stride, target_width, target_height, allocator)));
}

ncnn_mat_t ncnn_mat_from_pixels_roi(const unsigned char* pixels, int type, int w, int h, int stride, int roix, int roiy, int roiw, int roih, ncnn_allocator_t allocator)
{
    return (ncnn_mat_t)(Mat(Mat::from_pixels_roi(pixels, type, w, h, stride, roix, roiy, roiw, roih, allocator)));
}

ncnn_mat_t ncnn_mat_from_pixels_roi_resize(const unsigned char* pixels, int type, int w, int h, int stride, int roix, int roiy, int roiw, int roih, int target_width, int target_height, ncnn_allocator_t allocator)
{
    return (ncnn_mat_t)(Mat(Mat::from_pixels_roi_resize(pixels, type, w, h, stride, roix, roiy, roiw, roih, target_width, target_height, allocator)));
}

void ncnn_mat_to_pixels(const ncnn_mat_t mat, unsigned char* pixels, int type, int stride)
{
    ((const Mat*)mat)->to_pixels(pixels, type, stride);
}

void ncnn_mat_to_pixels_resize(const ncnn_mat_t mat, unsigned char* pixels, int type, int target_width, int target_height, int target_stride)
{
    ((const Mat*)mat)->to_pixels_resize(pixels, type, target_width, target_height, target_stride);
}

#endif /* NCNN_PIXEL */

void ncnn_mat_substract_mean_normalize(ncnn_mat_t mat, const float* mean_vals, const float* norm_vals)
{
    ((Mat*)mat)->substract_mean_normalize(mean_vals, norm_vals);
}

void ncnn_convert_packing(const ncnn_mat_t src, ncnn_mat_t dst, int elempack, const ncnn_option_t opt)
{
    ncnn::convert_packing(*(const Mat*)src, *(Mat*)dst, elempack, opt);
}

void ncnn_flatten(const ncnn_mat_t src, ncnn_mat_t dst, const ncnn_option_t opt)
{
    ncnn::flatten(*(const Mat*)src, *(Mat*)dst, opt);
}

/* blob api */
#if NCNN_STRING
const char* ncnn_blob_get_name(const ncnn_blob_t blob)
{
    return ((const Blob*)blob)->name.c_str();
}
#endif /* NCNN_STRING */

int ncnn_blob_get_producer(const ncnn_blob_t blob)
{
    return ((const Blob*)blob)->producer;
}

int ncnn_blob_get_consumer(const ncnn_blob_t blob)
{
    return ((const Blob*)blob)->consumer;
}

void ncnn_blob_get_shape(const ncnn_blob_t blob, int* dims, int* w, int* h, int* c)
{
    const Mat& shape = ((const Blob*)blob)->shape;
    *dims = shape.dims;
    *w = shape.w;
    *h = shape.h;
    *c = shape.c;
}

/* paramdict api */
ncnn_paramdict_t ncnn_paramdict_create()
{
    return (ncnn_paramdict_t)(new ParamDict());
}

void ncnn_paramdict_destroy(ncnn_paramdict_t pd)
{
    delete (ParamDict*)pd;
}

int ncnn_paramdict_get_type(const ncnn_paramdict_t pd, int id)
{
    return ((const ParamDict*)pd)->type(id);
}

int ncnn_paramdict_get_int(const ncnn_paramdict_t pd, int id, int def)
{
    return ((const ParamDict*)pd)->get(id, def);
}

float ncnn_paramdict_get_float(const ncnn_paramdict_t pd, int id, float def)
{
    return ((const ParamDict*)pd)->get(id, def);
}

ncnn_mat_t ncnn_paramdict_get_array(ncnn_paramdict_t pd, int id, const ncnn_mat_t* def)
{
    return ((const ParamDict*)pd)->get(id, *def);
}

void ncnn_paramdict_set_int(ncnn_paramdict_t pd, int id, int i)
{
    return ((ParamDict*)pd)->set(id, i);
}

void ncnn_paramdict_set_float(ncnn_paramdict_t pd, int id, float f)
{
    return ((ParamDict*)pd)->set(id, f);
}

void ncnn_paramdict_set_array(ncnn_paramdict_t pd, int id, ncnn_mat_t v)
{
    return ((ParamDict*)pd)->set(id, *(const Mat*)v);
}

/* datareader api */
class DataReader_c_api : public ncnn::DataReader
{
public:
    DataReader_c_api(ncnn_datareader_ext_t _ext): ncnn::DataReader()
    {
        ext = _ext;
    }

#if NCNN_STRING
    virtual int scan(const char* format, void* p) const
    {
        return ext.scan(ext.pdata, format, p);
    }
#endif /* NCNN_STRING */

    virtual size_t read(void* buf, size_t size) const
    {
        return ext.read(ext.pdata, buf, size);
    }

public:
    ncnn_datareader_ext_t ext;
};


ncnn_datareader_t ncnn_datareader_create(ncnn_datareader_ext_t ext)
{
    return new DataReader_c_api(ext);
}

#if NCNN_STDIO
ncnn_datareader_t ncnn_datareader_create_from_stdio(FILE* fp)
{
    return new ncnn::DataReaderFromStdio(fp);
}
#endif /* NCNN_STDIO */

ncnn_datareader_t ncnn_datareader_create_from_memory(const unsigned char** mem)
{
    return new ncnn::DataReaderFromMemory(*mem);
}

#if NCNN_STRING
void ncnn_datareader_scan(ncnn_datareader_t dr, const char* format, void* p)
{
    dr->scan(format, p);
}
#endif

size_t ncnn_datareader_read(ncnn_datareader_t dr, void* buf, size_t size)
{
    return dr->read(buf, size);
}

void ncnn_datareader_destroy(ncnn_datareader_t dr)
{
    delete dr;
}

/* modelbin api */
class ModelBin_c_api : public ncnn::ModelBin
{
public:
    ModelBin_c_api(ncnn_modelbin_ext_t _ext): ncnn::ModelBin() {
        ext = _ext;
    }

    Mat load(int w, int type) const override {
        return ext.load_1d(ext.pdata, w, type);
    }

    Mat load(int w, int h, int type) const override {
        return ext.load_2d(ext.pdata, w, h, type);
    }

    Mat load(int w, int h, int c, int type) const override {
        return ext.load_3d(ext.pdata, w, h, c, type);
    }
public:
    ncnn_modelbin_ext_t ext;
};


ncnn_modelbin_t ncnn_modelbin_create_from_datareader(const ncnn_datareader_t dr)
{
    return new ncnn::ModelBinFromDataReader(*dr);
}

ncnn_modelbin_t ncnn_modelbin_create_from_mat_array(const ncnn_mat_t* weights)
{
    return new ncnn::ModelBinFromMatArray(weights);
}

void ncnn_modelbin_destroy(ncnn_modelbin_t mb)
{
    delete mb;
}

ncnn_mat_t ncnn_modelbin_load_1d(const ncnn_modelbin_t mb, int w, int type)
{
    return mb->load(w, type);
}

ncnn_mat_t ncnn_modelbin_load_2d(const ncnn_modelbin_t mb, int w, int h, int type)
{
    return mb->load(w, h, type);
}

ncnn_mat_t ncnn_modelbin_load_3d(const ncnn_modelbin_t mb, int w, int h, int c, int type)
{
    return mb->load(w, h, c, type);
}

/* layer api */
class Layer_c_api : public Layer
{
public:
    Layer_c_api(ncnn_layer_ext_t _ext): Layer()
    {
        ext = _ext;
    }

    virtual int load_param(const ParamDict& pd)
    {
        return ext.load_param(ext.pdata, (ncnn_paramdict_t)&pd);
    }

    virtual int load_model(const ModelBin& mb)
    {
        return ext.load_model(ext.pdata, (ncnn_modelbin_t)&mb);
    }

    virtual int create_pipeline(const Option& opt)
    {
        return ext.create_pipeline(ext.pdata, opt);
    }

    virtual int destroy_pipeline(const Option& opt)
    {
        return ext.destroy_pipeline(ext.pdata, opt);
    }

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
    {
        return ext.forward_n(ext.pdata, bottom_blobs.data(), bottom_blobs.size(), top_blobs.data(), top_blobs.size(), opt);
    }

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
    {
        return ext.forward_1(ext.pdata, bottom_blob, top_blob, opt);
    }

    virtual int forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
    {
        return ext.forward_inplace_n(ext.pdata, bottom_top_blobs.data(), bottom_top_blobs.size(), opt);
    }

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const
    {
        return ext.forward_inplace_1(ext.pdata, bottom_top_blob, opt);
    }

public:
    ncnn_layer_ext ext;
};

static int ncnn_layer_load_param(ncnn_layer_t layer, const ncnn_paramdict_t pd)
{
    return layer->load_param(pd);
}

static int __ncnn_Layer_load_model(ncnn_layer_t layer, const ncnn_modelbin_t mb)
{
    return ((Layer*)layer->pthis)->Layer::load_model(*(const ModelBin*)mb);
}

static int __ncnn_Layer_create_pipeline(ncnn_layer_t layer, const ncnn_option_t opt)
{
    return ((Layer*)layer->pthis)->Layer::create_pipeline(*(const Option*)opt);
}

static int __ncnn_Layer_destroy_pipeline(ncnn_layer_t layer, const ncnn_option_t opt)
{
    return ((Layer*)layer->pthis)->Layer::destroy_pipeline(*(const Option*)opt);
}

static int __ncnn_Layer_forward_1(const ncnn_layer_t layer, const ncnn_mat_t bottom_blob, ncnn_mat_t* top_blob, const ncnn_option_t opt)
{
    Mat _top_blob;
    int ret = ((const Layer*)layer->pthis)->Layer::forward(*(const Mat*)bottom_blob, _top_blob, *(const Option*)opt);
    *top_blob = (ncnn_mat_t)(Mat(_top_blob));
    return ret;
}

static int __ncnn_Layer_forward_n(const ncnn_layer_t layer, const ncnn_mat_t* bottom_blobs, int n, ncnn_mat_t* top_blobs, int n2, const ncnn_option_t opt)
{
    std::vector<Mat> _bottom_blobs(n);
    std::vector<Mat> _top_blobs(n2);
    for (int i = 0; i < n; i++)
    {
        _bottom_blobs[i] = *(Mat*)bottom_blobs[i];
    }
    int ret = ((const Layer*)layer->pthis)->Layer::forward(_bottom_blobs, _top_blobs, *(const Option*)opt);
    for (int i = 0; i < n2; i++)
    {
        top_blobs[i] = (ncnn_mat_t)(Mat(_top_blobs[i]));
    }
    return ret;
}

static int __ncnn_Layer_forward_inplace_1(const ncnn_layer_t layer, ncnn_mat_t bottom_top_blob, const ncnn_option_t opt)
{
    return ((const Layer*)layer->pthis)->Layer::forward_inplace(*(Mat*)bottom_top_blob, *(const Option*)opt);
}

static int __ncnn_Layer_forward_inplace_n(const ncnn_layer_t layer, ncnn_mat_t* bottom_top_blobs, int n, const ncnn_option_t opt)
{
    std::vector<Mat> _bottom_top_blobs(n);
    for (int i = 0; i < n; i++)
    {
        _bottom_top_blobs[i] = *(Mat*)bottom_top_blobs[i];
    }
    return ((const Layer*)layer->pthis)->Layer::forward_inplace(_bottom_top_blobs, *(const Option*)opt);
}

static int __ncnn_layer_load_param(ncnn_layer_t layer, const ncnn_paramdict_t pd)
{
    return ((Layer*)layer->pthis)->load_param(*(const ParamDict*)pd);
}

static int __ncnn_layer_load_model(ncnn_layer_t layer, const ncnn_modelbin_t mb)
{
    return ((Layer*)layer->pthis)->load_model(*(const ModelBin*)mb);
}

static int __ncnn_layer_create_pipeline(ncnn_layer_t layer, const ncnn_option_t opt)
{
    return ((Layer*)layer->pthis)->create_pipeline(*(const Option*)opt);
}

static int __ncnn_layer_destroy_pipeline(ncnn_layer_t layer, const ncnn_option_t opt)
{
    return ((Layer*)layer->pthis)->destroy_pipeline(*(const Option*)opt);
}

static int __ncnn_layer_forward_1(const ncnn_layer_t layer, const ncnn_mat_t bottom_blob, ncnn_mat_t* top_blob, const ncnn_option_t opt)
{
    Mat _top_blob;
    int ret = ((const Layer*)layer->pthis)->forward(*(const Mat*)bottom_blob, _top_blob, *(const Option*)opt);
    *top_blob = (ncnn_mat_t)(Mat(_top_blob));
    return ret;
}

static int __ncnn_layer_forward_n(const ncnn_layer_t layer, const ncnn_mat_t* bottom_blobs, int n, ncnn_mat_t* top_blobs, int n2, const ncnn_option_t opt)
{
    std::vector<Mat> _bottom_blobs(n);
    std::vector<Mat> _top_blobs(n2);
    for (int i = 0; i < n; i++)
    {
        _bottom_blobs[i] = *(Mat*)bottom_blobs[i];
    }
    int ret = ((const Layer*)layer->pthis)->forward(_bottom_blobs, _top_blobs, *(const Option*)opt);
    for (int i = 0; i < n2; i++)
    {
        top_blobs[i] = (ncnn_mat_t)(Mat(_top_blobs[i]));
    }
    return ret;
}

static int __ncnn_layer_forward_inplace_1(const ncnn_layer_t layer, ncnn_mat_t bottom_top_blob, const ncnn_option_t opt)
{
    return ((const Layer*)layer->pthis)->forward_inplace(*(Mat*)bottom_top_blob, *(const Option*)opt);
}

static int __ncnn_layer_forward_inplace_n(const ncnn_layer_t layer, ncnn_mat_t* bottom_top_blobs, int n, const ncnn_option_t opt)
{
    std::vector<Mat> _bottom_top_blobs(n);
    for (int i = 0; i < n; i++)
    {
        _bottom_top_blobs[i] = *(Mat*)bottom_top_blobs[i];
    }
    return ((const Layer*)layer->pthis)->forward_inplace(_bottom_top_blobs, *(const Option*)opt);
}

ncnn_layer_t ncnn_layer_create()
{
    ncnn_layer_t layer = (ncnn_layer_t)malloc(sizeof(__ncnn_layer_t));
    layer->pthis = (void*)(new Layer_c_api(layer));
    layer->load_param = __ncnn_Layer_load_param;
    layer->load_model = __ncnn_Layer_load_model;
    layer->create_pipeline = __ncnn_Layer_create_pipeline;
    layer->destroy_pipeline = __ncnn_Layer_destroy_pipeline;
    layer->forward_1 = __ncnn_Layer_forward_1;
    layer->forward_n = __ncnn_Layer_forward_n;
    layer->forward_inplace_1 = __ncnn_Layer_forward_inplace_1;
    layer->forward_inplace_n = __ncnn_Layer_forward_inplace_n;
    return layer;
}

ncnn_layer_t ncnn_layer_create_by_typeindex(int typeindex)
{
    ncnn_layer_t layer = (ncnn_layer_t)malloc(sizeof(__ncnn_layer_t));
    layer->pthis = (void*)(ncnn::create_layer(typeindex));
    layer->load_param = __ncnn_layer_load_param;
    layer->load_model = __ncnn_layer_load_model;
    layer->create_pipeline = __ncnn_layer_create_pipeline;
    layer->destroy_pipeline = __ncnn_layer_destroy_pipeline;
    layer->forward_1 = __ncnn_layer_forward_1;
    layer->forward_n = __ncnn_layer_forward_n;
    layer->forward_inplace_1 = __ncnn_layer_forward_inplace_1;
    layer->forward_inplace_n = __ncnn_layer_forward_inplace_n;
    return layer;
}

#if NCNN_STRING
ncnn_layer_t ncnn_layer_create_by_type(const char* type)
{
    ncnn_layer_t layer = (ncnn_layer_t)malloc(sizeof(__ncnn_layer_t));
    layer->pthis = (void*)(ncnn::create_layer(type));
    layer->load_param = __ncnn_layer_load_param;
    layer->load_model = __ncnn_layer_load_model;
    layer->create_pipeline = __ncnn_layer_create_pipeline;
    layer->destroy_pipeline = __ncnn_layer_destroy_pipeline;
    layer->forward_1 = __ncnn_layer_forward_1;
    layer->forward_n = __ncnn_layer_forward_n;
    layer->forward_inplace_1 = __ncnn_layer_forward_inplace_1;
    layer->forward_inplace_n = __ncnn_layer_forward_inplace_n;
    return layer;
}
#endif /* NCNN_STRING */

void ncnn_layer_destroy(ncnn_layer_t layer)
{
    delete (Layer*)layer->pthis;
    free(layer);
}

#if NCNN_STRING
const char* ncnn_layer_get_name(const ncnn_layer_t layer)
{
    return ((const Layer*)layer->pthis)->name.c_str();
}
#endif /* NCNN_STRING */

int ncnn_layer_get_typeindex(const ncnn_layer_t layer)
{
    return ((const Layer*)layer->pthis)->typeindex;
}

#if NCNN_STRING
const char* ncnn_layer_get_type(const ncnn_layer_t layer)
{
    return ((const Layer*)layer->pthis)->type.c_str();
}
#endif /* NCNN_STRING */

int ncnn_layer_get_one_blob_only(const ncnn_layer_t layer)
{
    return ((const Layer*)layer->pthis)->one_blob_only;
}

int ncnn_layer_get_support_inplace(const ncnn_layer_t layer)
{
    return ((const Layer*)layer->pthis)->support_inplace;
}

int ncnn_layer_get_support_vulkan(const ncnn_layer_t layer)
{
    return ((const Layer*)layer->pthis)->support_vulkan;
}

int ncnn_layer_get_support_packing(const ncnn_layer_t layer)
{
    return ((const Layer*)layer->pthis)->support_packing;
}

int ncnn_layer_get_support_bf16_storage(const ncnn_layer_t layer)
{
    return ((const Layer*)layer->pthis)->support_bf16_storage;
}

int ncnn_layer_get_support_fp16_storage(const ncnn_layer_t layer)
{
    return ((const Layer*)layer->pthis)->support_fp16_storage;
}

int ncnn_layer_get_support_image_storage(const ncnn_layer_t layer)
{
    return ((const Layer*)layer->pthis)->support_image_storage;
}

void ncnn_layer_set_one_blob_only(ncnn_layer_t layer, int enable)
{
    ((Layer*)layer->pthis)->one_blob_only = enable;
}

void ncnn_layer_set_support_inplace(ncnn_layer_t layer, int enable)
{
    ((Layer*)layer->pthis)->support_inplace = enable;
}

void ncnn_layer_set_support_vulkan(ncnn_layer_t layer, int enable)
{
    ((Layer*)layer->pthis)->support_vulkan = enable;
}

void ncnn_layer_set_support_packing(ncnn_layer_t layer, int enable)
{
    ((Layer*)layer->pthis)->support_packing = enable;
}

void ncnn_layer_set_support_bf16_storage(ncnn_layer_t layer, int enable)
{
    ((Layer*)layer->pthis)->support_bf16_storage = enable;
}

void ncnn_layer_set_support_fp16_storage(ncnn_layer_t layer, int enable)
{
    ((Layer*)layer->pthis)->support_fp16_storage = enable;
}

void ncnn_layer_set_support_image_storage(ncnn_layer_t layer, int enable)
{
    ((Layer*)layer->pthis)->support_image_storage = enable;
}

int ncnn_layer_get_bottom_count(const ncnn_layer_t layer)
{
    return (int)((const Layer*)layer->pthis)->bottoms.size();
}

int ncnn_layer_get_bottom(const ncnn_layer_t layer, int i)
{
    return ((const Layer*)layer->pthis)->bottoms[i];
}

int ncnn_layer_get_top_count(const ncnn_layer_t layer)
{
    return (int)((const Layer*)layer->pthis)->tops.size();
}

int ncnn_layer_get_top(const ncnn_layer_t layer, int i)
{
    return ((const Layer*)layer->pthis)->tops[i];
}

void ncnn_blob_get_bottom_shape(const ncnn_layer_t layer, int i, int* dims, int* w, int* h, int* c)
{
    const Mat& shape = ((const Layer*)layer->pthis)->bottom_shapes[i];
    *dims = shape.dims;
    *w = shape.w;
    *h = shape.h;
    *c = shape.c;
}

void ncnn_blob_get_top_shape(const ncnn_layer_t layer, int i, int* dims, int* w, int* h, int* c)
{
    const Mat& shape = ((const Layer*)layer->pthis)->top_shapes[i];
    *dims = shape.dims;
    *w = shape.w;
    *h = shape.h;
    *c = shape.c;
}

/* net api */
ncnn_net_t ncnn_net_create()
{
    ncnn_net_t net = (ncnn_net_t)malloc(sizeof(struct __ncnn_net_t));
    net->pthis = (void*)(new Net());
    net->custom_layer_factory = 0;
    return net;
}

void ncnn_net_destroy(ncnn_net_t net)
{
    delete (Net*)net->pthis;
    ncnn_net_custom_layer_factory_t ud = net->custom_layer_factory;
    while (ud)
    {
        ncnn_net_custom_layer_factory_t ud_next = ud->next;
        free(ud);
        ud = ud_next;
    }
    free(net);
}

void ncnn_net_set_option(ncnn_net_t net, ncnn_option_t opt)
{
    ((Net*)net->pthis)->opt = *((Option*)opt);
}

static ::ncnn::Layer* __Layer_c_api_layer_creator(void* userdata)
{
    ncnn_net_custom_layer_factory_t ud = (ncnn_net_custom_layer_factory_t)userdata;

    ncnn_layer_t layer0 = ud->creator(ud->userdata);

    ::ncnn::Layer* layer = (::ncnn::Layer*)layer0->pthis;

    layer->userdata = (void*)layer0;

    layer->one_blob_only = ncnn_layer_get_one_blob_only(layer0);
    layer->support_inplace = ncnn_layer_get_support_inplace(layer0);
    layer->support_vulkan = ncnn_layer_get_support_vulkan(layer0);
    layer->support_packing = ncnn_layer_get_support_packing(layer0);

    layer->support_bf16_storage = ncnn_layer_get_support_bf16_storage(layer0);
    layer->support_fp16_storage = ncnn_layer_get_support_fp16_storage(layer0);
    layer->support_image_storage = ncnn_layer_get_support_image_storage(layer0);

    return layer;
}

static void __Layer_c_api_layer_destroyer(::ncnn::Layer* layer, void* userdata)
{
    ncnn_net_custom_layer_factory_t ud = (ncnn_net_custom_layer_factory_t)userdata;

    ncnn_layer_t layer0 = (ncnn_layer_t)layer->userdata;

    ud->destroyer(layer0, ud->userdata);
}

#if NCNN_STRING
void ncnn_net_register_custom_layer_by_type(ncnn_net_t net, const char* type, ncnn_layer_creator_t creator, ncnn_layer_destroyer_t destroyer, void* userdata)
{
    ncnn_net_custom_layer_factory_t ud = (ncnn_net_custom_layer_factory_t)malloc(sizeof(struct __ncnn_net_custom_layer_factory_t));
    ud->creator = creator;
    ud->destroyer = destroyer;
    ud->userdata = userdata;
    ud->next = net->custom_layer_factory;
    net->custom_layer_factory = ud;
    ((Net*)net->pthis)->register_custom_layer(type, __Layer_c_api_layer_creator, __Layer_c_api_layer_destroyer, (void*)ud);
}
#endif /* NCNN_STRING */

void ncnn_net_register_custom_layer_by_typeindex(ncnn_net_t net, int typeindex, ncnn_layer_creator_t creator, ncnn_layer_destroyer_t destroyer, void* userdata)
{
    ncnn_net_custom_layer_factory_t ud = (ncnn_net_custom_layer_factory_t)malloc(sizeof(struct __ncnn_net_custom_layer_factory_t));
    ud->creator = creator;
    ud->destroyer = destroyer;
    ud->userdata = userdata;
    ud->next = net->custom_layer_factory;
    net->custom_layer_factory = ud;
    ((Net*)net->pthis)->register_custom_layer(typeindex, __Layer_c_api_layer_creator, __Layer_c_api_layer_destroyer, (void*)ud);
}

#if NCNN_STDIO
#if NCNN_STRING
int ncnn_net_load_param(ncnn_net_t net, const char* path)
{
    return ((Net*)net->pthis)->load_param(path);
}
#endif /* NCNN_STRING */

int ncnn_net_load_param_bin(ncnn_net_t net, const char* path)
{
    return ((Net*)net->pthis)->load_param_bin(path);
}

int ncnn_net_load_model(ncnn_net_t net, const char* path)
{
    return ((Net*)net->pthis)->load_model(path);
}
#endif /* NCNN_STDIO */

#if NCNN_STDIO
#if NCNN_STRING
int ncnn_net_load_param_memory(ncnn_net_t net, const char* mem)
{
    return ((Net*)net->pthis)->load_param_mem(mem);
}
#endif /* NCNN_STRING */
#endif /* NCNN_STDIO */

int ncnn_net_load_param_bin_memory(ncnn_net_t net, const unsigned char* mem)
{
    return ((Net*)net->pthis)->load_param(mem);
}

int ncnn_net_load_model_memory(ncnn_net_t net, const unsigned char* mem)
{
    return ((Net*)net->pthis)->load_model(mem);
}

#if NCNN_STRING
int ncnn_net_load_param_datareader(ncnn_net_t net, const ncnn_datareader_t dr)
{
    return ((Net*)net->pthis)->load_param(*(const DataReader*)dr->pthis);
}
#endif /* NCNN_STRING */

int ncnn_net_load_param_bin_datareader(ncnn_net_t net, const ncnn_datareader_t dr)
{
    return ((Net*)net->pthis)->load_param_bin(*(const DataReader*)dr->pthis);
}

int ncnn_net_load_model_datareader(ncnn_net_t net, const ncnn_datareader_t dr)
{
    return ((Net*)net->pthis)->load_model(*(const DataReader*)dr->pthis);
}

void ncnn_net_clear(ncnn_net_t net)
{
    return ((Net*)net->pthis)->clear();
}

/* extractor api */
ncnn_extractor_t ncnn_extractor_create(ncnn_net_t net)
{
    return (ncnn_extractor_t)(new Extractor(((Net*)net->pthis)->create_extractor()));
}

void ncnn_extractor_destroy(ncnn_extractor_t ex)
{
    delete (Extractor*)ex;
}

void ncnn_extractor_set_option(ncnn_extractor_t ex, const ncnn_option_t opt)
{
    ((Extractor*)ex)->set_num_threads(((const Option*)opt)->num_threads);
#if NCNN_VULKAN
    ((Extractor*)ex)->set_vulkan_compute(((const Option*)opt)->use_vulkan_compute);
#endif
}

#if NCNN_STRING
int ncnn_extractor_input(ncnn_extractor_t ex, const char* name, const ncnn_mat_t mat)
{
    return ((Extractor*)ex)->input(name, *((const Mat*)mat));
}

int ncnn_extractor_extract(ncnn_extractor_t ex, const char* name, ncnn_mat_t* mat)
{
    Mat mat0;
    int ret = ((Extractor*)ex)->extract(name, mat0);
    *mat = (ncnn_mat_t)(Mat(mat0));
    return ret;
}
#endif /* NCNN_STRING */

int ncnn_extractor_input_index(ncnn_extractor_t ex, int index, const ncnn_mat_t mat)
{
    return ((Extractor*)ex)->input(index, *((const Mat*)mat));
}

int ncnn_extractor_extract_index(ncnn_extractor_t ex, int index, ncnn_mat_t* mat)
{
    Mat mat0;
    int ret = ((Extractor*)ex)->extract(index, mat0);
    *mat = (ncnn_mat_t)(Mat(mat0));
    return ret;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NCNN_C_API */
