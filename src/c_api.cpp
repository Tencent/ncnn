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
#include "mat.h"
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
class PoolAllocator_c_api : public ncnn::PoolAllocator
{
public:
    PoolAllocator_c_api(ncnn_allocator_t _allocator)
        : ncnn::PoolAllocator()
    {
        allocator = _allocator;
    }

    virtual void* fastMalloc(size_t size)
    {
        return allocator->fast_malloc(allocator, size);
    }

    virtual void fastFree(void* ptr)
    {
        return allocator->fast_free(allocator, ptr);
    }

public:
    ncnn_allocator_t allocator;
};

static void* __ncnn_PoolAllocator_fast_malloc(ncnn_allocator_t allocator, size_t size)
{
    return ((ncnn::PoolAllocator*)allocator->pthis)->ncnn::PoolAllocator::fastMalloc(size);
}

static void __ncnn_PoolAllocator_fast_free(ncnn_allocator_t allocator, void* ptr)
{
    ((ncnn::PoolAllocator*)allocator->pthis)->ncnn::PoolAllocator::fastFree(ptr);
}

class UnlockedPoolAllocator_c_api : public ncnn::UnlockedPoolAllocator
{
public:
    UnlockedPoolAllocator_c_api(ncnn_allocator_t _allocator)
        : ncnn::UnlockedPoolAllocator()
    {
        allocator = _allocator;
    }

    virtual void* fastMalloc(size_t size)
    {
        return allocator->fast_malloc(allocator, size);
    }

    virtual void fastFree(void* ptr)
    {
        return allocator->fast_free(allocator, ptr);
    }

public:
    ncnn_allocator_t allocator;
};

static void* __ncnn_UnlockedPoolAllocator_fast_malloc(ncnn_allocator_t allocator, size_t size)
{
    return ((ncnn::UnlockedPoolAllocator*)allocator->pthis)->ncnn::UnlockedPoolAllocator::fastMalloc(size);
}

static void __ncnn_UnlockedPoolAllocator_fast_free(ncnn_allocator_t allocator, void* ptr)
{
    ((ncnn::UnlockedPoolAllocator*)allocator->pthis)->ncnn::UnlockedPoolAllocator::fastFree(ptr);
}

ncnn_allocator_t ncnn_allocator_create_pool_allocator()
{
    ncnn_allocator_t allocator = (ncnn_allocator_t)malloc(sizeof(struct __ncnn_allocator_t));
    allocator->pthis = (void*)(new PoolAllocator_c_api(allocator));
    allocator->fast_malloc = __ncnn_PoolAllocator_fast_malloc;
    allocator->fast_free = __ncnn_PoolAllocator_fast_free;
    return allocator;
}

ncnn_allocator_t ncnn_allocator_create_unlocked_pool_allocator()
{
    ncnn_allocator_t allocator = (ncnn_allocator_t)malloc(sizeof(struct __ncnn_allocator_t));
    allocator->pthis = (void*)(new UnlockedPoolAllocator_c_api(allocator));
    allocator->fast_malloc = __ncnn_UnlockedPoolAllocator_fast_malloc;
    allocator->fast_free = __ncnn_UnlockedPoolAllocator_fast_free;
    return allocator;
}

void ncnn_allocator_destroy(ncnn_allocator_t allocator)
{
    if (allocator)
    {
        delete (Allocator*)allocator->pthis;
        free(allocator);
    }
}

/* option api */
ncnn_option_t ncnn_option_create()
{
    return (ncnn_option_t)(new Option());
}

void ncnn_option_destroy(ncnn_option_t opt)
{
    delete (Option*)opt;
}

int ncnn_option_get_num_threads(const ncnn_option_t opt)
{
    return ((const Option*)opt)->num_threads;
}

void ncnn_option_set_num_threads(ncnn_option_t opt, int num_threads)
{
    ((Option*)opt)->num_threads = num_threads;
}

int ncnn_option_get_use_local_pool_allocator(const ncnn_option_t opt)
{
    return ((Option*)opt)->use_local_pool_allocator;
}

void ncnn_option_set_use_local_pool_allocator(ncnn_option_t opt, int use_local_pool_allocator)
{
    ((Option*)opt)->use_local_pool_allocator = use_local_pool_allocator;
}

void ncnn_option_set_blob_allocator(ncnn_option_t opt, ncnn_allocator_t allocator)
{
    ((Option*)opt)->blob_allocator = allocator ? (Allocator*)allocator->pthis : NULL;
}

void ncnn_option_set_workspace_allocator(ncnn_option_t opt, ncnn_allocator_t allocator)
{
    ((Option*)opt)->workspace_allocator = allocator ? (Allocator*)allocator->pthis : NULL;
}

int ncnn_option_get_use_vulkan_compute(const ncnn_option_t opt)
{
#if NCNN_VULKAN
    return ((const Option*)opt)->use_vulkan_compute;
#else
    (void)opt;
    return 0;
#endif
}

void ncnn_option_set_use_vulkan_compute(ncnn_option_t opt, int use_vulkan_compute)
{
#if NCNN_VULKAN
    ((Option*)opt)->use_vulkan_compute = use_vulkan_compute;
#else
    (void)opt;
    (void)use_vulkan_compute;
#endif
}

/* mat api */
ncnn_mat_t ncnn_mat_create()
{
    return (ncnn_mat_t)(new Mat());
}

ncnn_mat_t ncnn_mat_create_1d(int w, ncnn_allocator_t allocator)
{
    return (ncnn_mat_t)(new Mat(w, (size_t)4u, allocator ? (Allocator*)allocator->pthis : NULL));
}

ncnn_mat_t ncnn_mat_create_2d(int w, int h, ncnn_allocator_t allocator)
{
    return (ncnn_mat_t)(new Mat(w, h, (size_t)4u, allocator ? (Allocator*)allocator->pthis : NULL));
}

ncnn_mat_t ncnn_mat_create_3d(int w, int h, int c, ncnn_allocator_t allocator)
{
    return (ncnn_mat_t)(new Mat(w, h, c, (size_t)4u, allocator ? (Allocator*)allocator->pthis : NULL));
}

ncnn_mat_t ncnn_mat_create_4d(int w, int h, int d, int c, ncnn_allocator_t allocator)
{
    return (ncnn_mat_t)(new Mat(w, h, d, c, (size_t)4u, allocator ? (Allocator*)allocator->pthis : NULL));
}

ncnn_mat_t ncnn_mat_create_external_1d(int w, void* data, ncnn_allocator_t allocator)
{
    return (ncnn_mat_t)(new Mat(w, data, (size_t)4u, allocator ? (Allocator*)allocator->pthis : NULL));
}

ncnn_mat_t ncnn_mat_create_external_2d(int w, int h, void* data, ncnn_allocator_t allocator)
{
    return (ncnn_mat_t)(new Mat(w, h, data, (size_t)4u, allocator ? (Allocator*)allocator->pthis : NULL));
}

ncnn_mat_t ncnn_mat_create_external_3d(int w, int h, int c, void* data, ncnn_allocator_t allocator)
{
    return (ncnn_mat_t)(new Mat(w, h, c, data, (size_t)4u, allocator ? (Allocator*)allocator->pthis : NULL));
}

ncnn_mat_t ncnn_mat_create_external_4d(int w, int h, int d, int c, void* data, ncnn_allocator_t allocator)
{
    return (ncnn_mat_t)(new Mat(w, h, d, c, data, (size_t)4u, allocator ? (Allocator*)allocator->pthis : NULL));
}

ncnn_mat_t ncnn_mat_create_1d_elem(int w, size_t elemsize, int elempack, ncnn_allocator_t allocator)
{
    return (ncnn_mat_t)(new Mat(w, elemsize, elempack, allocator ? (Allocator*)allocator->pthis : NULL));
}

ncnn_mat_t ncnn_mat_create_2d_elem(int w, int h, size_t elemsize, int elempack, ncnn_allocator_t allocator)
{
    return (ncnn_mat_t)(new Mat(w, h, elemsize, elempack, allocator ? (Allocator*)allocator->pthis : NULL));
}

ncnn_mat_t ncnn_mat_create_3d_elem(int w, int h, int c, size_t elemsize, int elempack, ncnn_allocator_t allocator)
{
    return (ncnn_mat_t)(new Mat(w, h, c, elemsize, elempack, allocator ? (Allocator*)allocator->pthis : NULL));
}

ncnn_mat_t ncnn_mat_create_4d_elem(int w, int h, int d, int c, size_t elemsize, int elempack, ncnn_allocator_t allocator)
{
    return (ncnn_mat_t)(new Mat(w, h, d, c, elemsize, elempack, allocator ? (Allocator*)allocator->pthis : NULL));
}

ncnn_mat_t ncnn_mat_create_external_1d_elem(int w, void* data, size_t elemsize, int elempack, ncnn_allocator_t allocator)
{
    return (ncnn_mat_t)(new Mat(w, data, elemsize, elempack, allocator ? (Allocator*)allocator->pthis : NULL));
}

ncnn_mat_t ncnn_mat_create_external_2d_elem(int w, int h, void* data, size_t elemsize, int elempack, ncnn_allocator_t allocator)
{
    return (ncnn_mat_t)(new Mat(w, h, data, elemsize, elempack, allocator ? (Allocator*)allocator->pthis : NULL));
}

ncnn_mat_t ncnn_mat_create_external_3d_elem(int w, int h, int c, void* data, size_t elemsize, int elempack, ncnn_allocator_t allocator)
{
    return (ncnn_mat_t)(new Mat(w, h, c, data, elemsize, elempack, allocator ? (Allocator*)allocator->pthis : NULL));
}

ncnn_mat_t ncnn_mat_create_external_4d_elem(int w, int h, int d, int c, void* data, size_t elemsize, int elempack, ncnn_allocator_t allocator)
{
    return (ncnn_mat_t)(new Mat(w, h, d, c, data, elemsize, elempack, allocator ? (Allocator*)allocator->pthis : NULL));
}

void ncnn_mat_destroy(ncnn_mat_t mat)
{
    delete (Mat*)mat;
}

void ncnn_mat_fill_float(ncnn_mat_t mat, float v)
{
    ((Mat*)mat)->fill(v);
}

ncnn_mat_t ncnn_mat_clone(const ncnn_mat_t mat, ncnn_allocator_t allocator)
{
    return (ncnn_mat_t)(new Mat(((const Mat*)mat)->clone(allocator ? (Allocator*)allocator->pthis : NULL)));
}

ncnn_mat_t ncnn_mat_reshape_1d(const ncnn_mat_t mat, int w, ncnn_allocator_t allocator)
{
    return (ncnn_mat_t)(new Mat(((const Mat*)mat)->reshape(w, allocator ? (Allocator*)allocator->pthis : NULL)));
}

ncnn_mat_t ncnn_mat_reshape_2d(const ncnn_mat_t mat, int w, int h, ncnn_allocator_t allocator)
{
    return (ncnn_mat_t)(new Mat(((const Mat*)mat)->reshape(w, h, allocator ? (Allocator*)allocator->pthis : NULL)));
}

ncnn_mat_t ncnn_mat_reshape_3d(const ncnn_mat_t mat, int w, int h, int c, ncnn_allocator_t allocator)
{
    return (ncnn_mat_t)(new Mat(((const Mat*)mat)->reshape(w, h, c, allocator ? (Allocator*)allocator->pthis : NULL)));
}

ncnn_mat_t ncnn_mat_reshape_4d(const ncnn_mat_t mat, int w, int h, int d, int c, ncnn_allocator_t allocator)
{
    return (ncnn_mat_t)(new Mat(((const Mat*)mat)->reshape(w, h, d, c, allocator ? (Allocator*)allocator->pthis : NULL)));
}

int ncnn_mat_get_dims(const ncnn_mat_t mat)
{
    return ((const Mat*)mat)->dims;
}

int ncnn_mat_get_w(const ncnn_mat_t mat)
{
    return ((const Mat*)mat)->w;
}

int ncnn_mat_get_h(const ncnn_mat_t mat)
{
    return ((const Mat*)mat)->h;
}

int ncnn_mat_get_d(const ncnn_mat_t mat)
{
    return ((const Mat*)mat)->d;
}

int ncnn_mat_get_c(const ncnn_mat_t mat)
{
    return ((const Mat*)mat)->c;
}

size_t ncnn_mat_get_elemsize(const ncnn_mat_t mat)
{
    return ((const Mat*)mat)->elemsize;
}

int ncnn_mat_get_elempack(const ncnn_mat_t mat)
{
    return ((const Mat*)mat)->elempack;
}

size_t ncnn_mat_get_cstep(const ncnn_mat_t mat)
{
    return ((const Mat*)mat)->cstep;
}

void* ncnn_mat_get_data(const ncnn_mat_t mat)
{
    return ((const Mat*)mat)->data;
}

void* ncnn_mat_get_channel_data(const ncnn_mat_t mat, int c)
{
    return ((const Mat*)mat)->channel(c).data;
}

#if NCNN_PIXEL

/* mat pixel api */
ncnn_mat_t ncnn_mat_from_pixels(const unsigned char* pixels, int type, int w, int h, int stride, ncnn_allocator_t allocator)
{
    return (ncnn_mat_t)(new Mat(Mat::from_pixels(pixels, type, w, h, stride, allocator ? (Allocator*)allocator->pthis : NULL)));
}

ncnn_mat_t ncnn_mat_from_pixels_resize(const unsigned char* pixels, int type, int w, int h, int stride, int target_width, int target_height, ncnn_allocator_t allocator)
{
    return (ncnn_mat_t)(new Mat(Mat::from_pixels_resize(pixels, type, w, h, stride, target_width, target_height, allocator ? (Allocator*)allocator->pthis : NULL)));
}

ncnn_mat_t ncnn_mat_from_pixels_roi(const unsigned char* pixels, int type, int w, int h, int stride, int roix, int roiy, int roiw, int roih, ncnn_allocator_t allocator)
{
    return (ncnn_mat_t)(new Mat(Mat::from_pixels_roi(pixels, type, w, h, stride, roix, roiy, roiw, roih, allocator ? (Allocator*)allocator->pthis : NULL)));
}

ncnn_mat_t ncnn_mat_from_pixels_roi_resize(const unsigned char* pixels, int type, int w, int h, int stride, int roix, int roiy, int roiw, int roih, int target_width, int target_height, ncnn_allocator_t allocator)
{
    return (ncnn_mat_t)(new Mat(Mat::from_pixels_roi_resize(pixels, type, w, h, stride, roix, roiy, roiw, roih, target_width, target_height, allocator ? (Allocator*)allocator->pthis : NULL)));
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

void ncnn_convert_packing(const ncnn_mat_t src, ncnn_mat_t* dst, int elempack, const ncnn_option_t opt)
{
    Mat _dst;
    ncnn::convert_packing(*(const Mat*)src, _dst, elempack, *(Option*)opt);
    *dst = (ncnn_mat_t)(new Mat(_dst));
}

void ncnn_flatten(const ncnn_mat_t src, ncnn_mat_t* dst, const ncnn_option_t opt)
{
    Mat _dst;
    ncnn::flatten(*(const Mat*)src, _dst, *(Option*)opt);
    *dst = (ncnn_mat_t)(new Mat(_dst));
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

ncnn_mat_t ncnn_paramdict_get_array(ncnn_paramdict_t pd, int id, const ncnn_mat_t def)
{
    return (ncnn_mat_t)(new Mat(((const ParamDict*)pd)->get(id, *(const Mat*)def)));
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
    DataReader_c_api(ncnn_datareader_t _dr)
        : ncnn::DataReader()
    {
        dr = _dr;
    }

#if NCNN_STRING
    virtual int scan(const char* format, void* p) const
    {
        return dr->scan(dr, format, p);
    }
#endif /* NCNN_STRING */

    virtual size_t read(void* buf, size_t size) const
    {
        return dr->read(dr, buf, size);
    }

public:
    ncnn_datareader_t dr;
};

#if NCNN_STRING
static int __ncnn_DataReader_scan(ncnn_datareader_t dr, const char* format, void* p)
{
    return ((ncnn::DataReader*)dr->pthis)->ncnn::DataReader::scan(format, p);
}
#endif /* NCNN_STRING */

static size_t __ncnn_DataReader_read(ncnn_datareader_t dr, void* buf, size_t size)
{
    return ((ncnn::DataReader*)dr->pthis)->ncnn::DataReader::read(buf, size);
}

#if NCNN_STDIO
class DataReaderFromStdio_c_api : public ncnn::DataReaderFromStdio
{
public:
    DataReaderFromStdio_c_api(FILE* fp, ncnn_datareader_t _dr)
        : ncnn::DataReaderFromStdio(fp)
    {
        dr = _dr;
    }

#if NCNN_STRING
    virtual int scan(const char* format, void* p) const
    {
        return dr->scan(dr, format, p);
    }
#endif /* NCNN_STRING */

    virtual size_t read(void* buf, size_t size) const
    {
        return dr->read(dr, buf, size);
    }

public:
    ncnn_datareader_t dr;
};

#if NCNN_STRING
static int __ncnn_DataReaderFromStdio_scan(ncnn_datareader_t dr, const char* format, void* p)
{
    return ((ncnn::DataReaderFromStdio*)dr->pthis)->ncnn::DataReaderFromStdio::scan(format, p);
}
#endif /* NCNN_STRING */

static size_t __ncnn_DataReaderFromStdio_read(ncnn_datareader_t dr, void* buf, size_t size)
{
    return ((ncnn::DataReaderFromStdio*)dr->pthis)->ncnn::DataReaderFromStdio::read(buf, size);
}
#endif /* NCNN_STDIO */

class DataReaderFromMemory_c_api : public ncnn::DataReaderFromMemory
{
public:
    DataReaderFromMemory_c_api(const unsigned char*& mem, ncnn_datareader_t _dr)
        : ncnn::DataReaderFromMemory(mem)
    {
        dr = _dr;
    }

#if NCNN_STRING
    virtual int scan(const char* format, void* p) const
    {
        return dr->scan(dr, format, p);
    }
#endif /* NCNN_STRING */

    virtual size_t read(void* buf, size_t size) const
    {
        return dr->read(dr, buf, size);
    }

public:
    ncnn_datareader_t dr;
};

#if NCNN_STRING
static int __ncnn_DataReaderFromMemory_scan(ncnn_datareader_t dr, const char* format, void* p)
{
    return ((ncnn::DataReaderFromMemory*)dr->pthis)->ncnn::DataReaderFromMemory::scan(format, p);
}
#endif /* NCNN_STRING */

static size_t __ncnn_DataReaderFromMemory_read(ncnn_datareader_t dr, void* buf, size_t size)
{
    return ((ncnn::DataReaderFromMemory*)dr->pthis)->ncnn::DataReaderFromMemory::read(buf, size);
}

ncnn_datareader_t ncnn_datareader_create()
{
    ncnn_datareader_t dr = (ncnn_datareader_t)malloc(sizeof(struct __ncnn_datareader_t));
    dr->pthis = (void*)(new DataReader_c_api(dr));
#if NCNN_STRING
    dr->scan = __ncnn_DataReader_scan;
#endif /* NCNN_STRING */
    dr->read = __ncnn_DataReader_read;
    return dr;
}

#if NCNN_STDIO
ncnn_datareader_t ncnn_datareader_create_from_stdio(FILE* fp)
{
    ncnn_datareader_t dr = (ncnn_datareader_t)malloc(sizeof(struct __ncnn_datareader_t));
    dr->pthis = (void*)(new DataReaderFromStdio_c_api(fp, dr));
#if NCNN_STRING
    dr->scan = __ncnn_DataReaderFromStdio_scan;
#endif /* NCNN_STRING */
    dr->read = __ncnn_DataReaderFromStdio_read;
    return dr;
}
#endif /* NCNN_STDIO */

ncnn_datareader_t ncnn_datareader_create_from_memory(const unsigned char** mem)
{
    ncnn_datareader_t dr = (ncnn_datareader_t)malloc(sizeof(struct __ncnn_datareader_t));
    dr->pthis = (void*)(new DataReaderFromMemory_c_api(*mem, dr));
#if NCNN_STRING
    dr->scan = __ncnn_DataReaderFromMemory_scan;
#endif /* NCNN_STRING */
    dr->read = __ncnn_DataReaderFromMemory_read;
    return dr;
}

void ncnn_datareader_destroy(ncnn_datareader_t dr)
{
    delete (DataReader*)dr->pthis;
    free(dr);
}

/* modelbin api */
class ModelBinFromDataReader_c_api : public ncnn::ModelBinFromDataReader
{
public:
    ModelBinFromDataReader_c_api(ncnn_modelbin_t _mb, const DataReader& dr)
        : ncnn::ModelBinFromDataReader(dr)
    {
        mb = _mb;
    }

    virtual Mat load(int w, int type) const
    {
        ncnn_mat_t m = mb->load_1d(mb, w, type);
        Mat m2 = *(Mat*)m;
        ncnn_mat_destroy(m);
        return m2;
    }

    virtual Mat load(int w, int h, int type) const
    {
        ncnn_mat_t m = mb->load_2d(mb, w, h, type);
        Mat m2 = *(Mat*)m;
        ncnn_mat_destroy(m);
        return m2;
    }

    virtual Mat load(int w, int h, int c, int type) const
    {
        ncnn_mat_t m = mb->load_3d(mb, w, h, c, type);
        Mat m2 = *(Mat*)m;
        ncnn_mat_destroy(m);
        return m2;
    }

public:
    ncnn_modelbin_t mb;
};

static ncnn_mat_t __ncnn_ModelBinFromDataReader_load_1d(const ncnn_modelbin_t mb, int w, int type)
{
    return (ncnn_mat_t)(new Mat(((const ncnn::ModelBinFromDataReader*)mb->pthis)->ncnn::ModelBinFromDataReader::load(w, type)));
}

static ncnn_mat_t __ncnn_ModelBinFromDataReader_load_2d(const ncnn_modelbin_t mb, int w, int h, int type)
{
    return (ncnn_mat_t)(new Mat(((const ncnn::ModelBinFromDataReader*)mb->pthis)->ncnn::ModelBin::load(w, h, type)));
}

static ncnn_mat_t __ncnn_ModelBinFromDataReader_load_3d(const ncnn_modelbin_t mb, int w, int h, int c, int type)
{
    return (ncnn_mat_t)(new Mat(((const ncnn::ModelBinFromDataReader*)mb->pthis)->ncnn::ModelBin::load(w, h, c, type)));
}

class ModelBinFromMatArray_c_api : public ncnn::ModelBinFromMatArray
{
public:
    ModelBinFromMatArray_c_api(ncnn_modelbin_t _mb, const Mat* weights)
        : ncnn::ModelBinFromMatArray(weights)
    {
        mb = _mb;
    }

    virtual Mat load(int w, int type) const
    {
        ncnn_mat_t m = mb->load_1d(mb, w, type);
        Mat m2 = *(Mat*)m;
        ncnn_mat_destroy(m);
        return m2;
    }

    virtual Mat load(int w, int h, int type) const
    {
        ncnn_mat_t m = mb->load_2d(mb, w, h, type);
        Mat m2 = *(Mat*)m;
        ncnn_mat_destroy(m);
        return m2;
    }

    virtual Mat load(int w, int h, int c, int type) const
    {
        ncnn_mat_t m = mb->load_3d(mb, w, h, c, type);
        Mat m2 = *(Mat*)m;
        ncnn_mat_destroy(m);
        return m2;
    }

public:
    ncnn_modelbin_t mb;
};

static ncnn_mat_t __ncnn_ModelBinFromMatArray_load_1d(const ncnn_modelbin_t mb, int w, int type)
{
    return (ncnn_mat_t)(new Mat(((const ncnn::ModelBinFromMatArray*)mb->pthis)->ncnn::ModelBinFromMatArray::load(w, type)));
}

static ncnn_mat_t __ncnn_ModelBinFromMatArray_load_2d(const ncnn_modelbin_t mb, int w, int h, int type)
{
    return (ncnn_mat_t)(new Mat(((const ncnn::ModelBinFromMatArray*)mb->pthis)->ncnn::ModelBin::load(w, h, type)));
}

static ncnn_mat_t __ncnn_ModelBinFromMatArray_load_3d(const ncnn_modelbin_t mb, int w, int h, int c, int type)
{
    return (ncnn_mat_t)(new Mat(((const ncnn::ModelBinFromMatArray*)mb->pthis)->ncnn::ModelBin::load(w, h, c, type)));
}

ncnn_modelbin_t ncnn_modelbin_create_from_datareader(const ncnn_datareader_t dr)
{
    ncnn_modelbin_t mb = (ncnn_modelbin_t)malloc(sizeof(struct __ncnn_modelbin_t));
    mb->pthis = (void*)(new ModelBinFromDataReader_c_api(mb, *(const DataReader*)dr->pthis));
    mb->load_1d = __ncnn_ModelBinFromDataReader_load_1d;
    mb->load_2d = __ncnn_ModelBinFromDataReader_load_2d;
    mb->load_3d = __ncnn_ModelBinFromDataReader_load_3d;
    return mb;
}

ncnn_modelbin_t ncnn_modelbin_create_from_mat_array(const ncnn_mat_t* weights, int n)
{
    std::vector<Mat> matarray(n);
    for (int i = 0; i < n; i++)
    {
        matarray[i] = *(const Mat*)weights[i];
    }
    ncnn_modelbin_t mb = (ncnn_modelbin_t)malloc(sizeof(struct __ncnn_modelbin_t));
    mb->pthis = (void*)(new ModelBinFromMatArray_c_api(mb, &matarray[0]));
    mb->load_1d = __ncnn_ModelBinFromMatArray_load_1d;
    mb->load_2d = __ncnn_ModelBinFromMatArray_load_2d;
    mb->load_3d = __ncnn_ModelBinFromMatArray_load_3d;
    return mb;
}

void ncnn_modelbin_destroy(ncnn_modelbin_t mb)
{
    delete (ModelBin*)mb->pthis;
    free(mb);
}

static ncnn_mat_t __ncnn_modelbin_load_1d(const ncnn_modelbin_t mb, int w, int type)
{
    return (ncnn_mat_t)(new Mat(((const ncnn::ModelBin*)mb->pthis)->load(w, type)));
}

static ncnn_mat_t __ncnn_modelbin_load_2d(const ncnn_modelbin_t mb, int w, int h, int type)
{
    return (ncnn_mat_t)(new Mat(((const ncnn::ModelBin*)mb->pthis)->load(w, h, type)));
}

static ncnn_mat_t __ncnn_modelbin_load_3d(const ncnn_modelbin_t mb, int w, int h, int c, int type)
{
    return (ncnn_mat_t)(new Mat(((const ncnn::ModelBin*)mb->pthis)->load(w, h, c, type)));
}

/* layer api */
class Layer_c_api : public Layer
{
public:
    Layer_c_api(ncnn_layer_t _layer)
        : Layer()
    {
        layer = _layer;
    }

    virtual int load_param(const ParamDict& pd)
    {
        return layer->load_param(layer, (ncnn_paramdict_t)&pd);
    }

    virtual int load_model(const ModelBin& mb)
    {
        struct __ncnn_modelbin_t mb0;
        mb0.pthis = (void*)&mb;
        mb0.load_1d = __ncnn_modelbin_load_1d;
        mb0.load_2d = __ncnn_modelbin_load_2d;
        mb0.load_3d = __ncnn_modelbin_load_3d;
        return layer->load_model(layer, &mb0);
    }

    virtual int create_pipeline(const Option& opt)
    {
        return layer->create_pipeline(layer, (ncnn_option_t)&opt);
    }

    virtual int destroy_pipeline(const Option& opt)
    {
        return layer->destroy_pipeline(layer, (ncnn_option_t)&opt);
    }

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
    {
        const int n = bottom_blobs.size();
        const int n2 = top_blobs.size();
        std::vector<ncnn_mat_t> bottom_blobs0(n);
        for (int i = 0; i < n; i++)
        {
            bottom_blobs0[i] = (ncnn_mat_t)&bottom_blobs[i];
        }
        std::vector<ncnn_mat_t> top_blobs0(n2, (ncnn_mat_t)0);
        int ret = layer->forward_n(layer, &bottom_blobs0[0], n, &top_blobs0[0], n2, (ncnn_option_t)&opt);
        for (int i = 0; i < n2; i++)
        {
            top_blobs[i] = *(Mat*)top_blobs0[i];
            ncnn_mat_destroy(top_blobs0[i]);
        }
        return ret;
    }

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
    {
        ncnn_mat_t top_blob0 = 0;
        int ret = layer->forward_1(layer, (ncnn_mat_t)&bottom_blob, &top_blob0, (ncnn_option_t)&opt);
        top_blob = *(Mat*)top_blob0;
        ncnn_mat_destroy(top_blob0);
        return ret;
    }

    virtual int forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
    {
        const int n = bottom_top_blobs.size();
        std::vector<ncnn_mat_t> bottom_top_blobs0(n);
        for (int i = 0; i < n; i++)
        {
            bottom_top_blobs0[i] = (ncnn_mat_t)&bottom_top_blobs[i];
        }
        return layer->forward_inplace_n(layer, &bottom_top_blobs0[0], n, (ncnn_option_t)&opt);
    }

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const
    {
        return layer->forward_inplace_1(layer, (ncnn_mat_t)&bottom_top_blob, (ncnn_option_t)&opt);
    }

public:
    ncnn_layer_t layer;
};

static int __ncnn_Layer_load_param(ncnn_layer_t layer, const ncnn_paramdict_t pd)
{
    return ((Layer*)layer->pthis)->Layer::load_param(*(const ParamDict*)pd);
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
    *top_blob = (ncnn_mat_t)(new Mat(_top_blob));
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
        top_blobs[i] = (ncnn_mat_t)(new Mat(_top_blobs[i]));
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
    *top_blob = (ncnn_mat_t)(new Mat(_top_blob));
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
        top_blobs[i] = (ncnn_mat_t)(new Mat(_top_blobs[i]));
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
    void* pthis = (void*)(ncnn::create_layer(typeindex));
    if (!pthis)
    {
        return 0;
    }

    ncnn_layer_t layer = (ncnn_layer_t)malloc(sizeof(__ncnn_layer_t));
    layer->pthis = pthis;
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
    void* pthis = (void*)(ncnn::create_layer(type));
    if (!pthis)
    {
        return 0;
    }

    ncnn_layer_t layer = (ncnn_layer_t)malloc(sizeof(__ncnn_layer_t));
    layer->pthis = pthis;
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

int ncnn_layer_type_to_index(const char* type)
{
    return ncnn::layer_to_index(type);
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

ncnn_option_t ncnn_net_get_option(ncnn_net_t net)
{
    return (ncnn_option_t)(&((Net*)(net->pthis))->opt);
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

int ncnn_net_get_input_count(const ncnn_net_t net)
{
    return (int)((Net*)net->pthis)->input_indexes().size();
}

int ncnn_net_get_output_count(const ncnn_net_t net)
{
    return (int)((Net*)net->pthis)->output_indexes().size();
}

#if NCNN_STRING
const char* ncnn_net_get_input_name(const ncnn_net_t net, int i)
{
    return ((Net*)net->pthis)->input_names()[i];
}

const char* ncnn_net_get_output_name(const ncnn_net_t net, int i)
{
    return ((Net*)net->pthis)->output_names()[i];
}
#endif /* NCNN_STRING */

int ncnn_net_get_input_index(const ncnn_net_t net, int i)
{
    return ((Net*)net->pthis)->input_indexes()[i];
}

int ncnn_net_get_output_index(const ncnn_net_t net, int i)
{
    return ((Net*)net->pthis)->output_indexes()[i];
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
    *mat = (ncnn_mat_t)(new Mat(mat0));
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
    *mat = (ncnn_mat_t)(new Mat(mat0));
    return ret;
}

void ncnn_copy_make_border(const ncnn_mat_t src, ncnn_mat_t dst, int top, int bottom, int left, int right, int type, float v, const ncnn_option_t opt)
{
    const Option _opt = opt ? *((const Option*)opt) : Option();
    copy_make_border(*(const Mat*)src, *(Mat*)dst, top, bottom, left, right, type, v, _opt);
}

void ncnn_copy_make_border_3d(const ncnn_mat_t src, ncnn_mat_t dst, int top, int bottom, int left, int right, int front, int behind, int type, float v, const ncnn_option_t opt)
{
    const Option _opt = opt ? *((const Option*)opt) : Option();
    copy_make_border_3d(*(const Mat*)src, *(Mat*)dst, top, bottom, left, right, front, behind, type, v, _opt);
}

void ncnn_copy_cut_border(const ncnn_mat_t src, ncnn_mat_t dst, int top, int bottom, int left, int right, const ncnn_option_t opt)
{
    const Option _opt = opt ? *((const Option*)opt) : Option();
    copy_cut_border(*(const Mat*)src, *(Mat*)dst, top, bottom, left, right, _opt);
}

void ncnn_copy_cut_border_3d(const ncnn_mat_t src, ncnn_mat_t dst, int top, int bottom, int left, int right, int front, int behind, const ncnn_option_t opt)
{
    const Option _opt = opt ? *((const Option*)opt) : Option();
    copy_cut_border_3d(*(const Mat*)src, *(Mat*)dst, top, bottom, left, right, front, behind, _opt);
}

#if NCNN_PIXEL_DRAWING
void ncnn_draw_rectangle_c1(unsigned char* pixels, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness)
{
    ncnn::draw_rectangle_c1(pixels, w, h, w, rx, ry, rw, rh, color, thickness);
}

void ncnn_draw_rectangle_c2(unsigned char* pixels, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness)
{
    ncnn::draw_rectangle_c2(pixels, w, h, w * 2, rx, ry, rw, rh, color, thickness);
}

void ncnn_draw_rectangle_c3(unsigned char* pixels, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness)
{
    ncnn::draw_rectangle_c3(pixels, w, h, w * 3, rx, ry, rw, rh, color, thickness);
}

void ncnn_draw_rectangle_c4(unsigned char* pixels, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness)
{
    ncnn::draw_rectangle_c4(pixels, w, h, w * 4, rx, ry, rw, rh, color, thickness);
}

void ncnn_draw_text_c1(unsigned char* pixels, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color)
{
    ncnn::draw_text_c1(pixels, w, h, w, text, x, y, fontpixelsize, color);
}

void ncnn_draw_text_c2(unsigned char* pixels, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color)
{
    ncnn::draw_text_c2(pixels, w, h, w * 2, text, x, y, fontpixelsize, color);
}

void ncnn_draw_text_c3(unsigned char* pixels, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color)
{
    ncnn::draw_text_c3(pixels, w, h, w * 3, text, x, y, fontpixelsize, color);
}

void ncnn_draw_text_c4(unsigned char* pixels, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color)
{
    ncnn::draw_text_c4(pixels, w, h, w * 4, text, x, y, fontpixelsize, color);
}

void ncnn_draw_circle_c1(unsigned char* pixels, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness)
{
    ncnn::draw_circle_c1(pixels, w, h, w, cx, cy, radius, color, thickness);
}

void ncnn_draw_circle_c2(unsigned char* pixels, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness)
{
    ncnn::draw_circle_c2(pixels, w, h, w * 2, cx, cy, radius, color, thickness);
}

void ncnn_draw_circle_c3(unsigned char* pixels, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness)
{
    ncnn::draw_circle_c3(pixels, w, h, w * 3, cx, cy, radius, color, thickness);
}

void ncnn_draw_circle_c4(unsigned char* pixels, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness)
{
    ncnn::draw_circle_c4(pixels, w, h, w * 4, cx, cy, radius, color, thickness);
}

void ncnn_draw_line_c1(unsigned char* pixels, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness)
{
    ncnn::draw_line_c1(pixels, w, h, w, x0, y0, x1, y1, color, thickness);
}

void ncnn_draw_line_c2(unsigned char* pixels, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness)
{
    ncnn::draw_line_c2(pixels, w, h, w * 2, x0, y0, x1, y1, color, thickness);
}

void ncnn_draw_line_c3(unsigned char* pixels, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness)
{
    ncnn::draw_line_c3(pixels, w, h, w * 3, x0, y0, x1, y1, color, thickness);
}

void ncnn_draw_line_c4(unsigned char* pixels, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness)
{
    ncnn::draw_line_c4(pixels, w, h, w * 4, x0, y0, x1, y1, color, thickness);
}
#endif /* NCNN_PIXEL_DRAWING */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NCNN_C_API */
