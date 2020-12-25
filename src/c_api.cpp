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

#include "c_api.h"

#include "blob.h"
#include "datareader.h"
#include "layer.h"
#include "mat.h"
#include "modelbin.h"
#include "net.h"
#include "option.h"
#include "paramdict.h"

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

/* mat api */
ncnn_mat_t ncnn_mat_create()
{
    return (ncnn_mat_t)(new Mat());
}

ncnn_mat_t ncnn_mat_create_1d(int w)
{
    return (ncnn_mat_t)(new Mat(w));
}

ncnn_mat_t ncnn_mat_create_2d(int w, int h)
{
    return (ncnn_mat_t)(new Mat(w, h));
}

ncnn_mat_t ncnn_mat_create_3d(int w, int h, int c)
{
    return (ncnn_mat_t)(new Mat(w, h, c));
}

ncnn_mat_t ncnn_mat_create_1d_packed(int w, size_t elemsize, int elempack)
{
    return (ncnn_mat_t)(new Mat(w, elemsize, elempack));
}

ncnn_mat_t ncnn_mat_create_2d_packed(int w, int h, size_t elemsize, int elempack)
{
    return (ncnn_mat_t)(new Mat(w, h, elemsize, elempack));
}

ncnn_mat_t ncnn_mat_create_3d_packed(int w, int h, int c, size_t elemsize, int elempack)
{
    return (ncnn_mat_t)(new Mat(w, h, c, elemsize, elempack));
}

ncnn_mat_t ncnn_mat_create_external_1d(int w, void* data)
{
    return (ncnn_mat_t)(new Mat(w, data));
}

ncnn_mat_t ncnn_mat_create_external_2d(int w, int h, void* data)
{
    return (ncnn_mat_t)(new Mat(w, h, data));
}

ncnn_mat_t ncnn_mat_create_external_3d(int w, int h, int c, void* data)
{
    return (ncnn_mat_t)(new Mat(w, h, c, data));
}

ncnn_mat_t ncnn_mat_create_external_1d_packed(int w, void* data, size_t elemsize, int elempack)
{
    return (ncnn_mat_t)(new Mat(w, data, elemsize, elempack));
}

ncnn_mat_t ncnn_mat_create_external_2d_packedl(int w, int h, void* data, size_t elemsize, int elempack)
{
    return (ncnn_mat_t)(new Mat(w, h, data, elemsize, elempack));
}

ncnn_mat_t ncnn_mat_create_external_3d_packed(int w, int h, int c, void* data, size_t elemsize, int elempack)
{
    return (ncnn_mat_t)(new Mat(w, h, c, data, elemsize, elempack));
}

void ncnn_mat_destroy(ncnn_mat_t mat)
{
    delete (Mat*)mat;
}

void ncnn_mat_fill_float(ncnn_mat_t mat, float v)
{
    ((Mat*)mat)->fill(v);
}

ncnn_mat_t ncnn_mat_clone(const ncnn_mat_t mat)
{
    return (ncnn_mat_t)(new Mat(((const Mat*)mat)->clone()));
}

ncnn_mat_t ncnn_mat_reshape_1d(const ncnn_mat_t mat, int w)
{
    return (ncnn_mat_t)(new Mat(((const Mat*)mat)->reshape(w)));
}

ncnn_mat_t ncnn_mat_reshape_2d(const ncnn_mat_t mat, int w, int h)
{
    return (ncnn_mat_t)(new Mat(((const Mat*)mat)->reshape(w, h)));
}

ncnn_mat_t ncnn_mat_reshape_3d(const ncnn_mat_t mat, int w, int h, int c)
{
    return (ncnn_mat_t)(new Mat(((const Mat*)mat)->reshape(w, h, c)));
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

#if NCNN_PIXEL

/* mat pixel api */
ncnn_mat_t ncnn_mat_from_pixels(const unsigned char* pixels, int type, int w, int h, int stride)
{
    return (ncnn_mat_t)(new Mat(Mat::from_pixels(pixels, type, w, h, stride)));
}

ncnn_mat_t ncnn_mat_from_pixels_resize(const unsigned char* pixels, int type, int w, int h, int stride, int target_width, int target_height)
{
    return (ncnn_mat_t)(new Mat(Mat::from_pixels_resize(pixels, type, w, h, stride, target_width, target_height)));
}

void ncnn_mat_to_pixels(const ncnn_mat_t mat, unsigned char* pixels, int type, int stride)
{
    ((const Mat*)mat)->to_pixels(pixels, type, stride);
}

void ncnn_mat_to_pixels_resize(const ncnn_mat_t mat, unsigned char* pixels, int type, int target_width, int target_height, int target_stride)
{
    ((const Mat*)mat)->to_pixels_resize(pixels, type, target_width, target_height, target_stride);
}

#endif

void ncnn_mat_substract_mean_normalize(ncnn_mat_t mat, const float* mean_vals, const float* norm_vals)
{
    ((Mat*)mat)->substract_mean_normalize(mean_vals, norm_vals);
}

void ncnn_convert_packing(const ncnn_mat_t src, ncnn_mat_t* dst, int elempack)
{
    Mat _dst;
    ncnn::convert_packing(*(const Mat*)src, _dst, elempack);
    *dst = (ncnn_mat_t)(new Mat(_dst));
}

void ncnn_flatten(const ncnn_mat_t src, ncnn_mat_t* dst)
{
    Mat _dst;
    ncnn::flatten(*(const Mat*)src, _dst);
    *dst = (ncnn_mat_t)(new Mat(_dst));
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

/* blob api */
const char* ncnn_blob_get_name(const ncnn_blob_t blob)
{
#if NCNN_STRING
    return ((const Blob*)blob)->name.c_str();
#else
    (void)blob;
    return "";
#endif
}

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

int ncnn_paramdict_get_float(const ncnn_paramdict_t pd, int id, float def)
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
#if NCNN_STDIO
ncnn_datareader_t ncnn_datareader_from_stdio(FILE* fp)
{
    return (ncnn_datareader_t)(new ncnn::DataReaderFromStdio(fp));
}
#endif

ncnn_datareader_t ncnn_datareader_from_memory(const unsigned char** mem)
{
    return (ncnn_datareader_t)(new ncnn::DataReaderFromMemory(*mem));
}

void ncnn_datareader_destroy(ncnn_datareader_t dr)
{
    delete (DataReader*)dr;
}

/* modelbin api */
ncnn_modelbin_t ncnn_modelbin_from_datareader(const ncnn_datareader_t dr)
{
    return (ncnn_modelbin_t)(new ncnn::ModelBinFromDataReader(*(const DataReader*)dr));
}

ncnn_modelbin_t ncnn_modelbin_from_mat_array(const ncnn_mat_t* weights, int n)
{
    std::vector<Mat> matarray(n);
    for (int i = 0; i < n; i++)
    {
        matarray[i] = *(const Mat*)weights[i];
    }
    return (ncnn_modelbin_t)(new ncnn::ModelBinFromMatArray(&matarray[0]));
}

void ncnn_modelbin_destroy(ncnn_modelbin_t mb)
{
    delete (ModelBin*)mb;
}

ncnn_mat_t ncnn_modelbin_load_1d(const ncnn_modelbin_t mb, int w, int type)
{
    return (ncnn_mat_t)(new Mat(((const ModelBin*)mb)->load(w, type)));
}

ncnn_mat_t ncnn_modelbin_load_2d(const ncnn_modelbin_t mb, int w, int h, int type)
{
    return (ncnn_mat_t)(new Mat(((const ModelBin*)mb)->load(w, h, type)));
}

ncnn_mat_t ncnn_modelbin_load_3d(const ncnn_modelbin_t mb, int w, int h, int c, int type)
{
    return (ncnn_mat_t)(new Mat(((const ModelBin*)mb)->load(w, h, c, type)));
}

/* layer api */
ncnn_layer_t ncnn_layer_create_by_typeindex(int typeindex)
{
    return (ncnn_layer_t)(ncnn::create_layer(typeindex));
}

ncnn_layer_t ncnn_layer_create_by_type(const char* type)
{
    return (ncnn_layer_t)(ncnn::create_layer(type));
}

void ncnn_layer_destroy(ncnn_layer_t layer)
{
    delete (Layer*)layer;
}

const char* ncnn_layer_get_name(const ncnn_layer_t layer)
{
#if NCNN_STRING
    return ((const Layer*)layer)->name.c_str();
#else
    (void)layer;
    return "";
#endif
}

int ncnn_layer_get_typeindex(const ncnn_layer_t layer)
{
    return ((const Layer*)layer)->typeindex;
}

const char* ncnn_layer_get_type(const ncnn_layer_t layer)
{
#if NCNN_STRING
    return ((const Layer*)layer)->type.c_str();
#else
    (void)layer;
    return "";
#endif
}

int ncnn_layer_get_bottom_count(const ncnn_layer_t layer)
{
    return (int)((const Layer*)layer)->bottoms.size();
}

int ncnn_layer_get_bottom(const ncnn_layer_t layer, int i)
{
    return ((const Layer*)layer)->bottoms[i];
}

int ncnn_layer_get_top_count(const ncnn_layer_t layer)
{
    return (int)((const Layer*)layer)->tops.size();
}

int ncnn_layer_get_top(const ncnn_layer_t layer, int i)
{
    return ((const Layer*)layer)->tops[i];
}

void ncnn_blob_get_bottom_shape(const ncnn_layer_t layer, int i, int* dims, int* w, int* h, int* c)
{
    const Mat& shape = ((const Layer*)layer)->bottom_shapes[i];
    *dims = shape.dims;
    *w = shape.w;
    *h = shape.h;
    *c = shape.c;
}

void ncnn_blob_get_top_shape(const ncnn_layer_t layer, int i, int* dims, int* w, int* h, int* c)
{
    const Mat& shape = ((const Layer*)layer)->top_shapes[i];
    *dims = shape.dims;
    *w = shape.w;
    *h = shape.h;
    *c = shape.c;
}

int ncnn_layer_load_param(ncnn_layer_t layer, const ncnn_paramdict_t pd)
{
    return ((Layer*)layer)->load_param(*(const ParamDict*)pd);
}

int ncnn_layer_load_model(ncnn_layer_t layer, const ncnn_modelbin_t mb)
{
    return ((Layer*)layer)->load_model(*(const ModelBin*)mb);
}

int ncnn_layer_create_pipeline(ncnn_layer_t layer, const ncnn_option_t opt)
{
    return ((Layer*)layer)->create_pipeline(*(const Option*)opt);
}

int ncnn_layer_destroy_pipeline(ncnn_layer_t layer, const ncnn_option_t opt)
{
    return ((Layer*)layer)->destroy_pipeline(*(const Option*)opt);
}

int ncnn_layer_forward_1(const ncnn_layer_t layer, const ncnn_mat_t bottom_blob, ncnn_mat_t* top_blob, const ncnn_option_t opt)
{
    Mat _top_blob;
    int ret = ((const Layer*)layer)->forward(*(const Mat*)bottom_blob, _top_blob, *(const Option*)opt);
    *top_blob = (ncnn_mat_t)(new Mat(_top_blob));
    return ret;
}

int ncnn_layer_forward_n(const ncnn_layer_t layer, const ncnn_mat_t* bottom_blobs, int n, ncnn_mat_t** top_blobs, int n2, const ncnn_option_t opt)
{
    std::vector<Mat> _bottom_blobs(n);
    std::vector<Mat> _top_blobs(n2);
    for (int i = 0; i < n; i++)
    {
        _bottom_blobs[i] = *(Mat*)bottom_blobs[i];
    }
    int ret = ((const Layer*)layer)->forward(_bottom_blobs, _top_blobs, *(const Option*)opt);
    for (int i = 0; i < n2; i++)
    {
        *top_blobs[i] = (ncnn_mat_t)(new Mat(_top_blobs[i]));
    }
    return ret;
}

int ncnn_layer_forward_inplace_1(const ncnn_layer_t layer, ncnn_mat_t bottom_top_blob, const ncnn_option_t opt)
{
    return ((const Layer*)layer)->forward_inplace(*(Mat*)bottom_top_blob, *(const Option*)opt);
}

int ncnn_layer_forward_inplace_n(const ncnn_layer_t layer, ncnn_mat_t* bottom_top_blobs, int n, const ncnn_option_t opt)
{
    std::vector<Mat> _bottom_top_blobs(n);
    for (int i = 0; i < n; i++)
    {
        _bottom_top_blobs[i] = *(Mat*)bottom_top_blobs[i];
    }
    return ((const Layer*)layer)->forward_inplace(_bottom_top_blobs, *(const Option*)opt);
}

/* net api */
ncnn_net_t ncnn_net_create()
{
    return (ncnn_net_t)(new Net);
}

void ncnn_net_destroy(ncnn_net_t net)
{
    delete (Net*)net;
}

void ncnn_net_set_option(ncnn_net_t net, ncnn_option_t opt)
{
    ((Net*)net)->opt = *((Option*)opt);
}

int ncnn_net_load_param(ncnn_net_t net, const char* path)
{
#if NCNN_STDIO && NCNN_STRING
    return ((Net*)net)->load_param(path);
#else
    (void)path;
    (void)net;
    return -1;
#endif
}

int ncnn_net_load_model(ncnn_net_t net, const char* path)
{
#if NCNN_STDIO && NCNN_STRING
    return ((Net*)net)->load_model(path);
#else
    (void)path;
    (void)net;
    return -1;
#endif
}

int ncnn_net_get_layer_count(const ncnn_net_t net)
{
    return (int)((const Net*)net)->layers.size();
}

ncnn_layer_t ncnn_net_get_layer(const ncnn_net_t net, int i)
{
    return (ncnn_layer_t)((const Net*)net)->layers[i];
}

int ncnn_net_get_blob_count(const ncnn_net_t net)
{
    return (int)((const Net*)net)->blobs.size();
}

ncnn_blob_t ncnn_net_get_blob(const ncnn_net_t net, int i)
{
    return (ncnn_blob_t)(&((const Net*)net)->blobs[i]);
}

/* extractor api */
ncnn_extractor_t ncnn_extractor_create(ncnn_net_t net)
{
    return (ncnn_extractor_t)(new Extractor(((Net*)net)->create_extractor()));
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

int ncnn_extractor_input(ncnn_extractor_t ex, const char* name, const ncnn_mat_t mat)
{
#if NCNN_STRING
    return ((Extractor*)ex)->input(name, *((const Mat*)mat));
#else
    (void)ex;
    (void)name;
    (void)mat;
    return -1;
#endif
}

int ncnn_extractor_extract(ncnn_extractor_t ex, const char* name, ncnn_mat_t* mat)
{
#if NCNN_STRING
    Mat mat0;
    int ret = ((Extractor*)ex)->extract(name, mat0);
    *mat = (ncnn_mat_t)(new Mat(mat0));
    return ret;
#else
    (void)ex;
    (void)name;
    (void)mat;
    return -1;
#endif
}

#ifdef __cplusplus
} /* extern "C" */
#endif
