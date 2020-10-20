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

#include "mat.h"
#include "option.h"
#include "net.h"

using ncnn::Blob;
using ncnn::Extractor;
using ncnn::Layer;
using ncnn::Mat;
using ncnn::Net;
using ncnn::Option;

#ifdef __cplusplus
extern "C" {
#endif

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

void ncnn_mat_destroy(ncnn_mat_t mat)
{
    delete (Mat*)mat;
}

int ncnn_mat_get_dims(ncnn_mat_t mat)
{
    return ((Mat*)mat)->dims;
}

int ncnn_mat_get_w(ncnn_mat_t mat)
{
    return ((Mat*)mat)->w;
}

int ncnn_mat_get_h(ncnn_mat_t mat)
{
    return ((Mat*)mat)->h;
}

int ncnn_mat_get_c(ncnn_mat_t mat)
{
    return ((Mat*)mat)->c;
}

size_t ncnn_mat_get_elemsize(ncnn_mat_t mat)
{
    return ((Mat*)mat)->elemsize;
}

int ncnn_mat_get_elempack(ncnn_mat_t mat)
{
    return ((Mat*)mat)->elempack;
}

size_t ncnn_mat_get_cstep(ncnn_mat_t mat)
{
    return ((Mat*)mat)->cstep;
}

void* ncnn_mat_get_data(ncnn_mat_t mat)
{
    return ((Mat*)mat)->data;
}

/* mat pixel api */
ncnn_mat_t ncnn_mat_from_pixels(const unsigned char* pixels, int type, int w, int h, int stride)
{
    return (ncnn_mat_t)(new Mat(Mat::from_pixels(pixels, type, w, h, stride)));
}

ncnn_mat_t ncnn_mat_from_pixels_resize(const unsigned char* pixels, int type, int w, int h, int stride, int target_width, int target_height)
{
    return (ncnn_mat_t)(new Mat(Mat::from_pixels_resize(pixels, type, w, h, stride, target_width, target_height)));
}

void ncnn_mat_to_pixels(ncnn_mat_t mat, unsigned char* pixels, int type, int stride)
{
    ((Mat*)mat)->to_pixels(pixels, type, stride);
}

void ncnn_mat_to_pixels_resize(ncnn_mat_t mat, unsigned char* pixels, int type, int target_width, int target_height, int target_stride)
{
    ((Mat*)mat)->to_pixels_resize(pixels, type, target_width, target_height, target_stride);
}

void ncnn_mat_substract_mean_normalize(ncnn_mat_t mat, const float* mean_vals, const float* norm_vals)
{
    ((Mat*)mat)->substract_mean_normalize(mean_vals, norm_vals);
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

int ncnn_option_get_num_threads(ncnn_option_t opt)
{
    return ((Option*)opt)->num_threads;
}

void ncnn_option_set_num_threads(ncnn_option_t opt, int num_threads)
{
    ((Option*)opt)->num_threads = num_threads;
}

int ncnn_option_get_use_vulkan_compute(ncnn_option_t opt)
{
#if NCNN_VULKAN
    return ((Option*)opt)->use_vulkan_compute;
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
const char* ncnn_blob_get_name(ncnn_blob_t blob)
{
#if NCNN_STRING
    return ((Blob*)blob)->name.c_str();
#else
    return "";
#endif
}

int ncnn_blob_get_producer(ncnn_blob_t blob)
{
    return ((Blob*)blob)->producer;
}

int ncnn_blob_get_consumer_count(ncnn_blob_t blob)
{
    return (int)((Blob*)blob)->consumers.size();
}

int ncnn_blob_get_consumer(ncnn_blob_t blob, int i)
{
    return ((Blob*)blob)->consumers[i];
}

void ncnn_blob_get_shape(ncnn_blob_t blob, int* dims, int* w, int* h, int* c)
{
    const Mat& shape = ((Blob*)blob)->shape;
    *dims = shape.dims;
    *w = shape.w;
    *h = shape.h;
    *c = shape.c;
}

/* layer api */
const char* ncnn_layer_get_name(ncnn_layer_t layer)
{
#if NCNN_STRING
    return ((Layer*)layer)->name.c_str();
#else
    return "";
#endif
}

int ncnn_layer_get_typeindex(ncnn_layer_t layer)
{
    return ((Layer*)layer)->typeindex;
}

const char* ncnn_layer_get_type(ncnn_layer_t layer)
{
#if NCNN_STRING
    return ((Layer*)layer)->type.c_str();
#else
    return "";
#endif
}

int ncnn_layer_get_bottom_count(ncnn_layer_t layer)
{
    return (int)((Layer*)layer)->bottoms.size();
}

int ncnn_layer_get_bottom(ncnn_layer_t layer, int i)
{
    return ((Layer*)layer)->bottoms[i];
}

int ncnn_layer_get_top_count(ncnn_layer_t layer)
{
    return (int)((Layer*)layer)->tops.size();
}

int ncnn_layer_get_top(ncnn_layer_t layer, int i)
{
    return ((Layer*)layer)->tops[i];
}

void ncnn_blob_get_bottom_shape(ncnn_layer_t layer, int i, int* dims, int* w, int* h, int* c)
{
    const Mat& shape = ((Layer*)layer)->bottom_shapes[i];
    *dims = shape.dims;
    *w = shape.w;
    *h = shape.h;
    *c = shape.c;
}

void ncnn_blob_get_top_shape(ncnn_layer_t layer, int i, int* dims, int* w, int* h, int* c)
{
    const Mat& shape = ((Layer*)layer)->top_shapes[i];
    *dims = shape.dims;
    *w = shape.w;
    *h = shape.h;
    *c = shape.c;
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
    return -1;
#endif
}

int ncnn_net_load_model(ncnn_net_t net, const char* path)
{
#if NCNN_STDIO && NCNN_STRING
    return ((Net*)net)->load_model(path);
#else
    return -1;
#endif
}

int ncnn_net_get_layer_count(ncnn_net_t net)
{
    return (int)((Net*)net)->layers.size();
}

ncnn_layer_t ncnn_net_get_layer(ncnn_net_t net, int i)
{
    return (ncnn_layer_t)((Net*)net)->layers[i];
}

int ncnn_net_get_blob_count(ncnn_net_t net)
{
    return (int)((Net*)net)->blobs.size();
}

ncnn_blob_t ncnn_net_get_blob(ncnn_net_t net, int i)
{
    return (ncnn_blob_t) & ((Net*)net)->blobs[i];
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

void ncnn_extractor_set_option(ncnn_extractor_t ex, ncnn_option_t opt)
{
    ((Extractor*)ex)->set_num_threads(((Option*)opt)->num_threads);
#if NCNN_VULKAN
    ((Extractor*)ex)->set_vulkan_compute(((Option*)opt)->use_vulkan_compute);
#endif
}

int ncnn_extractor_input(ncnn_extractor_t ex, const char* name, ncnn_mat_t mat)
{
#if NCNN_STRING
    return ((Extractor*)ex)->input(name, *((Mat*)mat));
#else
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
    return -1;
#endif
}

#ifdef __cplusplus
} /* extern "C" */
#endif
