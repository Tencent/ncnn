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

#ifndef NCNN_C_API_H
#define NCNN_C_API_H

#include <stddef.h>
#include "platform.h"

#ifdef __cplusplus
extern "C" {
#endif

NCNN_EXPORT const char* ncnn_version();

/* allocator api */
typedef struct __ncnn_allocator_t* ncnn_allocator_t;
struct NCNN_EXPORT __ncnn_allocator_t
{
    void* pthis;

    void* (*fast_malloc)(ncnn_allocator_t allocator, size_t size);
    void (*fast_free)(ncnn_allocator_t allocator, void* ptr);
};

NCNN_EXPORT ncnn_allocator_t ncnn_allocator_create_pool_allocator();
NCNN_EXPORT ncnn_allocator_t ncnn_allocator_create_unlocked_pool_allocator();
NCNN_EXPORT void ncnn_allocator_destroy(ncnn_allocator_t allocator);

/* option api */
typedef struct __ncnn_option_t* ncnn_option_t;

NCNN_EXPORT ncnn_option_t ncnn_option_create();
NCNN_EXPORT void ncnn_option_destroy(ncnn_option_t opt);

NCNN_EXPORT int ncnn_option_get_num_threads(const ncnn_option_t opt);
NCNN_EXPORT void ncnn_option_set_num_threads(ncnn_option_t opt, int num_threads);

NCNN_EXPORT int ncnn_option_get_use_vulkan_compute(const ncnn_option_t opt);
NCNN_EXPORT void ncnn_option_set_use_vulkan_compute(ncnn_option_t opt, int use_vulkan_compute);

/* mat api */
typedef struct __ncnn_mat_t* ncnn_mat_t;

NCNN_EXPORT ncnn_mat_t ncnn_mat_create_1d(int w, ncnn_allocator_t allocator);
NCNN_EXPORT ncnn_mat_t ncnn_mat_create_2d(int w, int h, ncnn_allocator_t allocator);
NCNN_EXPORT ncnn_mat_t ncnn_mat_create_3d(int w, int h, int c, ncnn_allocator_t allocator);
NCNN_EXPORT ncnn_mat_t ncnn_mat_create_external_1d(int w, void* data, ncnn_allocator_t allocator);
NCNN_EXPORT ncnn_mat_t ncnn_mat_create_external_2d(int w, int h, void* data, ncnn_allocator_t allocator);
NCNN_EXPORT ncnn_mat_t ncnn_mat_create_external_3d(int w, int h, int c, void* data, ncnn_allocator_t allocator);
NCNN_EXPORT ncnn_mat_t ncnn_mat_create_1d_elem(int w, size_t elemsize, int elempack, ncnn_allocator_t allocator);
NCNN_EXPORT ncnn_mat_t ncnn_mat_create_2d_elem(int w, int h, size_t elemsize, int elempack, ncnn_allocator_t allocator);
NCNN_EXPORT ncnn_mat_t ncnn_mat_create_3d_elem(int w, int h, int c, size_t elemsize, int elempack, ncnn_allocator_t allocator);
NCNN_EXPORT ncnn_mat_t ncnn_mat_create_external_1d_elem(int w, void* data, size_t elemsize, int elempack, ncnn_allocator_t allocator);
NCNN_EXPORT ncnn_mat_t ncnn_mat_create_external_2d_elem(int w, int h, void* data, size_t elemsize, int elempack, ncnn_allocator_t allocator);
NCNN_EXPORT ncnn_mat_t ncnn_mat_create_external_3d_elem(int w, int h, int c, void* data, size_t elemsize, int elempack, ncnn_allocator_t allocator);
NCNN_EXPORT void ncnn_mat_destroy(ncnn_mat_t mat);

NCNN_EXPORT void ncnn_mat_fill_float(ncnn_mat_t mat, float v);

NCNN_EXPORT ncnn_mat_t ncnn_mat_clone(const ncnn_mat_t mat, ncnn_allocator_t allocator);
NCNN_EXPORT ncnn_mat_t ncnn_mat_reshape_1d(const ncnn_mat_t mat, int w, ncnn_allocator_t allocator);
NCNN_EXPORT ncnn_mat_t ncnn_mat_reshape_2d(const ncnn_mat_t mat, int w, int h, ncnn_allocator_t allocator);
NCNN_EXPORT ncnn_mat_t ncnn_mat_reshape_3d(const ncnn_mat_t mat, int w, int h, int c, ncnn_allocator_t allocator);

NCNN_EXPORT int ncnn_mat_get_dims(const ncnn_mat_t mat);
NCNN_EXPORT int ncnn_mat_get_w(const ncnn_mat_t mat);
NCNN_EXPORT int ncnn_mat_get_h(const ncnn_mat_t mat);
NCNN_EXPORT int ncnn_mat_get_c(const ncnn_mat_t mat);
NCNN_EXPORT size_t ncnn_mat_get_elemsize(const ncnn_mat_t mat);
NCNN_EXPORT int ncnn_mat_get_elempack(const ncnn_mat_t mat);
NCNN_EXPORT size_t ncnn_mat_get_cstep(const ncnn_mat_t mat);
NCNN_EXPORT void* ncnn_mat_get_data(const ncnn_mat_t mat);

NCNN_EXPORT void* ncnn_mat_get_channel_data(const ncnn_mat_t mat, int c);

#if NCNN_PIXEL

/* mat pixel api */
#define NCNN_MAT_PIXEL_RGB       1
#define NCNN_MAT_PIXEL_BGR       2
#define NCNN_MAT_PIXEL_GRAY      3
#define NCNN_MAT_PIXEL_RGBA      4
#define NCNN_MAT_PIXEL_BGRA      5
#define NCNN_MAT_PIXEL_X2Y(X, Y) (X | (Y << 16))
NCNN_EXPORT ncnn_mat_t ncnn_mat_from_pixels(const unsigned char* pixels, int type, int w, int h, int stride, ncnn_allocator_t allocator);
NCNN_EXPORT ncnn_mat_t ncnn_mat_from_pixels_resize(const unsigned char* pixels, int type, int w, int h, int stride, int target_width, int target_height, ncnn_allocator_t allocator);
NCNN_EXPORT ncnn_mat_t ncnn_mat_from_pixels_roi(const unsigned char* pixels, int type, int w, int h, int stride, int roix, int roiy, int roiw, int roih, ncnn_allocator_t allocator);
NCNN_EXPORT ncnn_mat_t ncnn_mat_from_pixels_roi_resize(const unsigned char* pixels, int type, int w, int h, int stride, int roix, int roiy, int roiw, int roih, int target_width, int target_height, ncnn_allocator_t allocator);
NCNN_EXPORT void ncnn_mat_to_pixels(const ncnn_mat_t mat, unsigned char* pixels, int type, int stride);
NCNN_EXPORT void ncnn_mat_to_pixels_resize(const ncnn_mat_t mat, unsigned char* pixels, int type, int target_width, int target_height, int target_stride);

#endif /* NCNN_PIXEL */

NCNN_EXPORT void ncnn_mat_substract_mean_normalize(ncnn_mat_t mat, const float* mean_vals, const float* norm_vals);

NCNN_EXPORT void ncnn_convert_packing(const ncnn_mat_t src, ncnn_mat_t* dst, int elempack, const ncnn_option_t opt);
NCNN_EXPORT void ncnn_flatten(const ncnn_mat_t src, ncnn_mat_t* dst, const ncnn_option_t opt);

/* blob api */
typedef struct __ncnn_blob_t* ncnn_blob_t;

#if NCNN_STRING
NCNN_EXPORT const char* ncnn_blob_get_name(const ncnn_blob_t blob);
#endif /* NCNN_STRING */

NCNN_EXPORT int ncnn_blob_get_producer(const ncnn_blob_t blob);
NCNN_EXPORT int ncnn_blob_get_consumer(const ncnn_blob_t blob);

NCNN_EXPORT void ncnn_blob_get_shape(const ncnn_blob_t blob, int* dims, int* w, int* h, int* c);

/* paramdict api */
typedef struct __ncnn_paramdict_t* ncnn_paramdict_t;

NCNN_EXPORT ncnn_paramdict_t ncnn_paramdict_create();
NCNN_EXPORT void ncnn_paramdict_destroy(ncnn_paramdict_t pd);

NCNN_EXPORT int ncnn_paramdict_get_type(const ncnn_paramdict_t pd, int id);

NCNN_EXPORT int ncnn_paramdict_get_int(const ncnn_paramdict_t pd, int id, int def);
NCNN_EXPORT float ncnn_paramdict_get_float(const ncnn_paramdict_t pd, int id, float def);
NCNN_EXPORT ncnn_mat_t ncnn_paramdict_get_array(const ncnn_paramdict_t pd, int id, const ncnn_mat_t def);

NCNN_EXPORT void ncnn_paramdict_set_int(ncnn_paramdict_t pd, int id, int i);
NCNN_EXPORT void ncnn_paramdict_set_float(ncnn_paramdict_t pd, int id, float f);
NCNN_EXPORT void ncnn_paramdict_set_array(ncnn_paramdict_t pd, int id, const ncnn_mat_t v);

/* datareader api */
typedef struct __ncnn_datareader_t* ncnn_datareader_t;
struct NCNN_EXPORT __ncnn_datareader_t
{
    void* pthis;

#if NCNN_STRING
    int (*scan)(ncnn_datareader_t dr, const char* format, void* p);
#endif /* NCNN_STRING */
    size_t (*read)(ncnn_datareader_t dr, void* buf, size_t size);
};

NCNN_EXPORT ncnn_datareader_t ncnn_datareader_create();
#if NCNN_STDIO
NCNN_EXPORT ncnn_datareader_t ncnn_datareader_create_from_stdio(FILE* fp);
#endif /* NCNN_STDIO */
NCNN_EXPORT ncnn_datareader_t ncnn_datareader_create_from_memory(const unsigned char** mem);
NCNN_EXPORT void ncnn_datareader_destroy(ncnn_datareader_t dr);

/* modelbin api */
typedef struct __ncnn_modelbin_t* ncnn_modelbin_t;
struct NCNN_EXPORT __ncnn_modelbin_t
{
    void* pthis;

    ncnn_mat_t (*load_1d)(const ncnn_modelbin_t mb, int w, int type);
    ncnn_mat_t (*load_2d)(const ncnn_modelbin_t mb, int w, int h, int type);
    ncnn_mat_t (*load_3d)(const ncnn_modelbin_t mb, int w, int h, int c, int type);
};

NCNN_EXPORT ncnn_modelbin_t ncnn_modelbin_create_from_datareader(const ncnn_datareader_t dr);
NCNN_EXPORT ncnn_modelbin_t ncnn_modelbin_create_from_mat_array(const ncnn_mat_t* weights, int n);
NCNN_EXPORT void ncnn_modelbin_destroy(ncnn_modelbin_t mb);

/* layer api */
typedef struct __ncnn_layer_t* ncnn_layer_t;
struct NCNN_EXPORT __ncnn_layer_t
{
    void* pthis;

    int (*load_param)(ncnn_layer_t layer, const ncnn_paramdict_t pd);
    int (*load_model)(ncnn_layer_t layer, const ncnn_modelbin_t mb);

    int (*create_pipeline)(ncnn_layer_t layer, const ncnn_option_t opt);
    int (*destroy_pipeline)(ncnn_layer_t layer, const ncnn_option_t opt);

    int (*forward_1)(const ncnn_layer_t layer, const ncnn_mat_t bottom_blob, ncnn_mat_t* top_blob, const ncnn_option_t opt);
    int (*forward_n)(const ncnn_layer_t layer, const ncnn_mat_t* bottom_blobs, int n, ncnn_mat_t** top_blobs, int n2, const ncnn_option_t opt);

    int (*forward_inplace_1)(const ncnn_layer_t layer, ncnn_mat_t bottom_top_blob, const ncnn_option_t opt);
    int (*forward_inplace_n)(const ncnn_layer_t layer, ncnn_mat_t* bottom_top_blobs, int n, const ncnn_option_t opt);
};

NCNN_EXPORT ncnn_layer_t ncnn_layer_create();
NCNN_EXPORT ncnn_layer_t ncnn_layer_create_by_typeindex(int typeindex);
#if NCNN_STRING
NCNN_EXPORT ncnn_layer_t ncnn_layer_create_by_type(const char* type);
#endif /* NCNN_STRING */
NCNN_EXPORT void ncnn_layer_destroy(ncnn_layer_t layer);

#if NCNN_STRING
NCNN_EXPORT const char* ncnn_layer_get_name(const ncnn_layer_t layer);
#endif /* NCNN_STRING */

NCNN_EXPORT int ncnn_layer_get_typeindex(const ncnn_layer_t layer);
#if NCNN_STRING
NCNN_EXPORT const char* ncnn_layer_get_type(const ncnn_layer_t layer);
#endif /* NCNN_STRING */

NCNN_EXPORT int ncnn_layer_get_one_blob_only(const ncnn_layer_t layer);
NCNN_EXPORT int ncnn_layer_get_support_inplace(const ncnn_layer_t layer);
NCNN_EXPORT int ncnn_layer_get_support_vulkan(const ncnn_layer_t layer);
NCNN_EXPORT int ncnn_layer_get_support_packing(const ncnn_layer_t layer);
NCNN_EXPORT int ncnn_layer_get_support_bf16_storage(const ncnn_layer_t layer);
NCNN_EXPORT int ncnn_layer_get_support_fp16_storage(const ncnn_layer_t layer);
NCNN_EXPORT int ncnn_layer_get_support_image_storage(const ncnn_layer_t layer);

NCNN_EXPORT void ncnn_layer_set_one_blob_only(ncnn_layer_t layer, int enable);
NCNN_EXPORT void ncnn_layer_set_support_inplace(ncnn_layer_t layer, int enable);
NCNN_EXPORT void ncnn_layer_set_support_vulkan(ncnn_layer_t layer, int enable);
NCNN_EXPORT void ncnn_layer_set_support_packing(ncnn_layer_t layer, int enable);
NCNN_EXPORT void ncnn_layer_set_support_bf16_storage(ncnn_layer_t layer, int enable);
NCNN_EXPORT void ncnn_layer_set_support_fp16_storage(ncnn_layer_t layer, int enable);
NCNN_EXPORT void ncnn_layer_set_support_image_storage(ncnn_layer_t layer, int enable);

NCNN_EXPORT int ncnn_layer_get_bottom_count(const ncnn_layer_t layer);
NCNN_EXPORT int ncnn_layer_get_bottom(const ncnn_layer_t layer, int i);
NCNN_EXPORT int ncnn_layer_get_top_count(const ncnn_layer_t layer);
NCNN_EXPORT int ncnn_layer_get_top(const ncnn_layer_t layer, int i);

NCNN_EXPORT void ncnn_blob_get_bottom_shape(const ncnn_layer_t layer, int i, int* dims, int* w, int* h, int* c);
NCNN_EXPORT void ncnn_blob_get_top_shape(const ncnn_layer_t layer, int i, int* dims, int* w, int* h, int* c);

/* layer factory function */
typedef ncnn_layer_t (*ncnn_layer_creator_t)(void* userdata);
typedef void (*ncnn_layer_destroyer_t)(ncnn_layer_t layer, void* userdata);

typedef struct __ncnn_net_custom_layer_factory_t* ncnn_net_custom_layer_factory_t;
struct __ncnn_net_custom_layer_factory_t
{
    ncnn_layer_creator_t creator;
    ncnn_layer_destroyer_t destroyer;
    void* userdata;
    ncnn_net_custom_layer_factory_t next;
};

/* net api */
typedef struct __ncnn_net_t* ncnn_net_t;
struct __ncnn_net_t
{
    void* pthis;

    ncnn_net_custom_layer_factory_t custom_layer_factory;
};

NCNN_EXPORT ncnn_net_t ncnn_net_create();
NCNN_EXPORT void ncnn_net_destroy(ncnn_net_t net);

NCNN_EXPORT void ncnn_net_set_option(ncnn_net_t net, ncnn_option_t opt);

#if NCNN_STRING
NCNN_EXPORT void ncnn_net_register_custom_layer_by_type(ncnn_net_t net, const char* type, ncnn_layer_creator_t creator, ncnn_layer_destroyer_t destroyer, void* userdata);
#endif /* NCNN_STRING */
NCNN_EXPORT void ncnn_net_register_custom_layer_by_typeindex(ncnn_net_t net, int typeindex, ncnn_layer_creator_t creator, ncnn_layer_destroyer_t destroyer, void* userdata);

#if NCNN_STDIO
#if NCNN_STRING
NCNN_EXPORT int ncnn_net_load_param(ncnn_net_t net, const char* path);
#endif /* NCNN_STRING */
NCNN_EXPORT int ncnn_net_load_param_bin(ncnn_net_t net, const char* path);
NCNN_EXPORT int ncnn_net_load_model(ncnn_net_t net, const char* path);
#endif /* NCNN_STDIO */

#if NCNN_STDIO
#if NCNN_STRING
NCNN_EXPORT int ncnn_net_load_param_memory(ncnn_net_t net, const char* mem);
#endif /* NCNN_STRING */
#endif /* NCNN_STDIO */
NCNN_EXPORT int ncnn_net_load_param_bin_memory(ncnn_net_t net, const unsigned char* mem);
NCNN_EXPORT int ncnn_net_load_model_memory(ncnn_net_t net, const unsigned char* mem);

#if NCNN_STRING
NCNN_EXPORT int ncnn_net_load_param_datareader(ncnn_net_t net, const ncnn_datareader_t dr);
#endif /* NCNN_STRING */
NCNN_EXPORT int ncnn_net_load_param_bin_datareader(ncnn_net_t net, const ncnn_datareader_t dr);
NCNN_EXPORT int ncnn_net_load_model_datareader(ncnn_net_t net, const ncnn_datareader_t dr);

NCNN_EXPORT void ncnn_net_clear(ncnn_net_t net);

/* extractor api */
typedef struct __ncnn_extractor_t* ncnn_extractor_t;

NCNN_EXPORT ncnn_extractor_t ncnn_extractor_create(ncnn_net_t net);
NCNN_EXPORT void ncnn_extractor_destroy(ncnn_extractor_t ex);

NCNN_EXPORT void ncnn_extractor_set_option(ncnn_extractor_t ex, const ncnn_option_t opt);

#if NCNN_STRING
NCNN_EXPORT int ncnn_extractor_input(ncnn_extractor_t ex, const char* name, const ncnn_mat_t mat);
NCNN_EXPORT int ncnn_extractor_extract(ncnn_extractor_t ex, const char* name, ncnn_mat_t* mat);
#endif /* NCNN_STRING */
NCNN_EXPORT int ncnn_extractor_input_index(ncnn_extractor_t ex, int index, const ncnn_mat_t mat);
NCNN_EXPORT int ncnn_extractor_extract_index(ncnn_extractor_t ex, int index, ncnn_mat_t* mat);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NCNN_C_API_H */
