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

/* mat api */
typedef struct __ncnn_mat_t* ncnn_mat_t;

ncnn_mat_t ncnn_mat_create();
ncnn_mat_t ncnn_mat_create_1d(int w);
ncnn_mat_t ncnn_mat_create_2d(int w, int h);
ncnn_mat_t ncnn_mat_create_3d(int w, int h, int c);
ncnn_mat_t ncnn_mat_create_1d_packed(int w, size_t elemsize, int elempack);
ncnn_mat_t ncnn_mat_create_2d_packed(int w, int h, size_t elemsize, int elempack);
ncnn_mat_t ncnn_mat_create_3d_packed(int w, int h, int c, size_t elemsize, int elempack);
void ncnn_mat_destroy(ncnn_mat_t mat);

int ncnn_mat_get_dims(ncnn_mat_t mat);
int ncnn_mat_get_w(ncnn_mat_t mat);
int ncnn_mat_get_h(ncnn_mat_t mat);
int ncnn_mat_get_c(ncnn_mat_t mat);
size_t ncnn_mat_get_elemsize(ncnn_mat_t mat);
int ncnn_mat_get_elempack(ncnn_mat_t mat);
size_t ncnn_mat_get_cstep(ncnn_mat_t mat);
void* ncnn_mat_get_data(ncnn_mat_t mat);

#if NCNN_PIXEL

/* mat pixel api */
#define NCNN_MAT_PIXEL_RGB       1
#define NCNN_MAT_PIXEL_BGR       2
#define NCNN_MAT_PIXEL_GRAY      3
#define NCNN_MAT_PIXEL_RGBA      4
#define NCNN_MAT_PIXEL_BGRA      5
#define NCNN_MAT_PIXEL_X2Y(X, Y) (X | (Y << 16))
ncnn_mat_t ncnn_mat_from_pixels(const unsigned char* pixels, int type, int w, int h, int stride);
ncnn_mat_t ncnn_mat_from_pixels_resize(const unsigned char* pixels, int type, int w, int h, int stride, int target_width, int target_height);
void ncnn_mat_to_pixels(ncnn_mat_t mat, unsigned char* pixels, int type, int stride);
void ncnn_mat_to_pixels_resize(ncnn_mat_t mat, unsigned char* pixels, int type, int target_width, int target_height, int target_stride);

#endif // NCNN_PIXEL

void ncnn_mat_substract_mean_normalize(ncnn_mat_t mat, const float* mean_vals, const float* norm_vals);

/* option api */
typedef struct __ncnn_option_t* ncnn_option_t;

ncnn_option_t ncnn_option_create();
void ncnn_option_destroy(ncnn_option_t opt);

int ncnn_option_get_num_threads(ncnn_option_t opt);
void ncnn_option_set_num_threads(ncnn_option_t opt, int num_threads);

int ncnn_option_get_use_vulkan_compute(ncnn_option_t opt);
void ncnn_option_set_use_vulkan_compute(ncnn_option_t opt, int use_vulkan_compute);

/* blob api */
typedef struct __ncnn_blob_t* ncnn_blob_t;

const char* ncnn_blob_get_name(ncnn_blob_t blob);

int ncnn_blob_get_producer(ncnn_blob_t blob);
int ncnn_blob_get_consumer_count(ncnn_blob_t blob);
int ncnn_blob_get_consumer(ncnn_blob_t blob, int i);

void ncnn_blob_get_shape(ncnn_blob_t blob, int* dims, int* w, int* h, int* c);

/* layer api */
typedef struct __ncnn_layer_t* ncnn_layer_t;

const char* ncnn_layer_get_name(ncnn_layer_t layer);

int ncnn_layer_get_typeindex(ncnn_layer_t layer);
const char* ncnn_layer_get_type(ncnn_layer_t layer);

int ncnn_layer_get_bottom_count(ncnn_layer_t layer);
int ncnn_layer_get_bottom(ncnn_layer_t layer, int i);
int ncnn_layer_get_top_count(ncnn_layer_t layer);
int ncnn_layer_get_top(ncnn_layer_t layer, int i);

void ncnn_blob_get_bottom_shape(ncnn_layer_t layer, int i, int* dims, int* w, int* h, int* c);
void ncnn_blob_get_top_shape(ncnn_layer_t layer, int i, int* dims, int* w, int* h, int* c);

/* net api */
typedef struct __ncnn_net_t* ncnn_net_t;

ncnn_net_t ncnn_net_create();
void ncnn_net_destroy(ncnn_net_t net);

void ncnn_net_set_option(ncnn_net_t net, ncnn_option_t opt);

int ncnn_net_load_param(ncnn_net_t net, const char* path);
int ncnn_net_load_model(ncnn_net_t net, const char* path);

int ncnn_net_get_layer_count(ncnn_net_t net);
ncnn_layer_t ncnn_net_get_layer(ncnn_net_t net, int i);
int ncnn_net_get_blob_count(ncnn_net_t net);
ncnn_blob_t ncnn_net_get_blob(ncnn_net_t net, int i);

/* extractor api */
typedef struct __ncnn_extractor_t* ncnn_extractor_t;

ncnn_extractor_t ncnn_extractor_create(ncnn_net_t net);
void ncnn_extractor_destroy(ncnn_extractor_t ex);

void ncnn_extractor_set_option(ncnn_extractor_t ex, ncnn_option_t opt);

int ncnn_extractor_input(ncnn_extractor_t ex, const char* name, ncnn_mat_t mat);
int ncnn_extractor_extract(ncnn_extractor_t ex, const char* name, ncnn_mat_t* mat);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif // NCNN_C_API_H
