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

#include <stdlib.h>
#include <string.h>

#include "mat.h"
#include "net.h"

using ncnn::Mat;
using ncnn::Net;
using ncnn::Extractor;

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

void ncnn_mat_destroy(ncnn_mat_t mat)
{
    delete (Mat*)mat;
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

/* net api */
ncnn_net_t ncnn_net_create()
{
    return (ncnn_net_t)(new Net);
}

void ncnn_net_destroy(ncnn_net_t net)
{
    delete (Net*)net;
}

int ncnn_net_load_param(ncnn_net_t net, const char* path)
{
    return ((Net*)net)->load_param(path);
}

int ncnn_net_load_model(ncnn_net_t net, const char* path)
{
    return ((Net*)net)->load_model(path);
}

/* extractor api */
ncnn_extractor_t ncnn_extractor_create(ncnn_net_t net)
{
    void* ex_mem = malloc(sizeof(Extractor));
    return (ncnn_extractor_t)(new(ex_mem)Extractor(((Net*)net)->create_extractor()));
}

void ncnn_extractor_destroy(ncnn_extractor_t ex)
{
    ((Extractor*)ex)->~Extractor();
    free(ex);
}

int ncnn_extractor_input(ncnn_extractor_t ex, const char* name, ncnn_mat_t mat)
{
    return ((Extractor*)ex)->input(name, *((Mat*)mat));
}

int ncnn_extractor_extract(ncnn_extractor_t ex, const char* name, ncnn_mat_t* mat)
{
    Mat mat0;
    int ret = ((Extractor*)ex)->input(name, mat0);
    *mat = new Mat(mat0);
    return ret;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
