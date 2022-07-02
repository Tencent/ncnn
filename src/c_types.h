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

#ifndef NCNN_C_TYPES_H
#define NCNN_C_TYPES_H

#ifdef __cplusplus
namespace ncnn {
class Allocator;
#if NCNN_VULKAN
class VkAllocator;
class PipelineCache;
#endif
struct Option;
struct Mat;
class DataReader;
class ModelBin;
class Layer;
namespace c_types {
typedef Allocator* ncnn_allocator_t;
#if NCNN_VULKAN
typedef VkAllocator* ncnn_vkallocator_t;
typedef PipelineCache* ncnn_pipelinecache_t;
#endif
typedef Option ncnn_option_t;
typedef Mat ncnn_mat_t;
typedef DataReader* ncnn_datareader_t;
typedef ModelBin* ncnn_modelbin_t;
typedef Layer* ncnn_layer_t;
}
}
#else
typedef struct __ncnn_allocator_t* ncnn_allocator_t;
#if NCNN_VULKAN
typedef struct __ncnn_vkallocator_t* ncnn_vkallocator_t;
typedef struct __ncnn_pipelinecache_t* ncnn_pipelinecache_t;
#endif
typedef struct ncnn_option ncnn_option_t;
typedef struct ncnn_mat ncnn_mat_t;
typedef struct __ncnn_datareader_t* ncnn_datareader_t;
typedef struct __ncnn_modelbin_t* ncnn_modelbin_t;
typedef struct __ncnn_layer_t* ncnn_layer_t;
#endif

#endif // NCNN_C_TYPES_H