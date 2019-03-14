// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef NCNN_LAYER_H
#define NCNN_LAYER_H

#include <stdio.h>
#include <string>
#include <vector>
#include "mat.h"
#include "modelbin.h"
#include "paramdict.h"
#include "platform.h"

#if NCNN_VULKAN
#include <vulkan/vulkan.h>
#include "command.h"
#include "pipeline.h"
#endif // NCNN_VULKAN

namespace ncnn {

#if NCNN_VULKAN
class VkAllocator;
#endif // NCNN_VULKAN

class Allocator;
class Option
{
public:
    // default option
    Option();

public:
    // light mode
    // intermediate blob will be recycled when enabled
    // enabled by default
    bool lightmode;

    // thread count
    // default value is the one returned by get_cpu_count()
    int num_threads;

    // blob memory allocator
    Allocator* blob_allocator;

    // workspace memory allocator
    Allocator* workspace_allocator;

#if NCNN_VULKAN
    // enable vulkan compute
    bool vulkan_compute;

    // blob memory allocator
    VkAllocator* blob_vkallocator;

    // workspace memory allocator
    VkAllocator* workspace_vkallocator;

    // staging memory allocator
    VkAllocator* staging_vkallocator;
#endif // NCNN_VULKAN
};

// the global default option
const Option& get_default_option();
int set_default_option(const Option& opt);

class Layer
{
public:
    // empty
    Layer();
    // virtual destructor
    virtual ~Layer();

    // load layer specific parameter from parsed dict
    // return 0 if success
    virtual int load_param(const ParamDict& pd);

    // load layer specific weight data from model binary
    // return 0 if success
    virtual int load_model(const ModelBin& mb);

public:
    // one input and one output blob
    bool one_blob_only;

    // support inplace inference
    bool support_inplace;

    // support vulkan compute
    bool support_vulkan;

public:
    // implement inference
    // return 0 if success
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt = get_default_option()) const;
    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt = get_default_option()) const;

    // implement inplace inference
    // return 0 if success
    virtual int forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt = get_default_option()) const;
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt = get_default_option()) const;

#if NCNN_VULKAN
public:
    // upload weight blob from host to device
    virtual int upload_model(VkTransfer& cmd);

    virtual int create_pipeline();
    virtual int destroy_pipeline();

public:
    // implement inference
    // return 0 if success
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt = get_default_option()) const;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt = get_default_option()) const;

    // implement inplace inference
    // return 0 if success
    virtual int forward_inplace(std::vector<VkMat>& bottom_top_blobs, VkCompute& cmd, const Option& opt = get_default_option()) const;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt = get_default_option()) const;

public:
    // assigned immediately after creating this layer
    const VulkanDevice* vkdev;
#endif // NCNN_VULKAN

public:
    // layer type index
    int typeindex;
#if NCNN_STRING
    // layer type name
    std::string type;
    // layer name
    std::string name;
#endif // NCNN_STRING
    // blob index which this layer needs as input
    std::vector<int> bottoms;
    // blob index which this layer produces as output
    std::vector<int> tops;
};

// layer factory function
typedef Layer* (*layer_creator_func)();

struct layer_registry_entry
{
#if NCNN_STRING
    // layer type name
    const char* name;
#endif // NCNN_STRING
    // layer factory entry
    layer_creator_func creator;
};

#if NCNN_STRING
// get layer type from type name
int layer_to_index(const char* type);
// create layer from type name
Layer* create_layer(const char* type);
#endif // NCNN_STRING
// create layer from layer type
Layer* create_layer(int index);

#define DEFINE_LAYER_CREATOR(name) \
    ::ncnn::Layer* name##_layer_creator() { return new name; }

} // namespace ncnn

#endif // NCNN_LAYER_H
