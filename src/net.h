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

#ifndef NCNN_NET_H
#define NCNN_NET_H

#include <stdio.h>
#include <vector>
#include "platform.h"
#include "blob.h"
#include "layer.h"
#include "mat.h"
#include "option.h"

#if __ANDROID_API__ >= 9
#include <android/asset_manager.h>
#endif // __ANDROID_API__ >= 9

namespace ncnn {

#if NCNN_VULKAN
class VkCompute;
#endif // NCNN_VULKAN
class DataReader;
class Extractor;
class Net
{
public:
    // empty init
    Net();
    // clear and destroy
    ~Net();

public:
    // option can be changed before loading
    Option opt;

#if NCNN_VULKAN
    // set gpu device by index
    void set_vulkan_device(int device_index);

    // set gpu device by device handle, no owner transfer
    void set_vulkan_device(const VulkanDevice* vkdev);

    const VulkanDevice* vulkan_device() const;
#endif // NCNN_VULKAN

#if NCNN_STRING
    // register custom layer by layer type name
    // return 0 if success
    int register_custom_layer(const char* type, layer_creator_func creator);
#endif // NCNN_STRING
    // register custom layer by layer type
    // return 0 if success
    int register_custom_layer(int index, layer_creator_func creator);

#if NCNN_STRING
    int load_param(const DataReader& dr);
#endif // NCNN_STRING

    int load_param_bin(const DataReader& dr);

    int load_model(const DataReader& dr);

#if NCNN_STDIO
#if NCNN_STRING
    // load network structure from plain param file
    // return 0 if success
    int load_param(FILE* fp);
    int load_param(const char* protopath);
    int load_param_mem(const char* mem);
#endif // NCNN_STRING
    // load network structure from binary param file
    // return 0 if success
    int load_param_bin(FILE* fp);
    int load_param_bin(const char* protopath);

    // load network weight data from model file
    // return 0 if success
    int load_model(FILE* fp);
    int load_model(const char* modelpath);
#endif // NCNN_STDIO

    // load network structure from external memory
    // memory pointer must be 32-bit aligned
    // return bytes consumed
    int load_param(const unsigned char* mem);

    // reference network weight data from external memory
    // weight data is not copied but referenced
    // so external memory should be retained when used
    // memory pointer must be 32-bit aligned
    // return bytes consumed
    int load_model(const unsigned char* mem);

#if __ANDROID_API__ >= 9
#if NCNN_STRING
    // convenient load network structure from android asset plain param file
    int load_param(AAsset* asset);
    int load_param(AAssetManager* mgr, const char* assetpath);
#endif // NCNN_STRING
    // convenient load network structure from android asset binary param file
    int load_param_bin(AAsset* asset);
    int load_param_bin(AAssetManager* mgr, const char* assetpath);

    // convenient load network weight data from android asset model file
    int load_model(AAsset* asset);
    int load_model(AAssetManager* mgr, const char* assetpath);
#endif // __ANDROID_API__ >= 9

    // unload network structure and weight data
    void clear();

    // construct an Extractor from network
    Extractor create_extractor() const;

protected:
    // parse the structure of network
    // fuse int8 op dequantize and quantize by requantize
    int fuse_network();

#if NCNN_VULKAN

    int upload_model();

    int create_pipeline();

    int destroy_pipeline();

#endif // NCNN_VULKAN

    friend class Extractor;
#if NCNN_STRING
    int find_blob_index_by_name(const char* name) const;
    int find_layer_index_by_name(const char* name) const;
    int custom_layer_to_index(const char* type);
    Layer* create_custom_layer(const char* type);
#endif // NCNN_STRING
    Layer* create_custom_layer(int index);
    int forward_layer(int layer_index, std::vector<Mat>& blob_mats, Option& opt) const;

#if NCNN_VULKAN
    int forward_layer(int layer_index, std::vector<Mat>& blob_mats, std::vector<VkMat>& blob_mats_gpu, VkCompute& cmd, Option& opt) const;
#endif // NCNN_VULKAN

protected:
    std::vector<Blob> blobs;
    std::vector<Layer*> layers;

    std::vector<layer_registry_entry> custom_layer_registry;

#if NCNN_VULKAN
    const VulkanDevice* vkdev;

    VkAllocator* weight_vkallocator;
    VkAllocator* weight_staging_vkallocator;

    ncnn::Layer* cast_float32_to_float16;
    ncnn::Layer* cast_float16_to_float32;
    ncnn::Layer* packing_pack1;
    ncnn::Layer* packing_pack4;
    ncnn::Layer* packing_pack8;
#endif // NCNN_VULKAN
};

class Extractor
{
public:
    ~Extractor();

    // enable light mode
    // intermediate blob will be recycled when enabled
    // enabled by default
    void set_light_mode(bool enable);

    // set thread count for this extractor
    // this will overwrite the global setting
    // default count is system depended
    void set_num_threads(int num_threads);

    // set blob memory allocator
    void set_blob_allocator(Allocator* allocator);

    // set workspace memory allocator
    void set_workspace_allocator(Allocator* allocator);

#if NCNN_VULKAN
    void set_vulkan_compute(bool enable);

    void set_blob_vkallocator(VkAllocator* allocator);

    void set_workspace_vkallocator(VkAllocator* allocator);

    void set_staging_vkallocator(VkAllocator* allocator);
#endif // NCNN_VULKAN

#if NCNN_STRING
    // set input by blob name
    // return 0 if success
    int input(const char* blob_name, const Mat& in);

    // get result by blob name
    // return 0 if success
    int extract(const char* blob_name, Mat& feat);
#endif // NCNN_STRING

    // set input by blob index
    // return 0 if success
    int input(int blob_index, const Mat& in);

    // get result by blob index
    // return 0 if success
    int extract(int blob_index, Mat& feat);

#if NCNN_VULKAN
#if NCNN_STRING
    // set input by blob name
    // return 0 if success
    int input(const char* blob_name, const VkMat& in);

    // get result by blob name
    // return 0 if success
    int extract(const char* blob_name, VkMat& feat, VkCompute& cmd);
#endif // NCNN_STRING

    // set input by blob index
    // return 0 if success
    int input(int blob_index, const VkMat& in);

    // get result by blob index
    // return 0 if success
    int extract(int blob_index, VkMat& feat, VkCompute& cmd);
#endif // NCNN_VULKAN

protected:
    friend Extractor Net::create_extractor() const;
    Extractor(const Net* net, size_t blob_count);

private:
    const Net* net;
    std::vector<Mat> blob_mats;
    Option opt;

#if NCNN_VULKAN
    VkAllocator* local_blob_vkallocator;
    VkAllocator* local_staging_vkallocator;

    std::vector<VkMat> blob_mats_gpu;
#endif // NCNN_VULKAN
};

} // namespace ncnn

#endif // NCNN_NET_H
