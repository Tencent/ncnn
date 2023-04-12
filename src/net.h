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

#include "blob.h"
#include "layer.h"
#include "mat.h"
#include "option.h"
#include "platform.h"

#if NCNN_PLATFORM_API
#if __ANDROID_API__ >= 9
#include <android/asset_manager.h>
#endif // __ANDROID_API__ >= 9
#endif // NCNN_PLATFORM_API

namespace ncnn {

#if NCNN_VULKAN
class VkCompute;
#endif // NCNN_VULKAN
class DataReader;
class Extractor;
class NetPrivate;
class NCNN_EXPORT Net
{
public:
    // empty init
    Net();
    // clear and destroy
    virtual ~Net();

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
    // register custom layer or overwrite built-in layer by layer type name
    // return 0 if success
    int register_custom_layer(const char* type, layer_creator_func creator, layer_destroyer_func destroyer = 0, void* userdata = 0);
    virtual int custom_layer_to_index(const char* type);
#endif // NCNN_STRING
    // register custom layer or overwrite built-in layer by layer type
    // return 0 if success
    int register_custom_layer(int index, layer_creator_func creator, layer_destroyer_func destroyer = 0, void* userdata = 0);

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

#if NCNN_PLATFORM_API
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
#endif // NCNN_PLATFORM_API

    // unload network structure and weight data
    void clear();

    // construct an Extractor from network
    Extractor create_extractor() const;

    // get input/output indexes/names
    const std::vector<int>& input_indexes() const;
    const std::vector<int>& output_indexes() const;
#if NCNN_STRING
    const std::vector<const char*>& input_names() const;
    const std::vector<const char*>& output_names() const;
#endif

    const std::vector<Blob>& blobs() const;
    const std::vector<Layer*>& layers() const;

    std::vector<Blob>& mutable_blobs();
    std::vector<Layer*>& mutable_layers();

protected:
    friend class Extractor;
#if NCNN_STRING
    int find_blob_index_by_name(const char* name) const;
    int find_layer_index_by_name(const char* name) const;
    virtual Layer* create_custom_layer(const char* type);
    virtual Layer* create_overwrite_builtin_layer(const char* type);
#endif // NCNN_STRING
    virtual Layer* create_custom_layer(int index);
    virtual Layer* create_overwrite_builtin_layer(int typeindex);

private:
    Net(const Net&);
    Net& operator=(const Net&);

private:
    NetPrivate* const d;
};

class ExtractorPrivate;
class NCNN_EXPORT Extractor
{
public:
    virtual ~Extractor();

    // copy
    Extractor(const Extractor&);

    // assign
    Extractor& operator=(const Extractor&);

    // clear blob mats and alloctors
    void clear();

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
    // type = 0, default
    // type = 1, do not convert fp16/bf16 or / and packing
    int extract(const char* blob_name, Mat& feat, int type = 0);
#endif // NCNN_STRING

    // set input by blob index
    // return 0 if success
    int input(int blob_index, const Mat& in);

    // get result by blob index
    // return 0 if success
    // type = 0, default
    // type = 1, do not convert fp16/bf16 or / and packing
    int extract(int blob_index, Mat& feat, int type = 0);

#if NCNN_VULKAN
#if NCNN_STRING
    // set input by blob name
    // return 0 if success
    int input(const char* blob_name, const VkMat& in);

    // get result by blob name
    // return 0 if success
    int extract(const char* blob_name, VkMat& feat, VkCompute& cmd);

    // set input by blob name
    // return 0 if success
    int input(const char* blob_name, const VkImageMat& in);

    // get result by blob name
    // return 0 if success
    int extract(const char* blob_name, VkImageMat& feat, VkCompute& cmd);
#endif // NCNN_STRING

    // set input by blob index
    // return 0 if success
    int input(int blob_index, const VkMat& in);

    // get result by blob index
    // return 0 if success
    int extract(int blob_index, VkMat& feat, VkCompute& cmd);

    // set input by blob index
    // return 0 if success
    int input(int blob_index, const VkImageMat& in);

    // get result by blob index
    // return 0 if success
    int extract(int blob_index, VkImageMat& feat, VkCompute& cmd);
#endif // NCNN_VULKAN

protected:
    friend Extractor Net::create_extractor() const;
    Extractor(const Net* net, size_t blob_count);

private:
    ExtractorPrivate* const d;
};

} // namespace ncnn

#endif // NCNN_NET_H
