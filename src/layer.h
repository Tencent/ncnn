// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_LAYER_H
#define NCNN_LAYER_H

#include "mat.h"
#include "modelbin.h"
#include "option.h"
#include "paramdict.h"
#include "platform.h"

#if NCNN_VULKAN
#include "command.h"
#include "pipeline.h"
#endif // NCNN_VULKAN

namespace ncnn {

class NCNN_EXPORT Layer
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

    // layer implementation specific setup
    // return 0 if success
    virtual int create_pipeline(const Option& opt);

    // layer implementation specific clean
    // return 0 if success
    virtual int destroy_pipeline(const Option& opt);

public:
    // one input and one output blob
    bool one_blob_only;

    // support inplace inference
    bool support_inplace;

    // support vulkan compute
    bool support_vulkan;

    // accept input blob with packed storage
    bool support_packing;

    // accept bf16
    bool support_bf16_storage;

    // accept fp16
    bool support_fp16_storage;

    // accept int8
    bool support_int8_storage;

    // shader tensor storage
    bool support_tensor_storage;

    bool support_reserved_000;

    bool support_reserved_00;

    bool support_reserved_0;
    bool support_reserved_1;
    bool support_reserved_2;
    bool support_reserved_3;
    bool support_reserved_4;
    bool support_reserved_5;
    bool support_reserved_6;
    bool support_reserved_7;
    bool support_reserved_8;
    bool support_reserved_9;

    // feature disabled set
    int featmask;

public:
    // implement inference
    // return 0 if success
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    // implement inplace inference
    // return 0 if success
    virtual int forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const;
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

#if NCNN_VULKAN
public:
    // upload weight blob from host to device
    virtual int upload_model(VkTransfer& cmd, const Option& opt);

public:
    // implement inference
    // return 0 if success
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

    // implement inplace inference
    // return 0 if success
    virtual int forward_inplace(std::vector<VkMat>& bottom_top_blobs, VkCompute& cmd, const Option& opt) const;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    // assigned immediately after creating this layer
    const VulkanDevice* vkdev;
#endif // NCNN_VULKAN

public:
    // custom user data
    void* userdata;
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
    // shape hint
    std::vector<Mat> bottom_shapes;
    std::vector<Mat> top_shapes;
};

// layer factory function
typedef Layer* (*layer_creator_func)(void*);
typedef void (*layer_destroyer_func)(Layer*, void*);

struct layer_registry_entry
{
#if NCNN_STRING
    // layer type name
    const char* name;
#endif // NCNN_STRING
    // layer factory entry
    layer_creator_func creator;
};

struct custom_layer_registry_entry
{
#if NCNN_STRING
    // layer type name
    const char* name;
#endif // NCNN_STRING
    // layer factory entry
    layer_creator_func creator;
    layer_destroyer_func destroyer;
    void* userdata;
};

struct overwrite_builtin_layer_registry_entry
{
    // layer type index
    int typeindex;
    // layer factory entry
    layer_creator_func creator;
    layer_destroyer_func destroyer;
    void* userdata;
};

#if NCNN_STRING
// get layer type from type name
NCNN_EXPORT int layer_to_index(const char* type);
// create layer from type name
NCNN_EXPORT Layer* create_layer(const char* type);
NCNN_EXPORT Layer* create_layer_naive(const char* type);
NCNN_EXPORT Layer* create_layer_cpu(const char* type);
#if NCNN_VULKAN
NCNN_EXPORT Layer* create_layer_vulkan(const char* type);
#endif // NCNN_VULKAN
#endif // NCNN_STRING
// create layer from layer type
NCNN_EXPORT Layer* create_layer(int index);
NCNN_EXPORT Layer* create_layer_naive(int index);
NCNN_EXPORT Layer* create_layer_cpu(int index);
#if NCNN_VULKAN
NCNN_EXPORT Layer* create_layer_vulkan(int index);
#endif // NCNN_VULKAN

#define DEFINE_LAYER_CREATOR(name)                          \
    ::ncnn::Layer* name##_layer_creator(void* /*userdata*/) \
    {                                                       \
        return new name;                                    \
    }

#define DEFINE_LAYER_DESTROYER(name)                                      \
    void name##_layer_destroyer(::ncnn::Layer* layer, void* /*userdata*/) \
    {                                                                     \
        delete layer;                                                     \
    }

} // namespace ncnn

#endif // NCNN_LAYER_H
