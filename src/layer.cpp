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

#include "layer.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include "cpu.h"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverloaded-virtual"
#endif
#include "layer_declaration.h"
#ifdef __clang__
#pragma clang diagnostic pop
#endif

namespace ncnn {

Layer::Layer()
{
    one_blob_only = false;
    support_inplace = false;
    support_vulkan = false;
    support_packing = false;

    support_bf16_storage = false;

#if NCNN_VULKAN
    vkdev = 0;
#endif // NCNN_VULKAN
}

Layer::~Layer()
{
}

int Layer::load_param(const ParamDict& /*pd*/)
{
    return 0;
}

int Layer::load_model(const ModelBin& /*mb*/)
{
    return 0;
}

int Layer::create_pipeline(const Option& /*opt*/)
{
    return 0;
}

int Layer::destroy_pipeline(const Option& /*opt*/)
{
    return 0;
}

int Layer::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (!support_inplace)
        return -1;

    top_blobs = bottom_blobs;
    for (int i = 0; i < (int)top_blobs.size(); i++)
    {
        top_blobs[i] = bottom_blobs[i].clone(opt.blob_allocator);
        if (top_blobs[i].empty())
            return -100;
    }

    return forward_inplace(top_blobs, opt);
}

int Layer::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (!support_inplace)
        return -1;

    top_blob = bottom_blob.clone(opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    return forward_inplace(top_blob, opt);
}

int Layer::forward_inplace(std::vector<Mat>& /*bottom_top_blobs*/, const Option& /*opt*/) const
{
    return -1;
}

int Layer::forward_inplace(Mat& /*bottom_top_blob*/, const Option& /*opt*/) const
{
    return -1;
}

#if NCNN_VULKAN
int Layer::upload_model(VkTransfer& /*cmd*/, const Option& /*opt*/)
{
    return 0;
}

int Layer::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    if (!support_inplace)
        return -1;

    top_blobs.resize(bottom_blobs.size());
    for (int i = 0; i < (int)top_blobs.size(); i++)
    {
        top_blobs[i].create_like(bottom_blobs[i], bottom_blobs[i].allocator, bottom_blobs[i].staging_allocator);
        if (top_blobs[i].empty())
            return -100;

        cmd.record_clone(bottom_blobs[i], top_blobs[i]);
    }

    return forward_inplace(top_blobs, cmd, opt);
}

int Layer::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    if (!support_inplace)
        return -1;

    top_blob.create_like(bottom_blob, bottom_blob.allocator, bottom_blob.staging_allocator);
    if (top_blob.empty())
        return -100;

    cmd.record_clone(bottom_blob, top_blob);

    return forward_inplace(top_blob, cmd, opt);
}

int Layer::forward_inplace(std::vector<VkMat>& /*bottom_top_blobs*/, VkCompute& /*cmd*/, const Option& /*opt*/) const
{
    return -1;
}

int Layer::forward_inplace(VkMat& /*bottom_top_blob*/, VkCompute& /*cmd*/, const Option& /*opt*/) const
{
    return -1;
}
#endif // NCNN_VULKAN

static const layer_registry_entry layer_registry[] =
{
#include "layer_registry.h"
};

static const int layer_registry_entry_count = sizeof(layer_registry) / sizeof(layer_registry_entry);

#if NCNN_STRING
int layer_to_index(const char* type)
{
    for (int i=0; i<layer_registry_entry_count; i++)
    {
        if (strcmp(type, layer_registry[i].name) == 0)
            return i;
    }

    return -1;
}

Layer* create_layer(const char* type)
{
    int index = layer_to_index(type);
    if (index == -1)
        return 0;

    return create_layer(index);
}
#endif // NCNN_STRING

Layer* create_layer(int index)
{
    if (index < 0 || index >= layer_registry_entry_count)
        return 0;

    layer_creator_func layer_creator = layer_registry[index].creator;
    if (!layer_creator)
        return 0;

    Layer* layer = layer_creator();
    layer->typeindex = index;
    return layer;
}

} // namespace ncnn
