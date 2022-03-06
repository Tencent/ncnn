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

#include "cpu.h"

#include <math.h>
#include <string.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4250)
#endif
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverloaded-virtual"
#endif
#include "layer_declaration.h"
#ifdef __clang__
#pragma clang diagnostic pop
#endif
#ifdef _MSC_VER
#pragma warning(pop)
#endif

namespace ncnn {

Layer::Layer()
{
    one_blob_only = false;
    support_inplace = false;
    support_vulkan = false;
    support_packing = false;

    support_bf16_storage = false;
    support_fp16_storage = false;
    support_int8_storage = false;
    support_image_storage = false;
    support_tensor_storage = false;

    support_reserved_00 = false;

    typeindex = -1;

#if NCNN_VULKAN
    vkdev = 0;
#endif // NCNN_VULKAN

    userdata = 0;
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
        cmd.record_clone(bottom_blobs[i], top_blobs[i], opt);
    }

    return forward_inplace(top_blobs, cmd, opt);
}

int Layer::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    if (!support_inplace)
        return -1;

    cmd.record_clone(bottom_blob, top_blob, opt);

    return forward_inplace(top_blob, cmd, opt);
}

int Layer::forward(const std::vector<VkImageMat>& bottom_blobs, std::vector<VkImageMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    if (!support_inplace)
        return -1;

    top_blobs.resize(bottom_blobs.size());
    for (int i = 0; i < (int)top_blobs.size(); i++)
    {
        cmd.record_clone(bottom_blobs[i], top_blobs[i], opt);
    }

    return forward_inplace(top_blobs, cmd, opt);
}

int Layer::forward(const VkImageMat& bottom_blob, VkImageMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    if (!support_inplace)
        return -1;

    cmd.record_clone(bottom_blob, top_blob, opt);

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

int Layer::forward_inplace(std::vector<VkImageMat>& /*bottom_top_blobs*/, VkCompute& /*cmd*/, const Option& /*opt*/) const
{
    return -1;
}

int Layer::forward_inplace(VkImageMat& /*bottom_top_blob*/, VkCompute& /*cmd*/, const Option& /*opt*/) const
{
    return -1;
}
#endif // NCNN_VULKAN

#include "layer_registry.h"

static const int layer_registry_entry_count = sizeof(layer_registry) / sizeof(layer_registry_entry);

#if NCNN_STRING
int layer_to_index(const char* type)
{
    for (int i = 0; i < layer_registry_entry_count; i++)
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

    // clang-format off
    // *INDENT-OFF*
    layer_creator_func layer_creator = 0;
#if NCNN_RUNTIME_CPU && NCNN_AVX512
    if (ncnn::cpu_support_x86_avx512())
    {
        layer_creator = layer_registry_avx512[index].creator;
    }
    else
#endif// NCNN_RUNTIME_CPU && NCNN_AVX512
#if NCNN_RUNTIME_CPU && NCNN_FMA
    if (ncnn::cpu_support_x86_fma())
    {
        layer_creator = layer_registry_fma[index].creator;
    }
    else
#endif// NCNN_RUNTIME_CPU && NCNN_FMA
#if NCNN_RUNTIME_CPU && NCNN_AVX
    if (ncnn::cpu_support_x86_avx())
    {
        layer_creator = layer_registry_avx[index].creator;
    }
    else
#endif // NCNN_RUNTIME_CPU && NCNN_AVX
#if NCNN_RUNTIME_CPU && NCNN_ARM82
    if (ncnn::cpu_support_arm_asimdhp())
    {
        layer_creator = layer_registry_arm82[index].creator;
    }
    else
#endif // NCNN_RUNTIME_CPU && NCNN_ARM82
#if NCNN_RUNTIME_CPU && NCNN_MMI
    if (ncnn::cpu_support_mips_msa() && ncnn::cpu_support_loongson_mmi())
    {
        layer_creator = layer_registry_mmi[index].creator;
    }
    else
#endif // NCNN_RUNTIME_CPU && NCNN_MMI
#if NCNN_RUNTIME_CPU && NCNN_MSA
    if (ncnn::cpu_support_mips_msa())
    {
        layer_creator = layer_registry_msa[index].creator;
    }
    else
#endif // NCNN_RUNTIME_CPU && NCNN_MSA
#if NCNN_RUNTIME_CPU && NCNN_RVV
    if (ncnn::cpu_support_riscv_v())
    {
        layer_creator = layer_registry_rvv[index].creator;
    }
    else
#endif // NCNN_RUNTIME_CPU && NCNN_RVV
    {
        layer_creator = layer_registry[index].creator;
    }
    // *INDENT-ON*
    // clang-format on
    if (!layer_creator)
        return 0;

    Layer* layer = layer_creator(0);
    layer->typeindex = index;
    return layer;
}

} // namespace ncnn
