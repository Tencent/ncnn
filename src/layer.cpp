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

#include <string.h>

#include "layer_declaration.h"

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

Layer* create_layer_naive(const char* type)
{
    int index = layer_to_index(type);
    if (index == -1)
        return 0;

    return create_layer_naive(index);
}

Layer* create_layer_cpu(const char* type)
{
    int index = layer_to_index(type);
    if (index == -1)
        return 0;

    return create_layer_cpu(index);
}

#if NCNN_VULKAN
Layer* create_layer_vulkan(const char* type)
{
    int index = layer_to_index(type);
    if (index == -1)
        return 0;

    return create_layer_vulkan(index);
}
#endif // NCNN_VULKAN
#endif // NCNN_STRING

// internal wrapper
class Layer_final : public Layer
{
public:
    Layer* layer_cpu;
#if NCNN_VULKAN
    Layer* layer_vulkan;
#endif

    // utility functions for transfer layer properties
    void set_layer_properties()
    {
        layer_cpu->userdata = userdata;

        layer_cpu->bottoms = bottoms;
        layer_cpu->tops = tops;
        layer_cpu->bottom_shapes = bottom_shapes;
        layer_cpu->top_shapes = top_shapes;
        layer_cpu->featmask = featmask;

#if NCNN_VULKAN
        if (layer_vulkan)
        {
            layer_vulkan->vkdev = vkdev;

            layer_vulkan->userdata = userdata;

            layer_vulkan->bottoms = bottoms;
            layer_vulkan->tops = tops;
            layer_vulkan->bottom_shapes = bottom_shapes;
            layer_vulkan->top_shapes = top_shapes;
            layer_vulkan->featmask = featmask;
        }
#endif
    }

    void get_layer_properties()
    {
        one_blob_only = layer_cpu->one_blob_only;
        support_inplace = layer_cpu->support_inplace;
        support_packing = layer_cpu->support_packing;
        support_bf16_storage = layer_cpu->support_bf16_storage;
        support_fp16_storage = layer_cpu->support_fp16_storage;
        support_int8_storage = layer_cpu->support_int8_storage;

        support_vulkan = 0;
        support_image_storage = 0;
        support_tensor_storage = 0;

#if NCNN_VULKAN
        if (layer_vulkan)
        {
            support_vulkan = layer_vulkan->support_vulkan;
            support_image_storage = layer_vulkan->support_image_storage;
            support_tensor_storage = layer_vulkan->support_tensor_storage;
        }
#endif
    }

public:
    Layer_final()
    {
        layer_cpu = 0;
#if NCNN_VULKAN
        layer_vulkan = 0;
#endif
    }

    ~Layer_final()
    {
        delete layer_cpu;
#if NCNN_VULKAN
        delete layer_vulkan;
#endif
    }

    virtual int load_param(const ParamDict& pd)
    {
        set_layer_properties();
#if NCNN_VULKAN
        if (layer_vulkan)
        {
            if (vkdev)
            {
                int ret = layer_vulkan->load_param(pd);
                get_layer_properties();

                if (layer_vulkan->support_vulkan)
                    return ret;
            }

            // fallback to cpu layer
            delete layer_vulkan;
            layer_vulkan = 0;
        }
#endif // NCNN_VULKAN

        int ret = layer_cpu->load_param(pd);
        get_layer_properties();
        return ret;
    }

    virtual int load_model(const ModelBin& mb)
    {
#if NCNN_VULKAN
        if (layer_vulkan)
        {
            int ret = layer_vulkan->load_model(mb);
            get_layer_properties();
            return ret;
        }
#endif // NCNN_VULKAN

        int ret = layer_cpu->load_model(mb);
        get_layer_properties();
        return ret;
    }

    virtual int create_pipeline(const Option& opt)
    {
        set_layer_properties();
#if NCNN_VULKAN
        if (layer_vulkan)
        {
            int ret = layer_vulkan->create_pipeline(opt);
            get_layer_properties();
            return ret;
        }
#endif // NCNN_VULKAN

        int ret = layer_cpu->create_pipeline(opt);
        get_layer_properties();
        return ret;
    }

    virtual int destroy_pipeline(const Option& opt)
    {
#if NCNN_VULKAN
        if (layer_vulkan)
        {
            return layer_vulkan->destroy_pipeline(opt);
        }
#endif // NCNN_VULKAN

        return layer_cpu->destroy_pipeline(opt);
    }

public:
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
    {
        return layer_cpu->forward(bottom_blobs, top_blobs, opt);
    }

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
    {
        return layer_cpu->forward(bottom_blob, top_blob, opt);
    }

    virtual int forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
    {
        return layer_cpu->forward_inplace(bottom_top_blobs, opt);
    }

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const
    {
        return layer_cpu->forward_inplace(bottom_top_blob, opt);
    }

#if NCNN_VULKAN
public:
    virtual int upload_model(VkTransfer& cmd, const Option& opt)
    {
        return layer_vulkan ? layer_vulkan->upload_model(cmd, opt) : -1;
    }

    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
    {
        return layer_vulkan ? layer_vulkan->forward(bottom_blobs, top_blobs, cmd, opt) : -1;
    }

    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
    {
        return layer_vulkan ? layer_vulkan->forward(bottom_blob, top_blob, cmd, opt) : -1;
    }

    virtual int forward(const std::vector<VkImageMat>& bottom_blobs, std::vector<VkImageMat>& top_blobs, VkCompute& cmd, const Option& opt) const
    {
        return layer_vulkan ? layer_vulkan->forward(bottom_blobs, top_blobs, cmd, opt) : -1;
    }

    virtual int forward(const VkImageMat& bottom_blob, VkImageMat& top_blob, VkCompute& cmd, const Option& opt) const
    {
        return layer_vulkan ? layer_vulkan->forward(bottom_blob, top_blob, cmd, opt) : -1;
    }

    virtual int forward_inplace(std::vector<VkMat>& bottom_top_blobs, VkCompute& cmd, const Option& opt) const
    {
        return layer_vulkan ? layer_vulkan->forward_inplace(bottom_top_blobs, cmd, opt) : -1;
    }

    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
    {
        return layer_vulkan ? layer_vulkan->forward_inplace(bottom_top_blob, cmd, opt) : -1;
    }

    virtual int forward_inplace(std::vector<VkImageMat>& bottom_top_blobs, VkCompute& cmd, const Option& opt) const
    {
        return layer_vulkan ? layer_vulkan->forward_inplace(bottom_top_blobs, cmd, opt) : -1;
    }

    virtual int forward_inplace(VkImageMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
    {
        return layer_vulkan ? layer_vulkan->forward_inplace(bottom_top_blob, cmd, opt) : -1;
    }
#endif // NCNN_VULKAN
};

Layer* create_layer(int index)
{
    Layer* layer_cpu = create_layer_cpu(index);
    if (!layer_cpu)
        return 0;

    Layer_final* layer_final = new Layer_final;
    layer_final->layer_cpu = layer_cpu;

#if NCNN_VULKAN
    layer_final->layer_vulkan = create_layer_vulkan(index);
#endif

    layer_final->typeindex = index;
    layer_final->set_layer_properties();
    layer_final->get_layer_properties();

    return layer_final;
}

Layer* create_layer_naive(int index)
{
    if (index < 0 || index >= layer_registry_entry_count)
        return 0;

    layer_creator_func layer_creator = layer_registry[index].creator;
    if (!layer_creator)
        return 0;

    Layer* layer = layer_creator(0);
    layer->typeindex = index;
    return layer;
}

Layer* create_layer_cpu(int index)
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
#if NCNN_RUNTIME_CPU && NCNN_LASX
    if (ncnn::cpu_support_loongarch_lasx())
    {
        layer_creator = layer_registry_lasx[index].creator;
    }
    else
#endif // NCNN_RUNTIME_CPU && NCNN_LASX
#if NCNN_RUNTIME_CPU && NCNN_LSX
    if (ncnn::cpu_support_loongarch_lsx())
    {
        layer_creator = layer_registry_lsx[index].creator;
    }
    else
#endif // NCNN_RUNTIME_CPU && NCNN_LSX
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
        layer_creator = layer_registry_arch[index].creator;
    }

    if (!layer_creator)
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

#if NCNN_VULKAN
Layer* create_layer_vulkan(int index)
{
    if (index < 0 || index >= layer_registry_entry_count)
        return 0;

    layer_creator_func layer_creator = layer_registry_vulkan[index].creator;
    if (!layer_creator)
        return 0;

    Layer* layer = layer_creator(0);
    layer->typeindex = index;
    return layer;
}
#endif // NCNN_VULKAN

} // namespace ncnn
