// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_PADDING_VULKAN_H
#define LAYER_PADDING_VULKAN_H

#include "padding.h"

namespace ncnn {

class Padding_vulkan : public Padding
{
public:
    Padding_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using Padding::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;

protected:
#if NCNN_INT8
    int create_pipeline_int8(const Option& opt);
    int forward_int8(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;
    int forward_int8(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;
#endif // NCNN_INT8

public:
    VkMat per_channel_pad_data_gpu;

    Pipeline* pipeline_padding;
    Pipeline* pipeline_padding_pack4;
    Pipeline* pipeline_padding_pack1to4;
    Pipeline* pipeline_padding_pack4to1;

    Pipeline* pipeline_padding_3d;
    Pipeline* pipeline_padding_3d_pack4;

#if NCNN_INT8
    Mat per_channel_pad_data_int8;
    VkMat per_channel_pad_data_int8_gpu;

    Pipeline* pipeline_padding_int8;
    Pipeline* pipeline_padding_pack4_int8;
    Pipeline* pipeline_padding_pack1to4_int8;
    Pipeline* pipeline_padding_pack4to1_int8;

    Pipeline* pipeline_padding_3d_int8;
    Pipeline* pipeline_padding_3d_pack4_int8;
#endif // NCNN_INT8
};

} // namespace ncnn

#endif // LAYER_PADDING_VULKAN_H
