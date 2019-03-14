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

#ifndef LAYER_CROP_H
#define LAYER_CROP_H

#include "layer.h"

namespace ncnn {

class Crop : public Layer
{
public:
    Crop();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

#if NCNN_VULKAN
    virtual int create_pipeline();
    virtual int destroy_pipeline();

    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;
#endif // NCNN_VULKAN

public:
    int woffset;
    int hoffset;
    int coffset;

    // -233 = remaining
    // -234 = remaining - 1
    int outw;
    int outh;
    int outc;

#if NCNN_VULKAN
    Pipeline* pipeline_crop;
    Pipeline* pipeline_crop_pack4;
#endif // NCNN_VULKAN
};

} // namespace ncnn

#endif // LAYER_CROP_H
