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

#ifndef LAYER_BATCHNORM_H
#define LAYER_BATCHNORM_H

#include "layer.h"

namespace ncnn {

class BatchNorm : public Layer
{
public:
    BatchNorm();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

#if NCNN_VULKAN
    virtual int upload_model(VkTransfer& cmd);

    virtual int create_pipeline();
    virtual int destroy_pipeline();

    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;
#endif // NCNN_VULKAN

public:
    // param
    int channels;
    float eps;

    // model
    Mat slope_data;
    Mat mean_data;
    Mat var_data;
    Mat bias_data;

    Mat a_data;
    Mat b_data;

#if NCNN_VULKAN
    VkMat a_data_gpu;
    VkMat b_data_gpu;
    Pipeline* pipeline_batchnorm;

    VkMat a_data_gpu_pack4;
    VkMat b_data_gpu_pack4;
    Pipeline* pipeline_batchnorm_pack4;
#endif // NCNN_VULKAN

};

} // namespace ncnn

#endif // LAYER_BATCHNORM_H
