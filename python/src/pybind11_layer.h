/* Tencent is pleased to support the open source community by making ncnn available.
 *
 * Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software distributed
 * under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 */

#ifndef PYBIND11_NCNN_LAYER_H
#define PYBIND11_NCNN_LAYER_H

#include <layer.h>
#include "pybind11_bind.h"

class PyLayer : public ncnn::Layer
{
public:
    virtual int load_param(const ncnn::ParamDict& pd)
    {
        PYBIND11_OVERRIDE_REFERENCE(
            int,
            ncnn::Layer,
            load_param,
            pd);
    }

    virtual int load_model(const ncnn::ModelBin& mb)
    {
        PYBIND11_OVERRIDE_REFERENCE(
            int,
            ncnn::Layer,
            load_model,
            mb);
    }

    virtual int create_pipeline(const ncnn::Option& opt)
    {
        PYBIND11_OVERRIDE_REFERENCE(
            int,
            ncnn::Layer,
            create_pipeline,
            opt);
    }

    virtual int destroy_pipeline(const ncnn::Option& opt)
    {
        PYBIND11_OVERRIDE_REFERENCE(
            int,
            ncnn::Layer,
            destroy_pipeline,
            opt);
    }

public:
    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        PYBIND11_OVERRIDE_REFERENCE(
            int,
            ncnn::Layer,
            forward,
            bottom_blobs,
            top_blobs,
            opt);
    }
    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        PYBIND11_OVERRIDE_REFERENCE(
            int,
            ncnn::Layer,
            forward,
            bottom_blob,
            top_blob,
            opt);
    }

    virtual int forward_inplace(std::vector<ncnn::Mat>& bottom_top_blobs, const ncnn::Option& opt) const
    {
        PYBIND11_OVERRIDE_REFERENCE(
            int,
            ncnn::Layer,
            forward_inplace,
            bottom_top_blobs,
            opt);
    }
    virtual int forward_inplace(ncnn::Mat& bottom_top_blob, const ncnn::Option& opt) const
    {
        PYBIND11_OVERRIDE_REFERENCE(
            int,
            ncnn::Layer,
            forward_inplace,
            bottom_top_blob,
            opt);
    }

#if NCNN_VULKAN
public:
    virtual int upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt)
    {
        PYBIND11_OVERRIDE_REFERENCE(
            int,
            ncnn::Layer,
            upload_model,
            cmd,
            opt);
    }

public:
    virtual int forward(const std::vector<ncnn::VkMat>& bottom_blobs, std::vector<ncnn::VkMat>& top_blobs, ncnn::VkCompute& cmd, const ncnn::Option& opt) const
    {
        PYBIND11_OVERRIDE_REFERENCE(
            int,
            ncnn::Layer,
            forward,
            bottom_blobs,
            top_blobs,
            cmd,
            opt);
    }
    virtual int forward(const ncnn::VkMat& bottom_blob, ncnn::VkMat& top_blob, ncnn::VkCompute& cmd, const ncnn::Option& opt) const
    {
        PYBIND11_OVERRIDE_REFERENCE(
            int,
            ncnn::Layer,
            forward,
            bottom_blob,
            top_blob,
            cmd,
            opt);
    }

    virtual int forward_inplace(std::vector<ncnn::VkMat>& bottom_top_blobs, ncnn::VkCompute& cmd, const ncnn::Option& opt) const
    {
        PYBIND11_OVERRIDE_REFERENCE(
            int,
            ncnn::Layer,
            forward_inplace,
            bottom_top_blobs,
            cmd,
            opt);
    }
    virtual int forward_inplace(ncnn::VkMat& bottom_top_blob, ncnn::VkCompute& cmd, const ncnn::Option& opt) const
    {
        PYBIND11_OVERRIDE_REFERENCE(
            int,
            ncnn::Layer,
            forward_inplace,
            bottom_top_blob,
            cmd,
            opt);
    }
#endif // NCNN_VULKAN
};

#endif
