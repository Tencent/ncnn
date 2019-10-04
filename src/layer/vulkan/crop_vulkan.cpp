// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "crop_vulkan.h"
#include <algorithm>

namespace ncnn {

DEFINE_LAYER_CREATOR(Crop_vulkan)

Crop_vulkan::Crop_vulkan()
{
    support_vulkan = true;

    pipeline_crop = 0;
    pipeline_crop_pack4 = 0;
    pipeline_crop_pack1to4 = 0;
    pipeline_crop_pack4to1 = 0;
}

int Crop_vulkan::create_pipeline(const Option& opt)
{
    std::vector<vk_specialization_type> specializations;

    // pack1
    {
        pipeline_crop = new Pipeline(vkdev);
        pipeline_crop->set_optimal_local_size_xyz();
        pipeline_crop->create("crop", opt, specializations, 2, 13);
    }

    // pack4
    {
        pipeline_crop_pack4 = new Pipeline(vkdev);
        pipeline_crop_pack4->set_optimal_local_size_xyz();
        pipeline_crop_pack4->create("crop_pack4", opt, specializations, 2, 13);
    }

    // pack1to4
    {
        pipeline_crop_pack1to4 = new Pipeline(vkdev);
        pipeline_crop_pack1to4->set_optimal_local_size_xyz();
        pipeline_crop_pack1to4->create("crop_pack1to4", opt, specializations, 2, 13);
    }

    // pack4to1
    {
        pipeline_crop_pack4to1 = new Pipeline(vkdev);
        pipeline_crop_pack4to1->set_optimal_local_size_xyz();
        pipeline_crop_pack4to1->create("crop_pack4to1", opt, specializations, 2, 13);
    }

    return 0;
}

int Crop_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_crop;
    pipeline_crop = 0;

    delete pipeline_crop_pack4;
    pipeline_crop_pack4 = 0;

    delete pipeline_crop_pack1to4;
    pipeline_crop_pack1to4 = 0;

    delete pipeline_crop_pack4to1;
    pipeline_crop_pack4to1 = 0;

    return 0;
}

int Crop_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    // TODO vec and image crop

    int _woffset = woffset;
    int _hoffset = hoffset;
    int _coffset = coffset;
    int _woffset2 = woffset2;
    int _hoffset2 = hoffset2;
    int _coffset2 = coffset2;
    int _outw;
    int _outh;
    int _outc;

    if (_hoffset == -233 && _coffset == -233)
    {
        _woffset = 0;
        _woffset2 = 0;
        _outw = w;
        _hoffset = 0;
        _hoffset2 = 0;
        _outh = h;

        _coffset = woffset;
        _coffset2 = woffset2;

        if (outw == -233)
            _outc = channels * elempack - _coffset - _coffset2;
        else
            _outc = std::min(outw, channels * elempack - _coffset - _coffset2);
    }
    else if (_hoffset == -233)
    {
        _woffset = 0;
        _woffset2 = 0;
        _outw = w;

        _hoffset = woffset;
        _hoffset2 = woffset2;

        if (outw == -233)
            _outh = h - _hoffset - _hoffset2;
        else
            _outh = std::min(outw, h - _hoffset - _hoffset2);

        _coffset = hoffset;
        _coffset2 = hoffset2;

        if (outh == -233)
            _outc = channels * elempack - _coffset - _coffset2;
        else
            _outc = std::min(outh, channels * elempack - _coffset - _coffset2);
    }
    else
    {
        if (outw == -233)
            _outw = w - _woffset - _woffset2;
        else
            _outw = std::min(outw, w - _woffset - _woffset2);

        if (outh == -233)
            _outh = h - _hoffset - _hoffset2;
        else
            _outh = std::min(outh, h - _hoffset - _hoffset2);

        if (outc == -233)
            _outc = channels * elempack - _coffset - _coffset2;
        else
            _outc = std::min(outc, channels * elempack - _coffset - _coffset2);
    }

    int out_elempack = _outc % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (opt.use_fp16_packed && !opt.use_fp16_storage)
    {
        if (out_elempack == 4) out_elemsize = 4*2u;
        if (out_elempack == 1) out_elemsize = 4u;
    }

    top_blob.create(_outw, _outh, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator, opt.staging_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(13);
    constants[0].i = bottom_blob.dims;
    constants[1].i = bottom_blob.w;
    constants[2].i = bottom_blob.h;
    constants[3].i = bottom_blob.c;
    constants[4].i = bottom_blob.cstep;
    constants[5].i = top_blob.dims;
    constants[6].i = top_blob.w;
    constants[7].i = top_blob.h;
    constants[8].i = top_blob.c;
    constants[9].i = top_blob.cstep;
    constants[10].i = _woffset;
    constants[11].i = _hoffset;
    constants[12].i = _coffset;

    const Pipeline* pipeline = 0;
    if (elempack == 1 && out_elempack == 1)
    {
        pipeline = pipeline_crop;
    }
    else if (elempack == 4 && out_elempack == 4)
    {
        constants[12].i = _coffset / 4;// TODO pack4to1to4

        pipeline = pipeline_crop_pack4;
    }
    else if (elempack == 1 && out_elempack == 4)
    {
        pipeline = pipeline_crop_pack1to4;
    }
    else if (elempack == 4 && out_elempack == 1)
    {
        pipeline = pipeline_crop_pack4to1;
    }

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

int Crop_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkMat& bottom_blob = bottom_blobs[0];
    const VkMat& reference_blob = bottom_blobs[1];

    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    // TODO vec and image crop

    int _woffset = woffset;
    int _hoffset = hoffset;
    int _coffset = coffset;
    int _outw;
    int _outh;
    int _outc;
    if (_woffset == -233 && _hoffset == -233 && _coffset == -233)
    {
        const int* param_data = reference_blob.mapped();

        _woffset = param_data[0];
        _hoffset = param_data[1];
        _coffset = param_data[2];
        _outw = param_data[3];
        _outh = param_data[4];
        _outc = param_data[5];
    }
    else
    {
        _outw = reference_blob.w;
        _outh = reference_blob.h;
        _outc = reference_blob.dims == 3 ? reference_blob.c * reference_blob.elempack : channels * elempack;
    }

    int out_elempack = _outc % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (opt.use_fp16_packed && !opt.use_fp16_storage)
    {
        if (out_elempack == 4) out_elemsize = 4*2u;
        if (out_elempack == 1) out_elemsize = 4u;
    }

    VkMat& top_blob = top_blobs[0];

    top_blob.create(_outw, _outh, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator, opt.staging_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(13);
    constants[0].i = bottom_blob.dims;
    constants[1].i = bottom_blob.w;
    constants[2].i = bottom_blob.h;
    constants[3].i = bottom_blob.c;
    constants[4].i = bottom_blob.cstep;
    constants[5].i = top_blob.dims;
    constants[6].i = top_blob.w;
    constants[7].i = top_blob.h;
    constants[8].i = top_blob.c;
    constants[9].i = top_blob.cstep;
    constants[10].i = _woffset;
    constants[11].i = _hoffset;
    constants[12].i = _coffset;

    const Pipeline* pipeline = 0;
    if (elempack == 1 && out_elempack == 1)
    {
        pipeline = pipeline_crop;
    }
    else if (elempack == 4 && out_elempack == 4)
    {
        constants[12].i = _coffset / 4;// TODO pack4to1to4

        pipeline = pipeline_crop_pack4;
    }
    else if (elempack == 1 && out_elempack == 4)
    {
        pipeline = pipeline_crop_pack1to4;
    }
    else if (elempack == 4 && out_elempack == 1)
    {
        pipeline = pipeline_crop_pack4to1;
    }

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

} // namespace ncnn
