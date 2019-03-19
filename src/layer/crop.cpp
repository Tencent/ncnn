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

#include "crop.h"
#include <algorithm>

namespace ncnn {

DEFINE_LAYER_CREATOR(Crop)

Crop::Crop()
{
    one_blob_only = false;
    support_inplace = false;
    support_vulkan = true;

#if NCNN_VULKAN
    pipeline_crop = 0;
    pipeline_crop_pack4 = 0;
#endif // NCNN_VULKAN
}

int Crop::load_param(const ParamDict& pd)
{
    woffset = pd.get(0, 0);
    hoffset = pd.get(1, 0);
    coffset = pd.get(2, 0);
    outw = pd.get(3, 0);
    outh = pd.get(4, 0);
    outc = pd.get(5, 0);

    if (outw != 0 || outh != 0 || outc != 0)
    {
        one_blob_only = true;
    }

    return 0;
}

int Crop::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

    int _outw;
    int _outh;
    int _outc;

    if (outw == -233)
        _outw = w - woffset;
    else if (outw == -234)
        _outw = w - 1 - woffset;
    else
        _outw = std::min(outw, w - woffset);

    if (outh == -233)
        _outh = h - hoffset;
    else if (outh == -234)
        _outh = h - 1 - hoffset;
    else
        _outh = std::min(outh, h - hoffset);

    if (outc == -233)
        _outc = channels - coffset;
    else if (outc == -234)
        _outc = channels - 1 - coffset;
    else
        _outc = std::min(outc, channels - coffset);

    if (_outw == w && _outh == h && _outc == channels)
    {
        top_blob = bottom_blob;
        return 0;
    }

    const Mat bottom_blob_sliced = bottom_blob.channel_range(coffset, _outc);

    if (_outw == w && _outh == h)
    {
        top_blob = bottom_blob_sliced.clone();
        if (top_blob.empty())
            return -100;

        return 0;
    }

    int top = hoffset;
    int bottom = h - _outh - hoffset;
    int left = woffset;
    int right = w - _outw - woffset;

    copy_cut_border(bottom_blob_sliced, top_blob, top, bottom, left, right, opt.blob_allocator, opt.num_threads);
    if (top_blob.empty())
        return -100;

    return 0;
}

int Crop::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& reference_blob = bottom_blobs[1];

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

    Mat& top_blob = top_blobs[0];

    int _outw = reference_blob.w;
    int _outh = reference_blob.h;
    int _outc = reference_blob.dims == 3 ? reference_blob.c : channels;

    if (_outw == w && _outh == h && _outc == channels)
    {
        top_blob = bottom_blob;
        return 0;
    }

    const Mat bottom_blob_sliced = bottom_blob.channel_range(coffset, _outc);

    if (_outw == w && _outh == h)
    {
        top_blob = bottom_blob_sliced.clone();
        if (top_blob.empty())
            return -100;

        return 0;
    }

    int top = hoffset;
    int bottom = h - _outh - hoffset;
    int left = woffset;
    int right = w - _outw - woffset;

    copy_cut_border(bottom_blob_sliced, top_blob, top, bottom, left, right, opt.blob_allocator, opt.num_threads);
    if (top_blob.empty())
        return -100;

    return 0;
}

#if NCNN_VULKAN
int Crop::create_pipeline()
{
    std::vector<vk_specialization_type> specializations(3);
    specializations[0].i = woffset;
    specializations[1].i = hoffset;
    specializations[2].i = coffset;

    // pack1
    {
        pipeline_crop = new Pipeline(vkdev);
        pipeline_crop->set_optimal_local_size_xyz();
        pipeline_crop->create("crop", specializations, 2, 10);
    }

    // pack4
    {
        pipeline_crop_pack4 = new Pipeline(vkdev);
        pipeline_crop_pack4->set_optimal_local_size_xyz();
        pipeline_crop_pack4->create("crop_pack4", specializations, 2, 10);
    }

    return 0;
}

int Crop::destroy_pipeline()
{
    delete pipeline_crop;
    pipeline_crop = 0;

    delete pipeline_crop_pack4;
    pipeline_crop_pack4 = 0;

    return 0;
}

int Crop::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int packing = bottom_blob.packing;

    // TODO vec and image crop
    int dims = bottom_blob.dims;

    int _outw;
    int _outh;
    int _outc;

    if (outw == -233)
        _outw = w - woffset;
    else if (outw == -234)
        _outw = w - 1 - woffset;
    else
        _outw = std::min(outw, w - woffset);

    if (outh == -233)
        _outh = h - hoffset;
    else if (outh == -234)
        _outh = h - 1 - hoffset;
    else
        _outh = std::min(outh, h - hoffset);

    if (outc == -233)
        _outc = channels * packing - coffset;
    else if (outc == -234)
        _outc = channels * packing - 1 - coffset;
    else
        _outc = std::min(outc, channels * packing - coffset);

    int out_packing = _outc % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / packing * out_packing;

    top_blob.create(_outw, _outh, _outc / out_packing, out_elemsize, out_packing, opt.blob_vkallocator, opt.staging_vkallocator);
    if (top_blob.empty())
        return -100;

//     fprintf(stderr, "Crop::forward %p %p\n", bottom_blob.buffer(), top_blob.buffer());

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(10);
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

    const Pipeline* pipeline = 0;
    if (packing == 1 && out_packing == 1)
    {
        pipeline = pipeline_crop;
    }
    else if (packing == 4 && out_packing == 4)
    {
        pipeline = pipeline_crop_pack4;
    }
    else if (packing == 1 && out_packing == 4)
    {
        // TODO
        return -1;
    }
    else if (packing == 4 && out_packing == 1)
    {
        // TODO
        return -1;
    }

    // record
    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

int Crop::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkMat& bottom_blob = bottom_blobs[0];
    const VkMat& reference_blob = bottom_blobs[1];

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int packing = bottom_blob.packing;

    // TODO vec and image crop
    int dims = bottom_blob.dims;

    int _outw = reference_blob.w;
    int _outh = reference_blob.h;
    int _outc = reference_blob.dims == 3 ? reference_blob.c : channels * packing;

    int out_packing = _outc % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / packing * out_packing;

    VkMat& top_blob = top_blobs[0];

    top_blob.create(_outw, _outh, _outc / out_packing, out_elemsize, out_packing, opt.blob_vkallocator, opt.staging_vkallocator);
    if (top_blob.empty())
        return -100;

//     fprintf(stderr, "Crop::forward %p %p\n", bottom_blob.buffer(), top_blob.buffer());

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(10);
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

    const Pipeline* pipeline = 0;
    if (packing == 1 && out_packing == 1)
    {
        pipeline = pipeline_crop;
    }
    else if (packing == 4 && out_packing == 4)
    {
        pipeline = pipeline_crop_pack4;
    }
    else if (packing == 1 && out_packing == 4)
    {
        // TODO
        return -1;
    }
    else if (packing == 4 && out_packing == 1)
    {
        // TODO
        return -1;
    }

    // record
    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}
#endif // NCNN_VULKAN

} // namespace ncnn
