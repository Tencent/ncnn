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

#include "slice_vulkan.h"
#include <algorithm>
#include "layer_type.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Slice_vulkan)

Slice_vulkan::Slice_vulkan()
{
    support_vulkan = true;

    packing_pack1 = 0;

    pipeline_slice[0] = 0;
    pipeline_slice[1] = 0;
    pipeline_slice_pack4[0] = 0;
    pipeline_slice_pack4[1] = 0;
    pipeline_slice_pack1to4[0] = 0;
    pipeline_slice_pack1to4[1] = 0;
}

int Slice_vulkan::create_pipeline(const Option& opt)
{
    std::vector<vk_specialization_type> specializations(1);
    specializations[0].i = axis;

    // pack1
    {
        pipeline_slice[0] = new Pipeline(vkdev);
        pipeline_slice[0]->set_optimal_local_size_xyz();
        pipeline_slice[0]->create("slice", opt, specializations, 2, 11);
        pipeline_slice[1] = new Pipeline(vkdev);
        pipeline_slice[1]->set_optimal_local_size_xyz();
        pipeline_slice[1]->create("slice", opt, specializations, 2, 11);
    }

    // pack4
    {
        pipeline_slice_pack4[0] = new Pipeline(vkdev);
        pipeline_slice_pack4[0]->set_optimal_local_size_xyz();
        pipeline_slice_pack4[0]->create("slice_pack4", opt, specializations, 2, 11);
        pipeline_slice_pack4[1] = new Pipeline(vkdev);
        pipeline_slice_pack4[1]->set_optimal_local_size_xyz();
        pipeline_slice_pack4[1]->create("slice_pack4", opt, specializations, 2, 11);
    }

    // pack4to1
    {
        pipeline_slice_pack1to4[0] = new Pipeline(vkdev);
        pipeline_slice_pack1to4[0]->set_optimal_local_size_xyz();
        pipeline_slice_pack1to4[0]->create("slice_pack1to4", opt, specializations, 2, 11);
        pipeline_slice_pack1to4[1] = new Pipeline(vkdev);
        pipeline_slice_pack1to4[1]->set_optimal_local_size_xyz();
        pipeline_slice_pack1to4[1]->create("slice_pack1to4", opt, specializations, 2, 11);
    }

    {
        packing_pack1 = ncnn::create_layer(ncnn::LayerType::Packing);
        packing_pack1->vkdev = vkdev;

        ncnn::ParamDict pd;
        pd.set(0, 1);

        packing_pack1->load_param(pd);

        packing_pack1->create_pipeline(opt);
    }

    return 0;
}

int Slice_vulkan::destroy_pipeline(const Option& opt)
{
    if (packing_pack1)
    {
        packing_pack1->destroy_pipeline(opt);
        delete packing_pack1;
        packing_pack1 = 0;
    }

    delete pipeline_slice[0];
    delete pipeline_slice[1];
    pipeline_slice[0] = 0;
    pipeline_slice[1] = 0;

    delete pipeline_slice_pack4[0];
    delete pipeline_slice_pack4[1];
    pipeline_slice_pack4[0] = 0;
    pipeline_slice_pack4[1] = 0;

    delete pipeline_slice_pack1to4[0];
    delete pipeline_slice_pack1to4[1];
    pipeline_slice_pack1to4[0] = 0;
    pipeline_slice_pack1to4[1] = 0;

    return 0;
}

int Slice_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkMat& bottom_blob = bottom_blobs[0];
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    const int* slices_ptr = slices;

    if (dims == 1) // axis == 0
    {
        // slice vector
        int w = bottom_blob.w * elempack;
        int q = 0;
        for (size_t i=0; i<top_blobs.size(); i++)
        {
            int slice = slices_ptr[i];
            if (slice == -233)
            {
                slice = (w - q) / (top_blobs.size() - i);
            }

            int out_elempack = slice % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            VkMat& top_blob = top_blobs[i];
            top_blob.create(slice / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator, opt.staging_vkallocator);
            if (top_blob.empty())
                return -100;

            q += slice;
        }

        size_t out_elemsize = top_blobs[0].elemsize;
        int out_elempack = top_blobs[0].elempack;
        for (size_t i=0; i<top_blobs.size(); i++)
        {
            out_elemsize = std::min(out_elemsize, top_blobs[i].elemsize);
            out_elempack = std::min(out_elempack, top_blobs[i].elempack);
        }

        VkMat bottom_blob_unpacked = bottom_blob;
        if (elempack == 4 && out_elempack == 1)
        {
            packing_pack1->forward(bottom_blob, bottom_blob_unpacked, cmd, opt);
        }

        int woffset = 0;
        for (size_t i=0; i<top_blobs.size(); i++)
        {
            VkMat& top_blob = top_blobs[i];

            std::vector<VkMat> bindings(2);
            bindings[0] = bottom_blob_unpacked;
            bindings[1] = top_blob;

            std::vector<vk_constant_type> constants(11);
            constants[0].i = bottom_blob_unpacked.dims;
            constants[1].i = bottom_blob_unpacked.w;
            constants[2].i = bottom_blob_unpacked.h;
            constants[3].i = bottom_blob_unpacked.c;
            constants[4].i = bottom_blob_unpacked.cstep;
            constants[5].i = top_blob.dims;
            constants[6].i = top_blob.w;
            constants[7].i = top_blob.h;
            constants[8].i = top_blob.c;
            constants[9].i = top_blob.cstep;
            constants[10].i = woffset;

            const Pipeline* pipeline = 0;
            if (out_elempack == 1 && top_blob.elempack == 1)
            {
                pipeline = pipeline_slice[i%2];
            }
            else if (out_elempack == 4 && top_blob.elempack == 4)
            {
                pipeline = pipeline_slice_pack4[i%2];
            }
            else if (out_elempack == 1 && top_blob.elempack == 4)
            {
                pipeline = pipeline_slice_pack1to4[i%2];
            }

            cmd.record_pipeline(pipeline, bindings, constants, top_blob);

            woffset += top_blob.w * top_blob.elempack / elempack;
        }

        return 0;
    }

    if (dims == 2 && axis == 0)
    {
        // slice image height
        int w = bottom_blob.w;
        int h = bottom_blob.h * elempack;

        int q = 0;
        for (size_t i=0; i<top_blobs.size(); i++)
        {
            int slice = slices_ptr[i];
            if (slice == -233)
            {
                slice = (h - q) / (top_blobs.size() - i);
            }

            int out_elempack = slice % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            VkMat& top_blob = top_blobs[i];
            top_blob.create(w, slice / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator, opt.staging_vkallocator);
            if (top_blob.empty())
                return -100;

            q += slice;
        }

        size_t out_elemsize = top_blobs[0].elemsize;
        int out_elempack = top_blobs[0].elempack;
        for (size_t i=0; i<top_blobs.size(); i++)
        {
            out_elemsize = std::min(out_elemsize, top_blobs[i].elemsize);
            out_elempack = std::min(out_elempack, top_blobs[i].elempack);
        }

        VkMat bottom_blob_unpacked = bottom_blob;
        if (elempack == 4 && out_elempack == 1)
        {
            packing_pack1->forward(bottom_blob, bottom_blob_unpacked, cmd, opt);
        }

        int hoffset = 0;
        for (size_t i=0; i<top_blobs.size(); i++)
        {
            VkMat& top_blob = top_blobs[i];

            std::vector<VkMat> bindings(2);
            bindings[0] = bottom_blob_unpacked;
            bindings[1] = top_blob;

            std::vector<vk_constant_type> constants(11);
            constants[0].i = bottom_blob_unpacked.dims;
            constants[1].i = bottom_blob_unpacked.w;
            constants[2].i = bottom_blob_unpacked.h;
            constants[3].i = bottom_blob_unpacked.c;
            constants[4].i = bottom_blob_unpacked.cstep;
            constants[5].i = top_blob.dims;
            constants[6].i = top_blob.w;
            constants[7].i = top_blob.h;
            constants[8].i = top_blob.c;
            constants[9].i = top_blob.cstep;
            constants[10].i = hoffset;

            const Pipeline* pipeline = 0;
            if (out_elempack == 1 && top_blob.elempack == 1)
            {
                pipeline = pipeline_slice[i%2];
            }
            else if (out_elempack == 4 && top_blob.elempack == 4)
            {
                pipeline = pipeline_slice_pack4[i%2];
            }
            else if (out_elempack == 1 && top_blob.elempack == 4)
            {
                pipeline = pipeline_slice_pack1to4[i%2];
            }

            cmd.record_pipeline(pipeline, bindings, constants, top_blob);

            hoffset += top_blob.w * top_blob.elempack / elempack;
        }

        return 0;
    }

    if (dims == 2 && axis == 1)
    {
        // slice image width
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        int q = 0;
        for (size_t i=0; i<top_blobs.size(); i++)
        {
            int slice = slices_ptr[i];
            if (slice == -233)
            {
                slice = (w - q) / (top_blobs.size() - i);
            }

            VkMat& top_blob = top_blobs[i];
            top_blob.create(slice, h, elemsize, elempack, opt.blob_vkallocator, opt.staging_vkallocator);
            if (top_blob.empty())
                return -100;

            q += slice;
        }

        int woffset = 0;
        for (size_t i=0; i<top_blobs.size(); i++)
        {
            VkMat& top_blob = top_blobs[i];

            std::vector<VkMat> bindings(2);
            bindings[0] = bottom_blob;
            bindings[1] = top_blob;

            std::vector<vk_constant_type> constants(11);
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
            constants[10].i = woffset;

            const Pipeline* pipeline = elempack == 4 ? pipeline_slice_pack4[i%2] : pipeline_slice[i%2];

            cmd.record_pipeline(pipeline, bindings, constants, top_blob);

            woffset += top_blob.w;
        }

        return 0;
    }

    if (dims == 3 && axis == 0)
    {
        // slice dim channel
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c * elempack;

        int q = 0;
        for (size_t i=0; i<top_blobs.size(); i++)
        {
            int slice = slices_ptr[i];
            if (slice == -233)
            {
                slice = (channels - q) / (top_blobs.size() - i);
            }

            int out_elempack = slice % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            VkMat& top_blob = top_blobs[i];
            top_blob.create(w, h, slice / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator, opt.staging_vkallocator);
            if (top_blob.empty())
                return -100;

            q += slice;
        }

        size_t out_elemsize = top_blobs[0].elemsize;
        int out_elempack = top_blobs[0].elempack;
        for (size_t i=0; i<top_blobs.size(); i++)
        {
            out_elemsize = std::min(out_elemsize, top_blobs[i].elemsize);
            out_elempack = std::min(out_elempack, top_blobs[i].elempack);
        }

        VkMat bottom_blob_unpacked = bottom_blob;
        if (elempack == 4 && out_elempack == 1)
        {
            packing_pack1->forward(bottom_blob, bottom_blob_unpacked, cmd, opt);
        }

        int coffset = 0;
        for (size_t i=0; i<top_blobs.size(); i++)
        {
            VkMat& top_blob = top_blobs[i];

            std::vector<VkMat> bindings(2);
            bindings[0] = bottom_blob_unpacked;
            bindings[1] = top_blob;

            std::vector<vk_constant_type> constants(11);
            constants[0].i = bottom_blob_unpacked.dims;
            constants[1].i = bottom_blob_unpacked.w;
            constants[2].i = bottom_blob_unpacked.h;
            constants[3].i = bottom_blob_unpacked.c;
            constants[4].i = bottom_blob_unpacked.cstep;
            constants[5].i = top_blob.dims;
            constants[6].i = top_blob.w;
            constants[7].i = top_blob.h;
            constants[8].i = top_blob.c;
            constants[9].i = top_blob.cstep;
            constants[10].i = coffset;

            const Pipeline* pipeline = 0;
            if (out_elempack == 1 && top_blob.elempack == 1)
            {
                pipeline = pipeline_slice[i%2];
            }
            else if (out_elempack == 4 && top_blob.elempack == 4)
            {
                pipeline = pipeline_slice_pack4[i%2];
            }
            else if (out_elempack == 1 && top_blob.elempack == 4)
            {
                pipeline = pipeline_slice_pack1to4[i%2];
            }

            cmd.record_pipeline(pipeline, bindings, constants, top_blob);

            coffset += top_blob.c * top_blob.elempack / elempack;
        }

        return 0;
    }

    if (dims == 3 && axis == 1)
    {
        // slice dim height
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int q = 0;
        for (size_t i=0; i<top_blobs.size(); i++)
        {
            int slice = slices_ptr[i];
            if (slice == -233)
            {
                slice = (h - q) / (top_blobs.size() - i);
            }

            VkMat& top_blob = top_blobs[i];
            top_blob.create(w, slice, channels, elemsize, elempack, opt.blob_vkallocator, opt.staging_vkallocator);
            if (top_blob.empty())
                return -100;

            q += slice;
        }

        int hoffset = 0;
        for (size_t i=0; i<top_blobs.size(); i++)
        {
            VkMat& top_blob = top_blobs[i];

            std::vector<VkMat> bindings(2);
            bindings[0] = bottom_blob;
            bindings[1] = top_blob;

            std::vector<vk_constant_type> constants(11);
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
            constants[10].i = hoffset;

            const Pipeline* pipeline = elempack == 4 ? pipeline_slice_pack4[i%2] : pipeline_slice[i%2];

            cmd.record_pipeline(pipeline, bindings, constants, top_blob);

            hoffset += top_blob.h;
        }

        return 0;
    }

    if (dims == 3 && axis == 2)
    {
        // slice dim width
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int q = 0;
        for (size_t i=0; i<top_blobs.size(); i++)
        {
            int slice = slices_ptr[i];
            if (slice == -233)
            {
                slice = (w - q) / (top_blobs.size() - i);
            }

            VkMat& top_blob = top_blobs[i];
            top_blob.create(slice, h, channels, elemsize, elempack, opt.blob_vkallocator, opt.staging_vkallocator);
            if (top_blob.empty())
                return -100;

            q += slice;
        }

        int woffset = 0;
        for (size_t i=0; i<top_blobs.size(); i++)
        {
            VkMat& top_blob = top_blobs[i];

            std::vector<VkMat> bindings(2);
            bindings[0] = bottom_blob;
            bindings[1] = top_blob;

            std::vector<vk_constant_type> constants(11);
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
            constants[10].i = woffset;

            const Pipeline* pipeline = elempack == 4 ? pipeline_slice_pack4[i%2] : pipeline_slice[i%2];

            cmd.record_pipeline(pipeline, bindings, constants, top_blob);

            woffset += top_blob.w;
        }

        return 0;
    }

    return 0;
}

} // namespace ncnn
