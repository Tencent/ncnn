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

#include "lrn.h"
#include <math.h>

namespace ncnn {

DEFINE_LAYER_CREATOR(LRN)

LRN::LRN()
{
    one_blob_only = true;
    support_inplace = true;
    support_vulkan = true;

#if NCNN_VULKAN
    pipeline_lrn_square_pad = 0;
    pipeline_lrn_norm = 0;
    pipeline_lrn_square_pad_across_channel_pack4 = 0;
    pipeline_lrn_norm_across_channel_pack4 = 0;
    pipeline_lrn_square_pad_within_channel_pack4 = 0;
    pipeline_lrn_norm_within_channel_pack4 = 0;
#endif // NCNN_VULKAN
}

int LRN::load_param(const ParamDict& pd)
{
    region_type = pd.get(0, 0);
    local_size = pd.get(1, 5);
    alpha = pd.get(2, 1.f);
    beta = pd.get(3, 0.75f);
    bias = pd.get(4, 1.f);

    return 0;
}

int LRN::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    size_t elemsize = bottom_top_blob.elemsize;
    int size = w * h;

    // squared values with local_size padding
    Mat square_blob;
    square_blob.create(w, h, channels, elemsize, opt.workspace_allocator);
    if (square_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q=0; q<channels; q++)
    {
        const float* ptr = bottom_top_blob.channel(q);
        float* outptr = square_blob.channel(q);

        for (int i=0; i<size; i++)
        {
            outptr[i] = ptr[i] * ptr[i];
        }
    }

    if (region_type == NormRegion_ACROSS_CHANNELS)
    {
        Mat square_sum;
        square_sum.create(w, h, channels, elemsize, opt.workspace_allocator);
        if (square_sum.empty())
            return -100;
        square_sum.fill(0.f);

        const float alpha_div_size = alpha / local_size;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            // square sum
            float* ssptr = square_sum.channel(q);
            for (int p=q - local_size / 2; p<=q + local_size / 2; p++)
            {
                if (p < 0 || p >= channels)
                    continue;

                const float* sptr = square_blob.channel(p);
                for (int i=0; i<size; i++)
                {
                    ssptr[i] += sptr[i];
                }
            }

            float* ptr = bottom_top_blob.channel(q);
            for (int i=0; i<size; i++)
            {
                ptr[i] = ptr[i] * pow(bias + alpha_div_size * ssptr[i], -beta);
            }
        }
    }
    else if (region_type == NormRegion_WITHIN_CHANNEL)
    {
        int outw = w;
        int outh = h;

        Mat square_blob_bordered = square_blob;
        int pad = local_size / 2;
        if (pad > 0)
        {
            copy_make_border(square_blob, square_blob_bordered, pad, local_size - pad - 1, pad, local_size - pad - 1, BORDER_CONSTANT, 0.f, opt.workspace_allocator, opt.num_threads);
            if (square_blob_bordered.empty())
                return -100;

            w = square_blob_bordered.w;
            h = square_blob_bordered.h;
        }

        const int maxk = local_size * local_size;

        const float alpha_div_size = alpha / maxk;

        // norm window offsets
        std::vector<int> _space_ofs(maxk);
        int* space_ofs = &_space_ofs[0];
        {
            int p1 = 0;
            int p2 = 0;
            int gap = w - local_size;
            for (int i = 0; i < local_size; i++)
            {
                for (int j = 0; j < local_size; j++)
                {
                    space_ofs[p1] = p2;
                    p1++;
                    p2++;
                }
                p2 += gap;
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            const Mat m = square_blob_bordered.channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const float* sptr = m.row(i) + j;

                    float ss = 0.f;

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = sptr[ space_ofs[k] ];
                        ss += val;
                    }

                    ptr[j] = ptr[j] * pow(bias + alpha_div_size * ss, -beta);
                }

                ptr += outw;
            }
        }
    }

    return 0;
}

#if NCNN_VULKAN
int LRN::create_pipeline()
{
    {
        pipeline_lrn_square_pad = new Pipeline(vkdev);
        pipeline_lrn_square_pad->set_optimal_local_size_xyz();

        std::vector<vk_specialization_type> specializations(3);
        specializations[0].i = region_type;

        int pad = local_size / 2;
        if (pad == 0)
        {
            specializations[1].i = 0;
            specializations[2].i = 0;
        }
        else
        {
            specializations[1].i = pad;
            specializations[2].i = local_size - pad - 1;
        }

        pipeline_lrn_square_pad->create("lrn_square_pad", specializations, 2, 10);

        // pack4
        if (region_type == 0)
        {
            pipeline_lrn_square_pad_across_channel_pack4 = new Pipeline(vkdev);
            pipeline_lrn_square_pad_across_channel_pack4->set_optimal_local_size_xyz();
            pipeline_lrn_square_pad_across_channel_pack4->create("lrn_square_pad_across_channel_pack4", specializations, 2, 10);
        }
        if (region_type == 1)
        {
            pipeline_lrn_square_pad_within_channel_pack4 = new Pipeline(vkdev);
            pipeline_lrn_square_pad_within_channel_pack4->set_optimal_local_size_xyz();
            pipeline_lrn_square_pad_within_channel_pack4->create("lrn_square_pad_within_channel_pack4", specializations, 2, 10);
        }
    }

    {
        pipeline_lrn_norm = new Pipeline(vkdev);
        pipeline_lrn_norm->set_optimal_local_size_xyz();

        std::vector<vk_specialization_type> specializations(5);
        specializations[0].i = region_type;
        specializations[1].i = local_size;
        specializations[2].f = alpha;
        specializations[3].f = beta;
        specializations[4].f = bias;

        pipeline_lrn_norm->create("lrn_norm", specializations, 2, 10);

        // pack4
        if (region_type == 0)
        {
            pipeline_lrn_norm_across_channel_pack4 = new Pipeline(vkdev);
            pipeline_lrn_norm_across_channel_pack4->set_optimal_local_size_xyz();
            pipeline_lrn_norm_across_channel_pack4->create("lrn_norm_across_channel_pack4", specializations, 2, 10);
        }
        if (region_type == 1)
        {
            pipeline_lrn_norm_within_channel_pack4 = new Pipeline(vkdev);
            pipeline_lrn_norm_within_channel_pack4->set_optimal_local_size_xyz();
            pipeline_lrn_norm_within_channel_pack4->create("lrn_norm_within_channel_pack4", specializations, 2, 10);
        }
    }

    return 0;
}

int LRN::destroy_pipeline()
{
    delete pipeline_lrn_square_pad;
    pipeline_lrn_square_pad = 0;

    delete pipeline_lrn_norm;
    pipeline_lrn_norm = 0;

    delete pipeline_lrn_square_pad_across_channel_pack4;
    pipeline_lrn_square_pad_across_channel_pack4 = 0;

    delete pipeline_lrn_norm_across_channel_pack4;
    pipeline_lrn_norm_across_channel_pack4 = 0;

    delete pipeline_lrn_square_pad_within_channel_pack4;
    pipeline_lrn_square_pad_within_channel_pack4 = 0;

    delete pipeline_lrn_norm_within_channel_pack4;
    pipeline_lrn_norm_within_channel_pack4 = 0;

    return 0;
}

int LRN::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    size_t elemsize = bottom_top_blob.elemsize;
    int packing = bottom_top_blob.packing;

    VkMat square_workspace;

    int pad = local_size / 2;
    if (pad == 0)
    {
        square_workspace.create(w, h, channels, elemsize, packing, opt.workspace_vkallocator, opt.staging_vkallocator);
    }
    else if (region_type == NormRegion_ACROSS_CHANNELS)
    {
        // always create scalar square workspace blob for norm across channel
        square_workspace.create(w, h, channels * packing + local_size - 1, 4u, 1, opt.workspace_vkallocator, opt.staging_vkallocator);
    }
    else if (region_type == NormRegion_WITHIN_CHANNEL)
    {
        square_workspace.create(w + local_size - 1, h + local_size - 1, channels, elemsize, packing, opt.workspace_vkallocator, opt.staging_vkallocator);
    }

//     fprintf(stderr, "LRN::forward_inplace %p\n", bottom_top_blob.buffer());

    // square pad
    {
    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_top_blob;
    bindings[1] = square_workspace;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = bottom_top_blob.dims;
    constants[1].i = bottom_top_blob.w;
    constants[2].i = bottom_top_blob.h;
    constants[3].i = bottom_top_blob.c;
    constants[4].i = bottom_top_blob.cstep;
    constants[5].i = square_workspace.dims;
    constants[6].i = square_workspace.w;
    constants[7].i = square_workspace.h;
    constants[8].i = square_workspace.c;
    constants[9].i = square_workspace.cstep;

    const Pipeline* pipeline = 0;
    if (packing == 4)
    {
        if (region_type == 0) pipeline = pipeline_lrn_square_pad_across_channel_pack4;
        if (region_type == 1) pipeline = pipeline_lrn_square_pad_within_channel_pack4;
    }
    else
    {
        pipeline = pipeline_lrn_square_pad;
    }

    // record
    cmd.record_pipeline(pipeline, bindings, constants, square_workspace);
    }

    // norm
    {
    std::vector<VkMat> bindings(2);
    bindings[0] = square_workspace;
    bindings[1] = bottom_top_blob;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = square_workspace.dims;
    constants[1].i = square_workspace.w;
    constants[2].i = square_workspace.h;
    constants[3].i = square_workspace.c;
    constants[4].i = square_workspace.cstep;
    constants[5].i = bottom_top_blob.dims;
    constants[6].i = bottom_top_blob.w;
    constants[7].i = bottom_top_blob.h;
    constants[8].i = bottom_top_blob.c;
    constants[9].i = bottom_top_blob.cstep;

    const Pipeline* pipeline = 0;
    if (packing == 4)
    {
        if (region_type == 0) pipeline = pipeline_lrn_norm_across_channel_pack4;
        if (region_type == 1) pipeline = pipeline_lrn_norm_within_channel_pack4;
    }
    else
    {
        pipeline = pipeline_lrn_norm;
    }

    // record
    cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);
    }

    return 0;
}
#endif // NCNN_VULKAN

} // namespace ncnn
