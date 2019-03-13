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

#include "deconvolution.h"
#include <algorithm>
#include "layer_type.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Deconvolution)

Deconvolution::Deconvolution()
{
    one_blob_only = true;
    support_inplace = false;
    support_vulkan = true;

#if NCNN_VULKAN
    crop = 0;
    pipeline_deconvolution = 0;
    pipeline_deconvolution_pack4 = 0;
    pipeline_deconvolution_pack1to4 = 0;
    pipeline_deconvolution_pack4to1 = 0;
#endif // NCNN_VULKAN
}

Deconvolution::~Deconvolution()
{
#if NCNN_VULKAN
    delete crop;
#endif // NCNN_VULKAN
}

int Deconvolution::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    dilation_w = pd.get(2, 1);
    dilation_h = pd.get(12, dilation_w);
    stride_w = pd.get(3, 1);
    stride_h = pd.get(13, stride_w);
    pad_w = pd.get(4, 0);
    pad_h = pd.get(14, pad_w);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);

#if NCNN_VULKAN
    if (pd.use_vulkan_compute)
    {
        {
        crop = ncnn::create_layer(ncnn::LayerType::Crop);
        crop->vkdev = vkdev;

        ncnn::ParamDict pd;
        pd.set(0, pad_w);
        pd.set(1, pad_h);
        pd.set(2, 0);

        pd.use_vulkan_compute = 1;

        crop->load_param(pd);
        }
    }
#endif // NCNN_VULKAN

    return 0;
}

int Deconvolution::load_model(const ModelBin& mb)
{
    weight_data = mb.load(weight_data_size, 0);
    if (weight_data.empty())
        return -100;

    if (bias_term)
    {
        bias_data = mb.load(num_output, 1);
        if (bias_data.empty())
            return -100;
    }

    return 0;
}

int Deconvolution::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // backward strided convolv with NxN kernel
    // value = value + bias

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

//     fprintf(stderr, "Deconvolution input %d x %d  pad = %d %d  ksize=%d %d  stride=%d %d\n", w, h, pad_w, pad_h, kernel_w, kernel_h, stride_w, stride_h);

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - 1) * stride_w + kernel_extent_w;
    int outh = (h - 1) * stride_h + kernel_extent_h;

    Mat top_blob_bordered;
    if (pad_w > 0 || pad_h > 0)
    {
        top_blob_bordered.create(outw, outh, num_output, elemsize, opt.workspace_allocator);
        if (top_blob_bordered.empty())
            return -100;
    }
    else
    {
        top_blob_bordered = top_blob;
        top_blob_bordered.create(outw, outh, num_output, elemsize, opt.blob_allocator);
        if (top_blob_bordered.empty())
            return -100;
    }

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = outw * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=0; p<num_output; p++)
    {
        Mat out = top_blob_bordered.channel(p);

        const float bias = bias_term ? bias_data[p] : 0.f;

        out.fill(bias);

        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < w; j++)
            {
                float* outptr = out.row(i*stride_h) + j*stride_w;

                const float* kptr = (const float*)weight_data + maxk * channels * p;

                // channels
                for (int q=0; q<channels; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    float val = *(m.row(i) + j);

                    for (int k = 0; k < maxk; k++)
                    {
                        float w = kptr[k];
                        outptr[ space_ofs[k] ] += val * w;
                    }

                    kptr += maxk;
                }
            }
        }
    }

    if (pad_w > 0 || pad_h > 0)
    {
        copy_cut_border(top_blob_bordered, top_blob, pad_h, pad_h, pad_w, pad_w, opt.blob_allocator, opt.num_threads);
        if (top_blob.empty())
            return -100;

        outw = top_blob.w;
        outh = top_blob.h;
    }
    else
    {
        top_blob = top_blob_bordered;
    }

    return 0;
}

#if NCNN_VULKAN
int Deconvolution::upload_model(VkTransfer& cmd)
{
    const int maxk = kernel_w * kernel_h;
    int num_input = weight_data_size / maxk / num_output;

    Mat weight_data_transposed(weight_data.w);
    {
        float* pt = weight_data_transposed;
        const float* p = weight_data;

        for (int i=0; i<num_input*num_output; i++)
        {
            for (int k=0; k<maxk; k++)
            {
                pt[maxk-1 - k] = p[k];
            }

            p += maxk;
            pt += maxk;
        }
    }

    // pack1
    if (num_input % 4 != 0 && num_output % 4 != 0)
    {
        cmd.record_upload(weight_data_transposed, weight_data_gpu);
    }

    // pack4
    if (num_input % 4 == 0 && num_output % 4 == 0)
    {
        // src = kw-kh-inch-outch
        // dst = 4a-4b-kw-kh-inch/4a-outch/4b
        Mat weight_data_pack4;
        {
            Mat weight_data_r2 = weight_data_transposed.reshape(maxk, num_input, num_output);

            weight_data_pack4.create(16*maxk, num_input/4, num_output/4);

            for (int q=0; q+3<num_output; q+=4)
            {
                const Mat k0 = weight_data_r2.channel(q);
                const Mat k1 = weight_data_r2.channel(q+1);
                const Mat k2 = weight_data_r2.channel(q+2);
                const Mat k3 = weight_data_r2.channel(q+3);

                Mat g0 = weight_data_pack4.channel(q/4);

                for (int p=0; p+3<num_input; p+=4)
                {
                    const float* k00 = k0.row(p);
                    const float* k01 = k0.row(p+1);
                    const float* k02 = k0.row(p+2);
                    const float* k03 = k0.row(p+3);

                    const float* k10 = k1.row(p);
                    const float* k11 = k1.row(p+1);
                    const float* k12 = k1.row(p+2);
                    const float* k13 = k1.row(p+3);

                    const float* k20 = k2.row(p);
                    const float* k21 = k2.row(p+1);
                    const float* k22 = k2.row(p+2);
                    const float* k23 = k2.row(p+3);

                    const float* k30 = k3.row(p);
                    const float* k31 = k3.row(p+1);
                    const float* k32 = k3.row(p+2);
                    const float* k33 = k3.row(p+3);

                    float* g00 = g0.row(p/4);

                    for (int k=0; k<maxk; k++)
                    {
                        g00[0] = k00[k];
                        g00[1] = k01[k];
                        g00[2] = k02[k];
                        g00[3] = k03[k];

                        g00[4] = k10[k];
                        g00[5] = k11[k];
                        g00[6] = k12[k];
                        g00[7] = k13[k];

                        g00[8] = k20[k];
                        g00[9] = k21[k];
                        g00[10] = k22[k];
                        g00[11] = k23[k];

                        g00[12] = k30[k];
                        g00[13] = k31[k];
                        g00[14] = k32[k];
                        g00[15] = k33[k];

                        g00 += 16;
                    }
                }
            }
        }

        weight_data_pack4 = weight_data_pack4.reshape(16*maxk * (num_input/4) * (num_output/4));
        cmd.record_upload(weight_data_pack4, weight_data_gpu_pack4);
    }

    // pack1to4
    if (num_input % 4 != 0 && num_output % 4 == 0)
    {
        // src = kw-kh-inch-outch
        // dst = 4b-kw-kh-inch-outch/4b
        Mat weight_data_pack1to4;
        {
            Mat weight_data_r2 = weight_data_transposed.reshape(maxk, num_input, num_output);

            weight_data_pack1to4.create(4*maxk, num_input, num_output/4);

            for (int q=0; q+3<num_output; q+=4)
            {
                const Mat k0 = weight_data_r2.channel(q);
                const Mat k1 = weight_data_r2.channel(q+1);
                const Mat k2 = weight_data_r2.channel(q+2);
                const Mat k3 = weight_data_r2.channel(q+3);

                Mat g0 = weight_data_pack1to4.channel(q/4);

                for (int p=0; p<num_input; p++)
                {
                    const float* k00 = k0.row(p);
                    const float* k10 = k1.row(p);
                    const float* k20 = k2.row(p);
                    const float* k30 = k3.row(p);

                    float* g00 = g0.row(p);

                    for (int k=0; k<maxk; k++)
                    {
                        g00[0] = k00[k];
                        g00[1] = k10[k];
                        g00[2] = k20[k];
                        g00[3] = k30[k];

                        g00 += 4;
                    }
                }
            }
        }

        weight_data_pack1to4 = weight_data_pack1to4.reshape(4*maxk * num_input * (num_output/4));
        cmd.record_upload(weight_data_pack1to4, weight_data_gpu_pack1to4);
    }

    // pack4to1
    if (num_input % 4 == 0 && num_output % 4 != 0)
    {
        // src = kw-kh-inch-outch
        // dst = 4a-kw-kh-inch/4a-outch
        Mat weight_data_pack4to1;
        {
            Mat weight_data_r2 = weight_data_transposed.reshape(maxk, num_input, num_output);

            weight_data_pack4to1.create(4*maxk, num_input/4, num_output);

            for (int q=0; q<num_output; q++)
            {
                const Mat k0 = weight_data_r2.channel(q);
                Mat g0 = weight_data_pack4to1.channel(q);

                for (int p=0; p+3<num_input; p+=4)
                {
                    const float* k00 = k0.row(p);
                    const float* k01 = k0.row(p+1);
                    const float* k02 = k0.row(p+2);
                    const float* k03 = k0.row(p+3);

                    float* g00 = g0.row(p/4);

                    for (int k=0; k<maxk; k++)
                    {
                        g00[0] = k00[k];
                        g00[1] = k01[k];
                        g00[2] = k02[k];
                        g00[3] = k03[k];

                        g00 += 4;
                    }
                }
            }
        }

        weight_data_pack4to1 = weight_data_pack4to1.reshape(4*maxk * (num_input/4) * num_output);
        cmd.record_upload(weight_data_pack4to1, weight_data_gpu_pack4to1);
    }

    if (bias_term)
    {
        if (num_output % 4 != 0)
        {
            cmd.record_upload(bias_data, bias_data_gpu);
        }

        if (num_output % 4 == 0)
        {
            Mat bias_data_pack4;
            convert_packing(bias_data, bias_data_pack4, 4);
            cmd.record_upload(bias_data_pack4, bias_data_gpu_pack4);
        }
    }

    return 0;
}

int Deconvolution::create_pipeline()
{
    crop->create_pipeline();

    const int maxk = kernel_w * kernel_h;
    int num_input = weight_data_size / maxk / num_output;

    std::vector<vk_specialization_type> specializations(7);
    specializations[0].i = kernel_w;
    specializations[1].i = kernel_h;
    specializations[2].i = dilation_w;
    specializations[3].i = dilation_h;
    specializations[4].i = stride_w;
    specializations[5].i = stride_h;
    specializations[6].i = bias_term;

    // pack1
    if (num_input % 4 != 0 && num_output % 4 != 0)
    {
        pipeline_deconvolution = new Pipeline(vkdev);
        pipeline_deconvolution->set_optimal_local_size_xyz(32, 32, std::max(1, num_output / 8));
        pipeline_deconvolution->create("deconvolution", specializations, 4, 10);
    }

    // pack4
    if (num_input % 4 == 0 && num_output % 4 == 0)
    {
        pipeline_deconvolution_pack4 = new Pipeline(vkdev);
        pipeline_deconvolution_pack4->set_optimal_local_size_xyz(32, 32, std::max(1, num_output / 8));
        pipeline_deconvolution_pack4->create("deconvolution_pack4", specializations, 4, 10);
    }

    // pack1to4
    if (num_input % 4 != 0 && num_output % 4 == 0)
    {
        pipeline_deconvolution_pack1to4 = new Pipeline(vkdev);
        pipeline_deconvolution_pack1to4->set_optimal_local_size_xyz(32, 32, std::max(1, num_output / 8));
        pipeline_deconvolution_pack1to4->create("deconvolution_pack1to4", specializations, 4, 10);
    }

    // pack4to1
    if (num_input % 4 == 0 && num_output % 4 != 0)
    {
        pipeline_deconvolution_pack4to1 = new Pipeline(vkdev);
        pipeline_deconvolution_pack4to1->set_optimal_local_size_xyz(32, 32, std::max(1, num_output / 8));
        pipeline_deconvolution_pack4to1->create("deconvolution_pack4to1", specializations, 4, 10);
    }

    return 0;
}

int Deconvolution::destroy_pipeline()
{
    if (crop)
        crop->destroy_pipeline();

    delete pipeline_deconvolution;
    pipeline_deconvolution = 0;

    delete pipeline_deconvolution_pack4;
    pipeline_deconvolution_pack4 = 0;

    delete pipeline_deconvolution_pack1to4;
    pipeline_deconvolution_pack1to4 = 0;

    delete pipeline_deconvolution_pack4to1;
    pipeline_deconvolution_pack4to1 = 0;

    return 0;
}

int Deconvolution::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int packing = bottom_blob.packing;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - 1) * stride_w + kernel_extent_w;
    int outh = (h - 1) * stride_h + kernel_extent_h;
    int out_packing = num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / packing * out_packing;

    VkMat top_blob_bordered;
    if (pad_w > 0 || pad_h > 0)
    {
        top_blob_bordered.create(outw, outh, num_output / out_packing, out_elemsize, out_packing, opt.workspace_vkallocator, opt.staging_vkallocator);
        if (top_blob_bordered.empty())
            return -100;
    }
    else
    {
        top_blob_bordered.create(outw, outh, num_output / out_packing, out_elemsize, out_packing, opt.blob_vkallocator, opt.staging_vkallocator);
        if (top_blob_bordered.empty())
            return -100;
    }

//     fprintf(stderr, "Deconvolution::forward %p %p\n", bottom_blob.buffer(), top_blob.buffer());

    std::vector<VkMat> bindings(4);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob_bordered;
    if (packing == 1 && out_packing == 1)
    {
        bindings[2] = weight_data_gpu;
        bindings[3] = bias_term ? bias_data_gpu : bindings[2];// TODO use dummy buffer
    }
    else if (packing == 4 && out_packing == 4)
    {
        bindings[2] = weight_data_gpu_pack4;
        bindings[3] = bias_term ? bias_data_gpu_pack4 : bindings[2];// TODO use dummy buffer
    }
    else if (packing == 1 && out_packing == 4)
    {
        bindings[2] = weight_data_gpu_pack1to4;
        bindings[3] = bias_term ? bias_data_gpu_pack4 : bindings[2];// TODO use dummy buffer
    }
    else if (packing == 4 && out_packing == 1)
    {
        bindings[2] = weight_data_gpu_pack4to1;
        bindings[3] = bias_term ? bias_data_gpu : bindings[2];// TODO use dummy buffer
    }

    std::vector<vk_constant_type> constants(10);
    constants[0].i = bottom_blob.dims;
    constants[1].i = bottom_blob.w;
    constants[2].i = bottom_blob.h;
    constants[3].i = bottom_blob.c;
    constants[4].i = bottom_blob.cstep;
    constants[5].i = top_blob_bordered.dims;
    constants[6].i = top_blob_bordered.w;
    constants[7].i = top_blob_bordered.h;
    constants[8].i = top_blob_bordered.c;
    constants[9].i = top_blob_bordered.cstep;

    const Pipeline* pipeline = 0;
    if (packing == 1 && out_packing == 1)
    {
        pipeline = pipeline_deconvolution;
    }
    else if (packing == 4 && out_packing == 4)
    {
        pipeline = pipeline_deconvolution_pack4;
    }
    else if (packing == 1 && out_packing == 4)
    {
        pipeline = pipeline_deconvolution_pack1to4;
    }
    else if (packing == 4 && out_packing == 1)
    {
        pipeline = pipeline_deconvolution_pack4to1;
    }

    // record
    cmd.record_pipeline(pipeline, bindings, constants, top_blob_bordered);

    if (pad_w > 0 || pad_h > 0)
    {
        VkMat reference_blob;
        reference_blob.dims = 2;
        reference_blob.w = top_blob_bordered.w - pad_w - pad_w;
        reference_blob.h = top_blob_bordered.h - pad_h - pad_h;

        std::vector<VkMat> crop_bottom_blobs(2);
        crop_bottom_blobs[0] = top_blob_bordered;
        crop_bottom_blobs[1] = reference_blob;
        std::vector<VkMat> crop_top_blobs(1);
        crop->forward(crop_bottom_blobs, crop_top_blobs, cmd, opt);
        top_blob = crop_top_blobs[0];

        outw = top_blob.w;
        outh = top_blob.h;
    }
    else
    {
        top_blob = top_blob_bordered;
    }

    return 0;
}
#endif // NCNN_VULKAN

} // namespace ncnn
