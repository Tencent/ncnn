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

#include "deconvolutiondepthwise.h"
#include "layer_type.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(DeconvolutionDepthWise)

DeconvolutionDepthWise::DeconvolutionDepthWise()
{
    one_blob_only = true;
    support_inplace = false;
    support_vulkan = true;

#if NCNN_VULKAN
    pipeline_deconvolutiondepthwise = 0;
    pipeline_deconvolutiondepthwise_pack4 = 0;
#endif // NCNN_VULKAN
}

DeconvolutionDepthWise::~DeconvolutionDepthWise()
{
#if NCNN_VULKAN
    for (int i=0; i<(int)deconvolution_group_ops.size(); i++)
        delete deconvolution_group_ops[i];

    deconvolution_group_ops.clear();
#endif // NCNN_VULKAN
}

int DeconvolutionDepthWise::load_param(const ParamDict& pd)
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
    group = pd.get(7, 1);

    return 0;
}

int DeconvolutionDepthWise::load_model(const ModelBin& mb)
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

#if NCNN_VULKAN
    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    // group deconvolution
    if (!(channels == group && group == num_output))
    {
        // create Deconvolution op for each group

        for (int i=0; i<(int)deconvolution_group_ops.size(); i++)
            delete deconvolution_group_ops[i];

        deconvolution_group_ops.clear();

        const int channels_g = channels / group;
        const int num_output_g = num_output / group;

        deconvolution_group_ops.resize(group);

        for (int g=0; g<group; g++)
        {
            Mat weight_data_g = weight_data.range(maxk * channels_g * num_output_g * g, maxk * channels_g * num_output_g);
            Mat bias_data_g;
            if (bias_term)
                bias_data_g = bias_data.range(num_output_g * g, num_output_g);

            ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Deconvolution);
            op->vkdev = vkdev;

            // set param
            ncnn::ParamDict pd;
            pd.set(0, num_output_g);// num_output
            pd.set(1, kernel_w);
            pd.set(11, kernel_h);
            pd.set(2, dilation_w);
            pd.set(12, dilation_h);
            pd.set(3, stride_w);
            pd.set(13, stride_h);
            pd.set(4, 0);// pad_w
            pd.set(14, 0);// pad_h
            pd.set(5, bias_term);
            pd.set(6, maxk * channels_g * num_output_g);// weight_data_size

            pd.use_vulkan_compute = 1;

            op->load_param(pd);

            // set weights
            ncnn::Mat weights[2];
            weights[0] = weight_data_g;
            weights[1] = bias_data_g;

            op->load_model(ModelBinFromMatArray(weights));

            deconvolution_group_ops[g] = op;
        }
    }
#endif // NCNN_VULKAN

    return 0;
}

int DeconvolutionDepthWise::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // deconvolv with NxN kernel
    // value = value + bias

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    if (channels % group != 0 || num_output % group != 0)
    {
        // reject invalid group
        return -100;
    }

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

    // depth-wise
    if (channels == group && group == num_output)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g=0; g<group; g++)
        {
            const float* inptr = bottom_blob.channel(g);
            const float* kptr = (const float*)weight_data + maxk * g;
            Mat m = top_blob_bordered.channel(g);

            const float bias = bias_term ? bias_data[g] : 0.f;

            m.fill(bias);

            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    float* outptr = m.row(i*stride_h) + j*stride_w;

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = inptr[i*w + j];
                        float w = kptr[k];
                        outptr[ space_ofs[k] ] += val * w;
                    }
                }
            }
        }
    }
    else
    {
        // num_output
        const int channels_g = channels / group;
        const int num_output_g = num_output / group;

#ifdef _WIN32
        #pragma omp parallel for num_threads(opt.num_threads)
#else // _WIN32
        #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
#endif // _WIN32
        for (int g = 0; g < group; g++)
        {
            for (int p = 0; p < num_output_g; p++)
            {
                Mat out = top_blob_bordered.channel(g * num_output_g + p);

                const float* weight_data_ptr = (const float*)weight_data + maxk * channels_g * num_output_g * g;
                const float bias = bias_term ? bias_data[g * num_output_g + p] : 0.f;

                out.fill(bias);

                for (int i = 0; i < h; i++)
                {
                    for (int j = 0; j < w; j++)
                    {
                        float* outptr = out.row(i*stride_h) + j*stride_w;

                        const float* kptr = weight_data_ptr + maxk * channels_g * p;

                        // channels_g
                        for (int q = 0; q < channels_g; q++)
                        {
                            const Mat m = bottom_blob.channel(channels_g * g + q);
                            float val = *(m.row(i) + j);

                            for (int k = 0; k < maxk; k++)
                            {
                                outptr[ space_ofs[k] ] += val * kptr[k];
                            }

                            kptr += maxk;
                        }
                    }
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
int DeconvolutionDepthWise::upload_model(VkTransfer& cmd)
{
    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    Mat weight_data_transposed(weight_data.w);
    {
        float* pt = weight_data_transposed;
        const float* p = weight_data;

        for (int i=0; i<(channels/group)*(num_output/group)*group; i++)
        {
            for (int k=0; k<maxk; k++)
            {
                pt[maxk-1 - k] = p[k];
            }

            p += maxk;
            pt += maxk;
        }
    }

    // depth-wise
    if (channels == group && group == num_output)
    {
        cmd.record_upload(weight_data_transposed, weight_data_gpu);

        if (bias_term)
        {
            cmd.record_upload(bias_data, bias_data_gpu);
        }

        // pack4
        if (channels % 4 == 0 && num_output % 4 == 0)
        {
            const int maxk = kernel_w * kernel_h;

            Mat weight_data_pack4;
            Mat weight_data_r2 = weight_data_transposed.reshape(maxk, group);
            convert_packing(weight_data_r2, weight_data_pack4, 4);

            weight_data_pack4 = weight_data_pack4.reshape(maxk * (group/4));
            cmd.record_upload(weight_data_pack4, weight_data_gpu_pack4);

            if (bias_term)
            {
                Mat bias_data_pack4;
                convert_packing(bias_data, bias_data_pack4, 4);
                cmd.record_upload(bias_data_pack4, bias_data_gpu_pack4);
            }
        }

        return 0;
    }

    for (int g=0; g<group; g++)
    {
        deconvolution_group_ops[g]->upload_model(cmd);
    }

    return 0;
}

int DeconvolutionDepthWise::create_pipeline()
{
    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    // depth-wise
    if (channels == group && group == num_output)
    {
        pipeline_deconvolutiondepthwise = new Pipeline(vkdev);
        pipeline_deconvolutiondepthwise->set_optimal_local_size_xyz(32, 32, num_output);

        std::vector<vk_specialization_type> specializations(8);
        specializations[0].i = kernel_w;
        specializations[1].i = kernel_h;
        specializations[2].i = dilation_w;
        specializations[3].i = dilation_h;
        specializations[4].i = stride_w;
        specializations[5].i = stride_h;
        specializations[6].i = bias_term;
        specializations[7].i = group;

        pipeline_deconvolutiondepthwise->create("deconvolutiondepthwise", specializations, 4, 10);

        // pack4
        if (num_output % 4 == 0)
        {
            pipeline_deconvolutiondepthwise_pack4 = new Pipeline(vkdev);
            pipeline_deconvolutiondepthwise_pack4->set_optimal_local_size_xyz(32, 32, std::max(1, num_output / 4));
            pipeline_deconvolutiondepthwise_pack4->create("deconvolutiondepthwise_pack4", specializations, 4, 10);
        }

        return 0;
    }

    for (int g=0; g<group; g++)
    {
        deconvolution_group_ops[g]->create_pipeline();
    }

    return 0;
}

int DeconvolutionDepthWise::destroy_pipeline()
{
    for (int g=0; g<(int)deconvolution_group_ops.size(); g++)
    {
        deconvolution_group_ops[g]->destroy_pipeline();
    }

    delete pipeline_deconvolutiondepthwise;
    pipeline_deconvolutiondepthwise = 0;

    delete pipeline_deconvolutiondepthwise_pack4;
    pipeline_deconvolutiondepthwise_pack4 = 0;

    return 0;
}

int DeconvolutionDepthWise::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
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

    // TODO assert num_output % packing == 0

    top_blob.create(outw, outh, num_output / packing, elemsize, packing, opt.blob_vkallocator, opt.staging_vkallocator);
    if (top_blob.empty())
        return -100;

//     fprintf(stderr, "DeconvolutionDepthWise::forward %p %p\n", bottom_blob.buffer(), top_blob.buffer());

    // depth-wise
    if (channels == group / packing && group / packing == num_output / packing)
    {
        std::vector<VkMat> bindings(4);
        bindings[0] = bottom_blob;
        bindings[1] = top_blob;
        bindings[2] = packing == 4 ? weight_data_gpu_pack4 : weight_data_gpu;
        bindings[3] = bias_term ? (packing == 4 ? bias_data_gpu_pack4 : bias_data_gpu) : weight_data_gpu;// TODO use dummy buffer

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

        const Pipeline* pipeline = packing == 4 ? pipeline_deconvolutiondepthwise_pack4 : pipeline_deconvolutiondepthwise;

        // record
        cmd.record_prepare_compute_barrier(bottom_blob);
        cmd.record_prepare_compute_barrier(top_blob);
        cmd.record_pipeline(pipeline, bindings, constants, top_blob);

        return 0;
    }

    // record
    cmd.record_prepare_compute_barrier(top_blob);

    const int channels_g = channels / group;
    const int num_output_g = num_output / packing / group;

    for (int g=0; g<group; g++)
    {
        VkMat bottom_blob_bordered_g = bottom_blob.channel_range(channels_g * g, channels_g);
        VkMat top_blob_g = top_blob.channel_range(num_output_g * g, num_output_g);

        const ncnn::Layer* op = deconvolution_group_ops[g];

        ncnn::Option opt_g = opt;
        opt_g.blob_vkallocator = top_blob.allocator;
        opt_g.staging_vkallocator = top_blob.staging_allocator;

        // forward
        op->forward(bottom_blob_bordered_g, top_blob_g, cmd, opt_g);
    }

    return 0;
}
#endif // NCNN_VULKAN

} // namespace ncnn
