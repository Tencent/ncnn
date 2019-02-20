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

#include "innerproduct.h"

#include "layer_type.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(InnerProduct)

InnerProduct::InnerProduct()
{
    one_blob_only = true;
    support_inplace = false;
    support_vulkan = true;

#if NCNN_VULKAN
    flatten = 0;

    pipeline_innerproduct = 0;
    pipeline_innerproduct_pack4 = 0;
    pipeline_innerproduct_pack1to4 = 0;
    pipeline_innerproduct_pack4to1 = 0;
#endif // NCNN_VULKAN

    quantize = 0;
    dequantize = 0;
}

InnerProduct::~InnerProduct()
{
#if NCNN_VULKAN
    delete flatten;
#endif // NCNN_VULKAN

    delete quantize;
    delete dequantize;
}

int InnerProduct::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    bias_term = pd.get(1, 0);
    weight_data_size = pd.get(2, 0);
    int8_scale_term = pd.get(8, 0);

    use_int8_inference = pd.use_int8_inference;

    if (int8_scale_term == 0)
        use_int8_inference = false;

#if NCNN_VULKAN
    if (pd.use_vulkan_compute)
    {
        flatten = ncnn::create_layer(ncnn::LayerType::Flatten);
        flatten->vkdev = vkdev;

        ncnn::ParamDict pd;
        pd.use_vulkan_compute = 1;

        flatten->load_param(pd);
    }
#endif // NCNN_VULKAN

    return 0;
}

int InnerProduct::load_model(const ModelBin& mb)
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

    if (int8_scale_term)
    {
        weight_data_int8_scale = mb.load(1, 1)[0];
        bottom_blob_int8_scale = mb.load(1, 1)[0];
    }

    bool weight_data_is_int8 = (weight_data.elemsize == (size_t)1u);
    bool weight_data_is_float32 = (weight_data.elemsize == (size_t)4u);

    if (weight_data_is_int8 && !use_int8_inference)
    {
        fprintf(stderr, "quantized int8 weight loaded but use_int8_inference disabled\n");
        return -1;
    }

    if (use_int8_inference)
    {
        quantize = ncnn::create_layer(ncnn::LayerType::Quantize);
        dequantize = ncnn::create_layer(ncnn::LayerType::Dequantize);
    }

    if (weight_data_is_float32 && use_int8_inference)
    {
        // quantize weight to int8
        ncnn::ParamDict pd;
        pd.set(0, weight_data_int8_scale);// scale

        quantize->load_param(pd);

        Mat int8_weight_data;
        quantize->forward(weight_data, int8_weight_data);

        if (int8_weight_data.empty())
            return -100;

        weight_data = int8_weight_data;
    }

    return 0;
}

int InnerProduct::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int size = w * h;

    top_blob.create(num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (use_int8_inference)
    {
        Mat bottom_blob_int8;
        bottom_blob_int8.create(w, h, channels, (size_t)1u, opt.workspace_allocator);
        if (bottom_blob_int8.empty())
            return -100;

        // quantize, scale and round to nearest
        {
            ncnn::ParamDict pd;
            pd.set(0, bottom_blob_int8_scale);// scale

            quantize->load_param(pd);

            quantize->forward(bottom_blob, bottom_blob_int8, opt);
        }

        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p=0; p<num_output; p++)
        {
            int sum = 0;
            int* out = top_blob;

            // channels
            for (int q=0; q<channels; q++)
            {
                const signed char* w = (const signed char*)weight_data + size * channels * p + size * q;
                const signed char* m = bottom_blob_int8.channel(q);

                for (int i = 0; i < size; i++)
                {
                    sum += m[i] * w[i];
                }
            }

            out[p] = sum;
        }

        // dequantize, reverse scale inplace
        {
            float top_rescale = 1.f / (bottom_blob_int8_scale * weight_data_int8_scale);

            ncnn::ParamDict pd;
            pd.set(0, top_rescale);// scale
            pd.set(1, bias_term);// bias_term
            pd.set(2, num_output);// bias_data_size

            dequantize->load_param(pd);

            ncnn::Mat weights[1];
            weights[0] = bias_data;

            dequantize->load_model(ModelBinFromMatArray(weights));

            dequantize->forward_inplace(top_blob, opt);
        }

        return 0;
    }

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=0; p<num_output; p++)
    {
        float sum = 0.f;

        if (bias_term)
            sum = bias_data[p];

        // channels
        for (int q=0; q<channels; q++)
        {
            const float* w = (const float*)weight_data + size * channels * p + size * q;
            const float* m = bottom_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                sum += m[i] * w[i];
            }
        }

        top_blob[p] = sum;
    }

    return 0;
}

#if NCNN_VULKAN
int InnerProduct::upload_model(VkTransfer& cmd)
{
    cmd.record_upload(weight_data, weight_data_gpu);

    if (bias_term)
    {
        cmd.record_upload(bias_data, bias_data_gpu);
    }

    int num_input = weight_data_size / num_output;

    // pack4
    if (num_input % 4 == 0 && num_output % 4 == 0)
    {
        // src = inch-outch
        // dst = 4a-4b-inch/4a-outch/4b
        Mat weight_data_pack4;
        {
            Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

            weight_data_pack4.create(16, num_input/4, num_output/4);

            for (int q=0; q+3<num_output; q+=4)
            {
                const float* k0 = weight_data_r2.row(q);
                const float* k1 = weight_data_r2.row(q+1);
                const float* k2 = weight_data_r2.row(q+2);
                const float* k3 = weight_data_r2.row(q+3);

                float* g00 = weight_data_pack4.channel(q/4);

                for (int p=0; p+3<num_input; p+=4)
                {
                    g00[0] = k0[0];
                    g00[1] = k0[1];
                    g00[2] = k0[2];
                    g00[3] = k0[3];

                    g00[4] = k1[0];
                    g00[5] = k1[1];
                    g00[6] = k1[2];
                    g00[7] = k1[3];

                    g00[8] = k2[0];
                    g00[9] = k2[1];
                    g00[10] = k2[2];
                    g00[11] = k2[3];

                    g00[12] = k3[0];
                    g00[13] = k3[1];
                    g00[14] = k3[2];
                    g00[15] = k3[3];

                    k0 += 4;
                    k1 += 4;
                    k2 += 4;
                    k3 += 4;
                    g00 += 16;
                }
            }
        }

        weight_data_pack4 = weight_data_pack4.reshape(16 * (num_input/4) * (num_output/4));
        cmd.record_upload(weight_data_pack4, weight_data_gpu_pack4);
    }

    // pack1to4
    if (num_input % 4 != 0 && num_output % 4 == 0)
    {
        // src = inch-outch
        // dst = 4b-inch-outch/4b
        Mat weight_data_pack1to4;
        {
            Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

            weight_data_pack1to4.create(4, num_input, num_output/4);

            for (int q=0; q+3<num_output; q+=4)
            {
                const float* k0 = weight_data_r2.row(q);
                const float* k1 = weight_data_r2.row(q+1);
                const float* k2 = weight_data_r2.row(q+2);
                const float* k3 = weight_data_r2.row(q+3);

                float* g00 = weight_data_pack1to4.channel(q/4);

                for (int p=0; p<num_input; p++)
                {
                    g00[0] = k0[p];
                    g00[1] = k1[p];
                    g00[2] = k2[p];
                    g00[3] = k3[p];

                    g00 += 4;
                }
            }
        }

        weight_data_pack1to4 = weight_data_pack1to4.reshape(4 * num_input * (num_output/4));
        cmd.record_upload(weight_data_pack1to4, weight_data_gpu_pack1to4);
    }

    // pack4to1
    if (num_input % 4 == 0 && num_output % 4 != 0)
    {
        // src = inch-outch
        // dst = 4a-inch/4a-outch
        Mat weight_data_pack4to1;
        {
            Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

            weight_data_pack4to1.create(4, num_input/4, num_output);

            for (int q=0; q<num_output; q++)
            {
                const float* k0 = weight_data_r2.row(q);

                float* g00 = weight_data_pack4to1.channel(q);

                for (int p=0; p+3<num_input; p+=4)
                {
                    g00[0] = k0[0];
                    g00[1] = k0[1];
                    g00[2] = k0[2];
                    g00[3] = k0[3];

                    k0 += 4;
                    g00 += 4;
                }
            }
        }

        weight_data_pack4to1 = weight_data_pack4to1.reshape(4 * (num_input/4) * num_output);
        cmd.record_upload(weight_data_pack4to1, weight_data_gpu_pack4to1);
    }

    if (num_output % 4 == 0)
    {
        if (bias_term)
        {
            Mat bias_data_pack4;
            convert_packing(bias_data, bias_data_pack4, 4);
            cmd.record_upload(bias_data_pack4, bias_data_gpu_pack4);
        }
    }

    return 0;
}

int InnerProduct::create_pipeline()
{
    pipeline_innerproduct = new Pipeline(vkdev);
    pipeline_innerproduct->set_optimal_local_size_xyz(num_output, 1, 1);

    std::vector<vk_specialization_type> specializations(1);
    specializations[0].i = bias_term;

    pipeline_innerproduct->create("innerproduct", specializations, 4, 10);

    flatten->create_pipeline();

    int num_input = weight_data_size / num_output;

    // pack4
    if (num_input % 4 == 0 && num_output % 4 == 0)
    {
        pipeline_innerproduct_pack4 = new Pipeline(vkdev);
        pipeline_innerproduct_pack4->set_optimal_local_size_xyz(num_output / 4, 1, 1);
        pipeline_innerproduct_pack4->create("innerproduct_pack4", specializations, 4, 10);
    }

    // pack1to4
    if (num_input % 4 != 0 && num_output % 4 == 0)
    {
        pipeline_innerproduct_pack1to4 = new Pipeline(vkdev);
        pipeline_innerproduct_pack1to4->set_optimal_local_size_xyz(num_output / 4, 1, 1);
        pipeline_innerproduct_pack1to4->create("innerproduct_pack1to4", specializations, 4, 10);
    }

    // pack4to1
    if (num_input % 4 == 0 && num_output % 4 != 0)
    {
        pipeline_innerproduct_pack4to1 = new Pipeline(vkdev);
        pipeline_innerproduct_pack4to1->set_optimal_local_size_xyz(num_output, 1, 1);
        pipeline_innerproduct_pack4to1->create("innerproduct_pack4to1", specializations, 4, 10);
    }

    return 0;
}

int InnerProduct::destroy_pipeline()
{
    if (flatten)
        flatten->destroy_pipeline();

    delete pipeline_innerproduct;
    pipeline_innerproduct = 0;

    delete pipeline_innerproduct_pack4;
    pipeline_innerproduct_pack4 = 0;

    delete pipeline_innerproduct_pack1to4;
    pipeline_innerproduct_pack1to4 = 0;

    delete pipeline_innerproduct_pack4to1;
    pipeline_innerproduct_pack4to1 = 0;

    return 0;
}

int InnerProduct::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int packing = bottom_blob.packing;

    // flatten
    VkMat bottom_blob_flattened = bottom_blob;
    {
        ncnn::Option opt_flatten = opt;
        opt_flatten.blob_vkallocator = opt.workspace_vkallocator;

        flatten->forward(bottom_blob, bottom_blob_flattened, cmd, opt_flatten);
    }

    int out_packing = num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / packing * out_packing;

    top_blob.create(num_output / out_packing, out_elemsize, out_packing, opt.blob_vkallocator, opt.staging_vkallocator);
    if (top_blob.empty())
        return -100;

//     fprintf(stderr, "InnerProduct::forward %p %p\n", bottom_blob_flattened.buffer(), top_blob.buffer());

    std::vector<VkMat> bindings(4);
    bindings[0] = bottom_blob_flattened;
    bindings[1] = top_blob;
    if (packing == 1 && out_packing == 1)
    {
        bindings[2] = weight_data_gpu;
        bindings[3] = bias_term ? bias_data_gpu : weight_data_gpu;// TODO use dummy buffer
    }
    else if (packing == 4 && out_packing == 4)
    {
        bindings[2] = weight_data_gpu_pack4;
        bindings[3] = bias_term ? bias_data_gpu_pack4 : weight_data_gpu_pack4;// TODO use dummy buffer
    }
    else if (packing == 1 && out_packing == 4)
    {
        bindings[2] = weight_data_gpu_pack1to4;
        bindings[3] = bias_term ? bias_data_gpu_pack4 : weight_data_gpu_pack1to4;// TODO use dummy buffer
    }
    else if (packing == 4 && out_packing == 1)
    {
        bindings[2] = weight_data_gpu_pack4to1;
        bindings[3] = bias_term ? bias_data_gpu : weight_data_gpu_pack4to1;// TODO use dummy buffer
    }

    std::vector<vk_constant_type> constants(10);
    constants[0].i = bottom_blob_flattened.dims;
    constants[1].i = bottom_blob_flattened.w;
    constants[2].i = bottom_blob_flattened.h;
    constants[3].i = bottom_blob_flattened.c;
    constants[4].i = bottom_blob_flattened.cstep;
    constants[5].i = top_blob.dims;
    constants[6].i = top_blob.w;
    constants[7].i = top_blob.h;
    constants[8].i = top_blob.c;
    constants[9].i = top_blob.cstep;

    const Pipeline* pipeline = 0;
    if (packing == 1 && out_packing == 1)
    {
        pipeline = pipeline_innerproduct;
    }
    else if (packing == 4 && out_packing == 4)
    {
        pipeline = pipeline_innerproduct_pack4;
    }
    else if (packing == 1 && out_packing == 4)
    {
        pipeline = pipeline_innerproduct_pack1to4;
    }
    else if (packing == 4 && out_packing == 1)
    {
        pipeline = pipeline_innerproduct_pack4to1;
    }

    // record
    cmd.record_prepare_compute_barrier(bottom_blob_flattened);
    cmd.record_prepare_compute_barrier(top_blob);
    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}
#endif // NCNN_VULKAN

} // namespace ncnn
