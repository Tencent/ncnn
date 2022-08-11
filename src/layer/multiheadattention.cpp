// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "multiheadattention.h"

#include <float.h>
#ifdef NCNN_INT8
#include <math.h>
#include "mathfun.h"
#endif

namespace ncnn {

MultiHeadAttention::MultiHeadAttention()
{
}

int MultiHeadAttention::create_pipeline(const Option& opt)
{
    // runtime quantize the weight data
    int ret = 0;
#if NCNN_INT8
    if (opt.use_int8_inference && int8_scale_term)
    {
        if (q_weight_data.elemsize == (size_t)4u)
        {
            ret += quantize_weight(q_weight_data, q_weight_scales, opt);
        }
        if (k_weight_data.elemsize == (size_t)4u)
        {
            ret += quantize_weight(k_weight_data, k_weight_scales, opt);
        }
        if (v_weight_data.elemsize == (size_t)4u)
        {
            ret += quantize_weight(v_weight_data, v_weight_scales, opt);
        }
        if (out_weight_data.elemsize == (size_t)4u)
        {
            ret += quantize_weight(out_weight_data, o_weight_scales, opt);
        }
    }
#else
    (void)(opt);
#endif // NCNN_INT8
    return ret;
}

int MultiHeadAttention::load_param(const ParamDict& pd)
{
    embed_dim = pd.get(0, 0);
    num_head = pd.get(1, 1);
    weight_data_size = pd.get(2, 0);
    int8_scale_term = pd.get(3, 0);

    embed_dim_per_head = embed_dim / num_head;
    inv_sqrt_embed_dim_per_head = 1.f / sqrt(embed_dim_per_head);

    if (int8_scale_term)
    {
#if NCNN_INT8
        support_int8_storage = true;
#else
        NCNN_LOGE("please build ncnn with NCNN_INT8 enabled for int8 inference");
        return -1;
#endif
    }
    return 0;
}

int MultiHeadAttention::load_model(const ModelBin& mb)
{
#define LOAD_MAT(name, len) \
    name = mb.load(len, 0); \
    if (name.empty())       \
    {                       \
        return -100;        \
    }

#define LOAD_FLOAT_MAT(name, len) \
    name = mb.load(len, 1);       \
    if (name.empty())             \
    {                             \
        return -100;              \
    }

    LOAD_MAT(q_weight_data, weight_data_size);
    LOAD_FLOAT_MAT(q_bias_data, embed_dim);

    LOAD_MAT(k_weight_data, weight_data_size);
    LOAD_FLOAT_MAT(k_bias_data, embed_dim);

    LOAD_MAT(v_weight_data, weight_data_size);
    LOAD_FLOAT_MAT(v_bias_data, embed_dim);

    LOAD_MAT(out_weight_data, weight_data_size);
    LOAD_FLOAT_MAT(out_bias_data, embed_dim);

#if NCNN_INT8
    if (int8_scale_term)
    {
        LOAD_FLOAT_MAT(q_input_scale, 1);
        LOAD_FLOAT_MAT(k_input_scale, 1);
        LOAD_FLOAT_MAT(v_input_scale, 1);

        LOAD_FLOAT_MAT(q_weight_scales, embed_dim);
        LOAD_FLOAT_MAT(k_weight_scales, embed_dim);
        LOAD_FLOAT_MAT(v_weight_scales, embed_dim);
        LOAD_FLOAT_MAT(o_weight_scales, embed_dim);

        LOAD_FLOAT_MAT(internal_scales, 5);
    }
#endif // NCNN_INT8

#undef LOAD_MAT
#undef LOAD_FLOAT_MAT

    return 0;
}

#ifdef NCNN_INT8

int MultiHeadAttention::quantize_weight(Mat& weight_data, const Mat& weight_data_int8_scales, const Option& opt)
{
    const int num_output = embed_dim;
    const int num_input = weight_data_size / num_output;

    Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

    Mat weight_data_int8;
    Option opt_q = opt;
    opt_q.use_packing_layout = false;
    quantize_to_int8(weight_data_r2, weight_data_int8, weight_data_int8_scales, opt_q);
    if (weight_data_int8.empty())
        return -100;

    weight_data = weight_data_int8.reshape(weight_data_size);
    return 0;
}

int MultiHeadAttention::affine_input(
    const Mat& input, const Mat& weight, const Mat& bias, Mat& out_int8,
    const Mat& input_scale, const Mat& weight_scales, const float transform_scale,
    const int num_head, const Option& opt, bool transpose) const
{
    const int embed_dim = input.w;
    const int seqlen = input.h;
    const int embed_dim_per_head = embed_dim / num_head;
    const float scale = 1.0 / input_scale[0];

    Mat input_int8;
    if (input.elemsize != 1)
    {
        quantize_to_int8(input, input_int8, input_scale, opt);
    }
    else
    {
        input_int8 = input;
    }

    Mat buffer(out_int8.w, out_int8.h, out_int8.c, 4u, opt.workspace_allocator);

    if (transpose)
    {
        for (int q = 0; q < num_head; q++)
        {
            Mat outm = buffer.channel(q);

            for (int i = 0; i < embed_dim_per_head; i++)
            {
                for (int j = 0; j < seqlen; j++)
                {
                    const int8_t* ptr = input_int8.row<int8_t>(j);
                    const int8_t* kptr = (int8_t*)(weight.data) + embed_dim * (q * embed_dim_per_head + i);

                    int32_t sum = 0;
                    const int32_t index = q * embed_dim_per_head + i;
                    for (int k = 0; k < embed_dim; k++)
                    {
                        sum += *ptr++ * *kptr++;
                    }

                    float* outptr = outm.row(i);
                    outptr[j] = (float)sum * scale / weight_scales[index] + bias[index];
                }
            }
        }
    }
    else
    {
        for (int q = 0; q < num_head; q++)
        {
            Mat outm = buffer.channel(q);

            for (int i = 0; i < seqlen; i++)
            {
                float* outptr = outm.row(i);

                for (int j = 0; j < embed_dim_per_head; j++)
                {
                    const int8_t* ptr = input_int8.row<int8_t>(i);
                    const int8_t* kptr = (int8_t*)(weight.data) + embed_dim * (q * embed_dim_per_head + j);

                    int32_t sum = 0;
                    const int32_t index = q * embed_dim_per_head + j;
                    for (int k = 0; k < embed_dim; k++)
                    {
                        sum += *ptr++ * *kptr++;
                    }

                    outptr[j] = (float)sum * scale / weight_scales[index] + bias[index];
                }
            }
        }
    }

    Mat transform(1, 4u, opt.workspace_allocator);
    transform[0] = transform_scale;
    quantize_to_int8(buffer, out_int8, transform, opt);
    return 0;
}

int MultiHeadAttention::forward_int8(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& q_blob = bottom_blobs[0];
    const Mat& k_blob = bottom_blobs.size() == 1 ? q_blob : bottom_blobs[1];
    const Mat& v_blob = bottom_blobs.size() == 1 ? q_blob : bottom_blobs[2];

    const int seqlen = q_blob.h;

    Option opt_g = opt;
    opt_g.blob_allocator = opt.workspace_allocator;
    opt_g.use_packing_layout = false;

    Mat xq(embed_dim_per_head, seqlen, num_head, 1u, opt.workspace_allocator);
    Mat xk(embed_dim_per_head, seqlen, num_head, 1u, opt.workspace_allocator);
    Mat xv(seqlen, embed_dim_per_head, num_head, 1u, opt.workspace_allocator);

    affine_input(q_blob, q_weight_data, q_bias_data, xq, q_input_scale, q_weight_scales, internal_scales[0], num_head, opt_g, false);
    affine_input(k_blob, k_weight_data, k_bias_data, xk, k_input_scale, k_weight_scales, internal_scales[1], num_head, opt_g, false);
    affine_input(v_blob, v_weight_data, v_bias_data, xv, v_input_scale, v_weight_scales, internal_scales[2], num_head, opt_g, true);

    // xq @ qk * inv_sqrt_embed_dim_per_head
    Mat xqk(seqlen, seqlen, num_head, 4u, opt.workspace_allocator);
    {
        // xqk = xq * xk
        // xq  (embed_dim_per_head, seqlen)
        // xk  (embed_dim_per_head, seqlen)
        const float out_scale = inv_sqrt_embed_dim_per_head / (internal_scales[0] * internal_scales[1]);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_head; ++q)
        {
            const Mat xqm = xq.channel(q);
            const Mat xkm = xk.channel(q);

            Mat outm = xqk.channel(q);

            for (int i = 0; i < seqlen; i++)
            {
                float* outptr = outm.row<float>(i);

                for (int j = 0; j < seqlen; j++)
                {
                    const int8_t* qptr = xqm.row<int8_t>(i);
                    const int8_t* kptr = xkm.row<int8_t>(j);

                    int32_t sum = 0;
                    for (int k = 0; k < embed_dim_per_head; k++)
                    {
                        sum += *qptr++ * *kptr++;
                    }

                    outptr[j] = sum * out_scale;
                }
            }
        }

        // fp32_softmax(xqk)
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_head; q++)
        {
            // softmax(xqk)
            {
                Mat outm = xqk.channel(q);

                for (int i = 0; i < seqlen; i++)
                {
                    float* ptr = outm.row(i);

                    float max = -FLT_MAX;
                    for (int j = 0; j < seqlen; j++)
                    {
                        max = std::max(max, ptr[j]);
                    }

                    float sum = 0.f;
                    for (int j = 0; j < seqlen; j++)
                    {
                        ptr[j] = (float)(exp(ptr[j] - max));
                        sum += ptr[j];
                    }

                    for (int j = 0; j < seqlen; j++)
                    {
                        ptr[j] = ptr[j] / sum;
                    }
                }
            }
        }
    }

    // xqkv int4 @ int8, implement by shift
    Mat xqkv(embed_dim_per_head, num_head, seqlen, 1u, opt.workspace_allocator);

    const float xqkv_out_scale = internal_scales[4] / internal_scales[2];
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_head; ++q)
    {
        // xqkv = xqk * xv
        // xqk (seqlen, seqlen)
        // xv  (seqlen, embed_dim_per_head)
        // out (embed_dim_per_head, num_head, seqlen)
        const Mat xqkm = xqk.channel(q);
        const Mat xvm = xv.channel(q);

        for (int i = 0; i < seqlen; i++)
        {
            int8_t* outptr = xqkv.channel(i).row<int8_t>(q);

            for (int j = 0; j < embed_dim_per_head; j++)
            {
                const float* qkptr = xqkm.row<float>(i);
                const int8_t* vptr = xvm.row<int8_t>(j);

                float sum = 0;
                for (int k = 0; k < seqlen; k++)
                {
                    sum += (*vptr++) * (*qkptr++);
                }

                outptr[j] = float2int8(sum * xqkv_out_scale);
            }
        }
    }

    Mat& top_blob = top_blobs[0];
    top_blob.create(embed_dim, seqlen, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -1;

    const float out_scale = 1.0f / internal_scales[4];
    // out = affine(xqkv)
    // xqkv  (embed_dim, seqlen)
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < seqlen; i++)
    {
        float* outptr = top_blob.row(i);

        for (int j = 0; j < embed_dim; j++)
        {
            const int8_t* ptr = xqkv.channel(i);
            const int8_t* kptr = (const int8_t*)out_weight_data + embed_dim * j;

            int32_t sum = 0;
            for (int k = 0; k < embed_dim; k++)
            {
                sum += *ptr++ * *kptr++;
            }

            outptr[j] = sum * out_scale / o_weight_scales[j] + out_bias_data[j];
        }
    }

    return 0;
}

#endif

// refers to https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
int MultiHeadAttention::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if NCNN_INT8
    if (opt.use_int8_inference && q_weight_data.elemsize == (size_t)1u && k_weight_data.elemsize == (size_t)1u && v_weight_data.elemsize == (size_t)1u && out_weight_data.elemsize == (size_t)1u)
    {
        return forward_int8(bottom_blobs, top_blobs, opt);
    }
#endif

    const Mat& q_blob = bottom_blobs[0];
    const Mat& k_blob = bottom_blobs.size() == 1 ? q_blob : bottom_blobs[1];
    const Mat& v_blob = bottom_blobs.size() == 1 ? q_blob : bottom_blobs[2];

    const int seqlen = q_blob.h;
    const int embed_dim_per_head = embed_dim / num_head;

    Mat& top_blob = top_blobs[0];
    top_blob.create(embed_dim, seqlen, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -1;

    Mat xq(embed_dim_per_head, seqlen, num_head, 4u, opt.workspace_allocator);
    Mat xk(embed_dim_per_head, seqlen, num_head, 4u, opt.workspace_allocator);
    Mat xv(seqlen, embed_dim_per_head, num_head, 4u, opt.workspace_allocator);

    Mat xqk(seqlen, seqlen, num_head, 4u, opt.workspace_allocator);

    Mat xqkv(embed_dim_per_head, num_head, seqlen, 4u, opt.workspace_allocator);

    const float inv_sqrt_embed_dim_per_head = 1.f / sqrt(embed_dim_per_head);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_head; q++)
    {
        // xq = affine(q) * inv_sqrt_embed_dim_per_head
        {
            Mat outm = xq.channel(q);

            for (int i = 0; i < seqlen; i++)
            {
                float* outptr = outm.row(i);

                for (int j = 0; j < embed_dim_per_head; j++)
                {
                    const float* ptr = q_blob.row(i);
                    const float* kptr = (const float*)q_weight_data + embed_dim * (q * embed_dim_per_head + j);

                    float sum = q_bias_data[q * embed_dim_per_head + j];
                    for (int k = 0; k < embed_dim; k++)
                    {
                        sum += *ptr++ * *kptr++;
                    }

                    outptr[j] = sum * inv_sqrt_embed_dim_per_head;
                }
            }
        }

        // xk = affine(k)
        {
            Mat outm = xk.channel(q);

            for (int i = 0; i < seqlen; i++)
            {
                float* outptr = outm.row(i);

                for (int j = 0; j < embed_dim_per_head; j++)
                {
                    const float* ptr = k_blob.row(i);
                    const float* kptr = (const float*)k_weight_data + embed_dim * (q * embed_dim_per_head + j);

                    float sum = k_bias_data[q * embed_dim_per_head + j];
                    for (int k = 0; k < embed_dim; k++)
                    {
                        sum += *ptr++ * *kptr++;
                    }

                    outptr[j] = sum;
                }
            }
        }

        // xv = affine(v)
        {
            Mat outm = xv.channel(q);

            for (int i = 0; i < embed_dim_per_head; i++)
            {
                for (int j = 0; j < seqlen; j++)
                {
                    const float* ptr = v_blob.row(j);
                    const float* kptr = (const float*)v_weight_data + embed_dim * (q * embed_dim_per_head + i);

                    float sum = v_bias_data[q * embed_dim_per_head + i];
                    for (int k = 0; k < embed_dim; k++)
                    {
                        sum += *ptr++ * *kptr++;
                    }

                    float* outptr = outm.row(i);

                    outptr[j] = sum;
                }
            }
        }

        // xqk = xq * xk
        // xq  (embed_dim_per_head, seqlen)
        // xk  (embed_dim_per_head, seqlen)
        {
            const Mat xqm = xq.channel(q);
            const Mat xkm = xk.channel(q);

            Mat outm = xqk.channel(q);

            for (int i = 0; i < seqlen; i++)
            {
                float* outptr = outm.row(i);

                for (int j = 0; j < seqlen; j++)
                {
                    const float* qptr = xqm.row(i);
                    const float* kptr = xkm.row(j);

                    float sum = 0.f;
                    for (int k = 0; k < embed_dim_per_head; k++)
                    {
                        sum += *qptr++ * *kptr++;
                    }

                    outptr[j] = sum;
                }
            }
        }

        // softmax(xqk)
        {
            Mat outm = xqk.channel(q);

            for (int i = 0; i < seqlen; i++)
            {
                float* ptr = outm.row(i);

                float max = -FLT_MAX;
                for (int j = 0; j < seqlen; j++)
                {
                    max = std::max(max, ptr[j]);
                }

                float sum = 0.f;
                for (int j = 0; j < seqlen; j++)
                {
                    ptr[j] = (float)(exp(ptr[j] - max));
                    sum += ptr[j];
                }

                for (int j = 0; j < seqlen; j++)
                {
                    ptr[j] /= sum;
                }
            }
        }

        // xqkv = xqk * xv
        // xqk (seqlen, seqlen)
        // xv  (seqlen, embed_dim_per_head)
        // out (embed_dim_per_head, num_head, seqlen)
        {
            const Mat xqkm = xqk.channel(q);
            const Mat xvm = xv.channel(q);

            for (int i = 0; i < seqlen; i++)
            {
                float* outptr = xqkv.channel(i).row(q);

                for (int j = 0; j < embed_dim_per_head; j++)
                {
                    const float* qkptr = xqkm.row(i);
                    const float* vptr = xvm.row(j);

                    float sum = 0.f;
                    for (int k = 0; k < seqlen; k++)
                    {
                        sum += *qkptr++ * *vptr++;
                    }

                    outptr[j] = sum;
                }
            }
        }
    }

    // out = affine(xqkv)
    // xqkv  (embed_dim, seqlen)
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < seqlen; i++)
    {
        float* outptr = top_blob.row(i);

        for (int j = 0; j < embed_dim; j++)
        {
            const float* ptr = xqkv.channel(i);
            const float* kptr = (const float*)out_weight_data + embed_dim * j;

            float sum = out_bias_data[j];
            for (int k = 0; k < embed_dim; k++)
            {
                sum += *ptr++ * *kptr++;
            }

            outptr[j] = sum;
        }
    }

    return 0;
}

} // namespace ncnn
