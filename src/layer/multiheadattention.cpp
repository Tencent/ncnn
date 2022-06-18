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

namespace ncnn {

MultiHeadAttention::MultiHeadAttention()
{
}

int MultiHeadAttention::load_param(const ParamDict& pd)
{
    embed_dim = pd.get(0, 0);
    num_head = pd.get(1, 1);
    weight_data_size = pd.get(2, 0);
    int8_scale_term = pd.get(3, 0);

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

    LOAD_MAT(q_weight_data, weight_data_size);
    LOAD_MAT(k_weight_data, weight_data_size);
    LOAD_MAT(v_weight_data, weight_data_size);
    LOAD_MAT(out_weight_data, weight_data_size);
#undef LOAD_MAT

#define LOAD_FLOAT_MAT(name, len) \
    name = mb.load(len, 1);       \
    if (name.empty())             \
    {                             \
        return -100;              \
    }

    LOAD_FLOAT_MAT(q_bias_data, embed_dim);
    LOAD_FLOAT_MAT(k_bias_data, embed_dim);
    LOAD_FLOAT_MAT(v_bias_data, embed_dim);
    LOAD_FLOAT_MAT(out_bias_data, embed_dim);

#if NCNN_INT8
    if (int8_scale_term)
    {
        LOAD_FLOAT_MAT(q_weight_scales, weight_data_size);
        LOAD_FLOAT_MAT(k_weight_scales, weight_data_size);
        LOAD_FLOAT_MAT(v_weight_scales, weight_data_size);
        LOAD_FLOAT_MAT(o_weight_scales, weight_data_size);

        LOAD_FLOAT_MAT(internal_scales, 5);
    }
#endif // NCNN_INT8

#undef LOAD_FLOAT_MAT

    return 0;
}

#ifdef NCNN_INT8
/**
 * @brief
 *  q_input_int8 * q_weight --> q_out_int32
 *  q_out_int32 / input_scale / weight_scale + bias --> q_out_fp32
 *  q_out_fp32 --> q_internal_int8
 * @param input
 * @param internal
 * @param input_scale
 * @param weight_scales
 * @param transpose_out
 * @return int
 */
int MultiHeadAttention::transform_input(const Mat& input, const Mat& weight, Mat& out_int8, const Mat& input_scale, const Mat& weight_scales, const Option& opt, bool transpose_out = false) const
{
    const int seqlen = input.h;
    const int embed_dim_per_head = embed_dim / num_head;

    Mat input_int8;
    if (input.elemsize != 1)
    {
        quantize_to_int8(input, input_int8, input_scale, opt);
    }

    Mat buffer(embed_dim_per_head, seqlen, num_head, 4u, opt.workspace_allocator);

    for (int q = 0; q < num_head; q++)
    {
        // xq = affine(q)
        {
            Mat outm = buffer.channel(q);

            for (int i = 0; i < seqlen; i++)
            {
                int32_t* outptr = outm.row<int32_t>(i);

                for (int j = 0; j < embed_dim_per_head; j++)
                {
                    const int8_t* ptr = input_int8.row<int8_t>(i);
                    const int8_t* kptr = (int8_t*)(weight.data) + embed_dim * (q * embed_dim_per_head + j);

                    int32_t sum = 0;
                    for (int k = 0; k < embed_dim; k++)
                    {
                        sum += *ptr++ * *kptr++;
                    }
                    // TODO calculate scale
                    outptr[j] = sum / input_scale;
                }
            }
        }
    }

    return 0;
}

/**
 * @brief int8 mha, referenced to
 *
 *  https://github.com/megvii-research/FQ-ViT/blob/main/models/vit_quant.py#L95
 *
 * @param bottom_blobs
 * @param top_blobs
 * @param opt
 * @return int
 */
int MultiHeadAttention::forward_int8(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    // mha int8 kernel
    const Mat& q_blob = bottom_blobs[0];
    const Mat& k_blob = bottom_blobs.size() == 1 ? q_blob : bottom_blobs[1];
    const Mat& v_blob = bottom_blobs.size() == 1 ? q_blob : bottom_blobs[2];

    Option opt_g = opt;
    opt_g.blob_allocator = opt.workspace_allocator;
    opt_g.use_packing_layout = false;

    Mat q_blob_int8 = q_blob, k_blob_int8 = k_blob, v_blob_int8 = v_blob;

    if (q_blob_int8.elemsize != 1)
    {
        quantize_to_int8(q_blob, q_blob_int8, q_input_scale, opt_g);
    }

    if (k_blob_int8.elemsize != 1)
    {
        quantize_to_int8(k_blob, k_blob_int8, k_input_scale, opt_g);
    }

    if (v_blob_int8.elemsize != 1)
    {
        quantize_to_int8(v_blob, v_blob_int8, v_input_scale, opt_g);
    }

    // 145
    const int seqlen = q_blob.h;
    // 64
    const int embed_dim_per_head = embed_dim / num_head;

    Mat& top_blob = top_blobs[0];
    top_blob.create(embed_dim, seqlen, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -1;

    // 64, 145, 12
    Mat xq(embed_dim_per_head, seqlen, num_head, 4u, opt.workspace_allocator);
    Mat xk(embed_dim_per_head, seqlen, num_head, 4u, opt.workspace_allocator);
    // transpose(v) for better gemm performance
    Mat xv(seqlen, embed_dim_per_head, num_head, 4u, opt.workspace_allocator);

    {
    }

    return 0;
}
#endif

// refers to https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
int MultiHeadAttention::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if NCNN_INT8
    if (opt.use_int8_inference)
    {
        if (q_weight_data.elemsize != (size_t)1u || k_weight_data.elemsize != (size_t)1u || v_weight_data.elemsize != (size_t)1u || out_weight_data.elemsize != (size_t)1u)
        {
            return -1;
        }
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
