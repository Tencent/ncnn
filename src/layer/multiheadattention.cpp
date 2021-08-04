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

    return 0;
}

int MultiHeadAttention::load_model(const ModelBin& mb)
{
    q_weight_data = mb.load(weight_data_size, 0);
    if (q_weight_data.empty())
        return -100;

    q_bias_data = mb.load(embed_dim, 1);
    if (q_bias_data.empty())
        return -100;

    k_weight_data = mb.load(weight_data_size, 0);
    if (k_weight_data.empty())
        return -100;

    k_bias_data = mb.load(embed_dim, 1);
    if (k_bias_data.empty())
        return -100;

    v_weight_data = mb.load(weight_data_size, 0);
    if (v_weight_data.empty())
        return -100;

    v_bias_data = mb.load(embed_dim, 1);
    if (v_bias_data.empty())
        return -100;

    out_weight_data = mb.load(weight_data_size, 0);
    if (out_weight_data.empty())
        return -100;

    out_bias_data = mb.load(embed_dim, 1);
    if (out_bias_data.empty())
        return -100;

    return 0;
}

int MultiHeadAttention::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
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
