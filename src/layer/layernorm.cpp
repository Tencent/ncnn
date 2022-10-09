// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "layernorm.h"
#include <float.h>
#include <math.h>
#include "mathfun.h"
#include <stdint.h>

namespace ncnn {

LayerNorm::LayerNorm()
{
    one_blob_only = true;
    support_inplace = true;
}

int LayerNorm::load_param(const ParamDict& pd)
{
    affine_size = pd.get(0, 0);
    eps = pd.get(1, 0.001f);
    affine = pd.get(2, 1);
    int8_scale_term = pd.get(3, 0);
    return 0;
}

int LayerNorm::load_model(const ModelBin& mb)
{
    if (affine == 0)
        return 0;

    gamma_data = mb.load(affine_size, 1);
    if (gamma_data.empty())
        return -100;

    beta_data = mb.load(affine_size, 1);
    if (beta_data.empty())
        return -100;

#ifdef NCNN_INT8
    if (int8_scale_term)
    {
        input_scales = mb.load(affine_size, 1);
        output_scale = mb.load(1, 1);
    }
#endif
    return 0;
}

#ifdef NCNN_INT8
static inline void get_MN(const float x, uint32_t& M, uint32_t& N)
{
#define LOG2 (0.693147180f)
    // log2(x)  = log(x) / log(2)
    int bit = 7 - round(floor(log(x) / LOG2));
#undef LOG2
    bit = bit < 0 ? 0 : bit;
    bit = bit > 31 ? 31 : bit;

    N = 1u << bit;

    // N > 0 and x > 0
    M = round(floor(N * x));
    M = M > 255 ? 255 : M;

    return;
}

int LayerNorm::forward_inplace_int8(Mat& bottom_top_blob, const Option& opt) const
{
    if (!affine || bottom_top_blob.c != 1)
    {
        // non transformer int8 layernorm not implemented
        return -100;
    }

    if (bottom_top_blob.w != affine_size)
    {
        // check input parameter
        return -200;
    }

    // Transformer using BNC format
    float in_scale_max = -FLT_MAX;
    const float out_scale = output_scale[0];
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < affine_size; ++i)
        {
            if (in_scale_max < input_scales[i])
            {
                in_scale_max = input_scales[i];
            }
        }
    }

    // quantize input to int8
    Mat xq(bottom_top_blob.w, bottom_top_blob.h, 4u, opt.workspace_allocator);
    const int elem_count = bottom_top_blob.w * bottom_top_blob.h * bottom_top_blob.c;
    if (bottom_top_blob.elemsize == (size_t)1u)
    {
        // if input int8, rescale input
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < bottom_top_blob.h; ++i)
        {
            int8_t* from = bottom_top_blob.row<int8_t>(i);
            int32_t* to = xq.row<int32_t>(i);
            for (int j = 0; j < bottom_top_blob.w; ++j)
            {
                to[j] = round(from[j] * in_scale_max / input_scales[j]);
            }
        }
    }
    else
    {
        int32_t* ptr = (int32_t*)xq.data;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < elem_count; ++i)
        {
            ptr[i] = round(bottom_top_blob[i] * in_scale_max);
        }
    }

    // get mean and std
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < xq.h; ++i)
    {
        // get mean and std
        int32_t sum = 0;
        int32_t sum_pow2 = 0;
        int32_t* ptr = xq.row<int32_t>(i);
        for (int j = 0; j < xq.w; ++j)
        {
            sum += ptr[j];
            sum_pow2 += ptr[j] * ptr[j];
        }

        const float mean = sum * 1.0f / in_scale_max / affine_size;
        const float std = sqrt(1.0f * affine_size * sum_pow2 - sum * sum) / in_scale_max / affine_size;

        // update xq
        const float scale_a = out_scale / std / in_scale_max;
        const float scale_b = mean / std;
        for (int j = 0; j < affine_size; ++j)
        {
            float A = gamma_data[j] * scale_a;
            const float sign = A > 0.f ? 1.f : -1.f;

            uint32_t M, N;
            get_MN(abs(A), M, N);

            int32_t B = round((beta_data[j] - scale_b * gamma_data[j]) * out_scale * N);

            ptr[j] = round((sign * M * ptr[j] + B) / N);
        }
    }

    if (int8_scale_term >= 100)
    {
        // output int8
        bottom_top_blob.create(bottom_top_blob.w, bottom_top_blob.h, 1u, opt.workspace_allocator);
        int32_t* from = (int32_t*)xq.data;
        int8_t* to = (int8_t*)bottom_top_blob.data;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < elem_count; ++i)
        {
            if (from[i] > 127)
            {
                to[i] = 127;
            }
            else if (from[i] < -127)
            {
                to[i] = -127;
            }
            else
            {
                to[i] = from[i];
            }
        }
    }
    else
    {
        // dequant and output fp32
        if (bottom_top_blob.elemsize == (size_t)1u)
        {
            bottom_top_blob.create(bottom_top_blob.w, bottom_top_blob.h, (size_t)4u, opt.workspace_allocator);
        }

        int32_t* ptr = (int32_t*)xq.data;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < elem_count; ++i)
        {
            bottom_top_blob[i] = ptr[i] / out_scale;
        }
    }

    return 0;
}
#endif

int LayerNorm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#ifdef NCNN_INT8
    if (opt.use_int8_inference && int8_scale_term)
    {
        return forward_inplace_int8(bottom_top_blob, opt);
    }
#endif
    // x = (x - mean) / sqrt(var + eps) * gamma + beta

    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int w = bottom_top_blob.w;
        // assert affine_size == w

        float* ptr = bottom_top_blob;

        // mean and var
        float sum = 0.f;
        float sqsum = 0.f;
        for (int i = 0; i < w; i++)
        {
            sum += ptr[i];
            //sqsum += ptr[i] * ptr[i];
        }
        float mean = sum / w;
        float tmp = 0.f;
        for (int i = 0; i < w; i++)
        {
            tmp = ptr[i] - mean;
            sqsum += tmp * tmp;
        }
        float var = sqsum / w;
        // the var maybe minus due to accuracy
        //float var = sqsum / w - mean * mean;

        float a = static_cast<float>(1.f / (sqrt(var + eps)));
        float b = -mean * a;

        if (affine)
        {
            for (int i = 0; i < w; i++)
            {
                ptr[i] = (ptr[i] * a + b) * gamma_data[i] + beta_data[i];
            }
        }
        else
        {
            for (int i = 0; i < w; i++)
            {
                ptr[i] = ptr[i] * a + b;
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        // assert affine_size == w

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);

            // mean and var
            float sum = 0.f;
            float sqsum = 0.f;
            for (int j = 0; j < w; j++)
            {
                sum += ptr[j];
                //sqsum += ptr[j] * ptr[j];
            }
            float mean = sum / w;
            float tmp = 0.f;
            for (int j = 0; j < w; j++)
            {
                tmp = ptr[j] - mean;
                sqsum += tmp * tmp;
            }
            float var = sqsum / w;
            // the var maybe minus due to accuracy
            //float var = sqsum / w - mean * mean;

            float a = static_cast<float>(1.f / (sqrt(var + eps)));
            float b = -mean * a;

            if (affine)
            {
                for (int j = 0; j < w; j++)
                {
                    ptr[j] = (ptr[j] * a + b) * gamma_data[j] + beta_data[j];
                }
            }
            else
            {
                for (int j = 0; j < w; j++)
                {
                    ptr[j] = ptr[j] * a + b;
                }
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h;

        if (affine_size == w)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.channel(q).row(i);

                    // mean and var
                    float sum = 0.f;
                    float sqsum = 0.f;
                    for (int j = 0; j < w; j++)
                    {
                        sum += ptr[j];
                        //sqsum += ptr[j] * ptr[j];
                    }
                    float mean = sum / w;
                    float tmp = 0.f;
                    for (int j = 0; j < w; j++)
                    {
                        tmp = ptr[j] - mean;
                        sqsum += tmp * tmp;
                    }
                    float var = sqsum / w;
                    // the var maybe minus due to accuracy
                    //float var = sqsum / w - mean * mean;

                    float a = static_cast<float>(1.f / (sqrt(var + eps)));
                    float b = -mean * a;

                    if (affine)
                    {
                        for (int j = 0; j < w; j++)
                        {
                            ptr[j] = (ptr[j] * a + b) * gamma_data[j] + beta_data[j];
                        }
                    }
                    else
                    {
                        for (int j = 0; j < w; j++)
                        {
                            ptr[j] = ptr[j] * a + b;
                        }
                    }
                }
            }
        }
        else // if (affine_size == size)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                // mean and var
                float sum = 0.f;
                float sqsum = 0.f;
                for (int i = 0; i < size; i++)
                {
                    sum += ptr[i];
                    //sqsum += ptr[i] * ptr[i];
                }
                float mean = sum / size;
                float tmp = 0.f;
                for (int i = 0; i < size; i++)
                {
                    tmp = ptr[i] - mean;
                    sqsum += tmp * tmp;
                }
                float var = sqsum / size;
                // the var maybe minus due to accuracy
                //float var = sqsum / size - mean * mean;

                float a = static_cast<float>(1.f / (sqrt(var + eps)));
                float b = -mean * a;

                if (affine)
                {
                    for (int i = 0; i < size; i++)
                    {
                        ptr[i] = (ptr[i] * a + b) * gamma_data[i] + beta_data[i];
                    }
                }
                else
                {
                    for (int i = 0; i < size; i++)
                    {
                        ptr[i] = ptr[i] * a + b;
                    }
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
