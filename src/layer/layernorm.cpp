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
    static uint32_t pow2_table[] = {
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216,
        33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648};

    int bit = 7 - round(floor(log2(x)));
    bit = bit < 0 ? 0 : bit;
    bit = bit > 31 ? 31 : bit;

    N = pow2_table[bit];

    // N > 0 and x > 0
    M = round(floor(N * x));
    M = M > 255 ? 255 : M;

    return;
}

int LayerNorm::forward_inplace_int8(Mat& bottom_top_blob, const Option& opt) const
{
    if (!affine || bottom_top_blob.dims != 3 || bottom_top_blob.c != 1)
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
        for (int i = 0; i < affine_size; ++i)
        {
            if (in_scale_max < input_scales[i])
            {
                in_scale_max = input_scales[i];
            }
        }
    }

    // quantize input to int8
    Mat xq;
    const int elem_count = bottom_top_blob.w * bottom_top_blob.h * bottom_top_blob.c;
    if (bottom_top_blob.elemsize == (size_t)1u)
    {
        xq = bottom_top_blob;
        // if input int8, rescale input
        for (int i = 0; i < bottom_top_blob.h; ++i)
        {
            int8_t* ptr = xq.row<int8_t>(i);
            for (int j = 0; j < bottom_top_blob.w; ++j)
            {
                ptr[j] = float2int8(ptr[j] * in_scale_max / input_scales[j]);
            }
        }
    }
    else
    {
        xq.create(bottom_top_blob.w, bottom_top_blob.h, 1u, opt.workspace_allocator);
        // else fuse ((in * in_scale).round() * (in_scale_max / in_scale)).round to (in*in_scale_max).round()
        int8_t* ptr = (int8_t*)xq.data;
        for (int i = 0; i < elem_count; ++i)
        {
            ptr[i] = float2int8(bottom_top_blob[i] * in_scale_max);
        }
    }

    // get mean and std
    for (int i = 0; i < xq.h; ++i)
    {
        // get mean and std
        int32_t sum = 0;
        int32_t sum_pow2 = 0;
        int8_t* ptr = xq.row<int8_t>(i);
        for (int j = 0; j < xq.w; ++j)
        {
            sum += ptr[j];
            sum_pow2 += ptr[j] * ptr[j];
        }

        const float mean = sum * 1.0f /  in_scale_max / affine_size;
        const float std = sqrt(affine_size * sum_pow2 - sum * sum) * in_scale_max / affine_size;

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

            ptr[j] = float2int8((sign * M * ptr[j] + B) / N);
        }
    }

    if (int8_scale_term >= 100)
    {
        // output int8
        bottom_top_blob = xq;
    }
    else
    {
        // dequant and output fp32
        if (bottom_top_blob.elemsize == (size_t)1u)
        {
            bottom_top_blob.create(bottom_top_blob.w, bottom_top_blob.h, (size_t)4u, opt.workspace_allocator);
        }

        int8_t* ptr = (int8_t*)xq.data;
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
