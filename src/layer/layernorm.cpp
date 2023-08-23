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

    return 0;
}

int LayerNorm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
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

        float a = 1.f / (sqrtf(var + eps));
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

            float a = 1.f / (sqrtf(var + eps));
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

                    float a = 1.f / (sqrtf(var + eps));
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

                float a = 1.f / (sqrtf(var + eps));
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
