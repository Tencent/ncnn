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

#include "normalize.h"
#include <math.h>

namespace ncnn {

DEFINE_LAYER_CREATOR(Normalize)

Normalize::Normalize()
{
    one_blob_only = true;
    support_inplace = false;
}

int Normalize::load_param(const ParamDict& pd)
{
    across_spatial = pd.get(0, 0);
    channel_shared = pd.get(1, 0);
    eps = pd.get(2, 0.0001f);
    scale_data_size = pd.get(3, 0);

    return 0;
}

#if NCNN_STDIO
int Normalize::load_model(FILE* binfp)
{
    int nread;

    scale_data.create(1, scale_data_size);
    nread = fread(scale_data, scale_data_size * sizeof(float), 1, binfp);
    if (nread != 1)
    {
        fprintf(stderr, "Normalize read scale_data failed %d\n", nread);
        return -1;
    }

    return 0;
}
#endif // NCNN_STDIO

int Normalize::load_model(const unsigned char*& mem)
{
    scale_data = Mat(1, scale_data_size, (float*)mem);
    mem += scale_data_size * sizeof(float);

    return 0;
}

int Normalize::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int size = w * h;

    top_blob.create(w, h, channels);
    if (top_blob.empty())
        return -100;

    if (across_spatial)
    {
        // square
        Mat square_sum_blob;
        square_sum_blob.create(channels);
        if (square_sum_blob.empty())
            return -100;

        float* square_sum_ptr = square_sum_blob;
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);

            float ssum = 0.f;
            for (int i=0; i<size; i++)
            {
                ssum += ptr[i] * ptr[i];
            }

            square_sum_ptr[q] = ssum;
        }

        // sum + eps
        float ssum = eps;
        for (int q=0; q<channels; q++)
        {
            ssum += square_sum_ptr[q];
        }

        // 1 / sqrt(ssum)
        float a = 1.f / sqrt(ssum);

        if (channel_shared)
        {
            float scale = a * scale_data.data[0];

            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i=0; i<size; i++)
                {
                    outptr[i] = ptr[i] * scale;
                }
            }
        }
        else
        {
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);
                float scale = a * scale_data.data[q];

                for (int i=0; i<size; i++)
                {
                    outptr[i] = ptr[i] * scale;
                }
            }
        }
    }
    else
    {
        // square sum, 1 / sqrt(ssum)
        Mat square_sum_blob;
        square_sum_blob.create(w, h);
        if (square_sum_blob.empty())
            return -100;

        float* ssptr = square_sum_blob;

        if (channel_shared)
        {
            float scale = scale_data.data[0];

            #pragma omp parallel for
            for (int i=0; i<size; i++)
            {
                float ssum = eps;
                for (int q=0; q<channels; q++)
                {
                    const float* ptr = bottom_blob.channel(q);
                    ssum += ptr[i] * ptr[i];
                }

                ssptr[i] = 1.f / sqrt(ssum) * scale;
            }

            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i=0; i<size; i++)
                {
                    outptr[i] = ptr[i] * ssptr[i];
                }
            }
        }
        else
        {
            #pragma omp parallel for
            for (int i=0; i<size; i++)
            {
                float ssum = eps;
                for (int q=0; q<channels; q++)
                {
                    const float* ptr = bottom_blob.channel(q);
                    ssum += ptr[i] * ptr[i];
                }

                ssptr[i] = 1.f / sqrt(ssum);
            }

            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);
                float scale = scale_data.data[q];

                for (int i=0; i<size; i++)
                {
                    outptr[i] = ptr[i] * ssptr[i] * scale;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
