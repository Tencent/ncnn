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

#include "reduction.h"
#include <math.h>

namespace ncnn {

DEFINE_LAYER_CREATOR(Reduction)

Reduction::Reduction()
{
    one_blob_only = true;
    support_inplace = false;
}

Reduction::~Reduction()
{
}

#if NCNN_STDIO
#if NCNN_STRING
int Reduction::load_param(FILE* paramfp)
{
    int nscan = fscanf(paramfp, "%d %d %f", &operation, &dim, &coeff);
    if (nscan != 3)
    {
        fprintf(stderr, "Reduction load_param failed %d\n", nscan);
        return -1;
    }

    return 0;
}
#endif // NCNN_STRING
int Reduction::load_param_bin(FILE* paramfp)
{
    fread(&operation, sizeof(int), 1, paramfp);

    fread(&dim, sizeof(int), 1, paramfp);

    fread(&coeff, sizeof(float), 1, paramfp);

    return 0;
}
#endif // NCNN_STDIO

int Reduction::load_param(const unsigned char*& mem)
{
    operation = *(int*)(mem);
    mem += 4;

    dim = *(int*)(mem);
    mem += 4;

    coeff = *(float*)(mem);
    mem += 4;

    return 0;
}

int Reduction::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int size = w * h;

    if (dim == 0)
    {
        top_blob.create(1);
    }
    else if (dim == 1)
    {
        top_blob.create(channels);
    }
    else if (dim == 2)
    {
        top_blob.create(h, channels);
    }
    if (top_blob.empty())
        return -100;

    if (operation == ReductionOp_SUM)
    {
        if (dim == 0)
        {
            Mat sums(channels);
            if (sums.empty())
                return -100;
            float* sums_ptr = sums;
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);

                float sum = 0.f;
                for (int i=0; i<size; i++)
                {
                    sum += ptr[i];
                }

                sums_ptr[q] = sum;
            }

            float* outptr = top_blob;

            float sum = 0.f;
            for (int i=0; i<size; i++)
            {
                sum += sums_ptr[i];
            }

            outptr[0] = sum * coeff;
        }
        else if (dim == 1)
        {
            float* outptr = top_blob;
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);

                float sum = 0.f;
                for (int i=0; i<size; i++)
                {
                    sum += ptr[i];
                }

                outptr[q] = sum * coeff;
            }
        }
        else if (dim == 2)
        {
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i=0; i<h; i++)
                {
                    float sum = 0.f;
                    for (int j=0; j<w; j++)
                    {
                        sum += ptr[j];
                    }

                    outptr[i] = sum * coeff;

                    ptr += w;
                }
            }
        }
    }
    else if (operation == ReductionOp_ASUM)
    {
        if (dim == 0)
        {
            Mat sums(channels);
            if (sums.empty())
                return -100;
            float* sums_ptr = sums;
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);

                float sum = 0.f;
                for (int i=0; i<size; i++)
                {
                    sum += fabs(ptr[i]);
                }

                sums_ptr[q] = sum;
            }

            float* outptr = top_blob;

            float sum = 0.f;
            for (int i=0; i<size; i++)
            {
                sum += sums_ptr[i];
            }

            outptr[0] = sum * coeff;
        }
        else if (dim == 1)
        {
            float* outptr = top_blob;
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);

                float sum = 0.f;
                for (int i=0; i<size; i++)
                {
                    sum += fabs(ptr[i]);
                }

                outptr[q] = sum * coeff;
            }
        }
        else if (dim == 2)
        {
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i=0; i<h; i++)
                {
                    float sum = 0.f;
                    for (int j=0; j<w; j++)
                    {
                        sum += fabs(ptr[j]);
                    }

                    outptr[i] = sum * coeff;

                    ptr += w;
                }
            }
        }
    }
    else if (operation == ReductionOp_SUMSQ)
    {
        if (dim == 0)
        {
            Mat sums(channels);
            if (sums.empty())
                return -100;
            float* sums_ptr = sums;
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);

                float sum = 0.f;
                for (int i=0; i<size; i++)
                {
                    sum += ptr[i] * ptr[i];
                }

                sums_ptr[q] = sum;
            }

            float* outptr = top_blob;

            float sum = 0.f;
            for (int i=0; i<size; i++)
            {
                sum += sums_ptr[i];
            }

            outptr[0] = sum * coeff;
        }
        else if (dim == 1)
        {
            float* outptr = top_blob;
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);

                float sum = 0.f;
                for (int i=0; i<size; i++)
                {
                    sum += ptr[i] * ptr[i];
                }

                outptr[q] = sum * coeff;
            }
        }
        else if (dim == 2)
        {
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i=0; i<h; i++)
                {
                    float sum = 0.f;
                    for (int j=0; j<w; j++)
                    {
                        sum += ptr[i] * ptr[i];
                    }

                    outptr[i] = sum * coeff;

                    ptr += w;
                }
            }
        }
    }
    else if (operation == ReductionOp_MEAN)
    {
        if (dim == 0)
        {
            Mat sums(channels);
            if (sums.empty())
                return -100;
            float* sums_ptr = sums;
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);

                float sum = 0.f;
                for (int i=0; i<size; i++)
                {
                    sum += ptr[i];
                }

                sums_ptr[q] = sum;
            }

            float* outptr = top_blob;

            float sum = 0.f;
            for (int i=0; i<size; i++)
            {
                sum += sums_ptr[i];
            }

            outptr[0] = sum / (channels * size) * coeff;
        }
        else if (dim == 1)
        {
            float* outptr = top_blob;
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);

                float sum = 0.f;
                for (int i=0; i<size; i++)
                {
                    sum += ptr[i];
                }

                outptr[q] = sum / size * coeff;
            }
        }
        else if (dim == 2)
        {
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i=0; i<h; i++)
                {
                    float sum = 0.f;
                    for (int j=0; j<w; j++)
                    {
                        sum += ptr[j];
                    }

                    outptr[i] = sum / w * coeff;

                    ptr += w;
                }
            }
        }
    }
    else if (operation == ReductionOp_MAX)
    {
        if (dim == 0)
        {
            Mat maxs(channels);
            if (maxs.empty())
                return -100;
            float* maxs_ptr = maxs;
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);

                float max = ptr[0];
                for (int i=1; i<size; i++)
                {
                    max = std::max(max, ptr[i]);
                }

                maxs_ptr[q] = max;
            }

            float* outptr = top_blob;

            float max = maxs_ptr[0];
            for (int i=1; i<size; i++)
            {
                max = std::max(max, maxs_ptr[i]);
            }

            outptr[0] = max * coeff;
        }
        else if (dim == 1)
        {
            float* outptr = top_blob;
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);

                float max = ptr[0];
                for (int i=1; i<size; i++)
                {
                    max = std::max(max, ptr[i]);
                }

                outptr[q] = max * coeff;
            }
        }
        else if (dim == 2)
        {
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i=0; i<h; i++)
                {
                    float max = ptr[0];
                    for (int j=1; j<w; j++)
                    {
                        max = std::max(max, ptr[i]);
                    }

                    outptr[i] = max * coeff;

                    ptr += w;
                }
            }
        }
    }
    else if (operation == ReductionOp_MIN)
    {
        if (dim == 0)
        {
            Mat mins(channels);
            if (mins.empty())
                return -100;
            float* mins_ptr = mins;
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);

                float min = ptr[0];
                for (int i=1; i<size; i++)
                {
                    min = std::min(min, ptr[i]);
                }

                mins_ptr[q] = min;
            }

            float* outptr = top_blob;

            float min = mins_ptr[0];
            for (int i=1; i<size; i++)
            {
                min = std::min(min, mins_ptr[i]);
            }

            outptr[0] = min * coeff;
        }
        else if (dim == 1)
        {
            float* outptr = top_blob;
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);

                float min = ptr[0];
                for (int i=1; i<size; i++)
                {
                    min = std::min(min, ptr[i]);
                }

                outptr[q] = min * coeff;
            }
        }
        else if (dim == 2)
        {
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i=0; i<h; i++)
                {
                    float min = ptr[0];
                    for (int j=1; j<w; j++)
                    {
                        min = std::min(min, ptr[i]);
                    }

                    outptr[i] = min * coeff;

                    ptr += w;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
