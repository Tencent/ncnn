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

#include "pooling.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Pooling)

Pooling::Pooling()
{
    one_blob_only = true;
    support_inplace = false;
}

#if NCNN_STDIO
#if NCNN_STRING
int Pooling::load_param(FILE* paramfp)
{
    int nscan = fscanf(paramfp, "%d %d %d %d %d",
                       &pooling_type, &kernel_size, &stride, &pad, &global_pooling);
    if (nscan != 5)
    {
        fprintf(stderr, "Pooling load_param failed %d\n", nscan);
        return -1;
    }

    return 0;
}
#endif // NCNN_STRING
int Pooling::load_param_bin(FILE* paramfp)
{
    fread(&pooling_type, sizeof(int), 1, paramfp);

    fread(&kernel_size, sizeof(int), 1, paramfp);

    fread(&stride, sizeof(int), 1, paramfp);

    fread(&pad, sizeof(int), 1, paramfp);

    fread(&global_pooling, sizeof(int), 1, paramfp);

    return 0;
}
#endif // NCNN_STDIO

int Pooling::load_param(const unsigned char*& mem)
{
    pooling_type = *(int*)(mem);
    mem += 4;

    kernel_size = *(int*)(mem);
    mem += 4;

    stride = *(int*)(mem);
    mem += 4;

    pad = *(int*)(mem);
    mem += 4;

    global_pooling = *(int*)(mem);
    mem += 4;

    return 0;
}

int Pooling::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    // max value in NxN window
    // avg value in NxN window

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

//     fprintf(stderr, "Pooling     input %d x %d  pad = %d  ksize=%d  stride=%d\n", w, h, pad, kernel_size, stride);
    if (global_pooling)
    {
        top_blob.create(1, 1, channels);
        if (top_blob.empty())
            return -100;

        int size = w * h;

        if (pooling_type == PoolMethod_MAX)
        {
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                float max = ptr[0];
                for (int i=0; i<size; i++)
                {
                    max = std::max(max, ptr[i]);
                }

                outptr[0] = max;
            }
        }
        else if (pooling_type == PoolMethod_AVE)
        {
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                float sum = 0.f;
                for (int i=0; i<size; i++)
                {
                    sum += ptr[i];
                }

                outptr[0] = sum / size;
            }
        }

        return 0;
    }

    Mat bottom_blob_bordered = bottom_blob;
    if (pad > 0)
    {
        copy_make_border(bottom_blob, bottom_blob_bordered, pad, pad, pad, pad, BORDER_CONSTANT, 0.f);
        if (bottom_blob_bordered.empty())
            return -100;

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }

    int outw = (w - kernel_size) / stride + 1;
    int outh = (h - kernel_size) / stride + 1;

    int wtail = (w - kernel_size) % stride;
    int htail = (h - kernel_size) % stride;
    if (wtail != 0 || htail != 0)
    {
        int wtailpad = 0;
        int htailpad = 0;
        if (wtail != 0)
            wtailpad = kernel_size - wtail;
        if (htail != 0)
            htailpad = kernel_size - htail;

        Mat bottom_blob_bordered2;
        if (pooling_type == PoolMethod_MAX)
        {
            copy_make_border(bottom_blob_bordered, bottom_blob_bordered2, 0, htailpad, 0, wtailpad, BORDER_REPLICATE, 0.f);
        }
        else if (pooling_type == PoolMethod_AVE)
        {
            copy_make_border(bottom_blob_bordered, bottom_blob_bordered2, 0, htailpad, 0, wtailpad, BORDER_CONSTANT, 0.f);
        }
        if (bottom_blob_bordered2.empty())
            return -100;

        bottom_blob_bordered = bottom_blob_bordered2;

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;

        if (wtail != 0)
            outw += 1;
        if (htail != 0)
            outh += 1;
    }

    top_blob.create(outw, outh, channels);
    if (top_blob.empty())
        return -100;

    const int maxk = kernel_size * kernel_size;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w - kernel_size;
        for (int i = 0; i < kernel_size; i++)
        {
            for (int j = 0; j < kernel_size; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2++;
            }
            p2 += gap;
        }
    }

    if (pooling_type == PoolMethod_MAX)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const Mat m(w, h, bottom_blob_bordered.channel(q));
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const float* sptr = m.data + m.w * i*stride + j*stride;

                    float max = sptr[0];

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = sptr[ space_ofs[k] ];
                        max = std::max(max, val);
                    }

                    outptr[j] = max;
                }

                outptr += outw;
            }
        }
    }
    else if (pooling_type == PoolMethod_AVE)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const Mat m(w, h, bottom_blob_bordered.channel(q));
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const float* sptr = m.data + m.w * i*stride + j*stride;

                    float sum = 0;

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = sptr[ space_ofs[k] ];
                        sum += val;
                    }

                    outptr[j] = sum / maxk;
                }

                outptr += outw;
            }

            // fix tail pad
            if (wtail != 0)
            {
                const float scale = (float)kernel_size / wtail;

                outptr = top_blob.channel(q) + outw - 1;
                for (int i = 0; i < outh; i++)
                {
                    *outptr *= scale;
                    outptr += outw;
                }
            }
            if (htail != 0)
            {
                const float scale = (float)kernel_size / htail;

                outptr = top_blob.channel(q).row(outh - 1);
                for (int i = 0; i < outw; i++)
                {
                    outptr[i] *= scale;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
