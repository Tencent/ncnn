// BUG1989 is pleased to support the open source community by supporting ncnn available.
//
// Copyright (C) 2019 BUG1989. All rights reserved.
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

#include "requantize_arm.h"

#include <math.h>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

DEFINE_LAYER_CREATOR(Requantize_arm)

static inline signed char float2int8(float v)
{
    int int32 = round(v);
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

int Requantize_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{ 
    int dims = bottom_blob.dims;

    if (dims == 1)
    {
        int w = bottom_blob.w;

        const int* intptr = bottom_blob;
        signed char * ptr = top_blob;

        if (bias_term)
        {
            if (bias_data_size > 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i=0; i<w; i++)
                {
                    ptr[i] = float2int8(((intptr[i] * scale_in) + bias_data[i]) * scale_out);
                    if (fusion_relu && ptr[i] < 0)
                        ptr[i] = 0;
                }
            }
            else
            {
                float bias = bias_data[0];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i=0; i<w; i++)
                {
                    ptr[i] = float2int8(((intptr[i] * scale_in) + bias) * scale_out);
                    if (fusion_relu && ptr[i] < 0)
                        ptr[i] = 0;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<w; i++)
            {
                ptr[i] = float2int8(intptr[i] * scale_in * scale_out);
                if (fusion_relu && ptr[i] < 0)
                    ptr[i] = 0;
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        if (bias_term)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                signed char* ptr = top_blob.row<signed char>(i);

                float bias = bias_data_size > 1 ? bias_data[i] : bias_data[0];

                for (int j=0; j<w; j++)
                {
                    ptr[j] = float2int8(((intptr[j] * scale_in) + bias) * scale_out);
                    if (fusion_relu && ptr[j] < 0)
                        ptr[j] = 0;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                signed char* ptr = top_blob.row<signed char>(i);

                for (int j=0; j<w; j++)
                {
                    ptr[j] = float2int8(intptr[j] * scale_in * scale_out);
                    if (fusion_relu && ptr[j] < 0)
                        ptr[j] = 0;
                }
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;
        int size = w * h;

        double scale_fuse = scale_in * scale_out;

        if (bias_term)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                signed char* ptr = top_blob.channel(q);

                float bias = bias_data_size > 1 ? bias_data[q] : bias_data[0];

                int remain = size;
                for (; remain > 0; remain--)
                {
                    *ptr = float2int8(((*intptr * scale_in) + bias) * scale_out);

                    fprintf(stdout, "in requantize: %d %f %f %f result: %d\n", (int32_t)(*intptr), scale_in, bias, scale_out, (int32_t)(*ptr));
                    intptr++;
                    ptr ++;                     
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                signed char* ptr = top_blob.channel(q);

                int remain = size;
                for (; remain > 0; remain--)
                {
                    *ptr = float2int8(*intptr * scale_fuse);

                    intptr++;
                    ptr ++;
                }
            }
        }    
    }

    return 0;
}

} // namespace ncnn
