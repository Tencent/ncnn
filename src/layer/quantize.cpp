// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "quantize.h"

namespace ncnn {

Quantize::Quantize()
{
    one_blob_only = true;
    support_inplace = false;
}

int Quantize::load_param(const ParamDict& pd)
{
    scale_data_size = pd.get(0, 1);

    return 0;
}

int Quantize::load_model(const ModelBin& mb)
{
    scale_data = mb.load(scale_data_size, 1);
    if (scale_data.empty())
        return -100;

    return 0;
}

static inline signed char float2int8(float v)
{
    int int32 = static_cast<int>(round(v));
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

static void quantize(const float* ptr, signed char* s8ptr, float scale, int size)
{
    for (int i = 0; i < size; i++)
    {
        *s8ptr = float2int8(*ptr * scale);
        ptr++;
        s8ptr++;
    }
}

int Quantize::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;

    if (dims == 1)
    {
        top_blob.create(w, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        // assert scale_data_size == 1

        const float* ptr = bottom_blob;
        signed char* s8ptr = top_blob;

        const float scale = scale_data[0];

        quantize(ptr, s8ptr, scale, w);
    }

    if (dims == 2)
    {
        top_blob.create(w, h, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            const float* ptr = bottom_blob.row(i);
            signed char* s8ptr = top_blob.row<signed char>(i);

            const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[i];

            quantize(ptr, s8ptr, scale, w);
        }
    }

    if (dims == 3)
    {
        top_blob.create(w, h, channels, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            signed char* s8ptr = top_blob.channel(q);

            const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];

            quantize(ptr, s8ptr, scale, w * h);
        }
    }

    return 0;
}

} // namespace ncnn
