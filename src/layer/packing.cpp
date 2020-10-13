// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "packing.h"

namespace ncnn {

Packing::Packing()
{
    one_blob_only = true;
    support_inplace = false;
}

int Packing::load_param(const ParamDict& pd)
{
    out_elempack = pd.get(0, 1);
    use_padding = pd.get(1, 0);

    cast_type_from = pd.get(2, 0);
    cast_type_to = pd.get(3, 0);

    storage_type_from = pd.get(4, 0);
    storage_type_to = pd.get(5, 0);

    return 0;
}

int Packing::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elempack = bottom_blob.elempack;

    if (elempack == out_elempack)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;

    if (!use_padding)
    {
        // identity if use_padding not allowed
        if (dims == 1 && w * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if (dims == 2 && h * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if (dims == 3 && channels * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
    }

    if (dims == 1)
    {
        if (out_elempack == 1)
        {
            top_blob = bottom_blob;
            top_blob.w = w * elempack;
            top_blob.cstep = (size_t)w * elempack;
            top_blob.elemsize = elemsize / elempack;
            top_blob.elempack = out_elempack;
            return 0;
        }

        int outw = (w * elempack + out_elempack - 1) / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        top_blob.create(outw, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        memcpy(top_blob.data, bottom_blob.data, w * elemsize);

        return 0;
    }

    if (dims == 2)
    {
        int outh = (h * elempack + out_elempack - 1) / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;
        size_t lane_size = out_elemsize / out_elempack;

        top_blob.create(w, outh, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < outh; i++)
        {
            unsigned char* outptr = (unsigned char*)top_blob + (size_t)i * w * out_elemsize;

            for (int j = 0; j < w; j++)
            {
                unsigned char* out_elem_ptr = outptr + j * out_elemsize;

                for (int k = 0; k < out_elempack; k++)
                {
                    int srcy = (i * out_elempack + k) / elempack;
                    if (srcy >= h)
                        break;

                    int srck = (i * out_elempack + k) % elempack;

                    const unsigned char* ptr = (const unsigned char*)bottom_blob + (size_t)srcy * w * elemsize;
                    const unsigned char* elem_ptr = ptr + j * elemsize;

                    memcpy(out_elem_ptr + k * lane_size, elem_ptr + srck * lane_size, lane_size);
                }
            }
        }

        return 0;
    }

    if (dims == 3)
    {
        int outc = (channels * elempack + out_elempack - 1) / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;
        size_t lane_size = out_elemsize / out_elempack;

        top_blob.create(w, h, outc, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < outc; q++)
        {
            Mat out = top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                unsigned char* outptr = (unsigned char*)out + (size_t)i * w * out_elemsize;

                for (int j = 0; j < w; j++)
                {
                    unsigned char* out_elem_ptr = outptr + j * out_elemsize;

                    for (int k = 0; k < out_elempack; k++)
                    {
                        int srcq = (q * out_elempack + k) / elempack;
                        if (srcq >= channels)
                            break;

                        int srck = (q * out_elempack + k) % elempack;

                        const Mat m = bottom_blob.channel(srcq);
                        const unsigned char* ptr = (const unsigned char*)m + (size_t)i * w * elemsize;
                        const unsigned char* elem_ptr = ptr + j * elemsize;

                        memcpy(out_elem_ptr + k * lane_size, elem_ptr + srck * lane_size, lane_size);
                    }
                }
            }
        }

        return 0;
    }

    return 0;
}

} // namespace ncnn
