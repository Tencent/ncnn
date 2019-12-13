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

#include "reshape_arm.h"

#include "layer_type.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

DEFINE_LAYER_CREATOR(Reshape_arm)

Reshape_arm::Reshape_arm()
{
#if __ARM_NEON
    support_packing = true;
#endif // __ARM_NEON

    flatten = 0;
}

int Reshape_arm::create_pipeline(const Option& opt)
{
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        flatten = ncnn::create_layer(ncnn::LayerType::Flatten);

        ncnn::ParamDict pd;

        flatten->load_param(pd);

        flatten->create_pipeline(opt);
    }
#endif // __ARM_NEON

    return 0;
}

int Reshape_arm::destroy_pipeline(const Option& opt)
{
    if (flatten)
    {
        flatten->destroy_pipeline(opt);
        delete flatten;
        flatten = 0;
    }

    return 0;
}

int Reshape_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if __ARM_NEON
    if (opt.use_packing_layout)
    {

    if (ndim == 1)
    {
        // flatten
        return flatten->forward(bottom_blob, top_blob, opt);
    }

    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int total = bottom_blob.w * bottom_blob.h * bottom_blob.c * elempack;

    if (ndim == 2)
    {
        int _w = w;
        int _h = h;

        if (_w == 0)
            _w = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (_h == 0)
            _h = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;

        if (_w == -1)
            _w = total / _h;
        if (_h == -1)
            _h = total / _w;

        int out_elempack = _h % 4 == 0 ? 4 : 1;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 2 && bottom_blob.h == _h && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        if (out_elempack == 1)
        {
            // flatten
            flatten->forward(bottom_blob, top_blob, opt);

            top_blob.dims = 2;
            top_blob.w = _w;
            top_blob.h = _h;
            top_blob.cstep = _w * _h;
            top_blob.elemsize = out_elemsize;
            top_blob.elempack = out_elempack;

            return 0;
        }

        // flatten
        Mat bottom_blob_flattened = bottom_blob;
        {
            Option opt_flatten = opt;
            opt_flatten.blob_allocator = opt.workspace_allocator;

            flatten->forward(bottom_blob, bottom_blob_flattened, opt_flatten);
        }

        top_blob.create(_w, _h / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int outw = top_blob.w;
        int outh = top_blob.h;

        // assert out_elempack == 4

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i=0; i<outh; i++)
        {
            const float* ptr0 = (const float*)bottom_blob_flattened + outw * i*4;
            const float* ptr1 = (const float*)bottom_blob_flattened + outw * (i*4 + 1);
            const float* ptr2 = (const float*)bottom_blob_flattened + outw * (i*4 + 2);
            const float* ptr3 = (const float*)bottom_blob_flattened + outw * (i*4 + 3);
            float* outptr = (float*)top_blob.row(i);

            int j=0;
            for (; j+3<outw; j+=4)
            {
                float32x4x4_t _v4;
                _v4.val[0] = vld1q_f32(ptr0);
                _v4.val[1] = vld1q_f32(ptr1);
                _v4.val[2] = vld1q_f32(ptr2);
                _v4.val[3] = vld1q_f32(ptr3);

                vst4q_f32(outptr, _v4);

                ptr0 += 4;
                ptr1 += 4;
                ptr2 += 4;
                ptr3 += 4;
                outptr += 16;
            }
            for (; j<outw; j++)
            {
                outptr[0] = *ptr0++;
                outptr[1] = *ptr1++;
                outptr[2] = *ptr2++;
                outptr[3] = *ptr3++;

                outptr += 4;
            }
        }

        return 0;
    }

    if (ndim == 3)
    {
        int _w = w;
        int _h = h;
        int _c = c;

        if (_w == 0)
            _w = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (_h == 0)
            _h = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
        if (_c == 0)
            _c = dims == 3 ? bottom_blob.c * elempack : bottom_blob.c;

        if (_w == -1)
            _w = total / _c / _h;
        if (_h == -1)
            _h = total / _c / _w;
        if (_c == -1)
            _c = total / _h / _w;

        int out_elempack = _c % 4 == 0 ? 4 : 1;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 3 && bottom_blob.c == _c && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            top_blob.w = _w;
            top_blob.h = _h;
            return 0;
        }

        // flatten
        Mat bottom_blob_flattened = bottom_blob;
        {
            Option opt_flatten = opt;
            opt_flatten.blob_allocator = opt.workspace_allocator;

            flatten->forward(bottom_blob, bottom_blob_flattened, opt_flatten);
        }

        top_blob.create(_w, _h, _c / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int size = top_blob.w * top_blob.h;

        if (out_elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<top_blob.c; q++)
            {
                const float* ptr0 = (const float*)bottom_blob_flattened + size * q*4;
                const float* ptr1 = (const float*)bottom_blob_flattened + size * (q*4 + 1);
                const float* ptr2 = (const float*)bottom_blob_flattened + size * (q*4 + 2);
                const float* ptr3 = (const float*)bottom_blob_flattened + size * (q*4 + 3);
                float* outptr = top_blob.channel(q);

                int i=0;
                for (; i+3<size; i+=4)
                {
                    float32x4x4_t _v4;
                    _v4.val[0] = vld1q_f32(ptr0);
                    _v4.val[1] = vld1q_f32(ptr1);
                    _v4.val[2] = vld1q_f32(ptr2);
                    _v4.val[3] = vld1q_f32(ptr3);

                    vst4q_f32(outptr, _v4);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
                for (; i<size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }

            return 0;
        }

        if (out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<top_blob.c; q++)
            {
                const float* ptr = (const float*)bottom_blob_flattened + size * q;
                float* outptr = top_blob.channel(q);

                int i=0;
                for (; i+3<size; i+=4)
                {
                    float32x4_t _v = vld1q_f32(ptr);
                    vst1q_f32(outptr, _v);
                    ptr += 4;
                    outptr += 4;
                }
                for (; i<size; i++)
                {
                    *outptr++ = *ptr++;
                }
            }

            return 0;
        }
    }

    return 0;

    } // opt.use_packing_layout
#endif // __ARM_NEON

    return Reshape::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
