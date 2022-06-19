// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "softmax_x86.h"

#include <float.h>
#include <math.h>

#if __SSE2__
#include <emmintrin.h>
#include "sse_mathfun.h"
#if __AVX__
#include <immintrin.h>
#include "avx_mathfun.h"
#if __AVX512F__
#include "avx512_mathfun.h"
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

#include "x86_usability.h"

namespace ncnn {

Softmax_x86::Softmax_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

int Softmax_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    size_t elemsize = bottom_top_blob.elemsize;
    int elempack = bottom_top_blob.elempack;
    int positive_axis = axis < 0 ? dims + axis : axis;

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        if (dims == 1) // positive_axis == 0
        {
            int w = bottom_top_blob.w;

            float* ptr = bottom_top_blob;

            __m512 _max = _mm512_set1_ps(-FLT_MAX);
            for (int i = 0; i < w; i++)
            {
                __m512 _p = _mm512_load_ps(ptr + i * 16);
                _max = _mm512_max_ps(_max, _p);
            }
            _max = _mm512_max_ps(_max, _mm512_permute_ps(_max, _MM_SHUFFLE(2, 3, 0, 1)));
            _max = _mm512_max_ps(_max, _mm512_permute_ps(_max, _MM_SHUFFLE(1, 0, 3, 2)));
            _max = _mm512_max_ps(_max, _mm512_shuffle_f32x4(_max, _max, _MM_SHUFFLE(2, 3, 0, 1)));
            _max = _mm512_max_ps(_max, _mm512_shuffle_f32x4(_max, _max, _MM_SHUFFLE(1, 0, 3, 2)));

            __m512 _sum = _mm512_setzero_ps();
            for (int i = 0; i < w; i++)
            {
                __m512 _p = _mm512_load_ps(ptr + i * 16);
                _p = exp512_ps(_mm512_sub_ps(_p, _max));
                _mm512_store_ps(ptr + i * 16, _p);
                _sum = _mm512_add_ps(_sum, _p);
            }
            _sum = _mm512_add_ps(_sum, _mm512_permute_ps(_sum, _MM_SHUFFLE(2, 3, 0, 1)));
            _sum = _mm512_add_ps(_sum, _mm512_permute_ps(_sum, _MM_SHUFFLE(1, 0, 3, 2)));
            _sum = _mm512_add_ps(_sum, _mm512_shuffle_f32x4(_sum, _sum, _MM_SHUFFLE(2, 3, 0, 1)));
            _sum = _mm512_add_ps(_sum, _mm512_shuffle_f32x4(_sum, _sum, _MM_SHUFFLE(1, 0, 3, 2)));

            for (int i = 0; i < w; i++)
            {
                __m512 _p = _mm512_load_ps(ptr + i * 16);
                _p = _mm512_div_ps(_p, _sum);
                _mm512_store_ps(ptr + i * 16, _p);
            }
        }

        if (dims == 2 && positive_axis == 0)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;

            Mat max;
            max.create(w, 4u, 1, opt.workspace_allocator);
            if (max.empty())
                return -100;
            max.fill(-FLT_MAX);

            for (int i = 0; i < h; i++)
            {
                const float* ptr = bottom_top_blob.row(i);
                float* pmax = max;

                int j = 0;
                for (; j + 15 < w; j += 16)
                {
                    __m512 _p0 = _mm512_load_ps(ptr);
                    __m512 _p1 = _mm512_load_ps(ptr + 16);
                    __m512 _p2 = _mm512_load_ps(ptr + 16 * 2);
                    __m512 _p3 = _mm512_load_ps(ptr + 16 * 3);
                    __m512 _p4 = _mm512_load_ps(ptr + 16 * 4);
                    __m512 _p5 = _mm512_load_ps(ptr + 16 * 5);
                    __m512 _p6 = _mm512_load_ps(ptr + 16 * 6);
                    __m512 _p7 = _mm512_load_ps(ptr + 16 * 7);
                    __m512 _p8 = _mm512_load_ps(ptr + 16 * 8);
                    __m512 _p9 = _mm512_load_ps(ptr + 16 * 9);
                    __m512 _pa = _mm512_load_ps(ptr + 16 * 10);
                    __m512 _pb = _mm512_load_ps(ptr + 16 * 11);
                    __m512 _pc = _mm512_load_ps(ptr + 16 * 12);
                    __m512 _pd = _mm512_load_ps(ptr + 16 * 13);
                    __m512 _pe = _mm512_load_ps(ptr + 16 * 14);
                    __m512 _pf = _mm512_load_ps(ptr + 16 * 15);
                    transpose16_ps(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7, _p8, _p9, _pa, _pb, _pc, _pd, _pe, _pf);
                    __m512 _max01 = _mm512_max_ps(_p0, _p1);
                    __m512 _max23 = _mm512_max_ps(_p2, _p3);
                    __m512 _max45 = _mm512_max_ps(_p4, _p5);
                    __m512 _max67 = _mm512_max_ps(_p6, _p7);
                    __m512 _max89 = _mm512_max_ps(_p8, _p9);
                    __m512 _maxab = _mm512_max_ps(_pa, _pb);
                    __m512 _maxcd = _mm512_max_ps(_pc, _pd);
                    __m512 _maxef = _mm512_max_ps(_pe, _pf);
                    __m512 _max0123 = _mm512_max_ps(_max01, _max23);
                    __m512 _max4567 = _mm512_max_ps(_max45, _max67);
                    __m512 _max89ab = _mm512_max_ps(_max89, _maxab);
                    __m512 _maxcdef = _mm512_max_ps(_maxcd, _maxef);
                    __m512 _max01234567 = _mm512_max_ps(_max0123, _max4567);
                    __m512 _max89abcdef = _mm512_max_ps(_max89ab, _maxcdef);
                    __m512 _max = _mm512_load_ps(pmax);
                    _max = _mm512_max_ps(_max, _mm512_max_ps(_max01234567, _max89abcdef));
                    _mm512_store_ps(pmax, _max);

                    ptr += 256;
                    pmax += 16;
                }
                for (; j < w; j++)
                {
                    __m512 _p = _mm512_load_ps(ptr);
                    *pmax = std::max(*pmax, _mm512_comp_reduce_max_ps(_p));

                    ptr += 16;
                    pmax++;
                }
            }

            Mat sum;
            sum.create(w, 4u, 1, opt.workspace_allocator);
            if (sum.empty())
                return -100;
            sum.fill(0.f);

            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);
                float* psum = sum;

                int j = 0;
                for (; j + 15 < w; j += 16)
                {
                    __m512 _p0 = _mm512_load_ps(ptr);
                    __m512 _p1 = _mm512_load_ps(ptr + 16);
                    __m512 _p2 = _mm512_load_ps(ptr + 16 * 2);
                    __m512 _p3 = _mm512_load_ps(ptr + 16 * 3);
                    __m512 _p4 = _mm512_load_ps(ptr + 16 * 4);
                    __m512 _p5 = _mm512_load_ps(ptr + 16 * 5);
                    __m512 _p6 = _mm512_load_ps(ptr + 16 * 6);
                    __m512 _p7 = _mm512_load_ps(ptr + 16 * 7);
                    __m512 _p8 = _mm512_load_ps(ptr + 16 * 8);
                    __m512 _p9 = _mm512_load_ps(ptr + 16 * 9);
                    __m512 _pa = _mm512_load_ps(ptr + 16 * 10);
                    __m512 _pb = _mm512_load_ps(ptr + 16 * 11);
                    __m512 _pc = _mm512_load_ps(ptr + 16 * 12);
                    __m512 _pd = _mm512_load_ps(ptr + 16 * 13);
                    __m512 _pe = _mm512_load_ps(ptr + 16 * 14);
                    __m512 _pf = _mm512_load_ps(ptr + 16 * 15);
                    _p0 = exp512_ps(_mm512_sub_ps(_p0, _mm512_set1_ps(max[j])));
                    _p1 = exp512_ps(_mm512_sub_ps(_p1, _mm512_set1_ps(max[j + 1])));
                    _p2 = exp512_ps(_mm512_sub_ps(_p2, _mm512_set1_ps(max[j + 2])));
                    _p3 = exp512_ps(_mm512_sub_ps(_p3, _mm512_set1_ps(max[j + 3])));
                    _p4 = exp512_ps(_mm512_sub_ps(_p4, _mm512_set1_ps(max[j + 4])));
                    _p5 = exp512_ps(_mm512_sub_ps(_p5, _mm512_set1_ps(max[j + 5])));
                    _p6 = exp512_ps(_mm512_sub_ps(_p6, _mm512_set1_ps(max[j + 6])));
                    _p7 = exp512_ps(_mm512_sub_ps(_p7, _mm512_set1_ps(max[j + 7])));
                    _p8 = exp512_ps(_mm512_sub_ps(_p8, _mm512_set1_ps(max[j + 8])));
                    _p9 = exp512_ps(_mm512_sub_ps(_p9, _mm512_set1_ps(max[j + 9])));
                    _pa = exp512_ps(_mm512_sub_ps(_pa, _mm512_set1_ps(max[j + 10])));
                    _pb = exp512_ps(_mm512_sub_ps(_pb, _mm512_set1_ps(max[j + 11])));
                    _pc = exp512_ps(_mm512_sub_ps(_pc, _mm512_set1_ps(max[j + 12])));
                    _pd = exp512_ps(_mm512_sub_ps(_pd, _mm512_set1_ps(max[j + 13])));
                    _pe = exp512_ps(_mm512_sub_ps(_pe, _mm512_set1_ps(max[j + 14])));
                    _pf = exp512_ps(_mm512_sub_ps(_pf, _mm512_set1_ps(max[j + 15])));
                    _mm512_store_ps(ptr, _p0);
                    _mm512_store_ps(ptr + 16, _p1);
                    _mm512_store_ps(ptr + 16 * 2, _p2);
                    _mm512_store_ps(ptr + 16 * 3, _p3);
                    _mm512_store_ps(ptr + 16 * 4, _p4);
                    _mm512_store_ps(ptr + 16 * 5, _p5);
                    _mm512_store_ps(ptr + 16 * 6, _p6);
                    _mm512_store_ps(ptr + 16 * 7, _p7);
                    _mm512_store_ps(ptr + 16 * 8, _p8);
                    _mm512_store_ps(ptr + 16 * 9, _p9);
                    _mm512_store_ps(ptr + 16 * 10, _pa);
                    _mm512_store_ps(ptr + 16 * 11, _pb);
                    _mm512_store_ps(ptr + 16 * 12, _pc);
                    _mm512_store_ps(ptr + 16 * 13, _pd);
                    _mm512_store_ps(ptr + 16 * 14, _pe);
                    _mm512_store_ps(ptr + 16 * 15, _pf);
                    transpose16_ps(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7, _p8, _p9, _pa, _pb, _pc, _pd, _pe, _pf);
                    __m512 _sum01 = _mm512_add_ps(_p0, _p1);
                    __m512 _sum23 = _mm512_add_ps(_p2, _p3);
                    __m512 _sum45 = _mm512_add_ps(_p4, _p5);
                    __m512 _sum67 = _mm512_add_ps(_p6, _p7);
                    __m512 _sum89 = _mm512_add_ps(_p8, _p9);
                    __m512 _sumab = _mm512_add_ps(_pa, _pb);
                    __m512 _sumcd = _mm512_add_ps(_pc, _pd);
                    __m512 _sumef = _mm512_add_ps(_pe, _pf);
                    __m512 _sum0123 = _mm512_add_ps(_sum01, _sum23);
                    __m512 _sum4567 = _mm512_add_ps(_sum45, _sum67);
                    __m512 _sum89ab = _mm512_add_ps(_sum89, _sumab);
                    __m512 _sumcdef = _mm512_add_ps(_sumcd, _sumef);
                    __m512 _sum01234567 = _mm512_add_ps(_sum0123, _sum4567);
                    __m512 _sum89abcdef = _mm512_add_ps(_sum89ab, _sumcdef);
                    __m512 _sum = _mm512_load_ps(psum);
                    _sum = _mm512_add_ps(_sum, _mm512_add_ps(_sum01234567, _sum89abcdef));
                    _mm512_store_ps(psum, _sum);

                    ptr += 256;
                    psum += 16;
                }
                for (; j < w; j++)
                {
                    __m512 _p = _mm512_load_ps(ptr);
                    __m512 _max = _mm512_set1_ps(max[j]);
                    _p = exp512_ps(_mm512_sub_ps(_p, _max));
                    _mm512_store_ps(ptr, _p);
                    *psum += _mm512_comp_reduce_add_ps(_p);

                    ptr += 16;
                    psum++;
                }
            }

            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);
                for (int j = 0; j < w; j++)
                {
                    __m512 _p = _mm512_load_ps(ptr);
                    __m512 _sum = _mm512_set1_ps(sum[j]);
                    _p = _mm512_div_ps(_p, _sum);
                    _mm512_store_ps(ptr, _p);

                    ptr += 16;
                }
            }
        }

        if (dims == 2 && positive_axis == 1)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);

                __m512 _max = _mm512_set1_ps(-FLT_MAX);
                for (int j = 0; j < w; j++)
                {
                    __m512 _p = _mm512_load_ps(ptr + j * 16);
                    _max = _mm512_max_ps(_max, _p);
                }

                __m512 _sum = _mm512_setzero_ps();
                for (int j = 0; j < w; j++)
                {
                    __m512 _p = _mm512_load_ps(ptr + j * 16);
                    _p = exp512_ps(_mm512_sub_ps(_p, _max));
                    _mm512_store_ps(ptr + j * 16, _p);
                    _sum = _mm512_add_ps(_sum, _p);
                }

                for (int j = 0; j < w; j++)
                {
                    __m512 _p = _mm512_load_ps(ptr + j * 16);
                    _p = _mm512_div_ps(_p, _sum);
                    _mm512_store_ps(ptr + j * 16, _p);
                }
            }
        }

        if (dims == 3 && positive_axis == 0)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;
            int size = w * h;

            Mat max;
            max.create(w, h, 4u, 1, opt.workspace_allocator);
            if (max.empty())
                return -100;
            max.fill(-FLT_MAX);

            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_top_blob.channel(q);
                float* pmax = max;

                int i = 0;
                for (; i + 15 < size; i += 16)
                {
                    __m512 _p0 = _mm512_load_ps(ptr);
                    __m512 _p1 = _mm512_load_ps(ptr + 16);
                    __m512 _p2 = _mm512_load_ps(ptr + 16 * 2);
                    __m512 _p3 = _mm512_load_ps(ptr + 16 * 3);
                    __m512 _p4 = _mm512_load_ps(ptr + 16 * 4);
                    __m512 _p5 = _mm512_load_ps(ptr + 16 * 5);
                    __m512 _p6 = _mm512_load_ps(ptr + 16 * 6);
                    __m512 _p7 = _mm512_load_ps(ptr + 16 * 7);
                    __m512 _p8 = _mm512_load_ps(ptr + 16 * 8);
                    __m512 _p9 = _mm512_load_ps(ptr + 16 * 9);
                    __m512 _pa = _mm512_load_ps(ptr + 16 * 10);
                    __m512 _pb = _mm512_load_ps(ptr + 16 * 11);
                    __m512 _pc = _mm512_load_ps(ptr + 16 * 12);
                    __m512 _pd = _mm512_load_ps(ptr + 16 * 13);
                    __m512 _pe = _mm512_load_ps(ptr + 16 * 14);
                    __m512 _pf = _mm512_load_ps(ptr + 16 * 15);
                    transpose16_ps(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7, _p8, _p9, _pa, _pb, _pc, _pd, _pe, _pf);
                    __m512 _max01 = _mm512_max_ps(_p0, _p1);
                    __m512 _max23 = _mm512_max_ps(_p2, _p3);
                    __m512 _max45 = _mm512_max_ps(_p4, _p5);
                    __m512 _max67 = _mm512_max_ps(_p6, _p7);
                    __m512 _max89 = _mm512_max_ps(_p8, _p9);
                    __m512 _maxab = _mm512_max_ps(_pa, _pb);
                    __m512 _maxcd = _mm512_max_ps(_pc, _pd);
                    __m512 _maxef = _mm512_max_ps(_pe, _pf);
                    __m512 _max0123 = _mm512_max_ps(_max01, _max23);
                    __m512 _max4567 = _mm512_max_ps(_max45, _max67);
                    __m512 _max89ab = _mm512_max_ps(_max89, _maxab);
                    __m512 _maxcdef = _mm512_max_ps(_maxcd, _maxef);
                    __m512 _max01234567 = _mm512_max_ps(_max0123, _max4567);
                    __m512 _max89abcdef = _mm512_max_ps(_max89ab, _maxcdef);
                    __m512 _max = _mm512_load_ps(pmax);
                    _max = _mm512_max_ps(_max, _mm512_max_ps(_max01234567, _max89abcdef));
                    _mm512_store_ps(pmax, _max);

                    ptr += 256;
                    pmax += 16;
                }
                for (; i < size; i++)
                {
                    __m512 _p = _mm512_load_ps(ptr);
                    *pmax = std::max(*pmax, _mm512_comp_reduce_max_ps(_p));

                    ptr += 16;
                    pmax++;
                }
            }

            Mat sum;
            sum.create(w, h, 4u, 1, opt.workspace_allocator);
            if (sum.empty())
                return -100;
            sum.fill(0.f);

            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                float* psum = sum;

                int i = 0;
                for (; i + 15 < size; i += 16)
                {
                    __m512 _p0 = _mm512_load_ps(ptr);
                    __m512 _p1 = _mm512_load_ps(ptr + 16);
                    __m512 _p2 = _mm512_load_ps(ptr + 16 * 2);
                    __m512 _p3 = _mm512_load_ps(ptr + 16 * 3);
                    __m512 _p4 = _mm512_load_ps(ptr + 16 * 4);
                    __m512 _p5 = _mm512_load_ps(ptr + 16 * 5);
                    __m512 _p6 = _mm512_load_ps(ptr + 16 * 6);
                    __m512 _p7 = _mm512_load_ps(ptr + 16 * 7);
                    __m512 _p8 = _mm512_load_ps(ptr + 16 * 8);
                    __m512 _p9 = _mm512_load_ps(ptr + 16 * 9);
                    __m512 _pa = _mm512_load_ps(ptr + 16 * 10);
                    __m512 _pb = _mm512_load_ps(ptr + 16 * 11);
                    __m512 _pc = _mm512_load_ps(ptr + 16 * 12);
                    __m512 _pd = _mm512_load_ps(ptr + 16 * 13);
                    __m512 _pe = _mm512_load_ps(ptr + 16 * 14);
                    __m512 _pf = _mm512_load_ps(ptr + 16 * 15);
                    _p0 = exp512_ps(_mm512_sub_ps(_p0, _mm512_set1_ps(max[i])));
                    _p1 = exp512_ps(_mm512_sub_ps(_p1, _mm512_set1_ps(max[i + 1])));
                    _p2 = exp512_ps(_mm512_sub_ps(_p2, _mm512_set1_ps(max[i + 2])));
                    _p3 = exp512_ps(_mm512_sub_ps(_p3, _mm512_set1_ps(max[i + 3])));
                    _p4 = exp512_ps(_mm512_sub_ps(_p4, _mm512_set1_ps(max[i + 4])));
                    _p5 = exp512_ps(_mm512_sub_ps(_p5, _mm512_set1_ps(max[i + 5])));
                    _p6 = exp512_ps(_mm512_sub_ps(_p6, _mm512_set1_ps(max[i + 6])));
                    _p7 = exp512_ps(_mm512_sub_ps(_p7, _mm512_set1_ps(max[i + 7])));
                    _p8 = exp512_ps(_mm512_sub_ps(_p8, _mm512_set1_ps(max[i + 8])));
                    _p9 = exp512_ps(_mm512_sub_ps(_p9, _mm512_set1_ps(max[i + 9])));
                    _pa = exp512_ps(_mm512_sub_ps(_pa, _mm512_set1_ps(max[i + 10])));
                    _pb = exp512_ps(_mm512_sub_ps(_pb, _mm512_set1_ps(max[i + 11])));
                    _pc = exp512_ps(_mm512_sub_ps(_pc, _mm512_set1_ps(max[i + 12])));
                    _pd = exp512_ps(_mm512_sub_ps(_pd, _mm512_set1_ps(max[i + 13])));
                    _pe = exp512_ps(_mm512_sub_ps(_pe, _mm512_set1_ps(max[i + 14])));
                    _pf = exp512_ps(_mm512_sub_ps(_pf, _mm512_set1_ps(max[i + 15])));
                    _mm512_store_ps(ptr, _p0);
                    _mm512_store_ps(ptr + 16, _p1);
                    _mm512_store_ps(ptr + 16 * 2, _p2);
                    _mm512_store_ps(ptr + 16 * 3, _p3);
                    _mm512_store_ps(ptr + 16 * 4, _p4);
                    _mm512_store_ps(ptr + 16 * 5, _p5);
                    _mm512_store_ps(ptr + 16 * 6, _p6);
                    _mm512_store_ps(ptr + 16 * 7, _p7);
                    _mm512_store_ps(ptr + 16 * 8, _p8);
                    _mm512_store_ps(ptr + 16 * 9, _p9);
                    _mm512_store_ps(ptr + 16 * 10, _pa);
                    _mm512_store_ps(ptr + 16 * 11, _pb);
                    _mm512_store_ps(ptr + 16 * 12, _pc);
                    _mm512_store_ps(ptr + 16 * 13, _pd);
                    _mm512_store_ps(ptr + 16 * 14, _pe);
                    _mm512_store_ps(ptr + 16 * 15, _pf);
                    transpose16_ps(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7, _p8, _p9, _pa, _pb, _pc, _pd, _pe, _pf);
                    __m512 _sum01 = _mm512_add_ps(_p0, _p1);
                    __m512 _sum23 = _mm512_add_ps(_p2, _p3);
                    __m512 _sum45 = _mm512_add_ps(_p4, _p5);
                    __m512 _sum67 = _mm512_add_ps(_p6, _p7);
                    __m512 _sum89 = _mm512_add_ps(_p8, _p9);
                    __m512 _sumab = _mm512_add_ps(_pa, _pb);
                    __m512 _sumcd = _mm512_add_ps(_pc, _pd);
                    __m512 _sumef = _mm512_add_ps(_pe, _pf);
                    __m512 _sum0123 = _mm512_add_ps(_sum01, _sum23);
                    __m512 _sum4567 = _mm512_add_ps(_sum45, _sum67);
                    __m512 _sum89ab = _mm512_add_ps(_sum89, _sumab);
                    __m512 _sumcdef = _mm512_add_ps(_sumcd, _sumef);
                    __m512 _sum01234567 = _mm512_add_ps(_sum0123, _sum4567);
                    __m512 _sum89abcdef = _mm512_add_ps(_sum89ab, _sumcdef);
                    __m512 _sum = _mm512_load_ps(psum);
                    _sum = _mm512_add_ps(_sum, _mm512_add_ps(_sum01234567, _sum89abcdef));
                    _mm512_store_ps(psum, _sum);

                    ptr += 256;
                    psum += 16;
                }
                for (; i < size; i++)
                {
                    __m512 _p = _mm512_load_ps(ptr);
                    __m512 _max = _mm512_set1_ps(max[i]);
                    _p = exp512_ps(_mm512_sub_ps(_p, _max));
                    _mm512_store_ps(ptr, _p);
                    *psum += _mm512_comp_reduce_add_ps(_p);

                    ptr += 16;
                    psum++;
                }
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    __m512 _p = _mm512_load_ps(ptr);
                    __m512 _sum = _mm512_set1_ps(sum[i]);
                    _p = _mm512_div_ps(_p, _sum);
                    _mm512_store_ps(ptr, _p);

                    ptr += 16;
                }
            }
        }

        if (dims == 3 && positive_axis == 1)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;

            Mat max;
            max.create(w, channels, elemsize, elempack, opt.workspace_allocator);
            if (max.empty())
                return -100;
            max.fill(_mm512_set1_ps(-FLT_MAX));

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < h; i++)
                {
                    float* maxptr = max.row(q);

                    for (int j = 0; j < w; j++)
                    {
                        __m512 _p = _mm512_load_ps(ptr);
                        __m512 _max = _mm512_load_ps(maxptr);
                        _max = _mm512_max_ps(_max, _p);
                        _mm512_store_ps(maxptr, _max);

                        ptr += 16;
                        maxptr += 16;
                    }
                }
            }

            Mat sum;
            sum.create(w, channels, elemsize, elempack, opt.workspace_allocator);
            if (sum.empty())
                return -100;
            sum.fill(_mm512_setzero_ps());

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < h; i++)
                {
                    float* maxptr = max.row(q);
                    float* sumptr = sum.row(q);

                    for (int j = 0; j < w; j++)
                    {
                        __m512 _p = _mm512_load_ps(ptr);
                        __m512 _max = _mm512_load_ps(maxptr);
                        _p = exp512_ps(_mm512_sub_ps(_p, _max));
                        _mm512_store_ps(ptr, _p);
                        __m512 _sum = _mm512_load_ps(sumptr);
                        _sum = _mm512_add_ps(_sum, _p);
                        _mm512_store_ps(sumptr, _sum);

                        ptr += 16;
                        maxptr += 16;
                        sumptr += 16;
                    }
                }
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < h; i++)
                {
                    float* sumptr = sum.row(q);

                    for (int j = 0; j < w; j++)
                    {
                        __m512 _p = _mm512_load_ps(ptr);
                        __m512 _sum = _mm512_load_ps(sumptr);
                        _p = _mm512_div_ps(_p, _sum);
                        _mm512_store_ps(ptr, _p);

                        ptr += 16;
                        sumptr += 16;
                    }
                }
            }
        }

        if (dims == 3 && positive_axis == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < h; i++)
                {
                    __m512 _max = _mm512_set1_ps(-FLT_MAX);
                    for (int j = 0; j < w; j++)
                    {
                        __m512 _p = _mm512_load_ps(ptr + j * 16);
                        _max = _mm512_max_ps(_max, _p);
                    }

                    __m512 _sum = _mm512_setzero_ps();
                    for (int j = 0; j < w; j++)
                    {
                        __m512 _p = _mm512_load_ps(ptr + j * 16);
                        _p = exp512_ps(_mm512_sub_ps(_p, _max));
                        _mm512_store_ps(ptr + j * 16, _p);
                        _sum = _mm512_add_ps(_sum, _p);
                    }

                    for (int j = 0; j < w; j++)
                    {
                        __m512 _p = _mm512_load_ps(ptr + j * 16);
                        _p = _mm512_div_ps(_p, _sum);
                        _mm512_store_ps(ptr + j * 16, _p);
                    }

                    ptr += w * 16;
                }
            }
        }

        return 0;
    }
#endif // __AVX512F__

    if (elempack == 8)
    {
        if (dims == 1) // positive_axis == 0
        {
            int w = bottom_top_blob.w;

            float* ptr = bottom_top_blob;

            __m256 _max = _mm256_set1_ps(-FLT_MAX);
            for (int i = 0; i < w; i++)
            {
                __m256 _p = _mm256_load_ps(ptr + i * 8);
                _max = _mm256_max_ps(_max, _p);
            }
            _max = _mm256_max_ps(_max, _mm256_permute_ps(_max, _MM_SHUFFLE(2, 3, 0, 1)));
            _max = _mm256_max_ps(_max, _mm256_permute_ps(_max, _MM_SHUFFLE(1, 0, 3, 2)));
            _max = _mm256_max_ps(_max, _mm256_permute2f128_ps(_max, _max, _MM_SHUFFLE(0, 0, 0, 1)));

            __m256 _sum = _mm256_setzero_ps();
            for (int i = 0; i < w; i++)
            {
                __m256 _p = _mm256_load_ps(ptr + i * 8);
                _p = exp256_ps(_mm256_sub_ps(_p, _max));
                _mm256_store_ps(ptr + i * 8, _p);
                _sum = _mm256_add_ps(_sum, _p);
            }
            _sum = _mm256_add_ps(_sum, _mm256_permute_ps(_sum, _MM_SHUFFLE(2, 3, 0, 1)));
            _sum = _mm256_add_ps(_sum, _mm256_permute_ps(_sum, _MM_SHUFFLE(1, 0, 3, 2)));
            _sum = _mm256_add_ps(_sum, _mm256_permute2f128_ps(_sum, _sum, _MM_SHUFFLE(0, 0, 0, 1)));

            for (int i = 0; i < w; i++)
            {
                __m256 _p = _mm256_load_ps(ptr + i * 8);
                _p = _mm256_div_ps(_p, _sum);
                _mm256_store_ps(ptr + i * 8, _p);
            }
        }

        if (dims == 2 && positive_axis == 0)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;

            Mat max;
            max.create(w, 4u, 1, opt.workspace_allocator);
            if (max.empty())
                return -100;
            max.fill(-FLT_MAX);

            for (int i = 0; i < h; i++)
            {
                const float* ptr = bottom_top_blob.row(i);
                float* pmax = max;

                int j = 0;
                for (; j + 7 < w; j += 8)
                {
                    __m256 _p0 = _mm256_load_ps(ptr);
                    __m256 _p1 = _mm256_load_ps(ptr + 8);
                    __m256 _p2 = _mm256_load_ps(ptr + 8 * 2);
                    __m256 _p3 = _mm256_load_ps(ptr + 8 * 3);
                    __m256 _p4 = _mm256_load_ps(ptr + 8 * 4);
                    __m256 _p5 = _mm256_load_ps(ptr + 8 * 5);
                    __m256 _p6 = _mm256_load_ps(ptr + 8 * 6);
                    __m256 _p7 = _mm256_load_ps(ptr + 8 * 7);
                    transpose8_ps(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7);
                    __m256 _max01 = _mm256_max_ps(_p0, _p1);
                    __m256 _max23 = _mm256_max_ps(_p2, _p3);
                    __m256 _max45 = _mm256_max_ps(_p4, _p5);
                    __m256 _max67 = _mm256_max_ps(_p6, _p7);
                    __m256 _max0123 = _mm256_max_ps(_max01, _max23);
                    __m256 _max4567 = _mm256_max_ps(_max45, _max67);
                    __m256 _max01234567 = _mm256_max_ps(_max0123, _max4567);
                    __m256 _max = _mm256_load_ps(pmax);
                    _max = _mm256_max_ps(_max, _max01234567);
                    _mm256_store_ps(pmax, _max);

                    ptr += 64;
                    pmax += 8;
                }
                for (; j < w; j++)
                {
                    __m256 _p = _mm256_load_ps(ptr);
                    *pmax = std::max(*pmax, _mm256_reduce_max_ps(_p));

                    ptr += 8;
                    pmax++;
                }
            }

            Mat sum;
            sum.create(w, 4u, 1, opt.workspace_allocator);
            if (sum.empty())
                return -100;
            sum.fill(0.f);

            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);
                float* psum = sum;
                int j = 0;
                for (; j + 7 < w; j += 8)
                {
                    __m256 _p0 = _mm256_load_ps(ptr);
                    __m256 _p1 = _mm256_load_ps(ptr + 8);
                    __m256 _p2 = _mm256_load_ps(ptr + 8 * 2);
                    __m256 _p3 = _mm256_load_ps(ptr + 8 * 3);
                    __m256 _p4 = _mm256_load_ps(ptr + 8 * 4);
                    __m256 _p5 = _mm256_load_ps(ptr + 8 * 5);
                    __m256 _p6 = _mm256_load_ps(ptr + 8 * 6);
                    __m256 _p7 = _mm256_load_ps(ptr + 8 * 7);
                    _p0 = exp256_ps(_mm256_sub_ps(_p0, _mm256_set1_ps(max[j])));
                    _p1 = exp256_ps(_mm256_sub_ps(_p1, _mm256_set1_ps(max[j + 1])));
                    _p2 = exp256_ps(_mm256_sub_ps(_p2, _mm256_set1_ps(max[j + 2])));
                    _p3 = exp256_ps(_mm256_sub_ps(_p3, _mm256_set1_ps(max[j + 3])));
                    _p4 = exp256_ps(_mm256_sub_ps(_p4, _mm256_set1_ps(max[j + 4])));
                    _p5 = exp256_ps(_mm256_sub_ps(_p5, _mm256_set1_ps(max[j + 5])));
                    _p6 = exp256_ps(_mm256_sub_ps(_p6, _mm256_set1_ps(max[j + 6])));
                    _p7 = exp256_ps(_mm256_sub_ps(_p7, _mm256_set1_ps(max[j + 7])));
                    _mm256_store_ps(ptr, _p0);
                    _mm256_store_ps(ptr + 8, _p1);
                    _mm256_store_ps(ptr + 8 * 2, _p2);
                    _mm256_store_ps(ptr + 8 * 3, _p3);
                    _mm256_store_ps(ptr + 8 * 4, _p4);
                    _mm256_store_ps(ptr + 8 * 5, _p5);
                    _mm256_store_ps(ptr + 8 * 6, _p6);
                    _mm256_store_ps(ptr + 8 * 7, _p7);
                    transpose8_ps(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7);
                    __m256 _sum01 = _mm256_add_ps(_p0, _p1);
                    __m256 _sum23 = _mm256_add_ps(_p2, _p3);
                    __m256 _sum45 = _mm256_add_ps(_p4, _p5);
                    __m256 _sum67 = _mm256_add_ps(_p6, _p7);
                    __m256 _sum0123 = _mm256_add_ps(_sum01, _sum23);
                    __m256 _sum4567 = _mm256_add_ps(_sum45, _sum67);
                    __m256 _sum01234567 = _mm256_add_ps(_sum0123, _sum4567);
                    __m256 _sum = _mm256_load_ps(psum);
                    _sum = _mm256_add_ps(_sum, _sum01234567);
                    _mm256_store_ps(psum, _sum);

                    ptr += 64;
                    psum += 8;
                }
                for (; j < w; j++)
                {
                    __m256 _p = _mm256_load_ps(ptr);
                    __m256 _max = _mm256_set1_ps(max[j]);
                    _p = exp256_ps(_mm256_sub_ps(_p, _max));
                    _mm256_store_ps(ptr, _p);
                    *psum += _mm256_reduce_add_ps(_p);

                    ptr += 8;
                    psum++;
                }
            }

            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);
                for (int j = 0; j < w; j++)
                {
                    __m256 _p = _mm256_load_ps(ptr);
                    __m256 _sum = _mm256_set1_ps(sum[j]);
                    _p = _mm256_div_ps(_p, _sum);
                    _mm256_store_ps(ptr, _p);

                    ptr += 8;
                }
            }
        }

        if (dims == 2 && positive_axis == 1)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);

                __m256 _max = _mm256_set1_ps(-FLT_MAX);
                for (int j = 0; j < w; j++)
                {
                    __m256 _p = _mm256_load_ps(ptr + j * 8);
                    _max = _mm256_max_ps(_max, _p);
                }

                __m256 _sum = _mm256_setzero_ps();
                for (int j = 0; j < w; j++)
                {
                    __m256 _p = _mm256_load_ps(ptr + j * 8);
                    _p = exp256_ps(_mm256_sub_ps(_p, _max));
                    _mm256_store_ps(ptr + j * 8, _p);
                    _sum = _mm256_add_ps(_sum, _p);
                }

                for (int j = 0; j < w; j++)
                {
                    __m256 _p = _mm256_load_ps(ptr + j * 8);
                    _p = _mm256_div_ps(_p, _sum);
                    _mm256_store_ps(ptr + j * 8, _p);
                }
            }
        }

        if (dims == 3 && positive_axis == 0)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;
            int size = w * h;

            Mat max;
            max.create(w, h, 4u, 1, opt.workspace_allocator);
            if (max.empty())
                return -100;
            max.fill(-FLT_MAX);
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_top_blob.channel(q);
                float* pmax = max;

                int i = 0;
                for (; i + 7 < size; i += 8)
                {
                    __m256 _p0 = _mm256_load_ps(ptr);
                    __m256 _p1 = _mm256_load_ps(ptr + 8);
                    __m256 _p2 = _mm256_load_ps(ptr + 8 * 2);
                    __m256 _p3 = _mm256_load_ps(ptr + 8 * 3);
                    __m256 _p4 = _mm256_load_ps(ptr + 8 * 4);
                    __m256 _p5 = _mm256_load_ps(ptr + 8 * 5);
                    __m256 _p6 = _mm256_load_ps(ptr + 8 * 6);
                    __m256 _p7 = _mm256_load_ps(ptr + 8 * 7);
                    transpose8_ps(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7);
                    __m256 _max01 = _mm256_max_ps(_p0, _p1);
                    __m256 _max23 = _mm256_max_ps(_p2, _p3);
                    __m256 _max45 = _mm256_max_ps(_p4, _p5);
                    __m256 _max67 = _mm256_max_ps(_p6, _p7);
                    __m256 _max0123 = _mm256_max_ps(_max01, _max23);
                    __m256 _max4567 = _mm256_max_ps(_max45, _max67);
                    __m256 _max01234567 = _mm256_max_ps(_max0123, _max4567);
                    __m256 _max = _mm256_load_ps(pmax);
                    _max = _mm256_max_ps(_max, _max01234567);
                    _mm256_store_ps(pmax, _max);

                    ptr += 64;
                    pmax += 8;
                }
                for (; i < size; i++)
                {
                    __m256 _p = _mm256_load_ps(ptr);
                    *pmax = std::max(*pmax, _mm256_reduce_max_ps(_p));

                    ptr += 8;
                    pmax++;
                }
            }

            Mat sum;
            sum.create(w, h, 4u, 1, opt.workspace_allocator);
            if (sum.empty())
                return -100;
            sum.fill(0.f);
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                float* psum = sum;

                int i = 0;
                for (; i + 7 < size; i += 8)
                {
                    __m256 _p0 = _mm256_load_ps(ptr);
                    __m256 _p1 = _mm256_load_ps(ptr + 8);
                    __m256 _p2 = _mm256_load_ps(ptr + 8 * 2);
                    __m256 _p3 = _mm256_load_ps(ptr + 8 * 3);
                    __m256 _p4 = _mm256_load_ps(ptr + 8 * 4);
                    __m256 _p5 = _mm256_load_ps(ptr + 8 * 5);
                    __m256 _p6 = _mm256_load_ps(ptr + 8 * 6);
                    __m256 _p7 = _mm256_load_ps(ptr + 8 * 7);
                    _p0 = exp256_ps(_mm256_sub_ps(_p0, _mm256_set1_ps(max[i])));
                    _p1 = exp256_ps(_mm256_sub_ps(_p1, _mm256_set1_ps(max[i + 1])));
                    _p2 = exp256_ps(_mm256_sub_ps(_p2, _mm256_set1_ps(max[i + 2])));
                    _p3 = exp256_ps(_mm256_sub_ps(_p3, _mm256_set1_ps(max[i + 3])));
                    _p4 = exp256_ps(_mm256_sub_ps(_p4, _mm256_set1_ps(max[i + 4])));
                    _p5 = exp256_ps(_mm256_sub_ps(_p5, _mm256_set1_ps(max[i + 5])));
                    _p6 = exp256_ps(_mm256_sub_ps(_p6, _mm256_set1_ps(max[i + 6])));
                    _p7 = exp256_ps(_mm256_sub_ps(_p7, _mm256_set1_ps(max[i + 7])));
                    _mm256_store_ps(ptr, _p0);
                    _mm256_store_ps(ptr + 8, _p1);
                    _mm256_store_ps(ptr + 8 * 2, _p2);
                    _mm256_store_ps(ptr + 8 * 3, _p3);
                    _mm256_store_ps(ptr + 8 * 4, _p4);
                    _mm256_store_ps(ptr + 8 * 5, _p5);
                    _mm256_store_ps(ptr + 8 * 6, _p6);
                    _mm256_store_ps(ptr + 8 * 7, _p7);
                    transpose8_ps(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7);
                    __m256 _sum01 = _mm256_add_ps(_p0, _p1);
                    __m256 _sum23 = _mm256_add_ps(_p2, _p3);
                    __m256 _sum45 = _mm256_add_ps(_p4, _p5);
                    __m256 _sum67 = _mm256_add_ps(_p6, _p7);
                    __m256 _sum0123 = _mm256_add_ps(_sum01, _sum23);
                    __m256 _sum4567 = _mm256_add_ps(_sum45, _sum67);
                    __m256 _sum01234567 = _mm256_add_ps(_sum0123, _sum4567);
                    __m256 _sum = _mm256_load_ps(psum);
                    _sum = _mm256_add_ps(_sum, _sum01234567);
                    _mm256_store_ps(psum, _sum);

                    ptr += 64;
                    psum += 8;
                }
                for (; i < size; i++)
                {
                    __m256 _p = _mm256_load_ps(ptr);
                    __m256 _max = _mm256_set1_ps(max[i]);
                    _p = exp256_ps(_mm256_sub_ps(_p, _max));
                    _mm256_store_ps(ptr, _p);
                    *psum += _mm256_reduce_add_ps(_p);

                    ptr += 8;
                    psum++;
                }
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    __m256 _p = _mm256_load_ps(ptr);
                    __m256 _sum = _mm256_set1_ps(sum[i]);
                    _p = _mm256_div_ps(_p, _sum);
                    _mm256_store_ps(ptr, _p);

                    ptr += 8;
                }
            }
        }

        if (dims == 3 && positive_axis == 1)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;

            Mat max;
            max.create(w, channels, elemsize, elempack, opt.workspace_allocator);
            if (max.empty())
                return -100;
            max.fill(_mm256_set1_ps(-FLT_MAX));

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < h; i++)
                {
                    float* maxptr = max.row(q);

                    for (int j = 0; j < w; j++)
                    {
                        __m256 _p = _mm256_load_ps(ptr);
                        __m256 _max = _mm256_load_ps(maxptr);
                        _max = _mm256_max_ps(_max, _p);
                        _mm256_store_ps(maxptr, _max);

                        ptr += 8;
                        maxptr += 8;
                    }
                }
            }

            Mat sum;
            sum.create(w, channels, elemsize, elempack, opt.workspace_allocator);
            if (sum.empty())
                return -100;
            sum.fill(_mm256_setzero_ps());

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < h; i++)
                {
                    float* maxptr = max.row(q);
                    float* sumptr = sum.row(q);

                    for (int j = 0; j < w; j++)
                    {
                        __m256 _p = _mm256_load_ps(ptr);
                        __m256 _max = _mm256_load_ps(maxptr);
                        _p = exp256_ps(_mm256_sub_ps(_p, _max));
                        _mm256_store_ps(ptr, _p);
                        __m256 _sum = _mm256_load_ps(sumptr);
                        _sum = _mm256_add_ps(_sum, _p);
                        _mm256_store_ps(sumptr, _sum);

                        ptr += 8;
                        maxptr += 8;
                        sumptr += 8;
                    }
                }
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < h; i++)
                {
                    float* sumptr = sum.row(q);

                    for (int j = 0; j < w; j++)
                    {
                        __m256 _p = _mm256_load_ps(ptr);
                        __m256 _sum = _mm256_load_ps(sumptr);
                        _p = _mm256_div_ps(_p, _sum);
                        _mm256_store_ps(ptr, _p);

                        ptr += 8;
                        sumptr += 8;
                    }
                }
            }
        }

        if (dims == 3 && positive_axis == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < h; i++)
                {
                    __m256 _max = _mm256_set1_ps(-FLT_MAX);
                    for (int j = 0; j < w; j++)
                    {
                        __m256 _p = _mm256_load_ps(ptr + j * 8);
                        _max = _mm256_max_ps(_max, _p);
                    }

                    __m256 _sum = _mm256_setzero_ps();
                    for (int j = 0; j < w; j++)
                    {
                        __m256 _p = _mm256_load_ps(ptr + j * 8);
                        _p = exp256_ps(_mm256_sub_ps(_p, _max));
                        _mm256_store_ps(ptr + j * 8, _p);
                        _sum = _mm256_add_ps(_sum, _p);
                    }

                    for (int j = 0; j < w; j++)
                    {
                        __m256 _p = _mm256_load_ps(ptr + j * 8);
                        _p = _mm256_div_ps(_p, _sum);
                        _mm256_store_ps(ptr + j * 8, _p);
                    }

                    ptr += w * 8;
                }
            }
        }

        return 0;
    }
#endif // __AVX__

    if (elempack == 4)
    {
        if (dims == 1) // positive_axis == 0
        {
            int w = bottom_top_blob.w;

            float* ptr = bottom_top_blob;

            __m128 _max = _mm_set1_ps(-FLT_MAX);
            for (int i = 0; i < w; i++)
            {
                __m128 _p = _mm_load_ps(ptr + i * 4);
                _max = _mm_max_ps(_max, _p);
            }
            _max = _mm_max_ps(_max, _mm_shuffle_ps(_max, _max, _MM_SHUFFLE(2, 3, 0, 1)));
            _max = _mm_max_ps(_max, _mm_shuffle_ps(_max, _max, _MM_SHUFFLE(1, 0, 3, 2)));

            __m128 _sum = _mm_setzero_ps();
            for (int i = 0; i < w; i++)
            {
                __m128 _p = _mm_load_ps(ptr + i * 4);
                _p = exp_ps(_mm_sub_ps(_p, _max));
                _mm_store_ps(ptr + i * 4, _p);
                _sum = _mm_add_ps(_sum, _p);
            }
            _sum = _mm_add_ps(_sum, _mm_shuffle_ps(_sum, _sum, _MM_SHUFFLE(2, 3, 0, 1)));
            _sum = _mm_add_ps(_sum, _mm_shuffle_ps(_sum, _sum, _MM_SHUFFLE(1, 0, 3, 2)));

            for (int i = 0; i < w; i++)
            {
                __m128 _p = _mm_load_ps(ptr + i * 4);
                _p = _mm_div_ps(_p, _sum);
                _mm_store_ps(ptr + i * 4, _p);
            }
        }

        if (dims == 2 && positive_axis == 0)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;

            Mat max;
            max.create(w, 4u, 1, opt.workspace_allocator);
            if (max.empty())
                return -100;
            max.fill(-FLT_MAX);

            for (int i = 0; i < h; i++)
            {
                const float* ptr = bottom_top_blob.row(i);
                float* pmax = max;

                int j = 0;
                for (; j + 3 < w; j += 4)
                {
                    __m128 _p0 = _mm_load_ps(ptr);
                    __m128 _p1 = _mm_load_ps(ptr + 4);
                    __m128 _p2 = _mm_load_ps(ptr + 8);
                    __m128 _p3 = _mm_load_ps(ptr + 12);
                    _MM_TRANSPOSE4_PS(_p0, _p1, _p2, _p3);
                    __m128 _max01 = _mm_max_ps(_p0, _p1);
                    __m128 _max23 = _mm_max_ps(_p2, _p3);
                    __m128 _max0123 = _mm_max_ps(_max01, _max23);
                    __m128 _max = _mm_load_ps(pmax);
                    _max = _mm_max_ps(_max, _max0123);
                    _mm_store_ps(pmax, _max);

                    ptr += 16;
                    pmax += 4;
                }
                for (; j < w; j++)
                {
                    __m128 _p = _mm_load_ps(ptr);
                    *pmax = std::max(*pmax, _mm_reduce_max_ps(_p));

                    ptr += 4;
                    pmax++;
                }
            }

            Mat sum;
            sum.create(w, 4u, 1, opt.workspace_allocator);
            if (sum.empty())
                return -100;
            sum.fill(0.f);

            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);
                float* psum = sum;
                int j = 0;
                for (; j + 3 < w; j += 4)
                {
                    __m128 _p0 = _mm_load_ps(ptr);
                    __m128 _p1 = _mm_load_ps(ptr + 4);
                    __m128 _p2 = _mm_load_ps(ptr + 8);
                    __m128 _p3 = _mm_load_ps(ptr + 12);
                    __m128 _max0 = _mm_set1_ps(max[j]);
                    __m128 _max1 = _mm_set1_ps(max[j + 1]);
                    __m128 _max2 = _mm_set1_ps(max[j + 2]);
                    __m128 _max3 = _mm_set1_ps(max[j + 3]);
                    _p0 = exp_ps(_mm_sub_ps(_p0, _max0));
                    _p1 = exp_ps(_mm_sub_ps(_p1, _max1));
                    _p2 = exp_ps(_mm_sub_ps(_p2, _max2));
                    _p3 = exp_ps(_mm_sub_ps(_p3, _max3));
                    _mm_store_ps(ptr, _p0);
                    _mm_store_ps(ptr + 4, _p1);
                    _mm_store_ps(ptr + 8, _p2);
                    _mm_store_ps(ptr + 12, _p3);
                    _MM_TRANSPOSE4_PS(_p0, _p1, _p2, _p3);
                    __m128 _sum01 = _mm_add_ps(_p0, _p1);
                    __m128 _sum23 = _mm_add_ps(_p2, _p3);
                    __m128 _sum0123 = _mm_add_ps(_sum01, _sum23);
                    __m128 _sum = _mm_load_ps(psum);
                    _sum = _mm_add_ps(_sum, _sum0123);
                    _mm_store_ps(psum, _sum);

                    ptr += 16;
                    psum += 4;
                }
                for (; j < w; j++)
                {
                    __m128 _p = _mm_load_ps(ptr);
                    __m128 _max = _mm_set1_ps(max[j]);
                    _p = exp_ps(_mm_sub_ps(_p, _max));
                    _mm_store_ps(ptr, _p);
                    *psum += _mm_reduce_add_ps(_p);

                    ptr += 4;
                    psum++;
                }
            }

            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);
                for (int j = 0; j < w; j++)
                {
                    __m128 _p = _mm_load_ps(ptr);
                    __m128 _sum = _mm_set1_ps(sum[j]);
                    _p = _mm_div_ps(_p, _sum);
                    _mm_store_ps(ptr, _p);

                    ptr += 4;
                }
            }
        }

        if (dims == 2 && positive_axis == 1)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);

                __m128 _max = _mm_set1_ps(-FLT_MAX);
                for (int j = 0; j < w; j++)
                {
                    __m128 _p = _mm_load_ps(ptr + j * 4);
                    _max = _mm_max_ps(_max, _p);
                }

                __m128 _sum = _mm_setzero_ps();
                for (int j = 0; j < w; j++)
                {
                    __m128 _p = _mm_load_ps(ptr + j * 4);
                    _p = exp_ps(_mm_sub_ps(_p, _max));
                    _mm_store_ps(ptr + j * 4, _p);
                    _sum = _mm_add_ps(_sum, _p);
                }

                for (int j = 0; j < w; j++)
                {
                    __m128 _p = _mm_load_ps(ptr + j * 4);
                    _p = _mm_div_ps(_p, _sum);
                    _mm_store_ps(ptr + j * 4, _p);
                }
            }
        }

        if (dims == 3 && positive_axis == 0)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;
            int size = w * h;

            Mat max;
            max.create(w, h, 4u, 1, opt.workspace_allocator);
            if (max.empty())
                return -100;
            max.fill(-FLT_MAX);
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_top_blob.channel(q);
                float* pmax = max;

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    __m128 _p0 = _mm_load_ps(ptr);
                    __m128 _p1 = _mm_load_ps(ptr + 4);
                    __m128 _p2 = _mm_load_ps(ptr + 8);
                    __m128 _p3 = _mm_load_ps(ptr + 12);
                    _MM_TRANSPOSE4_PS(_p0, _p1, _p2, _p3);
                    __m128 _max01 = _mm_max_ps(_p0, _p1);
                    __m128 _max23 = _mm_max_ps(_p2, _p3);
                    __m128 _max0123 = _mm_max_ps(_max01, _max23);
                    __m128 _max = _mm_load_ps(pmax);
                    _max = _mm_max_ps(_max, _max0123);
                    _mm_store_ps(pmax, _max);

                    ptr += 16;
                    pmax += 4;
                }
                for (; i < size; i++)
                {
                    __m128 _p = _mm_load_ps(ptr);
                    *pmax = std::max(*pmax, _mm_reduce_max_ps(_p));

                    ptr += 4;
                    pmax++;
                }
            }

            Mat sum;
            sum.create(w, h, 4u, 1, opt.workspace_allocator);
            if (sum.empty())
                return -100;
            sum.fill(0.f);
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                float* psum = sum;

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    __m128 _p0 = _mm_load_ps(ptr);
                    __m128 _p1 = _mm_load_ps(ptr + 4);
                    __m128 _p2 = _mm_load_ps(ptr + 8);
                    __m128 _p3 = _mm_load_ps(ptr + 12);
                    __m128 _max0 = _mm_set1_ps(max[i]);
                    __m128 _max1 = _mm_set1_ps(max[i + 1]);
                    __m128 _max2 = _mm_set1_ps(max[i + 2]);
                    __m128 _max3 = _mm_set1_ps(max[i + 3]);
                    _p0 = exp_ps(_mm_sub_ps(_p0, _max0));
                    _p1 = exp_ps(_mm_sub_ps(_p1, _max1));
                    _p2 = exp_ps(_mm_sub_ps(_p2, _max2));
                    _p3 = exp_ps(_mm_sub_ps(_p3, _max3));
                    _mm_store_ps(ptr, _p0);
                    _mm_store_ps(ptr + 4, _p1);
                    _mm_store_ps(ptr + 8, _p2);
                    _mm_store_ps(ptr + 12, _p3);
                    _MM_TRANSPOSE4_PS(_p0, _p1, _p2, _p3);
                    __m128 _sum01 = _mm_add_ps(_p0, _p1);
                    __m128 _sum23 = _mm_add_ps(_p2, _p3);
                    __m128 _sum0123 = _mm_add_ps(_sum01, _sum23);
                    __m128 _sum = _mm_load_ps(psum);
                    _sum = _mm_add_ps(_sum, _sum0123);
                    _mm_store_ps(psum, _sum);

                    ptr += 16;
                    psum += 4;
                }
                for (; i < size; i++)
                {
                    __m128 _p = _mm_load_ps(ptr);
                    __m128 _max = _mm_set1_ps(max[i]);
                    _p = exp_ps(_mm_sub_ps(_p, _max));
                    _mm_store_ps(ptr, _p);
                    *psum += _mm_reduce_add_ps(_p);

                    ptr += 4;
                    psum++;
                }
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    __m128 _p = _mm_load_ps(ptr);
                    __m128 _sum = _mm_set1_ps(sum[i]);
                    _p = _mm_div_ps(_p, _sum);
                    _mm_store_ps(ptr, _p);

                    ptr += 4;
                }
            }
        }

        if (dims == 3 && positive_axis == 1)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;

            Mat max;
            max.create(w, channels, elemsize, elempack, opt.workspace_allocator);
            if (max.empty())
                return -100;
            max.fill(_mm_set1_ps(-FLT_MAX));
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < h; i++)
                {
                    float* maxptr = max.row(q);

                    for (int j = 0; j < w; j++)
                    {
                        __m128 _p = _mm_load_ps(ptr);
                        __m128 _max = _mm_load_ps(maxptr);
                        _max = _mm_max_ps(_max, _p);
                        _mm_store_ps(maxptr, _max);

                        ptr += 4;
                        maxptr += 4;
                    }
                }
            }

            Mat sum;
            sum.create(w, channels, elemsize, elempack, opt.workspace_allocator);
            if (sum.empty())
                return -100;
            sum.fill(_mm_setzero_ps());
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < h; i++)
                {
                    float* maxptr = max.row(q);
                    float* sumptr = sum.row(q);

                    for (int j = 0; j < w; j++)
                    {
                        __m128 _p = _mm_load_ps(ptr);
                        __m128 _max = _mm_load_ps(maxptr);
                        _p = exp_ps(_mm_sub_ps(_p, _max));
                        _mm_store_ps(ptr, _p);
                        __m128 _sum = _mm_load_ps(sumptr);
                        _sum = _mm_add_ps(_sum, _p);
                        _mm_store_ps(sumptr, _sum);

                        ptr += 4;
                        maxptr += 4;
                        sumptr += 4;
                    }
                }
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < h; i++)
                {
                    float* sumptr = sum.row(q);

                    for (int j = 0; j < w; j++)
                    {
                        __m128 _p = _mm_load_ps(ptr);
                        __m128 _sum = _mm_load_ps(sumptr);
                        _p = _mm_div_ps(_p, _sum);
                        _mm_store_ps(ptr, _p);

                        ptr += 4;
                        sumptr += 4;
                    }
                }
            }
        }

        if (dims == 3 && positive_axis == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < h; i++)
                {
                    __m128 _max = _mm_set1_ps(-FLT_MAX);
                    for (int j = 0; j < w; j++)
                    {
                        __m128 _p = _mm_load_ps(ptr + j * 4);
                        _max = _mm_max_ps(_max, _p);
                    }

                    __m128 _sum = _mm_setzero_ps();
                    for (int j = 0; j < w; j++)
                    {
                        __m128 _p = _mm_load_ps(ptr + j * 4);
                        _p = exp_ps(_mm_sub_ps(_p, _max));
                        _mm_store_ps(ptr + j * 4, _p);
                        _sum = _mm_add_ps(_sum, _p);
                    }

                    for (int j = 0; j < w; j++)
                    {
                        __m128 _p = _mm_load_ps(ptr + j * 4);
                        _p = _mm_div_ps(_p, _sum);
                        _mm_store_ps(ptr + j * 4, _p);
                    }

                    ptr += w * 4;
                }
            }
        }

        return 0;
    }
#endif // __SSE2__

    if (dims == 1) // positive_axis == 0
    {
        int w = bottom_top_blob.w;

        float* ptr = bottom_top_blob;

        float max = -FLT_MAX;
        {
            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            __m512 _max_avx512 = _mm512_set1_ps(-FLT_MAX);
            for (; i + 15 < w; i += 16)
            {
                __m512 _p = _mm512_load_ps(ptr + i);
                _max_avx512 = _mm512_max_ps(_max_avx512, _p);
            }
            max = std::max(max, _mm512_comp_reduce_max_ps(_max_avx512));
#endif // __AVX512F__
            __m256 _max_avx = _mm256_set1_ps(-FLT_MAX);
            for (; i + 7 < w; i += 8)
            {
                __m256 _p = _mm256_load_ps(ptr + i);
                _max_avx = _mm256_max_ps(_max_avx, _p);
            }
            max = std::max(max, _mm256_reduce_max_ps(_max_avx));
#endif // __AVX__
            __m128 _max = _mm_set1_ps(-FLT_MAX);
            for (; i + 3 < w; i += 4)
            {
                __m128 _p = _mm_load_ps(ptr + i);
                _max = _mm_max_ps(_max, _p);
            }
            max = std::max(max, _mm_reduce_max_ps(_max));
#endif // __SSE2__
            for (; i < w; i++)
            {
                max = std::max(max, ptr[i]);
            }
        }

        float sum = 0.f;
        {
            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            __m512 _sum_avx512 = _mm512_setzero_ps();
            __m512 _max_avx512 = _mm512_set1_ps(max);
            for (; i + 15 < w; i += 16)
            {
                __m512 _p = _mm512_load_ps(ptr + i);
                _p = exp512_ps(_mm512_sub_ps(_p, _max_avx512));
                _mm512_storeu_ps(ptr + i, _p);
                _sum_avx512 = _mm512_add_ps(_sum_avx512, _p);
            }
            sum += _mm512_comp_reduce_add_ps(_sum_avx512);
#endif // __AVX512F__
            __m256 _sum_avx = _mm256_setzero_ps();
            __m256 _max_avx = _mm256_set1_ps(max);
            for (; i + 7 < w; i += 8)
            {
                __m256 _p = _mm256_load_ps(ptr + i);
                _p = exp256_ps(_mm256_sub_ps(_p, _max_avx));
                _mm256_storeu_ps(ptr + i, _p);
                _sum_avx = _mm256_add_ps(_sum_avx, _p);
            }
            sum += _mm256_reduce_add_ps(_sum_avx);
#endif // __AVX__
            __m128 _sum = _mm_setzero_ps();
            __m128 _max = _mm_set1_ps(max);
            for (; i + 3 < w; i += 4)
            {
                __m128 _p = _mm_load_ps(ptr + i);
                _p = exp_ps(_mm_sub_ps(_p, _max));
                _mm_store_ps(ptr + i, _p);
                _sum = _mm_add_ps(_sum, _p);
            }
            sum += _mm_reduce_add_ps(_sum);
#endif // __SSE2__
            for (; i < w; i++)
            {
                ptr[i] = (float)(exp(ptr[i] - max));
                sum += ptr[i];
            }
        }

        {
            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            __m512 _sum_avx512 = _mm512_set1_ps(sum);
            for (; i + 15 < w; i += 16)
            {
                __m512 _p = _mm512_load_ps(ptr + i);
                _p = _mm512_div_ps(_p, _sum_avx512);
                _mm512_store_ps(ptr + i, _p);
            }
#endif // __AVX512F__
            __m256 _sum_avx = _mm256_set1_ps(sum);
            for (; i + 7 < w; i += 8)
            {
                __m256 _p = _mm256_load_ps(ptr + i);
                _p = _mm256_div_ps(_p, _sum_avx);
                _mm256_store_ps(ptr + i, _p);
            }
#endif // __AVX__
            __m128 _sum = _mm_set1_ps(sum);
            for (; i + 3 < w; i += 4)
            {
                __m128 _p = _mm_load_ps(ptr + i);
                _p = _mm_div_ps(_p, _sum);
                _mm_store_ps(ptr + i, _p);
            }
#endif // __SSE2__
            for (; i < w; i++)
            {
                ptr[i] /= sum;
            }
        }
    }

    if (dims == 2 && positive_axis == 0)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        Mat max;
        max.create(w, elemsize, opt.workspace_allocator);
        if (max.empty())
            return -100;
        max.fill(-FLT_MAX);

        for (int i = 0; i < h; i++)
        {
            const float* ptr = bottom_top_blob.row(i);
            float* pmax = max;

            int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; j + 15 < w; j += 16)
            {
                __m512 _p = _mm512_loadu_ps(ptr);
                __m512 _max = _mm512_load_ps(pmax);
                _max = _mm512_max_ps(_max, _p);
                _mm512_store_ps(pmax, _max);

                ptr += 16;
                pmax += 16;
            }
#endif // __AVX512F__
            for (; j + 7 < w; j += 8)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                __m256 _max = _mm256_load_ps(pmax);
                _max = _mm256_max_ps(_max, _p);
                _mm256_store_ps(pmax, _max);

                ptr += 8;
                pmax += 8;
            }
#endif // __AVX__
            for (; j + 3 < w; j += 4)
            {
                __m128 _p = _mm_loadu_ps(ptr);
                __m128 _max = _mm_load_ps(pmax);
                _max = _mm_max_ps(_max, _p);
                _mm_store_ps(pmax, _max);

                ptr += 4;
                pmax += 4;
            }
#endif // __SSE2__
            for (; j < w; j++)
            {
                *pmax = std::max(*pmax, *ptr);

                ptr++;
                pmax++;
            }
        }

        Mat sum;
        sum.create(w, elemsize, opt.workspace_allocator);
        if (sum.empty())
            return -100;
        sum.fill(0.f);

        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            const float* pmax = max;
            float* psum = sum;

            int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; j + 15 < w; j += 16)
            {
                __m512 _p = _mm512_loadu_ps(ptr);
                __m512 _max = _mm512_load_ps(pmax);
                __m512 _sum = _mm512_load_ps(psum);
                _p = exp512_ps(_mm512_sub_ps(_p, _max));
                _sum = _mm512_add_ps(_sum, _p);
                _mm512_storeu_ps(ptr, _p);
                _mm512_store_ps(psum, _sum);

                ptr += 16;
                pmax += 16;
                psum += 16;
            }
#endif // __AVX512F__
            for (; j + 7 < w; j += 8)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                __m256 _max = _mm256_load_ps(pmax);
                __m256 _sum = _mm256_load_ps(psum);
                _p = exp256_ps(_mm256_sub_ps(_p, _max));
                _sum = _mm256_add_ps(_sum, _p);
                _mm256_storeu_ps(ptr, _p);
                _mm256_store_ps(psum, _sum);

                ptr += 8;
                pmax += 8;
                psum += 8;
            }
#endif // __AVX__
            for (; j + 3 < w; j += 4)
            {
                __m128 _p = _mm_loadu_ps(ptr);
                __m128 _max = _mm_load_ps(pmax);
                __m128 _sum = _mm_load_ps(psum);
                _p = exp_ps(_mm_sub_ps(_p, _max));
                _sum = _mm_add_ps(_sum, _p);
                _mm_storeu_ps(ptr, _p);
                _mm_store_ps(psum, _sum);

                ptr += 4;
                pmax += 4;
                psum += 4;
            }
#endif // __SSE2__
            for (; j < w; j++)
            {
                *ptr = (float)(exp(*ptr - *pmax));
                *psum += *ptr;

                ptr++;
                pmax++;
                psum++;
            }
        }

        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            const float* psum = sum;

            int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; j + 15 < w; j += 16)
            {
                __m512 _p = _mm512_loadu_ps(ptr);
                __m512 _sum = _mm512_load_ps(psum);
                _p = _mm512_div_ps(_p, _sum);
                _mm512_storeu_ps(ptr, _p);

                ptr += 16;
                psum += 16;
            }
#endif // __AVX512F__
            for (; j + 7 < w; j += 8)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                __m256 _sum = _mm256_load_ps(psum);
                _p = _mm256_div_ps(_p, _sum);
                _mm256_storeu_ps(ptr, _p);

                ptr += 8;
                psum += 8;
            }
#endif // __AVX__
            for (; j + 3 < w; j += 4)
            {
                __m128 _p = _mm_loadu_ps(ptr);
                __m128 _sum = _mm_load_ps(psum);
                _p = _mm_div_ps(_p, _sum);
                _mm_storeu_ps(ptr, _p);

                ptr += 4;
                psum += 4;
            }
#endif // __SSE2__
            for (; j < w; j++)
            {
                *ptr /= *psum;

                ptr++;
                psum++;
            }
        }
    }

    if (dims == 2 && positive_axis == 1)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);

            float max = -FLT_MAX;
            {
                int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _max_avx512 = _mm512_set1_ps(-FLT_MAX);
                for (; j + 15 < w; j += 16)
                {
                    __m512 _p = _mm512_loadu_ps(ptr + j);
                    _max_avx512 = _mm512_max_ps(_max_avx512, _p);
                }
                max = std::max(max, _mm512_comp_reduce_max_ps(_max_avx512));
#endif // __AVX512F__
                __m256 _max_avx = _mm256_set1_ps(-FLT_MAX);
                for (; j + 7 < w; j += 8)
                {
                    __m256 _p = _mm256_loadu_ps(ptr + j);
                    _max_avx = _mm256_max_ps(_max_avx, _p);
                }
                max = std::max(max, _mm256_reduce_max_ps(_max_avx));
#endif // __AVX__
                __m128 _max = _mm_set1_ps(-FLT_MAX);
                for (; j + 3 < w; j += 4)
                {
                    __m128 _p = _mm_loadu_ps(ptr + j);
                    _max = _mm_max_ps(_max, _p);
                }
                max = std::max(max, _mm_reduce_max_ps(_max));
#endif // __SSE2__
                for (; j < w; j++)
                {
                    max = std::max(max, ptr[j]);
                }
            }

            float sum = 0.f;
            {
                int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _sum_avx512 = _mm512_setzero_ps();
                __m512 _max_avx512 = _mm512_set1_ps(max);
                for (; j + 15 < w; j += 16)
                {
                    __m512 _p = _mm512_loadu_ps(ptr + j);
                    _p = exp512_ps(_mm512_sub_ps(_p, _max_avx512));
                    _mm512_storeu_ps(ptr + j, _p);
                    _sum_avx512 = _mm512_add_ps(_sum_avx512, _p);
                }
                sum += _mm512_comp_reduce_add_ps(_sum_avx512);
#endif // __AVX512F__
                __m256 _sum_avx = _mm256_setzero_ps();
                __m256 _max_avx = _mm256_set1_ps(max);
                for (; j + 7 < w; j += 8)
                {
                    __m256 _p = _mm256_loadu_ps(ptr + j);
                    _p = exp256_ps(_mm256_sub_ps(_p, _max_avx));
                    _mm256_storeu_ps(ptr + j, _p);
                    _sum_avx = _mm256_add_ps(_sum_avx, _p);
                }
                sum += _mm256_reduce_add_ps(_sum_avx);
#endif // __AVX__
                __m128 _sum = _mm_setzero_ps();
                __m128 _max = _mm_set1_ps(max);
                for (; j + 3 < w; j += 4)
                {
                    __m128 _p = _mm_loadu_ps(ptr + j);
                    _p = exp_ps(_mm_sub_ps(_p, _max));
                    _mm_storeu_ps(ptr + j, _p);
                    _sum = _mm_add_ps(_sum, _p);
                }
                sum += _mm_reduce_add_ps(_sum);
#endif // __SSE2__
                for (; j < w; j++)
                {
                    ptr[j] = (float)(exp(ptr[j] - max));
                    sum += ptr[j];
                }
            }

            {
                int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _sum_avx512 = _mm512_set1_ps(sum);
                for (; j + 15 < w; j += 16)
                {
                    __m512 _p = _mm512_loadu_ps(ptr + j);
                    _p = _mm512_div_ps(_p, _sum_avx512);
                    _mm512_storeu_ps(ptr + j, _p);
                }
#endif // __AVX512F__
                __m256 _sum_avx = _mm256_set1_ps(sum);
                for (; j + 7 < w; j += 8)
                {
                    __m256 _p = _mm256_loadu_ps(ptr + j);
                    _p = _mm256_div_ps(_p, _sum_avx);
                    _mm256_storeu_ps(ptr + j, _p);
                }
#endif // __AVX__
                __m128 _sum = _mm_set1_ps(sum);
                for (; j + 3 < w; j += 4)
                {
                    __m128 _p = _mm_loadu_ps(ptr + j);
                    _p = _mm_div_ps(_p, _sum);
                    _mm_storeu_ps(ptr + j, _p);
                }
#endif // __SSE2__
                for (; j < w; j++)
                {
                    ptr[j] /= sum;
                }
            }
        }
    }

    if (dims == 3 && positive_axis == 0)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h;

        Mat max;
        max.create(w, h, elemsize, opt.workspace_allocator);
        if (max.empty())
            return -100;
        max.fill(-FLT_MAX);
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            float* maxptr = max;

            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; i + 15 < size; i += 16)
            {
                __m512 _p = _mm512_loadu_ps(ptr);
                __m512 _max = _mm512_load_ps(maxptr);
                _max = _mm512_max_ps(_max, _p);
                _mm512_store_ps(maxptr, _max);

                ptr += 16;
                maxptr += 16;
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                __m256 _max = _mm256_load_ps(maxptr);
                _max = _mm256_max_ps(_max, _p);
                _mm256_store_ps(maxptr, _max);

                ptr += 8;
                maxptr += 8;
            }
#endif // __AVX__
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = _mm_load_ps(ptr);
                __m128 _max = _mm_load_ps(maxptr);
                _max = _mm_max_ps(_max, _p);
                _mm_store_ps(maxptr, _max);

                ptr += 4;
                maxptr += 4;
            }
#endif // __SSE2__
            for (; i < size; i++)
            {
                *maxptr = std::max(*maxptr, *ptr);

                ptr++;
                maxptr++;
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            float* maxptr = max;

            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; i + 15 < size; i += 16)
            {
                __m512 _p = _mm512_loadu_ps(ptr);
                __m512 _max = _mm512_load_ps(maxptr);
                _p = exp512_ps(_mm512_sub_ps(_p, _max));
                _mm512_storeu_ps(ptr, _p);

                ptr += 16;
                maxptr += 16;
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                __m256 _max = _mm256_load_ps(maxptr);
                _p = exp256_ps(_mm256_sub_ps(_p, _max));
                _mm256_storeu_ps(ptr, _p);

                ptr += 8;
                maxptr += 8;
            }
#endif // __AVX__
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = _mm_load_ps(ptr);
                __m128 _max = _mm_load_ps(maxptr);
                _p = exp_ps(_mm_sub_ps(_p, _max));
                _mm_store_ps(ptr, _p);

                ptr += 4;
                maxptr += 4;
            }
#endif // __SSE2__
            for (; i < size; i++)
            {
                *ptr = exp(*ptr - *maxptr);

                ptr++;
                maxptr++;
            }
        }

        Mat sum;
        sum.create(w, h, elemsize, opt.workspace_allocator);
        if (sum.empty())
            return -100;
        sum.fill(0.f);
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            float* sumptr = sum;

            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; i + 15 < size; i += 16)
            {
                __m512 _p = _mm512_loadu_ps(ptr);
                __m512 _sum = _mm512_load_ps(sumptr);
                _sum = _mm512_add_ps(_sum, _p);
                _mm512_store_ps(sumptr, _sum);

                ptr += 16;
                sumptr += 16;
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                __m256 _sum = _mm256_load_ps(sumptr);
                _sum = _mm256_add_ps(_sum, _p);
                _mm256_store_ps(sumptr, _sum);

                ptr += 8;
                sumptr += 8;
            }
#endif // __AVX__
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = _mm_load_ps(ptr);
                __m128 _sum = _mm_load_ps(sumptr);
                _sum = _mm_add_ps(_sum, _p);
                _mm_store_ps(sumptr, _sum);

                ptr += 4;
                sumptr += 4;
            }
#endif // __SSE2__
            for (; i < size; i++)
            {
                *sumptr += *ptr;

                ptr++;
                sumptr++;
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            float* sumptr = sum;

            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; i + 15 < size; i += 16)
            {
                __m512 _p = _mm512_loadu_ps(ptr);
                __m512 _sum = _mm512_load_ps(sumptr);
                _p = _mm512_div_ps(_p, _sum);
                _mm512_storeu_ps(ptr, _p);

                ptr += 16;
                sumptr += 16;
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                __m256 _sum = _mm256_load_ps(sumptr);
                _p = _mm256_div_ps(_p, _sum);
                _mm256_storeu_ps(ptr, _p);

                ptr += 8;
                sumptr += 8;
            }
#endif // __AVX__
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = _mm_load_ps(ptr);
                __m128 _sum = _mm_load_ps(sumptr);
                _p = _mm_div_ps(_p, _sum);
                _mm_store_ps(ptr, _p);

                ptr += 4;
                sumptr += 4;
            }
#endif // __SSE2__
            for (; i < size; i++)
            {
                *ptr /= *sumptr;

                ptr++;
                sumptr++;
            }
        }
    }

    if (dims == 3 && positive_axis == 1)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;

        Mat max;
        max.create(w, channels, elemsize, opt.workspace_allocator);
        if (max.empty())
            return -100;
        max.fill(-FLT_MAX);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_top_blob.channel(q);
            float* maxptr = max.row(q);

            for (int i = 0; i < h; i++)
            {
                int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                for (; j + 15 < w; j += 16)
                {
                    __m512 _p = _mm512_loadu_ps(ptr + j);
                    __m512 _max = _mm512_loadu_ps(maxptr + j);
                    _max = _mm512_max_ps(_max, _p);
                    _mm512_storeu_ps(maxptr + j, _max);
                }
#endif // __AVX512F__
                for (; j + 7 < w; j += 8)
                {
                    __m256 _p = _mm256_loadu_ps(ptr + j);
                    __m256 _max = _mm256_loadu_ps(maxptr + j);
                    _max = _mm256_max_ps(_max, _p);
                    _mm256_storeu_ps(maxptr + j, _max);
                }
#endif // __AVX__
                for (; j + 3 < w; j += 4)
                {
                    __m128 _p = _mm_loadu_ps(ptr + j);
                    __m128 _max = _mm_loadu_ps(maxptr + j);
                    _max = _mm_max_ps(_max, _p);
                    _mm_storeu_ps(maxptr + j, _max);
                }
#endif // __SSE2__
                for (; j < w; j++)
                {
                    maxptr[j] = std::max(maxptr[j], ptr[j]);
                }

                ptr += w;
            }
        }

        Mat sum;
        sum.create(w, channels, elemsize, opt.workspace_allocator);
        if (sum.empty())
            return -100;
        sum.fill(0.f);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            float* maxptr = max.row(q);
            float* sumptr = sum.row(q);

            for (int i = 0; i < h; i++)
            {
                int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                for (; j + 15 < w; j += 16)
                {
                    __m512 _p = _mm512_loadu_ps(ptr + j);
                    __m512 _max = _mm512_loadu_ps(maxptr + j);
                    __m512 _sum = _mm512_loadu_ps(sumptr + j);
                    _p = exp512_ps(_mm512_sub_ps(_p, _max));
                    _sum = _mm512_add_ps(_sum, _p);
                    _mm512_storeu_ps(ptr + j, _p);
                    _mm512_storeu_ps(sumptr + j, _sum);
                }
#endif // __AVX512F__
                for (; j + 7 < w; j += 8)
                {
                    __m256 _p = _mm256_loadu_ps(ptr + j);
                    __m256 _max = _mm256_loadu_ps(maxptr + j);
                    __m256 _sum = _mm256_loadu_ps(sumptr + j);
                    _p = exp256_ps(_mm256_sub_ps(_p, _max));
                    _sum = _mm256_add_ps(_sum, _p);
                    _mm256_storeu_ps(ptr + j, _p);
                    _mm256_storeu_ps(sumptr + j, _sum);
                }
#endif // __AVX__
                for (; j + 3 < w; j += 4)
                {
                    __m128 _p = _mm_loadu_ps(ptr + j);
                    __m128 _max = _mm_loadu_ps(maxptr + j);
                    __m128 _sum = _mm_loadu_ps(sumptr + j);
                    _p = exp_ps(_mm_sub_ps(_p, _max));
                    _sum = _mm_add_ps(_sum, _p);
                    _mm_storeu_ps(ptr + j, _p);
                    _mm_storeu_ps(sumptr + j, _sum);
                }
#endif // __SSE2__
                for (; j < w; j++)
                {
                    ptr[j] = (float)(exp(ptr[j] - maxptr[j]));
                    sumptr[j] += ptr[j];
                }

                ptr += w;
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            float* sumptr = sum.row(q);

            for (int i = 0; i < h; i++)
            {
                int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                for (; j + 15 < w; j += 16)
                {
                    __m512 _p = _mm512_loadu_ps(ptr + j);
                    __m512 _sum = _mm512_loadu_ps(sumptr + j);
                    _p = _mm512_div_ps(_p, _sum);
                    _mm512_storeu_ps(ptr + j, _p);
                }
#endif // __AVX512F__
                for (; j + 7 < w; j += 8)
                {
                    __m256 _p = _mm256_loadu_ps(ptr + j);
                    __m256 _sum = _mm256_loadu_ps(sumptr + j);
                    _p = _mm256_div_ps(_p, _sum);
                    _mm256_storeu_ps(ptr + j, _p);
                }
#endif // __AVX__
                for (; j + 3 < w; j += 4)
                {
                    __m128 _p = _mm_loadu_ps(ptr + j);
                    __m128 _sum = _mm_loadu_ps(sumptr + j);
                    _p = _mm_div_ps(_p, _sum);
                    _mm_storeu_ps(ptr + j, _p);
                }
#endif // __SSE2__
                for (; j < w; j++)
                {
                    ptr[j] /= sumptr[j];
                }

                ptr += w;
            }
        }
    }

    if (dims == 3 && positive_axis == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                float max = -FLT_MAX;
                {
                    int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                    __m512 _max_avx512 = _mm512_set1_ps(-FLT_MAX);
                    for (; j + 15 < w; j += 16)
                    {
                        __m512 _p = _mm512_loadu_ps(ptr + j);
                        _max_avx512 = _mm512_max_ps(_max_avx512, _p);
                    }
                    max = std::max(max, _mm512_comp_reduce_max_ps(_max_avx512));
#endif // __AVX512F__
                    __m256 _max_avx = _mm256_set1_ps(-FLT_MAX);
                    for (; j + 7 < w; j += 8)
                    {
                        __m256 _p = _mm256_loadu_ps(ptr + j);
                        _max_avx = _mm256_max_ps(_max_avx, _p);
                    }
                    max = std::max(max, _mm256_reduce_max_ps(_max_avx));
#endif // __AVX__
                    __m128 _max = _mm_set1_ps(-FLT_MAX);
                    for (; j + 3 < w; j += 4)
                    {
                        __m128 _p = _mm_loadu_ps(ptr + j);
                        _max = _mm_max_ps(_max, _p);
                    }
                    max = std::max(max, _mm_reduce_max_ps(_max));
#endif // __SSE2__
                    for (; j < w; j++)
                    {
                        max = std::max(max, ptr[j]);
                    }
                }

                float sum = 0.f;
                {
                    int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                    __m512 _sum_avx512 = _mm512_setzero_ps();
                    __m512 _max_avx512 = _mm512_set1_ps(max);
                    for (; j + 15 < w; j += 16)
                    {
                        __m512 _p = _mm512_loadu_ps(ptr + j);
                        _p = exp512_ps(_mm512_sub_ps(_p, _max_avx512));
                        _mm512_storeu_ps(ptr + j, _p);
                        _sum_avx512 = _mm512_add_ps(_sum_avx512, _p);
                    }
                    sum += _mm512_comp_reduce_add_ps(_sum_avx512);
#endif // __AVX512F__
                    __m256 _sum_avx = _mm256_setzero_ps();
                    __m256 _max_avx = _mm256_set1_ps(max);
                    for (; j + 7 < w; j += 8)
                    {
                        __m256 _p = _mm256_loadu_ps(ptr + j);
                        _p = exp256_ps(_mm256_sub_ps(_p, _max_avx));
                        _mm256_storeu_ps(ptr + j, _p);
                        _sum_avx = _mm256_add_ps(_sum_avx, _p);
                    }
                    sum += _mm256_reduce_add_ps(_sum_avx);
#endif // __AVX__
                    __m128 _sum = _mm_setzero_ps();
                    __m128 _max = _mm_set1_ps(max);
                    for (; j + 3 < w; j += 4)
                    {
                        __m128 _p = _mm_loadu_ps(ptr + j);
                        _p = exp_ps(_mm_sub_ps(_p, _max));
                        _mm_storeu_ps(ptr + j, _p);
                        _sum = _mm_add_ps(_sum, _p);
                    }
                    sum += _mm_reduce_add_ps(_sum);
#endif // __SSE2__
                    for (; j < w; j++)
                    {
                        ptr[j] = static_cast<float>(exp(ptr[j] - max));
                        sum += ptr[j];
                    }
                }

                {
                    int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                    __m512 _sum_avx512 = _mm512_set1_ps(sum);
                    for (; j + 15 < w; j += 16)
                    {
                        __m512 _p = _mm512_loadu_ps(ptr + j);
                        _p = _mm512_div_ps(_p, _sum_avx512);
                        _mm512_storeu_ps(ptr + j, _p);
                    }
#endif // __AVX512F__
                    __m256 _sum_avx = _mm256_set1_ps(sum);
                    for (; j + 7 < w; j += 8)
                    {
                        __m256 _p = _mm256_loadu_ps(ptr + j);
                        _p = _mm256_div_ps(_p, _sum_avx);
                        _mm256_storeu_ps(ptr + j, _p);
                    }
#endif // __AVX__
                    __m128 _sum = _mm_set1_ps(sum);
                    for (; j + 3 < w; j += 4)
                    {
                        __m128 _p = _mm_loadu_ps(ptr + j);
                        _p = _mm_div_ps(_p, _sum);
                        _mm_storeu_ps(ptr + j, _p);
                    }
#endif // __SSE2__
                    for (; j < w; j++)
                    {
                        ptr[j] /= sum;
                    }
                }

                ptr += w;
            }
        }
    }

    return 0;
}

} // namespace ncnn
