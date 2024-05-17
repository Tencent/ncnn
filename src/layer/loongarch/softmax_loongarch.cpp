// yala is pleased to support the open source community by making ncnn available.
//
//
// Copyright (C) 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>. All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "softmax_loongarch.h"

#include <float.h>

#if __loongarch_sx
#include <lsxintrin.h>
#include "lsx_mathfun.h"
#endif // __loongarch_sx

namespace ncnn {

int Softmax_loongarch::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    size_t elemsize = bottom_top_blob.elemsize;
    int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims != 3 || positive_axis != 0)
        return Softmax::forward_inplace(bottom_top_blob, opt);

    // value = exp( value - global max value )
    // sum all value
    // value = value / sum

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

        for (int i = 0; i < size; i++)
        {
            maxptr[i] = std::max(maxptr[i], ptr[i]);
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        float* maxptr = max;

#if __loongarch_sx
        int nn = size >> 2;
        int remain = size - (nn << 2);
#else
        int remain = size;
#endif // __loongarch_sx

#if __loongarch_sx
        for (; nn > 0; nn--)
        {
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            __m128 _max = (__m128)__lsx_vld(maxptr, 0);

            _p = exp_ps(__lsx_vfsub_s(_p, _max));

            __lsx_vst(_p, ptr, 0);

            ptr += 4;
            maxptr += 4;
        }
#endif // __loongarch_sx

        for (; remain > 0; remain--)
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

#if __loongarch_sx
        int nn = size >> 2;
        int remain = size - (nn << 2);
#else
        int remain = size;
#endif // __loongarch_sx

#if __loongarch_sx
        for (; nn > 0; nn--)
        {
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            __m128 _sum = (__m128)__lsx_vld(sumptr, 0);
            _sum = __lsx_vfadd_s(_sum, _p);
            __lsx_vst(_sum, sumptr, 0);

            ptr += 4;
            sumptr += 4;
        }
#endif // __loongarch_sx

        for (; remain > 0; remain--)
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

#if __loongarch_sx
        int nn = size >> 2;
        int remain = size - (nn << 2);
#else
        int remain = size;
#endif // __loongarch_sx

#if __loongarch_sx
        for (; nn > 0; nn--)
        {
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            __m128 _sum = (__m128)__lsx_vld(sumptr, 0);
            _p = __lsx_vfdiv_s(_p, _sum);
            __lsx_vst(_p, ptr, 0);

            ptr += 4;
            sumptr += 4;
        }
#endif // __loongarch_sx

        for (; remain > 0; remain--)
        {
            *ptr /= *sumptr;

            ptr++;
            sumptr++;
        }
    }

    return 0;
}

} // namespace ncnn
