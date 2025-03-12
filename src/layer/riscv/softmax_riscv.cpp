// Xavier Hsinyuan is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 Xavier Hsinyuan <me@lstlx.com>. All rights reserved.
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

#include "softmax_riscv.h"
#include <float.h>

#if __riscv_vector
#include <riscv_vector.h>
#include "riscv_usability.h"
#include "rvv_mathfun.h"
#endif // __riscv_vector

namespace ncnn {

Softmax_riscv::Softmax_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

static void softmax(float* _ptr, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

    // NCNN_LOGE("softmax %d %d  %d", elemcount, elempack, size);

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;

    // reduce max
    vfloat32m8_t _max = __riscv_vfmv_v_f_f32m8(-FLT_MAX, __riscv_vsetvlmax_e32m8());
    {
        const float* ptr = _ptr;

        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            _max = __riscv_vfmax_vv_f32m8(_max, _p, vl);
            ptr += vl;
            n -= vl;
        }
    }

    if (elempack == packn)
    {
        // reduce max n,n,n,n,n,n,n,n to n
        // broadcast n to n,n,n,n,n,n,n,n

        vfloat32m4_t _max0 = __riscv_vget_v_f32m8_f32m4(_max, 0);
        vfloat32m4_t _max1 = __riscv_vget_v_f32m8_f32m4(_max, 1);
        _max0 = __riscv_vfmax_vv_f32m4(_max0, _max1, __riscv_vsetvlmax_e32m4());
        vfloat32m2_t _max2 = __riscv_vget_v_f32m4_f32m2(_max0, 0);
        vfloat32m2_t _max3 = __riscv_vget_v_f32m4_f32m2(_max0, 1);
        _max2 = __riscv_vfmax_vv_f32m2(_max2, _max3, __riscv_vsetvlmax_e32m2());
        vfloat32m1_t _max4 = __riscv_vget_v_f32m2_f32m1(_max2, 0);
        vfloat32m1_t _max5 = __riscv_vget_v_f32m2_f32m1(_max2, 1);
        _max4 = __riscv_vfmax_vv_f32m1(_max4, _max5, __riscv_vsetvlmax_e32m1());
        _max = __riscv_vcreate_v_f32m1_f32m8(_max4, _max4, _max4, _max4, _max4, _max4, _max4, _max4);
    }
    if (elempack == 1)
    {
        // reduce max n,n,n,n,n,n,n,n to 1
        // broadcast 1 to n,n,n,n,n,n,n,n

        vfloat32m1_t _max0 = __riscv_vfmv_s_f_f32m1(-FLT_MAX, __riscv_vsetvlmax_e32m1());
        _max0 = __riscv_vfredmax_vs_f32m8_f32m1(_max, _max0, __riscv_vsetvlmax_e32m8());
        _max = __riscv_vset_v_f32m1_f32m8(_max, 0, _max0);
        _max = __riscv_vrgather_vx_f32m8(_max, 0, __riscv_vsetvlmax_e32m8());
    }

    // reduce exp(x - max)
    vfloat32m8_t _sum = __riscv_vfmv_v_f_f32m8(0.f, __riscv_vsetvlmax_e32m8());
    {
        float* ptr = _ptr;

        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            _p = __riscv_vfsub_vv_f32m8(_p, _max, vl);
            _p = exp_ps(_p, vl);
            __riscv_vse32_v_f32m8(ptr, _p, vl);
            _sum = __riscv_vfadd_vv_f32m8_tu(_sum, _sum, _p, vl);
            ptr += vl;
            n -= vl;
        }
    }

    if (elempack == packn)
    {
        // reduce sum n,n,n,n,n,n,n,n to n
        // broadcast n to n,n,n,n,n,n,n,n

        vfloat32m4_t _sum0 = __riscv_vget_v_f32m8_f32m4(_sum, 0);
        vfloat32m4_t _sum1 = __riscv_vget_v_f32m8_f32m4(_sum, 1);
        _sum0 = __riscv_vfadd_vv_f32m4(_sum0, _sum1, __riscv_vsetvlmax_e32m4());
        vfloat32m2_t _sum2 = __riscv_vget_v_f32m4_f32m2(_sum0, 0);
        vfloat32m2_t _sum3 = __riscv_vget_v_f32m4_f32m2(_sum0, 1);
        _sum2 = __riscv_vfadd_vv_f32m2(_sum2, _sum3, __riscv_vsetvlmax_e32m2());
        vfloat32m1_t _sum4 = __riscv_vget_v_f32m2_f32m1(_sum2, 0);
        vfloat32m1_t _sum5 = __riscv_vget_v_f32m2_f32m1(_sum2, 1);
        _sum4 = __riscv_vfadd_vv_f32m1(_sum4, _sum5, __riscv_vsetvlmax_e32m1());
        _sum = __riscv_vcreate_v_f32m1_f32m8(_sum4, _sum4, _sum4, _sum4, _sum4, _sum4, _sum4, _sum4);
    }
    if (elempack == 1)
    {
        // reduce sum n,n,n,n,n,n,n,n to 1
        // broadcast 1 to n,n,n,n,n,n,n,n

        vfloat32m1_t _sum0 = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
        _sum0 = __riscv_vfredusum_vs_f32m8_f32m1(_sum, _sum0, __riscv_vsetvlmax_e32m8());
        _sum = __riscv_vset_v_f32m1_f32m8(_sum, 0, _sum0);
        _sum = __riscv_vrgather_vx_f32m8(_sum, 0, __riscv_vsetvlmax_e32m8());
    }

    _sum = __riscv_vfrdiv_vf_f32m8(_sum, 1.f, __riscv_vsetvlmax_e32m8());

    // div sum
    {
        float* ptr = _ptr;

        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            _p = __riscv_vfmul_vv_f32m8(_p, _sum, vl);
            __riscv_vse32_v_f32m8(ptr, _p, vl);
            n -= vl;
            ptr += vl;
        }
    }
#else  // __riscv_vector
    float max = -FLT_MAX;
    {
        const float* ptr = _ptr;

        for (int i = 0; i < size; i++)
        {
            max = std::max(max, *ptr++);
        }
    }

    // reduce exp(x - max)
    float sum = 0.f;
    {
        float* ptr = _ptr;

        for (int i = 0; i < size; i++)
        {
            float v = expf(*ptr - max);
            *ptr = v;
            sum += v;
            ptr++;
        }
    }

    sum = 1.f / sum;

    // div sum
    {
        float* ptr = _ptr;

        for (int i = 0; i < size; i++)
        {
            *ptr++ *= sum;
        }
    }
#endif // __riscv_vector
}

#if __riscv_vector
static void softmax_unrollm8(float* _ptr, int elemcount, int elempack, int stride, size_t vl)
{
    // NCNN_LOGE("softmax_unrollm8 %d %d %d  %lu", elemcount, elempack, stride, vl);

    const int packn = csrr_vlenb() / 4;

    // reduce max
    vfloat32m8_t _max = __riscv_vfmv_v_f_f32m8(-FLT_MAX, vl);
    {
        const float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            _max = __riscv_vfmax_vv_f32m8(_max, _p, vl);
            ptr += stride;
        }
    }

    if (elempack == packn)
    {
        // reduce max n,n,n,n,n,n,n,n to 1,1,1,1,1,1,1,1
        // broadcast 1,1,1,1,1,1,1,1 to n,n,n,n,n,n,n,n

        // but there is no __riscv_vfredmax_vs_f32m8_f32m8  :(
        const size_t vlm1 = __riscv_vsetvlmax_e32m1();
        const size_t vl0 = vl < vlm1 * 1 ? vl - vlm1 * 0 : vlm1;
        const size_t vl1 = vl < vlm1 * 2 ? vl - vlm1 * 1 : vlm1;
        const size_t vl2 = vl < vlm1 * 3 ? vl - vlm1 * 2 : vlm1;
        const size_t vl3 = vl < vlm1 * 4 ? vl - vlm1 * 3 : vlm1;
        const size_t vl4 = vl < vlm1 * 5 ? vl - vlm1 * 4 : vlm1;
        const size_t vl5 = vl < vlm1 * 6 ? vl - vlm1 * 5 : vlm1;
        const size_t vl6 = vl < vlm1 * 7 ? vl - vlm1 * 6 : vlm1;
        const size_t vl7 = vl < vlm1 * 8 ? vl - vlm1 * 7 : vlm1;
        vfloat32m1_t _neginf = __riscv_vfmv_s_f_f32m1(-FLT_MAX, vlm1);
        vfloat32m1_t _max0 = __riscv_vfredmax_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_max, 0), _neginf, vl0);
        vfloat32m1_t _max1 = __riscv_vfredmax_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_max, 1), _neginf, vl1);
        vfloat32m1_t _max2 = __riscv_vfredmax_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_max, 2), _neginf, vl2);
        vfloat32m1_t _max3 = __riscv_vfredmax_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_max, 3), _neginf, vl3);
        vfloat32m1_t _max4 = __riscv_vfredmax_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_max, 4), _neginf, vl4);
        vfloat32m1_t _max5 = __riscv_vfredmax_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_max, 5), _neginf, vl5);
        vfloat32m1_t _max6 = __riscv_vfredmax_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_max, 6), _neginf, vl6);
        vfloat32m1_t _max7 = __riscv_vfredmax_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_max, 7), _neginf, vl7);
        _max0 = __riscv_vrgather_vx_f32m1(_max0, 0, vl0);
        _max1 = __riscv_vrgather_vx_f32m1(_max1, 0, vl1);
        _max2 = __riscv_vrgather_vx_f32m1(_max2, 0, vl2);
        _max3 = __riscv_vrgather_vx_f32m1(_max3, 0, vl3);
        _max4 = __riscv_vrgather_vx_f32m1(_max4, 0, vl4);
        _max5 = __riscv_vrgather_vx_f32m1(_max5, 0, vl5);
        _max6 = __riscv_vrgather_vx_f32m1(_max6, 0, vl6);
        _max7 = __riscv_vrgather_vx_f32m1(_max7, 0, vl7);
        _max = __riscv_vset_v_f32m1_f32m8(_max, 0, _max0);
        _max = __riscv_vset_v_f32m1_f32m8(_max, 1, _max1);
        _max = __riscv_vset_v_f32m1_f32m8(_max, 2, _max2);
        _max = __riscv_vset_v_f32m1_f32m8(_max, 3, _max3);
        _max = __riscv_vset_v_f32m1_f32m8(_max, 4, _max4);
        _max = __riscv_vset_v_f32m1_f32m8(_max, 5, _max5);
        _max = __riscv_vset_v_f32m1_f32m8(_max, 6, _max6);
        _max = __riscv_vset_v_f32m1_f32m8(_max, 7, _max7);
    }

    // reduce exp(x - max)
    vfloat32m8_t _sum = __riscv_vfmv_v_f_f32m8(0.f, vl);
    {
        float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            _p = __riscv_vfsub_vv_f32m8(_p, _max, vl);
            _p = exp_ps(_p, vl);
            __riscv_vse32_v_f32m8(ptr, _p, vl);
            _sum = __riscv_vfadd_vv_f32m8(_sum, _p, vl);
            ptr += stride;
        }
    }

    if (elempack == packn)
    {
        // reduce sum n,n,n,n,n,n,n,n to 1,1,1,1,1,1,1,1
        // broadcast 1,1,1,1,1,1,1,1 to n,n,n,n,n,n,n,n

        // but there is no __riscv_vfredusum_vs_f32m8_f32m8  :(
        const size_t vlm1 = __riscv_vsetvlmax_e32m1();
        const size_t vl0 = vl < vlm1 * 1 ? vl - vlm1 * 0 : vlm1;
        const size_t vl1 = vl < vlm1 * 2 ? vl - vlm1 * 1 : vlm1;
        const size_t vl2 = vl < vlm1 * 3 ? vl - vlm1 * 2 : vlm1;
        const size_t vl3 = vl < vlm1 * 4 ? vl - vlm1 * 3 : vlm1;
        const size_t vl4 = vl < vlm1 * 5 ? vl - vlm1 * 4 : vlm1;
        const size_t vl5 = vl < vlm1 * 6 ? vl - vlm1 * 5 : vlm1;
        const size_t vl6 = vl < vlm1 * 7 ? vl - vlm1 * 6 : vlm1;
        const size_t vl7 = vl < vlm1 * 8 ? vl - vlm1 * 7 : vlm1;
        vfloat32m1_t _zero = __riscv_vfmv_s_f_f32m1(0.f, vlm1);
        vfloat32m1_t _sum0 = __riscv_vfredusum_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_sum, 0), _zero, vl0);
        vfloat32m1_t _sum1 = __riscv_vfredusum_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_sum, 1), _zero, vl1);
        vfloat32m1_t _sum2 = __riscv_vfredusum_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_sum, 2), _zero, vl2);
        vfloat32m1_t _sum3 = __riscv_vfredusum_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_sum, 3), _zero, vl3);
        vfloat32m1_t _sum4 = __riscv_vfredusum_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_sum, 4), _zero, vl4);
        vfloat32m1_t _sum5 = __riscv_vfredusum_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_sum, 5), _zero, vl5);
        vfloat32m1_t _sum6 = __riscv_vfredusum_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_sum, 6), _zero, vl6);
        vfloat32m1_t _sum7 = __riscv_vfredusum_vs_f32m1_f32m1(__riscv_vget_v_f32m8_f32m1(_sum, 7), _zero, vl7);
        _sum0 = __riscv_vrgather_vx_f32m1(_sum0, 0, vl0);
        _sum1 = __riscv_vrgather_vx_f32m1(_sum1, 0, vl1);
        _sum2 = __riscv_vrgather_vx_f32m1(_sum2, 0, vl2);
        _sum3 = __riscv_vrgather_vx_f32m1(_sum3, 0, vl3);
        _sum4 = __riscv_vrgather_vx_f32m1(_sum4, 0, vl4);
        _sum5 = __riscv_vrgather_vx_f32m1(_sum5, 0, vl5);
        _sum6 = __riscv_vrgather_vx_f32m1(_sum6, 0, vl6);
        _sum7 = __riscv_vrgather_vx_f32m1(_sum7, 0, vl7);
        _sum = __riscv_vset_v_f32m1_f32m8(_sum, 0, _sum0);
        _sum = __riscv_vset_v_f32m1_f32m8(_sum, 1, _sum1);
        _sum = __riscv_vset_v_f32m1_f32m8(_sum, 2, _sum2);
        _sum = __riscv_vset_v_f32m1_f32m8(_sum, 3, _sum3);
        _sum = __riscv_vset_v_f32m1_f32m8(_sum, 4, _sum4);
        _sum = __riscv_vset_v_f32m1_f32m8(_sum, 5, _sum5);
        _sum = __riscv_vset_v_f32m1_f32m8(_sum, 6, _sum6);
        _sum = __riscv_vset_v_f32m1_f32m8(_sum, 7, _sum7);
    }

    _sum = __riscv_vfrdiv_vf_f32m8(_sum, 1.f, vl);

    // div sum
    {
        float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            _p = __riscv_vfmul_vv_f32m8(_p, _sum, vl);
            __riscv_vse32_v_f32m8(ptr, _p, vl);
            ptr += stride;
        }
    }
}
#endif // __riscv_vector

static void softmax_unroll2(float* _ptr, int elemcount, int /*elempack*/, int stride)
{
    // assert elempack == 1

    // reduce max
    float max0 = -FLT_MAX;
    float max1 = -FLT_MAX;
    {
        const float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            max0 = std::max(max0, ptr[0]);
            max1 = std::max(max1, ptr[1]);
            ptr += stride;
        }
    }

    // reduce exp(x - max)
    float sum0 = 0.f;
    float sum1 = 0.f;
    {
        float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            float v0 = expf(ptr[0] - max0);
            float v1 = expf(ptr[1] - max1);
            ptr[0] = v0;
            ptr[1] = v1;
            sum0 += v0;
            sum1 += v1;
            ptr += stride;
        }
    }

    sum0 = 1.f / sum0;
    sum1 = 1.f / sum1;

    // div sum
    {
        float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            ptr[0] *= sum0;
            ptr[1] *= sum1;
            ptr += stride;
        }
    }
}

static void softmax(float* _ptr, int elemcount, int /*elempack*/, int stride)
{
    // assert elempack == 1

    // reduce max
    float max = -FLT_MAX;
    {
        const float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            max = std::max(max, *ptr);
            ptr += stride;
        }
    }

    // reduce exp(x - max)
    float sum = 0.f;
    {
        float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            float v = expf(*ptr - max);
            *ptr = v;
            sum += v;
            ptr += stride;
        }
    }

    sum = 1.f / sum;

    // div sum
    {
        float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            *ptr *= sum;
            ptr += stride;
        }
    }
}

int Softmax_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;
    const int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1) // positive_axis == 0
    {
        float* ptr = bottom_top_blob;

        const int size = w * elempack;

        softmax(ptr, size, 1);
    }

    if (dims == 2 && positive_axis == 0)
    {
        const int size = w * elempack;

#if __riscv_vector
        const int packn = csrr_vlenb() / 4;
        const int sizen = (size / opt.num_threads + (packn - 1)) / packn * packn;
        int nn_size = size / sizen;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            const int i = ii * sizen;
            const int size1 = std::min(sizen, size - i);

            float* ptr = (float*)bottom_top_blob + i;

            int n = size1;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);

                softmax_unrollm8(ptr, h, elempack, size, vl);

                ptr += vl;
                n -= vl;
            }
        }
#else // __riscv_vector
        int nn_size = size / 2;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            const int i = ii * 2;
            float* ptr = (float*)bottom_top_blob + i;

            softmax_unroll2(ptr, h, elempack, size);
        }
        int i = nn_size * 2;
        for (; i < size; i++)
        {
            float* ptr = (float*)bottom_top_blob + i;

            softmax(ptr, h, elempack, size);
        }
#endif // __riscv_vector
    }

    if (dims == 2 && positive_axis == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);

            softmax(ptr, w, elempack);
        }
    }

    if ((dims == 3 || dims == 4) && positive_axis == 0)
    {
        const int size = w * h * d * elempack;
        const int stride = bottom_top_blob.cstep * elempack;

#if __riscv_vector
        const int packn = csrr_vlenb() / 4;
        const int sizen = (size / opt.num_threads + (packn - 1)) / packn * packn;
        int nn_size = size / sizen;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            const int i = ii * sizen;
            const int size1 = std::min(sizen, size - i);

            float* ptr = (float*)bottom_top_blob + i;

            int n = size1;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);

                softmax_unrollm8(ptr, channels, elempack, stride, vl);

                ptr += vl;
                n -= vl;
            }
        }
#else // __riscv_vector
        int nn_size = size / 2;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            const int i = ii * 2;
            float* ptr = (float*)bottom_top_blob + i;

            softmax_unroll2(ptr, channels, elempack, stride);
        }
        int i = nn_size * 2;
        for (; i < size; i++)
        {
            float* ptr = (float*)bottom_top_blob + i;

            softmax(ptr, channels, elempack, stride);
        }
#endif // __riscv_vector
    }

    if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            for (int i = 0; i < d; i++)
            {
                float* ptr = bottom_top_blob.channel(q).depth(i);

                const int size = w * elempack;

#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);

                    softmax_unrollm8(ptr, h, 1, size, vl);

                    ptr += vl;
                    n -= vl;
                }
#else  // __riscv_vector
                int j = 0;
                for (; j + 1 < size; j += 2)
                {
                    softmax_unroll2(ptr, h, 1, size);
                    ptr += 2;
                }
                for (; j < size; j++)
                {
                    softmax(ptr, h, 1, size);
                    ptr++;
                }
#endif // __riscv_vector
            }
        }
    }

    if (dims == 3 && positive_axis == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                softmax(ptr, w, elempack);
                ptr += w * elempack;
            }
        }
    }

    if (dims == 4 && positive_axis == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            const int size = w * h * elempack;

#if __riscv_vector
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);

                softmax_unrollm8(ptr, d, 1, size, vl);

                ptr += vl;
                n -= vl;
            }
#else  // __riscv_vector
            int i = 0;
            for (; i + 1 < size; i += 2)
            {
                softmax_unroll2(ptr, d, 1, size);
                ptr += 2;
            }
            for (; i < size; i++)
            {
                softmax(ptr, d, 1, size);
                ptr++;
            }
#endif // __riscv_vector
        }
    }

    if (dims == 4 && positive_axis == 3)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    softmax(ptr, w, elempack);
                    ptr += w * elempack;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
