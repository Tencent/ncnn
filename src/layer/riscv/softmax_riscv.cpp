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
#include "rvv_mathfun.h"
#endif // __riscv_vector

namespace ncnn {

Softmax_riscv::Softmax_riscv()
{
}

int Softmax_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    size_t elemsize = bottom_top_blob.elemsize;
    int elempack = bottom_top_blob.elempack;

    int positive_axis = axis < 0 ? dims + axis : axis;
#ifdef __riscv_vector
    if (dims == 1) // positive_axis == 0
    {
        int w = bottom_top_blob.w;
        float* ptr = bottom_top_blob;
        float max = -FLT_MAX;

        int n = w * elempack;
        float* ptr_vol = ptr;
        while (n > 0)
        {
            size_t vl = vsetvl_e32m8(n);

            vfloat32m8_t _p = vle32_v_f32m8(ptr_vol, vl);
            vfloat32m1_t _max = vfmv_s_f_f32m1(vundefined_f32m1(), max, vl);
            _max = vfredmax_vs_f32m8_f32m1(_max, _p, /* scalar*/ _max, vl);

            max = vfmv_f_s_f32m1_f32(_max);
            ptr_vol += vl;
            n -= vl;
        }
        ptr_vol = NULL;

        float sum = 0.f;
        n = w * elempack;
        ptr_vol = ptr;
        while (n > 0)
        {
            size_t vl = vsetvl_e32m8(n);
            vfloat32m1_t _sum = vfmv_s_f_f32m1(vundefined_f32m1(), sum, vl);
            vfloat32m8_t _p = vle32_v_f32m8(ptr_vol, vl);

            _p = vfsub_vf_f32m8(_p, max, vl);
            _p = exp_ps(_p, vl);
            _sum = vfredusum_vs_f32m8_f32m1(_sum, _p, /*scalar*/ _sum, vl);

            vse32_v_f32m8(ptr_vol, _p, vl);
            sum = vfmv_f_s_f32m1_f32(_sum);
            ptr_vol += vl;
            n -= vl;
        }
        ptr_vol = NULL;

        n = w * elempack;
        ptr_vol = ptr;
        while (n > 0)
        {
            size_t vl = vsetvl_e32m8(n);

            vfloat32m8_t _p = vle32_v_f32m8(ptr_vol, vl);
            _p = vfdiv_vf_f32m8(_p, sum, vl);
            vse32_v_f32m8(ptr_vol, _p, vl);

            n -= vl;
            ptr_vol += vl;
        }
        ptr_vol = NULL;

        return 0;
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
            float* ptr_max = max;
            int n = w * elempack;
            while (n > 0)
            {
                size_t vl = vsetvl_e32m8(n);

                vfloat32m8_t _max = vle32_v_f32m8(ptr_max, vl);
                vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);

                _max = vfmax_vv_f32m8(_max, _p, vl);

                vse32_v_f32m8(ptr_max, _max, vl);
                ptr += vl;
                ptr_max += vl;
                n -= vl;
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
            float* ptr_max = max;
            float* ptr_sum = sum;
            int n = w * elempack;

            while (n > 0)
            {
                size_t vl = vsetvl_e32m8(n);

                vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                vfloat32m8_t _max = vle32_v_f32m8(ptr_max, vl);
                vfloat32m8_t _sum = vle32_v_f32m8(ptr_sum, vl);

                _p = vfsub_vv_f32m8(_p, _max, vl);
                _p = exp_ps(_p, vl);
                _sum = vfadd_vv_f32m8(_sum, _p, vl);

                vse32_v_f32m8(ptr, _p, vl);
                vse32_v_f32m8(ptr_sum, _sum, vl);
                n -= vl;
                ptr_max += vl;
                ptr_sum += vl;
                ptr += vl;
            }
        }

        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            float* ptr_sum = sum;

            int n = w * elempack;
            while (n > 0)
            {
                size_t vl = vsetvl_e32m8(n);
                vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                vfloat32m8_t _sum = vle32_v_f32m8(ptr_sum, vl);

                _p = vfdiv_vv_f32m8(_p, _sum, vl);

                vse32_v_f32m8(ptr, _p, vl);
                n -= vl;
                ptr += vl;
                ptr_sum += vl;
            }
        }

        return 0;
    }

    if (dims == 2 && positive_axis == 1)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            float m = -FLT_MAX;

            int n1 = w * elempack;
            float* ptr1 = ptr;
            while (n1 > 0)
            {
                size_t vl = vsetvl_e32m8(n1);
                vfloat32m8_t _p = vle32_v_f32m8(ptr1, vl);
                vfloat32m1_t _m = vfmv_s_f_f32m1(vundefined_f32m1(), m, vl);

                _m = vfredmax_vs_f32m8_f32m1(_m, _p, _m, vl);

                m = vfmv_f_s_f32m1_f32(_m);
                ptr1 += vl;
                n1 -= vl;
            }
            ptr1 = NULL;

            float s = 0.f;
            int n2 = w * elempack;
            float* ptr2 = ptr;
            while (n2 > 0)
            {
                size_t vl = vsetvl_e32m8(n2);
                vfloat32m8_t _p = vle32_v_f32m8(ptr2, vl);
                vfloat32m1_t _s = vfmv_s_f_f32m1(vundefined_f32m1(), s, vl);

                _p = exp_ps(vfsub_vf_f32m8(_p, m, vl), vl);
                _s = vfredusum_vs_f32m8_f32m1(_s, _p, _s, vl);

                vse32_v_f32m8(ptr2, _p, vl);
                s = vfmv_f_s_f32m1_f32(_s);
                ptr2 += vl;
                n2 -= vl;
            }
            ptr2 = NULL;

            int n3 = w * elempack;
            float* ptr3 = ptr;
            while (n3 > 0)
            {
                size_t vl = vsetvl_e32m8(n3);

                vfloat32m8_t _p = vle32_v_f32m8(ptr3, vl);

                _p = vfdiv_vf_f32m8(_p, s, vl);

                vse32_v_f32m8(ptr3, _p, vl);
                n3 -= vl;
                ptr3 += vl;
            }
            ptr3 = NULL;
        }

        return 0;
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
            const float* ptr = bottom_top_blob.channel(q);

            float* ptr_max = max;
            int n = size * elempack;
            while (n > 0)
            {
                size_t vl = vsetvl_e32m8(n);

                vfloat32m8_t _max = vle32_v_f32m8(max, vl);
                vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                _max = vfmax_vv_f32m8(_max, _p, vl);
                vse32_v_f32m8(ptr_max, _max, vl);

                ptr += vl;
                ptr_max += vl;
                n -= vl;
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
            float* ptr_sum = sum;
            float* ptr_max = max;
            int n = size * elempack;
            while (n > 0)
            {
                size_t vl = vsetvl_e32m8(n);
                vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                vfloat32m8_t _max = vle32_v_f32m8(ptr_max, vl);
                vfloat32m8_t _sum = vle32_v_f32m8(ptr_sum, vl);
                _p = exp_ps(vfsub_vv_f32m8(_p, _max, vl), vl);
                _sum = vfadd_vv_f32m8(_sum, _p, vl);
                vse32_v_f32m8(ptr, _p, vl);
                vse32_v_f32m8(ptr_sum, _sum, vl);

                n -= vl;
                ptr += vl;
                ptr_sum += vl;
                ptr_max += vl;
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            float* ptr_sum = sum;
            int n = size * elempack;
            while (n > 0)
            {
                size_t vl = vsetvl_e32m8(n);
                vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                vfloat32m8_t _sum = vle32_v_f32m8(ptr_sum, vl);

                _p = vfdiv_vv_f32m8(_p, _sum, vl);
                vse32_v_f32m8(ptr, _p, vl);

                ptr_sum += vl;
                ptr += vl;
                n -= vl;
            }
        }

        return 0;
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
                float* maxptr_vol = maxptr;
                int n = w * elempack;
                while (n > 0)
                {
                    size_t vl = vsetvl_e32m8(n);
                    vfloat32m8_t _maxptr = vle32_v_f32m8(maxptr_vol, vl);
                    vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);

                    _maxptr = vfmax_vv_f32m8(_maxptr, _p, vl);
                    vse32_v_f32m8(maxptr_vol, _maxptr, vl);

                    ptr += vl;
                    maxptr_vol += vl;
                    n -= vl;
                }
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
                float* sumptr_vol = sumptr;
                float* maxptr_vol = maxptr;
                int n = w * elempack;

                while (n)
                {
                    size_t vl = vsetvl_e32m8(n);
                    vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                    vfloat32m8_t _maxptr = vle32_v_f32m8(maxptr_vol, vl);
                    vfloat32m8_t _sumptr = vle32_v_f32m8(sumptr_vol, vl);

                    _p = exp_ps(vfsub_vv_f32m8(_p, _maxptr, vl), vl);
                    _sumptr = vfadd_vv_f32m8(_sumptr, _p, vl);

                    vse32_v_f32m8(ptr, _p, vl);
                    vse32_v_f32m8(sumptr_vol, _sumptr, vl);
                    n -= vl;
                    sumptr_vol += vl;
                    maxptr_vol += vl;
                    ptr += vl;
                }
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            float* sumptr = sum.row(q);

            for (int i = 0; i < h; i++)
            {
                float* sumptr_vol = sumptr;
                int n = w * elempack;
                while (n > 0)
                {
                    size_t vl = vsetvl_e32m8(n);
                    vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                    vfloat32m8_t _sumptr = vle32_v_f32m8(sumptr_vol, vl);

                    _p = vfdiv_vv_f32m8(_p, _sumptr, vl);

                    vse32_v_f32m8(ptr, _p, vl);
                    n -= vl;
                    sumptr_vol += vl;
                    ptr += vl;
                }
            }
        }

        return 0;
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
                int n1 = w * elempack;
                float* ptr_1 = ptr;
                while (n1 > 0)
                {
                    size_t vl = vsetvl_e32m8(n1);
                    vfloat32m8_t _p = vle32_v_f32m8(ptr_1, vl);
                    vfloat32m1_t _scalar_max = vfmv_s_f_f32m1(vundefined_f32m1(), max, vl);
                    _scalar_max = vfredmax_vs_f32m8_f32m1(_scalar_max, _p, _scalar_max, vl);

                    max = vfmv_f_s_f32m1_f32(_scalar_max);
                    n1 -= vl;
                    ptr_1 += vl;
                }
                ptr_1 = NULL;

                float sum = 0.f;
                int n2 = w * elempack;
                float* ptr_2 = ptr;
                while (n2 > 0)
                {
                    size_t vl = vsetvl_e32m8(n2);
                    vfloat32m8_t _p = vle32_v_f32m8(ptr_2, vl);
                    vfloat32m1_t _scalar_sum = vfmv_s_f_f32m1(vundefined_f32m1(), sum, vl);

                    _p = exp_ps(vfsub_vf_f32m8(_p, max, vl), vl);
                    _scalar_sum = vfredusum_vs_f32m8_f32m1(_scalar_sum, _p, _scalar_sum, vl);

                    vse32_v_f32m8(ptr_2, _p, vl);
                    sum = vfmv_f_s_f32m1_f32(_scalar_sum);
                    n2 -= vl;
                    ptr_2 += vl;
                }
                ptr_2 = NULL;

                int n3 = w * elempack;
                float* ptr_3 = ptr;
                while (n3 > 0)
                {
                    size_t vl = vsetvl_e32m8(n3);
                    vfloat32m8_t _p = vle32_v_f32m8(ptr_3, vl);

                    _p = vfdiv_vf_f32m8(_p, sum, vl);

                    vse32_v_f32m8(ptr_3, _p, vl);
                    n3 -= vl;
                    ptr_3 += vl;
                }
                ptr_3 = NULL;
                ptr += w;
            }
        }

        return 0;
    }

    return 0;
#endif
    return Softmax::forward_inplace(bottom_top_blob, opt);
}

} // namespace ncnn
