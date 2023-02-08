// Xavier Hsinyuan is pleased to support the open source community by making
// ncnn available.
//
// Copyright (C) 2021 Xavier Hsinyuan <me@lstlx.com> All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#include "binaryop_riscv.h"

#include <math.h>

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun.h"
#include "rvv_mathfun_fp16s.h"
#endif // __riscv_vector

#include "riscv_usability.h"

namespace ncnn {

BinaryOp_riscv::BinaryOp_riscv()
{
#if __riscv_vector
    support_packing = true;
#if __riscv_zfh
    support_fp16_storage = true;
#endif
#endif
}

template<typename Op>
static int binary_op_scalar(const Mat& a, float b, Mat& c, const Option& opt)
{
    Op op;

    const int channels = a.c;
    const int size = a.w * a.h * a.d * a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = a.channel(q);
        float* outptr = c.channel(q);

#if __riscv_vector
        int n = size;
        while (n > 0)
        {
            size_t vl = vsetvl_e32m8(n);
            vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
            _p = op(_p, b, vl);
            vse32_v_f32m8(outptr, _p, vl);
            n -= vl;
            ptr += vl;
            outptr += vl;
        }
#else
        for (int i = 0; i < size; i++)
        {
            *outptr = op(*ptr, b);
            ptr++;
            outptr++;
        }
#endif
    }

    return 0;
}

template<typename Op>
static int binary_op_no_broadcast(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    Op op;

    const int channels = a.c;
    const int size = a.w * a.h * a.d * a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = a.channel(q);
        const float* ptr1 = b.channel(q);
        float* outptr = c.channel(q);

#if __riscv_vector
        int n = size;
        while (n > 0)
        {
            size_t vl = vsetvl_e32m8(n);
            vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
            vfloat32m8_t _p1 = vle32_v_f32m8(ptr1, vl);
            vfloat32m8_t _outp = op(_p, _p1, vl);
            vse32_v_f32m8(outptr, _outp, vl);
            n -= vl;
            ptr += vl;
            ptr1 += vl;
            outptr += vl;
        }
#else
        for (int i = 0; i < size; i++)
        {
            *outptr = op(*ptr, *ptr1);
            ptr += 1;
            ptr1 += 1;
            outptr += 1;
        }
#endif
    }

    return 0;
}

template<typename Op>
static int binary_op_broadcast_inner(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int d = a.d;
    int channels = a.c;
    int elempack = a.elempack;

    if (a.dims == 2 && b.dims == 1)
    {
        // type 8
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < h; y++)
        {
            const float* ptr = a.row(y);
            float* outptr = c.row(y);

            const float _b = b[y];

            const int size = w * elempack;

#if __riscv_vector
            int n = size;
            vfloat32m8_t _bx = (elempack == 1) ? vfmv_v_f_f32m8(_b, vsetvl_e32m8(n)) : vle32_v_f32m8_f32m1((const float*)b + y * elempack);
            while (n > 0)
            {
                size_t vl = vsetvl_e32m8(n);
                vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                vfloat32m8_t _outp = op(_p, _bx, vl);
                vse32_v_f32m8(outptr, _outp, vl);
                n -= vl;
                ptr += vl;
                outptr += vl;
            }
#else
            for (int i = 0; i < size; i++)
            {
                *outptr = op(*ptr, _b);
                ptr += 1;
                outptr += 1;
            }
#endif
        }
    }

    if ((a.dims == 3 || a.dims == 4) && b.dims == 1)
    {
        // type 9 11
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = a.channel(q);
            float* outptr = c.channel(q);

            const float _b = b[q];

            const int size = w * h * d * elempack;

#if __riscv_vector
            int n = size;
            vfloat32m8_t _bx = (elempack == 1) ? vfmv_v_f_f32m8(_b, vsetvl_e32m8(n)) : vle32_v_f32m8_f32m1((const float*)b + q * elempack);
            while (n > 0)
            {
                size_t vl = vsetvl_e32m8(n);
                vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                vfloat32m8_t _outp = op(_p, _bx, vl);
                vse32_v_f32m8(outptr, _outp, vl);
                n -= vl;
                ptr += vl;
                outptr += vl;
            }
#else
            for (int i = 0; i < size; i++)
            {
                *outptr = op(*ptr, _b);
                ptr += 1;
                outptr += 1;
            }
#endif
        }
    }

    if (a.dims == 3 && b.dims == 2)
    {
        // type 10
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = a.channel(q);
            const float* ptr1 = b.row(q);
            float* outptr = c.channel(q);

            const int size = w * elempack;

            for (int y = 0; y < h; y++)
            {
                const float _b = ptr1[y];

#if __riscv_vector
                int n = size;
                vfloat32m8_t _bx = (elempack == 1) ? vfmv_v_f_f32m8(_b, vsetvl_e32m8(n)) : vle32_v_f32m8_f32m1(ptr1 + y * elempack);
                while (n > 0)
                {
                    size_t vl = vsetvl_e32m8(n);
                    vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                    vfloat32m8_t _outp = op(_p, _bx, vl);
                    vse32_v_f32m8(outptr, _outp, vl);
                    n -= vl;
                    ptr += vl;
                    outptr += vl;
                }
#else
                for (int i = 0; i < size; i++)
                {
                    *outptr = op(*ptr, _b);
                    ptr += 1;
                    outptr += 1;
                }
#endif
            }
        }
    }

    if (a.dims == 4 && b.dims == 2)
    {
        // type 12
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = a.channel(q);
            const float* ptr1 = b.row(q);
            float* outptr = c.channel(q);

            const int size = w * h * elempack;

            for (int z = 0; z < d; z++)
            {
                const float _b = ptr1[z];

#if __riscv_vector
                int n = size;
                vfloat32m8_t _bx = (elempack == 1) ? vfmv_v_f_f32m8(_b, vsetvl_e32m8(n)) : vle32_v_f32m8_f32m1(ptr1 + z * elempack);
                while (n > 0)
                {
                    size_t vl = vsetvl_e32m8(n);
                    vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                    vfloat32m8_t _outp = op(_p, _bx, vl);
                    vse32_v_f32m8(outptr, _outp, vl);
                    n -= vl;
                    ptr += vl;
                    outptr += vl;
                }
#else
                for (int i = 0; i < size; i++)
                {
                    *outptr = op(*ptr, _b);
                    ptr += 1;
                    outptr += 1;
                }
#endif
            }
        }
    }

    if (a.dims == 4 && b.dims == 3)
    {
        // type 13
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = a.channel(q);
            float* outptr = c.channel(q);

            const int size = w * elempack;

            for (int z = 0; z < d; z++)
            {
                const float* ptr1 = b.channel(q).row(z);

                for (int y = 0; y < h; y++)
                {
                    const float _b = ptr1[y];

#if __riscv_vector
                    int n = size;
                    vfloat32m8_t _bx = (elempack == 1) ? vfmv_v_f_f32m8(_b, vsetvl_e32m8(n)) : vle32_v_f32m8_f32m1(ptr1 + y * elempack);
                    while (n > 0)
                    {
                        size_t vl = vsetvl_e32m8(n);
                        vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                        vfloat32m8_t _outp = op(_p, _bx, vl);
                        vse32_v_f32m8(outptr, _outp, vl);
                        n -= vl;
                        ptr += vl;
                        outptr += vl;
                    }
#else
                    for (int i = 0; i < size; i++)
                    {
                        *outptr = op(*ptr, _b);
                        ptr += 1;
                        outptr += 1;
                    }
#endif
                }
            }
        }
    }

    return 0;
}

template<typename Op>
static int binary_op_broadcast_outer(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int d = a.d;
    int channels = a.c;
    int elempack = a.elempack;

    if (a.dims == 2)
    {
        // type 14
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < h; y++)
        {
            const float* ptr = a.row(y);
            const float* ptr1 = b;
            float* outptr = c.row(y);

#if __riscv_vector
            if (elempack != 1)
            {
                for (int x = 0; x < w; x++)
                {
                    int n = elempack;
                    while (n > 0)
                    {
                        size_t vl = vsetvl_e32m8(n);
                        vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                        vfloat32m8_t _outp = op(_p, *ptr1, vl);
                        vse32_v_f32m8(outptr, _outp, vl);
                        n -= vl;
                        ptr += vl;
                        outptr += vl;
                    }
                    ptr1 += 1;
                }
            }
#endif
            if (elempack == 1)
            {
                for (int x = 0; x < w; x++)
                {
                    *outptr = op(*ptr, *ptr1);
                    ptr += 1;
                    ptr1 += 1;
                    outptr += 1;
                }
            }
        }
    }

    if (a.dims == 3 || a.dims == 4)
    {
        // type 15 16 17 18 19
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = a.channel(q);
            float* outptr = c.channel(q);

            for (int z = 0; z < d; z++)
            {
                int z1 = std::min(z, b.d - 1);
                for (int y = 0; y < h; y++)
                {
                    int y1 = std::min(y, b.h - 1);

                    const float* ptr1 = b.depth(z1).row(y1);

#if __riscv_vector
                    if (elempack != 1)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            int n = elempack;
                            while (n > 0)
                            {
                                size_t vl = vsetvl_e32m8(n);
                                vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                                vfloat32m8_t _outp = op(_p, *ptr1, vl);
                                vse32_v_f32m8(outptr, _outp, vl);
                                n -= vl;
                                ptr += vl;
                                outptr += vl;
                            }
                            ptr1 += 1;
                        }
                    }
#endif
                    if (elempack == 1)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            *outptr = op(*ptr, *ptr1);
                            ptr += 1;
                            ptr1 += 1;
                            outptr += 1;
                        }
                    }
                }
            }
        }
    }

    return 0;
}

template<typename Op>
static int binary_op_broadcast_20(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int elempack = a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = a.channel(q);
        float* outptr = c.channel(q);

        for (int y = 0; y < h; y++)
        {
            const float* ptr1 = b.channel(q);

            const int size = w * elempack;

#if __riscv_vector
            int n = size;
            while (n > 0)
            {
                size_t vl = vsetvl_e32m8(n);
                vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                vfloat32m8_t _p1 = vle32_v_f32m8(ptr1, vl);
                vfloat32m8_t _outp = op(_p, _p1, vl);
                vse32_v_f32m8(outptr, _outp, vl);
                n -= vl;
                ptr += vl;
                ptr1 += vl;
                outptr += vl;
            }
#else
            for (int i = 0; i < size; i++)
            {
                *outptr = op(*ptr, *ptr1);
                ptr += 1;
                ptr1 += 1;
                outptr += 1;
            }
#endif
        }
    }

    return 0;
}

template<typename Op>
static int binary_op_scalar_inplace(Mat& a, float b, const Option& opt)
{
    Op op;

    const int channels = a.c;
    const int size = a.w * a.h * a.d * a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = a.channel(q);

#if __riscv_vector
        int n = size;
        while (n > 0)
        {
            size_t vl = vsetvl_e32m8(n);
            vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
            _p = op(_p, b, vl);
            vse32_v_f32m8(ptr, _p, vl);
            n -= vl;
            ptr += vl;
        }
#else
        for (int i = 0; i < size; i++)
        {
            *ptr = op(*ptr, b);
            ptr++;
        }
#endif
    }

    return 0;
}

namespace BinaryOp_riscv_functor {

#if __riscv_vector
#define MAKE_FUNCTION(NAME, IMPL, IMPLVV, IMPLVS, IMPLSV)                                            \
    struct NAME                                                                                      \
    {                                                                                                \
        float operator()(const float& x, const float& y) const                                       \
        {                                                                                            \
            return IMPL;                                                                             \
        }                                                                                            \
        vfloat32m8_t operator()(const vfloat32m8_t& x, const vfloat32m8_t& y, const size_t vl) const \
        {                                                                                            \
            return IMPLVV;                                                                           \
        }                                                                                            \
        vfloat32m8_t operator()(const vfloat32m8_t& x, const float& y, const size_t vl) const        \
        {                                                                                            \
            return IMPLVS;                                                                           \
        }                                                                                            \
        vfloat32m8_t operator()(const float& x, const vfloat32m8_t& y, const size_t vl) const        \
        {                                                                                            \
            return IMPLSV;                                                                           \
        }                                                                                            \
    };
#else
#define MAKE_FUNCTION(NAME, IMPL, IMPLVV, IMPLVS, IMPLSV)      \
    struct NAME                                                \
    {                                                          \
        float operator()(const float& x, const float& y) const \
        {                                                      \
            return IMPL;                                       \
        }                                                      \
    };
#endif

// clang-format off
// *INDENT-OFF*
MAKE_FUNCTION(binary_op_add, x + y, vfadd_vv_f32m8(x, y, vl), vfadd_vf_f32m8(x, y, vl), vfadd_vf_f32m8(y, x, vl))
MAKE_FUNCTION(binary_op_sub, x - y, vfsub_vv_f32m8(x, y, vl), vfsub_vf_f32m8(x, y, vl), vfrsub_vf_f32m8(y, x, vl))
MAKE_FUNCTION(binary_op_mul, x * y, vfmul_vv_f32m8(x, y, vl), vfmul_vf_f32m8(x, y, vl), vfmul_vf_f32m8(y, x, vl))
MAKE_FUNCTION(binary_op_div, x / y, vfdiv_vv_f32m8(x, y, vl), vfdiv_vf_f32m8(x, y, vl), vfrdiv_vf_f32m8(y, x, vl))
MAKE_FUNCTION(binary_op_max, std::max(x, y), vfmax_vv_f32m8(x, y, vl), vfmax_vf_f32m8(x, y, vl), vfmax_vf_f32m8(y, x, vl))
MAKE_FUNCTION(binary_op_min, std::min(x, y), vfmin_vv_f32m8(x, y, vl), vfmin_vf_f32m8(x, y, vl), vfmin_vf_f32m8(y, x, vl))
MAKE_FUNCTION(binary_op_pow, (float)pow(x, y), pow_ps(x, y, vl), pow_ps(x, vfmv_v_f_f32m8(y, vl), vl), pow_ps(vfmv_v_f_f32m8(x, vl), y, vl))
MAKE_FUNCTION(binary_op_rsub, y - x, vfsub_vv_f32m8(y, x, vl), vfrsub_vf_f32m8(x, y, vl), vfsub_vf_f32m8(y, x, vl))
MAKE_FUNCTION(binary_op_rdiv, y / x, vfdiv_vv_f32m8(y, x, vl), vfrdiv_vf_f32m8(x, y, vl), vfdiv_vf_f32m8(y, x, vl))
MAKE_FUNCTION(binary_op_rpow, (float)pow(y, x), pow_ps(y, x, vl), pow_ps(vfmv_v_f_f32m8(y, vl), x, vl), pow_ps(y, vfmv_v_f_f32m8(x, vl), vl))
MAKE_FUNCTION(binary_op_atan2, (float)atan2(x, y), atan2_ps(x, y, vl), atan2_ps(x, vfmv_v_f_f32m8(y, vl), vl), atan2_ps(vfmv_v_f_f32m8(x, vl), y, vl))
MAKE_FUNCTION(binary_op_ratan2, (float)atan2(y, x), atan2_ps(y, x, vl), atan2_ps(vfmv_v_f_f32m8(y, vl), x, vl), atan2_ps(y, vfmv_v_f_f32m8(x, vl), vl))
// *INDENT-ON*
// clang-format on

#undef MAKE_FUNCTION

} // namespace BinaryOp_riscv_functor

static int binary_op_scalar(const Mat& a, float b, Mat& c, int op_type, const Option& opt)
{
    using namespace BinaryOp_riscv_functor;

    if (op_type == BinaryOp::Operation_ADD) return binary_op_scalar<binary_op_add>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_SUB) return binary_op_scalar<binary_op_sub>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_MUL) return binary_op_scalar<binary_op_mul>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_DIV) return binary_op_scalar<binary_op_div>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_MAX) return binary_op_scalar<binary_op_max>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_MIN) return binary_op_scalar<binary_op_min>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_POW) return binary_op_scalar<binary_op_pow>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_RSUB) return binary_op_scalar<binary_op_rsub>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_RDIV) return binary_op_scalar<binary_op_rdiv>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_RPOW) return binary_op_scalar<binary_op_rpow>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_ATAN2) return binary_op_scalar<binary_op_atan2>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_RATAN2) return binary_op_scalar<binary_op_ratan2>(a, b, c, opt);

    // should never reach here
    return 0;
}

static int binary_op_no_broadcast(const Mat& a, const Mat& b, Mat& c, int op_type, const Option& opt)
{
    using namespace BinaryOp_riscv_functor;

    if (op_type == BinaryOp::Operation_ADD) return binary_op_no_broadcast<binary_op_add>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_SUB) return binary_op_no_broadcast<binary_op_sub>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_MUL) return binary_op_no_broadcast<binary_op_mul>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_DIV) return binary_op_no_broadcast<binary_op_div>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_MAX) return binary_op_no_broadcast<binary_op_max>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_MIN) return binary_op_no_broadcast<binary_op_min>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_POW) return binary_op_no_broadcast<binary_op_pow>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_RSUB) return binary_op_no_broadcast<binary_op_sub>(b, a, c, opt);
    if (op_type == BinaryOp::Operation_RDIV) return binary_op_no_broadcast<binary_op_div>(b, a, c, opt);
    if (op_type == BinaryOp::Operation_RPOW) return binary_op_no_broadcast<binary_op_pow>(b, a, c, opt);
    if (op_type == BinaryOp::Operation_ATAN2) return binary_op_no_broadcast<binary_op_atan2>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_RATAN2) return binary_op_no_broadcast<binary_op_atan2>(b, a, c, opt);

    // should never reach here
    return 0;
}

static int binary_op_broadcast_inner(const Mat& a, const Mat& b, Mat& c, int op_type, const Option& opt)
{
    // squeeze inner axes
    Mat b2 = b;
    if (b.dims == 2 && b.w == 1)
        b2 = b.reshape(b.h);
    else if (b.dims == 3 && b.h == 1)
        b2 = b.reshape(b.c);
    else if (b.dims == 3 && b.w == 1)
        b2 = b.reshape(b.h, b.c);
    else if (b.dims == 4 && b.d == 1)
        b2 = b.reshape(b.c);
    else if (b.dims == 4 && b.h == 1)
        b2 = b.reshape(b.d, b.c);
    else if (b.dims == 4 && b.w == 1)
        b2 = b.reshape(b.h, b.d, b.c);

    using namespace BinaryOp_riscv_functor;

    if (op_type == BinaryOp::Operation_ADD) return binary_op_broadcast_inner<binary_op_add>(a, b2, c, opt);
    if (op_type == BinaryOp::Operation_SUB) return binary_op_broadcast_inner<binary_op_sub>(a, b2, c, opt);
    if (op_type == BinaryOp::Operation_MUL) return binary_op_broadcast_inner<binary_op_mul>(a, b2, c, opt);
    if (op_type == BinaryOp::Operation_DIV) return binary_op_broadcast_inner<binary_op_div>(a, b2, c, opt);
    if (op_type == BinaryOp::Operation_MAX) return binary_op_broadcast_inner<binary_op_max>(a, b2, c, opt);
    if (op_type == BinaryOp::Operation_MIN) return binary_op_broadcast_inner<binary_op_min>(a, b2, c, opt);
    if (op_type == BinaryOp::Operation_POW) return binary_op_broadcast_inner<binary_op_pow>(a, b2, c, opt);
    if (op_type == BinaryOp::Operation_RSUB) return binary_op_broadcast_inner<binary_op_rsub>(a, b2, c, opt);
    if (op_type == BinaryOp::Operation_RDIV) return binary_op_broadcast_inner<binary_op_rdiv>(a, b2, c, opt);
    if (op_type == BinaryOp::Operation_RPOW) return binary_op_broadcast_inner<binary_op_rpow>(a, b2, c, opt);
    if (op_type == BinaryOp::Operation_ATAN2) return binary_op_broadcast_inner<binary_op_atan2>(a, b2, c, opt);
    if (op_type == BinaryOp::Operation_RATAN2) return binary_op_broadcast_inner<binary_op_ratan2>(a, b2, c, opt);

    // should never reach here
    return 0;
}

static int binary_op_broadcast_outer(const Mat& a, const Mat& b, Mat& c, int op_type, const Option& opt)
{
    using namespace BinaryOp_riscv_functor;

    if (op_type == BinaryOp::Operation_ADD) return binary_op_broadcast_outer<binary_op_add>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_SUB) return binary_op_broadcast_outer<binary_op_sub>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_MUL) return binary_op_broadcast_outer<binary_op_mul>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_DIV) return binary_op_broadcast_outer<binary_op_div>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_MAX) return binary_op_broadcast_outer<binary_op_max>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_MIN) return binary_op_broadcast_outer<binary_op_min>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_POW) return binary_op_broadcast_outer<binary_op_pow>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_RSUB) return binary_op_broadcast_outer<binary_op_rsub>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_RDIV) return binary_op_broadcast_outer<binary_op_rdiv>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_RPOW) return binary_op_broadcast_outer<binary_op_rpow>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_ATAN2) return binary_op_broadcast_outer<binary_op_atan2>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_RATAN2) return binary_op_broadcast_outer<binary_op_ratan2>(a, b, c, opt);

    // should never reach here
    return 0;
}

static int binary_op_broadcast_20(const Mat& a, const Mat& b, Mat& c, int op_type, const Option& opt)
{
    using namespace BinaryOp_riscv_functor;

    if (op_type == BinaryOp::Operation_ADD) return binary_op_broadcast_20<binary_op_add>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_SUB) return binary_op_broadcast_20<binary_op_sub>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_MUL) return binary_op_broadcast_20<binary_op_mul>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_DIV) return binary_op_broadcast_20<binary_op_div>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_MAX) return binary_op_broadcast_20<binary_op_max>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_MIN) return binary_op_broadcast_20<binary_op_min>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_POW) return binary_op_broadcast_20<binary_op_pow>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_RSUB) return binary_op_broadcast_20<binary_op_rsub>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_RDIV) return binary_op_broadcast_20<binary_op_rdiv>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_RPOW) return binary_op_broadcast_20<binary_op_rpow>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_ATAN2) return binary_op_broadcast_20<binary_op_atan2>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_RATAN2) return binary_op_broadcast_20<binary_op_ratan2>(a, b, c, opt);

    // should never reach here
    return 0;
}

static int get_reverse_op_type(int op_type)
{
    if (op_type == BinaryOp::Operation_SUB) return BinaryOp::Operation_RSUB;
    if (op_type == BinaryOp::Operation_DIV) return BinaryOp::Operation_RDIV;
    if (op_type == BinaryOp::Operation_POW) return BinaryOp::Operation_RPOW;
    if (op_type == BinaryOp::Operation_ATAN2) return BinaryOp::Operation_RATAN2;
    if (op_type == BinaryOp::Operation_RSUB) return BinaryOp::Operation_SUB;
    if (op_type == BinaryOp::Operation_RDIV) return BinaryOp::Operation_DIV;
    if (op_type == BinaryOp::Operation_RPOW) return BinaryOp::Operation_POW;
    if (op_type == BinaryOp::Operation_RATAN2) return BinaryOp::Operation_ATAN2;
    return op_type;
}

int BinaryOp_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    int elembits = std::max(bottom_blobs[0].elembits(), bottom_blobs[1].elembits());

#if __riscv_vector && __riscv_zfh
    if (opt.use_fp16_storage && elembits == 16)
    {
        return forward_fp16s(bottom_blobs, top_blobs, opt);
    }
#endif

    const bool b_is_scalar = bottom_blobs[1].w * bottom_blobs[1].h * bottom_blobs[1].d * bottom_blobs[1].c * bottom_blobs[1].elempack == 1;
    const bool a_rank_is_lower = bottom_blobs[0].dims < bottom_blobs[1].dims && !b_is_scalar;
    const bool a_size_is_lower = bottom_blobs[0].w * bottom_blobs[0].h * bottom_blobs[0].d * bottom_blobs[0].c * bottom_blobs[0].elempack < bottom_blobs[1].w * bottom_blobs[1].h * bottom_blobs[1].d * bottom_blobs[1].c * bottom_blobs[1].elempack;
    const bool a_is_lower = a_rank_is_lower || (!a_rank_is_lower && a_size_is_lower);
    const Mat& A = a_is_lower ? bottom_blobs[1] : bottom_blobs[0];
    const Mat& B = a_is_lower ? bottom_blobs[0] : bottom_blobs[1];
    const int op_type_r = a_is_lower ? get_reverse_op_type(op_type) : op_type;

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(A, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // B is a scalar
    if (B.w * B.h * B.d * B.c * B.elempack == 1)
    {
        return binary_op_scalar(A, B[0], top_blob, op_type_r, opt);
    }

    // no broadcast
    if (A.dims == B.dims && A.w == B.w && A.h == B.h && A.d == B.d && A.c == B.c && A.elempack == B.elempack)
    {
        return binary_op_no_broadcast(A, B, top_blob, op_type_r, opt);
    }

    // broadcast B for inner axis
    if ((B.dims < A.dims)
            || (A.dims == 2 && B.w == 1 && B.h == A.h)
            || (A.dims == 3 && B.w == 1 && B.h == 1 && B.c == A.c)
            || (A.dims == 3 && B.w == 1 && B.h == A.h && B.c == A.c)
            || (A.dims == 4 && B.w == 1 && B.h == 1 && B.d == 1 && B.c == A.c)
            || (A.dims == 4 && B.w == 1 && B.h == 1 && B.d == A.d && B.c == A.c)
            || (A.dims == 4 && B.w == 1 && B.h == A.h && B.d == A.d && B.c == A.c))
    {
        return binary_op_broadcast_inner(A, B, top_blob, op_type_r, opt);
    }

    // broadcast B for outer axis
    if (B.elempack == 1 && ((A.dims == 2 && B.w == A.w && B.h == 1) || (A.dims == 3 && B.w == A.w && B.h == 1 && B.c == 1) || (A.dims == 3 && B.w == A.w && B.h == A.h && B.c == 1) || (A.dims == 4 && B.w == A.w && B.h == 1 && B.d == 1 && B.c == 1) || (A.dims == 4 && B.w == A.w && B.h == A.h && B.d == 1 && B.c == 1) || (A.dims == 4 && B.w == A.w && B.h == A.h && B.d == A.d && B.c == 1)))
    {
        return binary_op_broadcast_outer(A, B, top_blob, op_type_r, opt);
    }

    // some special broadcast rule here
    if (A.dims == 3 && B.dims == 3 && A.w == B.w && B.h == 1 && A.c == B.c)
    {
        return binary_op_broadcast_20(A, B, top_blob, op_type_r, opt);
    }

    return 0;
}

int BinaryOp_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elembits = bottom_top_blob.elembits();

#if __riscv_zfh
    if (opt.use_fp16_storage && elembits == 16)
    {
        return forward_inplace_fp16s(bottom_top_blob, opt);
    }
#endif

    using namespace BinaryOp_riscv_functor;

    if (op_type == Operation_ADD) return binary_op_scalar_inplace<binary_op_add>(bottom_top_blob, b, opt);
    if (op_type == Operation_SUB) return binary_op_scalar_inplace<binary_op_sub>(bottom_top_blob, b, opt);
    if (op_type == Operation_MUL) return binary_op_scalar_inplace<binary_op_mul>(bottom_top_blob, b, opt);
    if (op_type == Operation_DIV) return binary_op_scalar_inplace<binary_op_div>(bottom_top_blob, b, opt);
    if (op_type == Operation_MAX) return binary_op_scalar_inplace<binary_op_max>(bottom_top_blob, b, opt);
    if (op_type == Operation_MIN) return binary_op_scalar_inplace<binary_op_min>(bottom_top_blob, b, opt);
    if (op_type == Operation_POW) return binary_op_scalar_inplace<binary_op_pow>(bottom_top_blob, b, opt);
    if (op_type == Operation_RSUB) return binary_op_scalar_inplace<binary_op_rsub>(bottom_top_blob, b, opt);
    if (op_type == Operation_RDIV) return binary_op_scalar_inplace<binary_op_rdiv>(bottom_top_blob, b, opt);
    if (op_type == Operation_RPOW) return binary_op_scalar_inplace<binary_op_rpow>(bottom_top_blob, b, opt);
    if (op_type == Operation_ATAN2) return binary_op_scalar_inplace<binary_op_atan2>(bottom_top_blob, b, opt);
    if (op_type == Operation_RATAN2) return binary_op_scalar_inplace<binary_op_ratan2>(bottom_top_blob, b, opt);

    return 0;
}

#if __riscv_vector && __riscv_zfh
template<typename Op>
static int binary_op_scalar_fp16s(const Mat& a, __fp16 b, Mat& c, const Option& opt)
{
    Op op;

    const int channels = a.c;
    const int size = a.w * a.h * a.d * a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const __fp16* ptr = a.channel(q);
        __fp16* outptr = c.channel(q);

        int n = size;
        while (n > 0)
        {
            size_t vl = vsetvl_e16m8(n);
            vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
            _p = op(_p, b, vl);
            vse16_v_f16m8(outptr, _p, vl);
            n -= vl;
            ptr += vl;
            outptr += vl;
        }
    }

    return 0;
}

template<typename Op>
static int binary_op_no_broadcast_fp16s(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    Op op;

    const int channels = a.c;
    const int size = a.w * a.h * a.d * a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const __fp16* ptr = a.channel(q);
        const __fp16* ptr1 = b.channel(q);
        __fp16* outptr = c.channel(q);

        int n = size;
        while (n > 0)
        {
            size_t vl = vsetvl_e16m8(n);
            vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
            vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
            vfloat16m8_t _outp = op(_p, _p1, vl);
            vse16_v_f16m8(outptr, _outp, vl);
            n -= vl;
            ptr += vl;
            ptr1 += vl;
            outptr += vl;
        }
    }

    return 0;
}

template<typename Op>
static int binary_op_broadcast_inner_fp16s(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int d = a.d;
    int channels = a.c;
    int elempack = a.elempack;

    if (a.dims == 2 && b.dims == 1)
    {
        // type 8
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < h; y++)
        {
            const __fp16* ptr = a.row<const __fp16>(y);
            __fp16* outptr = c.row<__fp16>(y);

            const __fp16 _b = ((const __fp16*)b)[y];

            const int size = w * elempack;

            int n = size;
            vfloat16m8_t _bx = (elempack == 1) ? vfmv_v_f_f16m8(_b, vsetvl_e16m8(n)) : vle16_v_f16m8_f16m1((const __fp16*)b + y * elempack);
            while (n > 0)
            {
                size_t vl = vsetvl_e16m8(n);
                vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                vfloat16m8_t _outp = op(_p, _bx, vl);
                vse16_v_f16m8(outptr, _outp, vl);
                n -= vl;
                ptr += vl;
                outptr += vl;
            }
        }
    }

    if ((a.dims == 3 || a.dims == 4) && b.dims == 1)
    {
        // type 9 11
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const __fp16* ptr = a.channel(q);
            __fp16* outptr = c.channel(q);

            const __fp16 _b = ((const __fp16*)b)[q];

            const int size = w * h * d * elempack;

            int n = size;
            vfloat16m8_t _bx = (elempack == 1) ? vfmv_v_f_f16m8(_b, vsetvl_e16m8(n)) : vle16_v_f16m8_f16m1((const __fp16*)b + q * elempack);
            while (n > 0)
            {
                size_t vl = vsetvl_e16m8(n);
                vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                vfloat16m8_t _outp = op(_p, _bx, vl);
                vse16_v_f16m8(outptr, _outp, vl);
                n -= vl;
                ptr += vl;
                outptr += vl;
            }
        }
    }

    if (a.dims == 3 && b.dims == 2)
    {
        // type 10
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const __fp16* ptr = a.channel(q);
            const __fp16* ptr1 = b.row<const __fp16>(q);
            __fp16* outptr = c.channel(q);

            const int size = w * elempack;

            for (int y = 0; y < h; y++)
            {
                const __fp16 _b = ptr1[y];

                int n = size;
                vfloat16m8_t _bx = (elempack == 1) ? vfmv_v_f_f16m8(_b, vsetvl_e16m8(n)) : vle16_v_f16m8_f16m1(ptr1 + y * elempack);
                while (n > 0)
                {
                    size_t vl = vsetvl_e16m8(n);
                    vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                    vfloat16m8_t _outp = op(_p, _bx, vl);
                    vse16_v_f16m8(outptr, _outp, vl);
                    n -= vl;
                    ptr += vl;
                    outptr += vl;
                }
            }
        }
    }

    if (a.dims == 4 && b.dims == 2)
    {
        // type 12
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const __fp16* ptr = a.channel(q);
            const __fp16* ptr1 = b.row<const __fp16>(q);
            __fp16* outptr = c.channel(q);

            const int size = w * h * elempack;

            for (int z = 0; z < d; z++)
            {
                const __fp16 _b = ptr1[z];

                int n = size;
                vfloat16m8_t _bx = (elempack == 1) ? vfmv_v_f_f16m8(_b, vsetvl_e16m8(n)) : vle16_v_f16m8_f16m1(ptr1 + z * elempack);
                while (n > 0)
                {
                    size_t vl = vsetvl_e16m8(n);
                    vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                    vfloat16m8_t _outp = op(_p, _bx, vl);
                    vse16_v_f16m8(outptr, _outp, vl);
                    n -= vl;
                    ptr += vl;
                    outptr += vl;
                }
            }
        }
    }

    if (a.dims == 4 && b.dims == 3)
    {
        // type 13
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const __fp16* ptr = a.channel(q);
            __fp16* outptr = c.channel(q);

            const int size = w * elempack;

            for (int z = 0; z < d; z++)
            {
                const __fp16* ptr1 = b.channel(q).row<const __fp16>(z);

                for (int y = 0; y < h; y++)
                {
                    const __fp16 _b = ptr1[y];

                    int n = size;
                    vfloat16m8_t _bx = (elempack == 1) ? vfmv_v_f_f16m8(_b, vsetvl_e16m8(n)) : vle16_v_f16m8_f16m1(ptr1 + y * elempack);
                    while (n > 0)
                    {
                        size_t vl = vsetvl_e16m8(n);
                        vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                        vfloat16m8_t _outp = op(_p, _bx, vl);
                        vse16_v_f16m8(outptr, _outp, vl);
                        n -= vl;
                        ptr += vl;
                        outptr += vl;
                    }
                }
            }
        }
    }

    return 0;
}

template<typename Op>
static int binary_op_broadcast_outer_fp16s(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int d = a.d;
    int channels = a.c;
    int elempack = a.elempack;

    if (a.dims == 2)
    {
        // type 14
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < h; y++)
        {
            const __fp16* ptr = a.row<const __fp16>(y);
            const __fp16* ptr1 = b;
            __fp16* outptr = c.row<__fp16>(y);

            if (elempack != 1)
            {
                for (int x = 0; x < w; x++)
                {
                    int n = elempack;
                    while (n > 0)
                    {
                        size_t vl = vsetvl_e16m8(n);
                        vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                        vfloat16m8_t _outp = op(_p, *ptr1, vl);
                        vse16_v_f16m8(outptr, _outp, vl);
                        n -= vl;
                        ptr += vl;
                        outptr += vl;
                    }
                    ptr1 += 1;
                }
            }
            if (elempack == 1)
            {
                for (int x = 0; x < w; x++)
                {
                    *outptr = op(*ptr, *ptr1);
                    ptr += 1;
                    ptr1 += 1;
                    outptr += 1;
                }
            }
        }
    }

    if (a.dims == 3 || a.dims == 4)
    {
        // type 15 16 17 18 19
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const __fp16* ptr = a.channel(q);
            __fp16* outptr = c.channel(q);

            for (int z = 0; z < d; z++)
            {
                int z1 = std::min(z, b.d - 1);
                for (int y = 0; y < h; y++)
                {
                    int y1 = std::min(y, b.h - 1);

                    const __fp16* ptr1 = b.depth(z1).row<const __fp16>(y1);

                    if (elempack != 1)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            int n = elempack;
                            while (n > 0)
                            {
                                size_t vl = vsetvl_e16m8(n);
                                vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                                vfloat16m8_t _outp = op(_p, *ptr1, vl);
                                vse16_v_f16m8(outptr, _outp, vl);
                                n -= vl;
                                ptr += vl;
                                outptr += vl;
                            }
                            ptr1 += 1;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            *outptr = op(*ptr, *ptr1);
                            ptr += 1;
                            ptr1 += 1;
                            outptr += 1;
                        }
                    }
                }
            }
        }
    }

    return 0;
}

template<typename Op>
static int binary_op_broadcast_20_fp16s(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int elempack = a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const __fp16* ptr = a.channel(q);
        __fp16* outptr = c.channel(q);

        for (int y = 0; y < h; y++)
        {
            const __fp16* ptr1 = b.channel(q);

            const int size = w * elempack;

            int n = size;
            while (n > 0)
            {
                size_t vl = vsetvl_e16m8(n);
                vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                vfloat16m8_t _outp = op(_p, _p1, vl);
                vse16_v_f16m8(outptr, _outp, vl);
                n -= vl;
                ptr += vl;
                ptr1 += vl;
                outptr += vl;
            }
        }
    }

    return 0;
}

template<typename Op>
static int binary_op_scalar_inplace_fp16s(Mat& a, __fp16 b, const Option& opt)
{
    Op op;
    int w = a.w;
    int h = a.h;
    int d = a.d;
    int channels = a.c;
    int elempack = a.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = a.channel(q);

        int n = size;
        while (n > 0)
        {
            size_t vl = vsetvl_e16m8(n);
            vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
            _p = op(_p, b, vl);
            vse16_v_f16m8(ptr, _p, vl);
            n -= vl;
            ptr += vl;
        }
    }

    return 0;
}

namespace BinaryOp_riscv_functor {

#define MAKE_FUNCTION(NAME, IMPL, IMPLVV, IMPLVS, IMPLSV)                                            \
    struct NAME                                                                                      \
    {                                                                                                \
        __fp16 operator()(const __fp16& x, const __fp16& y) const                                    \
        {                                                                                            \
            return IMPL;                                                                             \
        }                                                                                            \
        vfloat16m8_t operator()(const vfloat16m8_t& x, const vfloat16m8_t& y, const size_t vl) const \
        {                                                                                            \
            return IMPLVV;                                                                           \
        }                                                                                            \
        vfloat16m8_t operator()(const vfloat16m8_t& x, const __fp16& y, const size_t vl) const       \
        {                                                                                            \
            return IMPLVS;                                                                           \
        }                                                                                            \
        vfloat16m8_t operator()(const __fp16& x, const vfloat16m8_t& y, const size_t vl) const       \
        {                                                                                            \
            return IMPLSV;                                                                           \
        }                                                                                            \
    };

// clang-format off
// *INDENT-OFF*
MAKE_FUNCTION(binary_op_add_fp16s, x + y, vfadd_vv_f16m8(x, y, vl), vfadd_vf_f16m8(x, y, vl), vfadd_vf_f16m8(y, x, vl))
MAKE_FUNCTION(binary_op_sub_fp16s, x - y, vfsub_vv_f16m8(x, y, vl), vfsub_vf_f16m8(x, y, vl), vfrsub_vf_f16m8(y, x, vl))
MAKE_FUNCTION(binary_op_mul_fp16s, x * y, vfmul_vv_f16m8(x, y, vl), vfmul_vf_f16m8(x, y, vl), vfmul_vf_f16m8(y, x, vl))
MAKE_FUNCTION(binary_op_div_fp16s, x / y, vfdiv_vv_f16m8(x, y, vl), vfdiv_vf_f16m8(x, y, vl), vfrdiv_vf_f16m8(y, x, vl))
MAKE_FUNCTION(binary_op_max_fp16s, std::max(x, y), vfmax_vv_f16m8(x, y, vl), vfmax_vf_f16m8(x, y, vl), vfmax_vf_f16m8(y, x, vl))
MAKE_FUNCTION(binary_op_min_fp16s, std::min(x, y), vfmin_vv_f16m8(x, y, vl), vfmin_vf_f16m8(x, y, vl), vfmin_vf_f16m8(y, x, vl))
MAKE_FUNCTION(binary_op_pow_fp16s, (__fp16)pow((float)x, (float)y), pow_ps(x, y, vl), pow_ps(x, vfmv_v_f_f16m8(y, vl), vl), pow_ps(vfmv_v_f_f16m8(x, vl), y, vl))
MAKE_FUNCTION(binary_op_rsub_fp16s, y - x, vfsub_vv_f16m8(y, x, vl), vfrsub_vf_f16m8(x, y, vl), vfsub_vf_f16m8(y, x, vl))
MAKE_FUNCTION(binary_op_rdiv_fp16s, y / x, vfdiv_vv_f16m8(y, x, vl), vfrdiv_vf_f16m8(x, y, vl), vfdiv_vf_f16m8(y, x, vl))
MAKE_FUNCTION(binary_op_rpow_fp16s, (__fp16)pow((float)y, (float)x), pow_ps(y, x, vl), pow_ps(vfmv_v_f_f16m8(y, vl), x, vl), pow_ps(y, vfmv_v_f_f16m8(x, vl), vl))
MAKE_FUNCTION(binary_op_atan2_fp16s, (__fp16)atan2((float)x, (float)y), atan2_ps(x, y, vl), atan2_ps(x, vfmv_v_f_f16m8(y, vl), vl), atan2_ps(vfmv_v_f_f16m8(x, vl), y, vl))
MAKE_FUNCTION(binary_op_ratan2_fp16s, (__fp16)atan2((float)y, (float)x), atan2_ps(y, x, vl), atan2_ps(vfmv_v_f_f16m8(y, vl), x, vl), atan2_ps(y, vfmv_v_f_f16m8(x, vl), vl))
// *INDENT-ON*
// clang-format on

#undef MAKE_FUNCTION

} // namespace BinaryOp_riscv_functor

static int binary_op_scalar_fp16s(const Mat& a, __fp16 b, Mat& c, int op_type, const Option& opt)
{
    using namespace BinaryOp_riscv_functor;

    if (op_type == BinaryOp::Operation_ADD) return binary_op_scalar_fp16s<binary_op_add_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_SUB) return binary_op_scalar_fp16s<binary_op_sub_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_MUL) return binary_op_scalar_fp16s<binary_op_mul_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_DIV) return binary_op_scalar_fp16s<binary_op_div_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_MAX) return binary_op_scalar_fp16s<binary_op_max_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_MIN) return binary_op_scalar_fp16s<binary_op_min_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_POW) return binary_op_scalar_fp16s<binary_op_pow_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_RSUB) return binary_op_scalar_fp16s<binary_op_rsub_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_RDIV) return binary_op_scalar_fp16s<binary_op_rdiv_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_RPOW) return binary_op_scalar_fp16s<binary_op_rpow_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_ATAN2) return binary_op_scalar_fp16s<binary_op_atan2_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_RATAN2) return binary_op_scalar_fp16s<binary_op_ratan2_fp16s>(a, b, c, opt);

    // should never reach here
    return 0;
}

static int binary_op_no_broadcast_fp16s(const Mat& a, const Mat& b, Mat& c, int op_type, const Option& opt)
{
    using namespace BinaryOp_riscv_functor;

    if (op_type == BinaryOp::Operation_ADD) return binary_op_no_broadcast_fp16s<binary_op_add_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_SUB) return binary_op_no_broadcast_fp16s<binary_op_sub_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_MUL) return binary_op_no_broadcast_fp16s<binary_op_mul_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_DIV) return binary_op_no_broadcast_fp16s<binary_op_div_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_MAX) return binary_op_no_broadcast_fp16s<binary_op_max_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_MIN) return binary_op_no_broadcast_fp16s<binary_op_min_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_POW) return binary_op_no_broadcast_fp16s<binary_op_pow_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_RSUB) return binary_op_no_broadcast_fp16s<binary_op_sub_fp16s>(b, a, c, opt);
    if (op_type == BinaryOp::Operation_RDIV) return binary_op_no_broadcast_fp16s<binary_op_div_fp16s>(b, a, c, opt);
    if (op_type == BinaryOp::Operation_RPOW) return binary_op_no_broadcast_fp16s<binary_op_pow_fp16s>(b, a, c, opt);
    if (op_type == BinaryOp::Operation_ATAN2) return binary_op_no_broadcast_fp16s<binary_op_atan2_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_RATAN2) return binary_op_no_broadcast_fp16s<binary_op_atan2_fp16s>(b, a, c, opt);

    // should never reach here
    return 0;
}

static int binary_op_broadcast_inner_fp16s(const Mat& a, const Mat& b, Mat& c, int op_type, const Option& opt)
{
    // squeeze inner axes
    Mat b2 = b;
    if (b.dims == 2 && b.w == 1)
        b2 = b.reshape(b.h);
    else if (b.dims == 3 && b.h == 1)
        b2 = b.reshape(b.c);
    else if (b.dims == 3 && b.w == 1)
        b2 = b.reshape(b.h, b.c);
    else if (b.dims == 4 && b.d == 1)
        b2 = b.reshape(b.c);
    else if (b.dims == 4 && b.h == 1)
        b2 = b.reshape(b.d, b.c);
    else if (b.dims == 4 && b.w == 1)
        b2 = b.reshape(b.h, b.d, b.c);

    using namespace BinaryOp_riscv_functor;

    if (op_type == BinaryOp::Operation_ADD) return binary_op_broadcast_inner_fp16s<binary_op_add_fp16s>(a, b2, c, opt);
    if (op_type == BinaryOp::Operation_SUB) return binary_op_broadcast_inner_fp16s<binary_op_sub_fp16s>(a, b2, c, opt);
    if (op_type == BinaryOp::Operation_MUL) return binary_op_broadcast_inner_fp16s<binary_op_mul_fp16s>(a, b2, c, opt);
    if (op_type == BinaryOp::Operation_DIV) return binary_op_broadcast_inner_fp16s<binary_op_div_fp16s>(a, b2, c, opt);
    if (op_type == BinaryOp::Operation_MAX) return binary_op_broadcast_inner_fp16s<binary_op_max_fp16s>(a, b2, c, opt);
    if (op_type == BinaryOp::Operation_MIN) return binary_op_broadcast_inner_fp16s<binary_op_min_fp16s>(a, b2, c, opt);
    if (op_type == BinaryOp::Operation_POW) return binary_op_broadcast_inner_fp16s<binary_op_pow_fp16s>(a, b2, c, opt);
    if (op_type == BinaryOp::Operation_RSUB) return binary_op_broadcast_inner_fp16s<binary_op_rsub_fp16s>(a, b2, c, opt);
    if (op_type == BinaryOp::Operation_RDIV) return binary_op_broadcast_inner_fp16s<binary_op_rdiv_fp16s>(a, b2, c, opt);
    if (op_type == BinaryOp::Operation_RPOW) return binary_op_broadcast_inner_fp16s<binary_op_rpow_fp16s>(a, b2, c, opt);
    if (op_type == BinaryOp::Operation_ATAN2) return binary_op_broadcast_inner_fp16s<binary_op_atan2_fp16s>(a, b2, c, opt);
    if (op_type == BinaryOp::Operation_RATAN2) return binary_op_broadcast_inner_fp16s<binary_op_ratan2_fp16s>(a, b2, c, opt);

    // should never reach here
    return 0;
}

static int binary_op_broadcast_outer_fp16s(const Mat& a, const Mat& b, Mat& c, int op_type, const Option& opt)
{
    using namespace BinaryOp_riscv_functor;

    if (op_type == BinaryOp::Operation_ADD) return binary_op_broadcast_outer_fp16s<binary_op_add_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_SUB) return binary_op_broadcast_outer_fp16s<binary_op_sub_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_MUL) return binary_op_broadcast_outer_fp16s<binary_op_mul_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_DIV) return binary_op_broadcast_outer_fp16s<binary_op_div_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_MAX) return binary_op_broadcast_outer_fp16s<binary_op_max_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_MIN) return binary_op_broadcast_outer_fp16s<binary_op_min_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_POW) return binary_op_broadcast_outer_fp16s<binary_op_pow_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_RSUB) return binary_op_broadcast_outer_fp16s<binary_op_rsub_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_RDIV) return binary_op_broadcast_outer_fp16s<binary_op_rdiv_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_RPOW) return binary_op_broadcast_outer_fp16s<binary_op_rpow_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_ATAN2) return binary_op_broadcast_outer_fp16s<binary_op_atan2_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_RATAN2) return binary_op_broadcast_outer_fp16s<binary_op_ratan2_fp16s>(a, b, c, opt);

    // should never reach here
    return 0;
}

static int binary_op_broadcast_20_fp16s(const Mat& a, const Mat& b, Mat& c, int op_type, const Option& opt)
{
    using namespace BinaryOp_riscv_functor;

    if (op_type == BinaryOp::Operation_ADD) return binary_op_broadcast_20_fp16s<binary_op_add_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_SUB) return binary_op_broadcast_20_fp16s<binary_op_sub_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_MUL) return binary_op_broadcast_20_fp16s<binary_op_mul_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_DIV) return binary_op_broadcast_20_fp16s<binary_op_div_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_MAX) return binary_op_broadcast_20_fp16s<binary_op_max_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_MIN) return binary_op_broadcast_20_fp16s<binary_op_min_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_POW) return binary_op_broadcast_20_fp16s<binary_op_pow_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_RSUB) return binary_op_broadcast_20_fp16s<binary_op_rsub_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_RDIV) return binary_op_broadcast_20_fp16s<binary_op_rdiv_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_RPOW) return binary_op_broadcast_20_fp16s<binary_op_rpow_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_ATAN2) return binary_op_broadcast_20_fp16s<binary_op_atan2_fp16s>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_RATAN2) return binary_op_broadcast_20_fp16s<binary_op_ratan2_fp16s>(a, b, c, opt);

    // should never reach here
    return 0;
}

int BinaryOp_riscv::forward_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const bool b_is_scalar = bottom_blobs[1].w * bottom_blobs[1].h * bottom_blobs[1].d * bottom_blobs[1].c * bottom_blobs[1].elempack == 1;
    const bool a_rank_is_lower = bottom_blobs[0].dims < bottom_blobs[1].dims && !b_is_scalar;
    const bool a_size_is_lower = bottom_blobs[0].w * bottom_blobs[0].h * bottom_blobs[0].d * bottom_blobs[0].c * bottom_blobs[0].elempack < bottom_blobs[1].w * bottom_blobs[1].h * bottom_blobs[1].d * bottom_blobs[1].c * bottom_blobs[1].elempack;
    const bool a_is_lower = a_rank_is_lower || (!a_rank_is_lower && a_size_is_lower);
    const Mat& A = a_is_lower ? bottom_blobs[1] : bottom_blobs[0];
    const Mat& B = a_is_lower ? bottom_blobs[0] : bottom_blobs[1];
    const int op_type_r = a_is_lower ? get_reverse_op_type(op_type) : op_type;

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(A, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // B is A scalar
    if (B.w * B.h * B.d * B.c * B.elempack == 1)
    {
        return binary_op_scalar_fp16s(A, ((const __fp16*)B)[0], top_blob, op_type_r, opt);
    }

    // no broadcast
    if (A.dims == B.dims && A.w == B.w && A.h == B.h && A.d == B.d && A.c == B.c && A.elempack == B.elempack)
    {
        return binary_op_no_broadcast_fp16s(A, B, top_blob, op_type_r, opt);
    }

    // broadcast B for inner axis
    if ((B.dims < A.dims)
            || (A.dims == 2 && B.w == 1 && B.h == A.h)
            || (A.dims == 3 && B.w == 1 && B.h == 1 && B.c == A.c)
            || (A.dims == 3 && B.w == 1 && B.h == A.h && B.c == A.c)
            || (A.dims == 4 && B.w == 1 && B.h == 1 && B.d == 1 && B.c == A.c)
            || (A.dims == 4 && B.w == 1 && B.h == 1 && B.d == A.d && B.c == A.c)
            || (A.dims == 4 && B.w == 1 && B.h == A.h && B.d == A.d && B.c == A.c))
    {
        return binary_op_broadcast_inner_fp16s(A, B, top_blob, op_type_r, opt);
    }

    // broadcast B for outer axis
    if (B.elempack == 1 && ((A.dims == 2 && B.w == A.w && B.h == 1) || (A.dims == 3 && B.w == A.w && B.h == 1 && B.c == 1) || (A.dims == 3 && B.w == A.w && B.h == A.h && B.c == 1) || (A.dims == 4 && B.w == A.w && B.h == 1 && B.d == 1 && B.c == 1) || (A.dims == 4 && B.w == A.w && B.h == A.h && B.d == 1 && B.c == 1) || (A.dims == 4 && B.w == A.w && B.h == A.h && B.d == A.d && B.c == 1)))
    {
        return binary_op_broadcast_outer_fp16s(A, B, top_blob, op_type_r, opt);
    }

    // some special broadcast rule here
    if (A.dims == 3 && B.dims == 3 && A.w == B.w && B.h == 1 && A.c == B.c)
    {
        return binary_op_broadcast_20_fp16s(A, B, top_blob, op_type_r, opt);
    }

    return 0;
}

int BinaryOp_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    using namespace BinaryOp_riscv_functor;

    if (op_type == Operation_ADD) return binary_op_scalar_inplace_fp16s<binary_op_add_fp16s>(bottom_top_blob, (__fp16)b, opt);
    if (op_type == Operation_SUB) return binary_op_scalar_inplace_fp16s<binary_op_sub_fp16s>(bottom_top_blob, (__fp16)b, opt);
    if (op_type == Operation_MUL) return binary_op_scalar_inplace_fp16s<binary_op_mul_fp16s>(bottom_top_blob, (__fp16)b, opt);
    if (op_type == Operation_DIV) return binary_op_scalar_inplace_fp16s<binary_op_div_fp16s>(bottom_top_blob, (__fp16)b, opt);
    if (op_type == Operation_MAX) return binary_op_scalar_inplace_fp16s<binary_op_max_fp16s>(bottom_top_blob, (__fp16)b, opt);
    if (op_type == Operation_MIN) return binary_op_scalar_inplace_fp16s<binary_op_min_fp16s>(bottom_top_blob, (__fp16)b, opt);
    if (op_type == Operation_POW) return binary_op_scalar_inplace_fp16s<binary_op_pow_fp16s>(bottom_top_blob, (__fp16)b, opt);
    if (op_type == Operation_RSUB) return binary_op_scalar_inplace_fp16s<binary_op_rsub_fp16s>(bottom_top_blob, (__fp16)b, opt);
    if (op_type == Operation_RDIV) return binary_op_scalar_inplace_fp16s<binary_op_rdiv_fp16s>(bottom_top_blob, (__fp16)b, opt);
    if (op_type == Operation_RPOW) return binary_op_scalar_inplace_fp16s<binary_op_rpow_fp16s>(bottom_top_blob, (__fp16)b, opt);
    if (op_type == Operation_ATAN2) return binary_op_scalar_inplace_fp16s<binary_op_atan2_fp16s>(bottom_top_blob, (__fp16)b, opt);
    if (op_type == Operation_RATAN2) return binary_op_scalar_inplace_fp16s<binary_op_ratan2_fp16s>(bottom_top_blob, (__fp16)b, opt);

    return 0;
}
#endif // __riscv_vector && __riscv_zfh

} // namespace ncnn
