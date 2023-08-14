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
static void binary_op_vector_no_broadcast(const float* ptr, const float* ptr1, float* outptr, int size)
{
    const Op op;

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

template<typename Op>
static void binary_op_vector_broadcast_b(const float* ptr, const float* ptr1, float* outptr, int size, int elempack)
{
    const Op op;

    const float b = *ptr1;

#if __riscv_vector
    int n = size;
    vfloat32m8_t _bx = (elempack == 1) ? vfmv_v_f_f32m8(b, vsetvl_e32m8(n)) : vle32_v_f32m8_f32m1(ptr1);
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
        *outptr = op(*ptr, b);
        ptr += 1;
        outptr += 1;
    }
#endif
}

template<typename Op>
static void binary_op_vector_broadcast_a(const float* ptr, const float* ptr1, float* outptr, int size, int elempack)
{
    const Op op;

    const float a = *ptr;

#if __riscv_vector
    int n = size;
    vfloat32m8_t _ax = (elempack == 1) ? vfmv_v_f_f32m8(a, vsetvl_e32m8(n)) : vle32_v_f32m8_f32m1(ptr);
    while (n > 0)
    {
        size_t vl = vsetvl_e32m8(n);
        vfloat32m8_t _p = vle32_v_f32m8(ptr1, vl);
        vfloat32m8_t _outp = op(_ax, _p, vl);
        vse32_v_f32m8(outptr, _outp, vl);
        n -= vl;
        ptr1 += vl;
        outptr += vl;
    }
#else
    for (int i = 0; i < size; i++)
    {
        *outptr = op(a, *ptr1);
        ptr1 += 1;
        outptr += 1;
    }
#endif
}

template<typename Op>
static void binary_op_vector_broadcast_pb(const float* ptr, const float* ptr1, float* outptr, int w, int elempack)
{
    const Op op;

#if __riscv_vector
    // if (elempack == packn)
    {
        size_t vl = vsetvl_e32m8(elempack);
        int i = 0;
        for (; i < w; i++)
        {
            vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
            vfloat32m8_t _outp = op(_p, *ptr1, vl);
            vse32_v_f32m8(outptr, _outp, vl);
            ptr += vl;
            ptr1 += 1;
            outptr += vl;
        }
    }
#endif // __riscv_vector
}

template<typename Op>
static void binary_op_vector_broadcast_pb_b(const float* ptr, const float* ptr1, float* outptr, int w, int elempack)
{
    const Op op;

#if __riscv_vector
    int n = w * elempack;

    vfloat32m8_t _bx = vfmv_v_f_f32m8(*ptr1, vsetvl_e32m8(n));
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
#endif // __riscv_vector
}

template<typename Op>
static void binary_op_vector_broadcast_pb_a(const float* ptr, const float* ptr1, float* outptr, int w, int elempack)
{
    const Op op;

#if __riscv_vector
    // if (elempack == packn)
    {
        size_t vl = vsetvl_e32m8(elempack);
        vfloat32m8_t _ax = vle32_v_f32m8_f32m1(ptr);
        for (int i = 0; i < w; i++)
        {
            vfloat32m8_t _outp = op(_ax, *ptr1, vl);
            vse32_v_f32m8(outptr, _outp, vl);
            ptr1 += 1;
            outptr += vl;
        }
    }
#endif // __riscv_vector
}

template<typename Op>
static void binary_op_vector(const float* ptr, const float* ptr1, float* outptr, int aw, int bw, int ap, int bp)
{
    const int w = std::max(aw, bw);
    const int elempack = std::max(ap, bp);
    const int size = w * elempack;

    if (ap == bp)
    {
        if (aw == bw)
        {
            // no broadcast
            return binary_op_vector_no_broadcast<Op>(ptr, ptr1, outptr, size);
        }

        if (bw == 1)
        {
            // broadcast single b
            return binary_op_vector_broadcast_b<Op>(ptr, ptr1, outptr, size, elempack);
        }

        if (aw == 1)
        {
            // broadcast single a
            return binary_op_vector_broadcast_a<Op>(ptr, ptr1, outptr, size, elempack);
        }
    }

    if (bp == 1)
    {
        if (aw == bw)
        {
            // broadcast pack1 b
            return binary_op_vector_broadcast_pb<Op>(ptr, ptr1, outptr, w, elempack);
        }

        if (bw == 1)
        {
            // broadcast pack1 single b
            return binary_op_vector_broadcast_pb_b<Op>(ptr, ptr1, outptr, w, elempack);
        }

        if (aw == 1)
        {
            // broadcast single a and pack1 b
            return binary_op_vector_broadcast_pb_a<Op>(ptr, ptr1, outptr, w, elempack);
        }
    }

    // shall never reach here
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

static void binary_op_vector(const float* ptr, const float* ptr1, float* outptr, int aw, int bw, int ap, int bp, int op_type)
{
    using namespace BinaryOp_riscv_functor;

    if (op_type == BinaryOp::Operation_ADD) return binary_op_vector<binary_op_add>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_SUB) return binary_op_vector<binary_op_sub>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_MUL) return binary_op_vector<binary_op_mul>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_DIV) return binary_op_vector<binary_op_div>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_MAX) return binary_op_vector<binary_op_max>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_MIN) return binary_op_vector<binary_op_min>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_POW) return binary_op_vector<binary_op_pow>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RSUB) return binary_op_vector<binary_op_rsub>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RDIV) return binary_op_vector<binary_op_rdiv>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RPOW) return binary_op_vector<binary_op_rpow>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_ATAN2) return binary_op_vector<binary_op_atan2>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RATAN2) return binary_op_vector<binary_op_ratan2>(ptr, ptr1, outptr, aw, bw, ap, bp);

    // should never reach here
}

static void binary_op_scalar(const Mat& a, float b, Mat& c, int op_type, const Option& opt)
{
    const int channels = a.c;
    const int size = a.w * a.h * a.d * a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = a.channel(q);
        float* outptr = c.channel(q);

        binary_op_vector(ptr, &b, outptr, size, 1, 1, 1, op_type);
    }
}

static void binary_op_no_broadcast(const Mat& a, const Mat& b, Mat& c, int op_type, const Option& opt)
{
    const int channels = a.c;
    const int size = a.w * a.h * a.d * a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = a.channel(q);
        const float* ptr1 = b.channel(q);
        float* outptr = c.channel(q);

        binary_op_vector(ptr, ptr1, outptr, size, size, 1, 1, op_type);
    }
}

static void binary_op_broadcast(const Mat& a, const Mat& b, Mat& c, int op_type, const Option& opt)
{
    if (b.w * b.h * b.d * b.c * b.elempack == 1)
    {
        return binary_op_scalar(a, b[0], c, op_type, opt);
    }

    if (a.dims == b.dims && a.w == b.w && a.h == b.h && a.d == b.d && a.c == b.c && a.elempack == b.elempack)
    {
        return binary_op_no_broadcast(a, b, c, op_type, opt);
    }

    const int dims = c.dims;

    if (dims == 2)
    {
        const int h = c.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < h; y++)
        {
            const int y0 = std::min(y, a.h - 1);
            const int y1 = std::min(y, b.h - 1);

            const float* ptr = a.row(y0);
            const float* ptr1 = b.row(y1);
            float* outptr = c.row(y);

            binary_op_vector(ptr, ptr1, outptr, a.w, b.w, a.elempack, b.elempack, op_type);
        }
    }

    if (dims == 3 || dims == 4)
    {
        const int channels = c.c;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const int q0 = std::min(q, a.c - 1);
            const int q1 = std::min(q, b.c - 1);

            if (b.d * b.h * b.w == 1)
            {
                const float* ptr = a.channel(q0);
                const float* ptr1 = b.channel(q1);
                float* outptr = c.channel(q);

                binary_op_vector(ptr, ptr1, outptr, a.w * a.h * a.d, 1, a.elempack, b.elempack, op_type);
                continue;
            }

            if (b.h * b.w == 1)
            {
                for (int z = 0; z < c.d; z++)
                {
                    const int z0 = std::min(z, a.d - 1);
                    const int z1 = std::min(z, b.d - 1);

                    const float* ptr = a.channel(q0).depth(z0);
                    const float* ptr1 = b.channel(q1).depth(z1);
                    float* outptr = c.channel(q).depth(z);

                    binary_op_vector(ptr, ptr1, outptr, a.w * a.h, 1, a.elempack, b.elempack, op_type);
                }
                continue;
            }

            for (int z = 0; z < c.d; z++)
            {
                const int z0 = std::min(z, a.d - 1);
                const int z1 = std::min(z, b.d - 1);

                for (int y = 0; y < c.h; y++)
                {
                    const int y0 = std::min(y, a.h - 1);
                    const int y1 = std::min(y, b.h - 1);

                    const float* ptr = a.channel(q0).depth(z0).row(y0);
                    const float* ptr1 = b.channel(q1).depth(z1).row(y1);
                    float* outptr = c.channel(q).depth(z).row(y);

                    binary_op_vector(ptr, ptr1, outptr, a.w, b.w, a.elempack, b.elempack, op_type);
                }
            }
        }
    }
}

static void binary_op_scalar_inplace(Mat& a, float b, int op_type, const Option& opt)
{
    const int channels = a.c;
    const int size = a.w * a.h * a.d * a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = a.channel(q);

        binary_op_vector(ptr, &b, ptr, size, 1, 1, 1, op_type);
    }
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

    const Mat& A = bottom_blobs[0];
    const Mat& B = bottom_blobs[1];
    const int outdims = std::max(A.dims, B.dims);

    Mat A2 = A;
    Mat B2 = B;
    if (A.dims < outdims)
    {
        // expand inner axes
        if (outdims == 2)
        {
            if (A.w * A.elempack == B.h * B.elempack)
                A2 = A.reshape(1, A.w, opt.workspace_allocator);
            else // if (A.w == B.w)
            {
                A2.dims = 2;
                A2.w = A.w * A.elempack;
                A2.elempack = 1;
                A2.elemsize = A.elemsize / A.elempack;
                A2.cstep = A2.w;
            }
        }
        if (outdims == 3 && A.dims == 1)
        {
            if (A.w * A.elempack == B.c * B.elempack)
                A2 = A.reshape(1, 1, A.w, opt.workspace_allocator);
            else // if (A.w == B.w)
            {
                A2.dims = 3;
                A2.w = A.w * A.elempack;
                A2.elempack = 1;
                A2.elemsize = A.elemsize / A.elempack;
                A2.cstep = A2.w;
            }
        }
        if (outdims == 3 && A.dims == 2)
            A2 = A.reshape(1, A.w, A.h, opt.workspace_allocator);
        if (outdims == 4 && A.dims == 1)
        {
            if (A.w * A.elempack == B.c * B.elempack)
                A2 = A.reshape(1, 1, 1, A.w, opt.workspace_allocator);
            else // if (A.w == B.w)
            {
                A2.dims = 4;
                A2.w = A.w * A.elempack;
                A2.elempack = 1;
                A2.elemsize = A.elemsize / A.elempack;
                A2.cstep = A2.w;
            }
        }
        if (outdims == 4 && A.dims == 2)
            A2 = A.reshape(1, 1, A.w, A.h, opt.workspace_allocator);
        if (outdims == 4 && A.dims == 3)
            A2 = A.reshape(1, A.w, A.h, A.c, opt.workspace_allocator);
    }
    if (B.dims < outdims)
    {
        // expand inner axes
        if (outdims == 2)
        {
            if (B.w * B.elempack == A.h * A.elempack)
                B2 = B.reshape(1, B.w, opt.workspace_allocator);
            else // if (B.w == A.w)
            {
                B2.dims = 2;
                B2.w = B.w * B.elempack;
                B2.elempack = 1;
                B2.elemsize = B.elemsize / B.elempack;
                B2.cstep = B2.w;
            }
        }
        if (outdims == 3 && B.dims == 1)
        {
            if (B.w * B.elempack == A.c * A.elempack)
                B2 = B.reshape(1, 1, B.w, opt.workspace_allocator);
            else // if (B.w == A.w)
            {
                B2.dims = 3;
                B2.w = B.w * B.elempack;
                B2.elempack = 1;
                B2.elemsize = B.elemsize / B.elempack;
                B2.cstep = B2.w;
            }
        }
        if (outdims == 3 && B.dims == 2)
            B2 = B.reshape(1, B.w, B.h, opt.workspace_allocator);
        if (outdims == 4 && B.dims == 1)
        {
            if (B.w * B.elempack == A.c * A.elempack)
                B2 = B.reshape(1, 1, 1, B.w, opt.workspace_allocator);
            else // if (B.w == A.w)
            {
                B2.dims = 4;
                B2.w = B.w * B.elempack;
                B2.elempack = 1;
                B2.elemsize = B.elemsize / B.elempack;
                B2.cstep = B2.w;
            }
        }
        if (outdims == 4 && B.dims == 2)
            B2 = B.reshape(1, 1, B.w, B.h, opt.workspace_allocator);
        if (outdims == 4 && B.dims == 3)
            B2 = B.reshape(1, B.w, B.h, B.c, opt.workspace_allocator);
    }

    const int outw = std::max(A2.w, B2.w);
    const int outh = std::max(A2.h, B2.h);
    const int outd = std::max(A2.d, B2.d);
    const int outc = std::max(A2.c, B2.c);
    const size_t out_elemsize = std::max(A2.elemsize, B2.elemsize);
    const int out_elempack = std::max(A2.elempack, B2.elempack);

    Mat& top_blob = top_blobs[0];
    if (outdims == 1)
    {
        top_blob.create(outw, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (outdims == 2)
    {
        top_blob.create(outw, outh, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (outdims == 3)
    {
        top_blob.create(outw, outh, outc, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (outdims == 4)
    {
        top_blob.create(outw, outh, outd, outc, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    const bool a_pack_is_lower = A2.elempack < B2.elempack;
    const bool a_pack_is_equal = A2.elempack == B2.elempack;
    const bool a_size_is_lower = A2.w * A2.h * A2.d * A2.c * A2.elempack < B2.w * B2.h * B2.d * B2.c * B2.elempack;
    if (a_pack_is_lower || (a_pack_is_equal && a_size_is_lower))
    {
        binary_op_broadcast(B2, A2, top_blob, get_reverse_op_type(op_type), opt);
    }
    else
    {
        binary_op_broadcast(A2, B2, top_blob, op_type, opt);
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

    binary_op_scalar_inplace(bottom_top_blob, b, op_type, opt);

    return 0;
}

#if __riscv_vector && __riscv_zfh
template<typename Op>
static void binary_op_vector_no_broadcast_fp16s(const __fp16* ptr, const __fp16* ptr1, __fp16* outptr, int size)
{
    const Op op;

#if __riscv_vector
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

template<typename Op>
static void binary_op_vector_broadcast_b_fp16s(const __fp16* ptr, const __fp16* ptr1, __fp16* outptr, int size, int elempack)
{
    const Op op;

    const __fp16 b = *ptr1;

#if __riscv_vector
    int n = size;
    vfloat16m8_t _bx = (elempack == 1) ? vfmv_v_f_f16m8(b, vsetvl_e16m8(n)) : vle16_v_f16m8_f16m1(ptr1);
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
#else
    for (int i = 0; i < size; i++)
    {
        *outptr = op(*ptr, b);
        ptr += 1;
        outptr += 1;
    }
#endif
}

template<typename Op>
static void binary_op_vector_broadcast_a_fp16s(const __fp16* ptr, const __fp16* ptr1, __fp16* outptr, int size, int elempack)
{
    const Op op;

    const __fp16 a = *ptr;

#if __riscv_vector
    int n = size;
    vfloat16m8_t _ax = (elempack == 1) ? vfmv_v_f_f16m8(a, vsetvl_e16m8(n)) : vle16_v_f16m8_f16m1(ptr);
    while (n > 0)
    {
        size_t vl = vsetvl_e16m8(n);
        vfloat16m8_t _p = vle16_v_f16m8(ptr1, vl);
        vfloat16m8_t _outp = op(_ax, _p, vl);
        vse16_v_f16m8(outptr, _outp, vl);
        n -= vl;
        ptr1 += vl;
        outptr += vl;
    }
#else
    for (int i = 0; i < size; i++)
    {
        *outptr = op(a, *ptr1);
        ptr1 += 1;
        outptr += 1;
    }
#endif
}

template<typename Op>
static void binary_op_vector_broadcast_pb_fp16s(const __fp16* ptr, const __fp16* ptr1, __fp16* outptr, int w, int elempack)
{
    const Op op;

#if __riscv_vector
    // if (elempack == packn)
    {
        size_t vl = vsetvl_e16m8(elempack);
        int i = 0;
        for (; i < w; i++)
        {
            vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
            vfloat16m8_t _outp = op(_p, *ptr1, vl);
            vse16_v_f16m8(outptr, _outp, vl);
            ptr += vl;
            ptr1 += 1;
            outptr += vl;
        }
    }
#endif // __riscv_vector
}

template<typename Op>
static void binary_op_vector_broadcast_pb_b_fp16s(const __fp16* ptr, const __fp16* ptr1, __fp16* outptr, int w, int elempack)
{
    const Op op;

#if __riscv_vector
    int n = w * elempack;

    vfloat16m8_t _bx = vfmv_v_f_f16m8(*ptr1, vsetvl_e16m8(n));
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
#endif // __riscv_vector
}

template<typename Op>
static void binary_op_vector_broadcast_pb_a_fp16s(const __fp16* ptr, const __fp16* ptr1, __fp16* outptr, int w, int elempack)
{
    const Op op;

#if __riscv_vector
    // if (elempack == packn)
    {
        size_t vl = vsetvl_e16m8(elempack);
        vfloat16m8_t _ax = vle16_v_f16m8_f16m1(ptr);
        for (int i = 0; i < w; i++)
        {
            vfloat16m8_t _outp = op(_ax, *ptr1, vl);
            vse16_v_f16m8(outptr, _outp, vl);
            ptr1 += 1;
            outptr += vl;
        }
    }
#endif // __riscv_vector
}

template<typename Op>
static void binary_op_vector_fp16s(const __fp16* ptr, const __fp16* ptr1, __fp16* outptr, int aw, int bw, int ap, int bp)
{
    const int w = std::max(aw, bw);
    const int elempack = std::max(ap, bp);
    const int size = w * elempack;

    if (ap == bp)
    {
        if (aw == bw)
        {
            // no broadcast
            return binary_op_vector_no_broadcast_fp16s<Op>(ptr, ptr1, outptr, size);
        }

        if (bw == 1)
        {
            // broadcast single b
            return binary_op_vector_broadcast_b_fp16s<Op>(ptr, ptr1, outptr, size, elempack);
        }

        if (aw == 1)
        {
            // broadcast single a
            return binary_op_vector_broadcast_a_fp16s<Op>(ptr, ptr1, outptr, size, elempack);
        }
    }

    if (bp == 1)
    {
        if (aw == bw)
        {
            // broadcast pack1 b
            return binary_op_vector_broadcast_pb_fp16s<Op>(ptr, ptr1, outptr, w, elempack);
        }

        if (bw == 1)
        {
            // broadcast pack1 single b
            return binary_op_vector_broadcast_pb_b_fp16s<Op>(ptr, ptr1, outptr, w, elempack);
        }

        if (aw == 1)
        {
            // broadcast single a and pack1 b
            return binary_op_vector_broadcast_pb_a_fp16s<Op>(ptr, ptr1, outptr, w, elempack);
        }
    }

    // shall never reach here
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

static void binary_op_vector_fp16s(const __fp16* ptr, const __fp16* ptr1, __fp16* outptr, int aw, int bw, int ap, int bp, int op_type)
{
    using namespace BinaryOp_riscv_functor;

    if (op_type == BinaryOp::Operation_ADD) return binary_op_vector_fp16s<binary_op_add_fp16s>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_SUB) return binary_op_vector_fp16s<binary_op_sub_fp16s>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_MUL) return binary_op_vector_fp16s<binary_op_mul_fp16s>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_DIV) return binary_op_vector_fp16s<binary_op_div_fp16s>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_MAX) return binary_op_vector_fp16s<binary_op_max_fp16s>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_MIN) return binary_op_vector_fp16s<binary_op_min_fp16s>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_POW) return binary_op_vector_fp16s<binary_op_pow_fp16s>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RSUB) return binary_op_vector_fp16s<binary_op_rsub_fp16s>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RDIV) return binary_op_vector_fp16s<binary_op_rdiv_fp16s>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RPOW) return binary_op_vector_fp16s<binary_op_rpow_fp16s>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_ATAN2) return binary_op_vector_fp16s<binary_op_atan2_fp16s>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RATAN2) return binary_op_vector_fp16s<binary_op_ratan2_fp16s>(ptr, ptr1, outptr, aw, bw, ap, bp);

    // should never reach here
}

static void binary_op_scalar_fp16s(const Mat& a, __fp16 b, Mat& c, int op_type, const Option& opt)
{
    const int channels = a.c;
    const int size = a.w * a.h * a.d * a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const __fp16* ptr = a.channel(q);
        __fp16* outptr = c.channel(q);

        binary_op_vector_fp16s(ptr, &b, outptr, size, 1, 1, 1, op_type);
    }
}

static void binary_op_no_broadcast_fp16s(const Mat& a, const Mat& b, Mat& c, int op_type, const Option& opt)
{
    const int channels = a.c;
    const int size = a.w * a.h * a.d * a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const __fp16* ptr = a.channel(q);
        const __fp16* ptr1 = b.channel(q);
        __fp16* outptr = c.channel(q);

        binary_op_vector_fp16s(ptr, ptr1, outptr, size, size, 1, 1, op_type);
    }
}

static void binary_op_broadcast_fp16s(const Mat& a, const Mat& b, Mat& c, int op_type, const Option& opt)
{
    if (b.w * b.h * b.d * b.c * b.elempack == 1)
    {
        return binary_op_scalar_fp16s(a, ((const __fp16*)b)[0], c, op_type, opt);
    }

    if (a.dims == b.dims && a.w == b.w && a.h == b.h && a.d == b.d && a.c == b.c && a.elempack == b.elempack)
    {
        return binary_op_no_broadcast_fp16s(a, b, c, op_type, opt);
    }

    const int dims = c.dims;

    if (dims == 2)
    {
        const int h = c.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < h; y++)
        {
            const int y0 = std::min(y, a.h - 1);
            const int y1 = std::min(y, b.h - 1);

            const __fp16* ptr = a.row<const __fp16>(y0);
            const __fp16* ptr1 = b.row<const __fp16>(y1);
            __fp16* outptr = c.row<__fp16>(y);

            binary_op_vector_fp16s(ptr, ptr1, outptr, a.w, b.w, a.elempack, b.elempack, op_type);
        }
    }

    if (dims == 3 || dims == 4)
    {
        const int channels = c.c;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const int q0 = std::min(q, a.c - 1);
            const int q1 = std::min(q, b.c - 1);

            if (b.d * b.h * b.w == 1)
            {
                const __fp16* ptr = a.channel(q0);
                const __fp16* ptr1 = b.channel(q1);
                __fp16* outptr = c.channel(q);

                binary_op_vector_fp16s(ptr, ptr1, outptr, a.w * a.h * a.d, 1, a.elempack, b.elempack, op_type);
                continue;
            }

            if (b.h * b.w == 1)
            {
                for (int z = 0; z < c.d; z++)
                {
                    const int z0 = std::min(z, a.d - 1);
                    const int z1 = std::min(z, b.d - 1);

                    const __fp16* ptr = a.channel(q0).depth(z0);
                    const __fp16* ptr1 = b.channel(q1).depth(z1);
                    __fp16* outptr = c.channel(q).depth(z);

                    binary_op_vector_fp16s(ptr, ptr1, outptr, a.w * a.h, 1, a.elempack, b.elempack, op_type);
                }
                continue;
            }

            for (int z = 0; z < c.d; z++)
            {
                const int z0 = std::min(z, a.d - 1);
                const int z1 = std::min(z, b.d - 1);

                for (int y = 0; y < c.h; y++)
                {
                    const int y0 = std::min(y, a.h - 1);
                    const int y1 = std::min(y, b.h - 1);

                    const __fp16* ptr = a.channel(q0).depth(z0).row<const __fp16>(y0);
                    const __fp16* ptr1 = b.channel(q1).depth(z1).row<const __fp16>(y1);
                    __fp16* outptr = c.channel(q).depth(z).row<__fp16>(y);

                    binary_op_vector_fp16s(ptr, ptr1, outptr, a.w, b.w, a.elempack, b.elempack, op_type);
                }
            }
        }
    }
}

static void binary_op_scalar_inplace_fp16s(Mat& a, __fp16 b, int op_type, const Option& opt)
{
    const int channels = a.c;
    const int size = a.w * a.h * a.d * a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = a.channel(q);

        binary_op_vector_fp16s(ptr, &b, ptr, size, 1, 1, 1, op_type);
    }
}

int BinaryOp_riscv::forward_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& A = bottom_blobs[0];
    const Mat& B = bottom_blobs[1];
    const int outdims = std::max(A.dims, B.dims);

    Mat A2 = A;
    Mat B2 = B;
    if (A.dims < outdims)
    {
        // expand inner axes
        if (outdims == 2)
        {
            if (A.w * A.elempack == B.h * B.elempack)
                A2 = A.reshape(1, A.w, opt.workspace_allocator);
            else // if (A.w == B.w)
            {
                A2.dims = 2;
                A2.w = A.w * A.elempack;
                A2.elempack = 1;
                A2.elemsize = A.elemsize / A.elempack;
                A2.cstep = A2.w;
            }
        }
        if (outdims == 3 && A.dims == 1)
        {
            if (A.w * A.elempack == B.c * B.elempack)
                A2 = A.reshape(1, 1, A.w, opt.workspace_allocator);
            else // if (A.w == B.w)
            {
                A2.dims = 3;
                A2.w = A.w * A.elempack;
                A2.elempack = 1;
                A2.elemsize = A.elemsize / A.elempack;
                A2.cstep = A2.w;
            }
        }
        if (outdims == 3 && A.dims == 2)
            A2 = A.reshape(1, A.w, A.h, opt.workspace_allocator);
        if (outdims == 4 && A.dims == 1)
        {
            if (A.w * A.elempack == B.c * B.elempack)
                A2 = A.reshape(1, 1, 1, A.w, opt.workspace_allocator);
            else // if (A.w == B.w)
            {
                A2.dims = 4;
                A2.w = A.w * A.elempack;
                A2.elempack = 1;
                A2.elemsize = A.elemsize / A.elempack;
                A2.cstep = A2.w;
            }
        }
        if (outdims == 4 && A.dims == 2)
            A2 = A.reshape(1, 1, A.w, A.h, opt.workspace_allocator);
        if (outdims == 4 && A.dims == 3)
            A2 = A.reshape(1, A.w, A.h, A.c, opt.workspace_allocator);
    }
    if (B.dims < outdims)
    {
        // expand inner axes
        if (outdims == 2)
        {
            if (B.w * B.elempack == A.h * A.elempack)
                B2 = B.reshape(1, B.w, opt.workspace_allocator);
            else // if (B.w == A.w)
            {
                B2.dims = 2;
                B2.w = B.w * B.elempack;
                B2.elempack = 1;
                B2.elemsize = B.elemsize / B.elempack;
                B2.cstep = B2.w;
            }
        }
        if (outdims == 3 && B.dims == 1)
        {
            if (B.w * B.elempack == A.c * A.elempack)
                B2 = B.reshape(1, 1, B.w, opt.workspace_allocator);
            else // if (B.w == A.w)
            {
                B2.dims = 3;
                B2.w = B.w * B.elempack;
                B2.elempack = 1;
                B2.elemsize = B.elemsize / B.elempack;
                B2.cstep = B2.w;
            }
        }
        if (outdims == 3 && B.dims == 2)
            B2 = B.reshape(1, B.w, B.h, opt.workspace_allocator);
        if (outdims == 4 && B.dims == 1)
        {
            if (B.w * B.elempack == A.c * A.elempack)
                B2 = B.reshape(1, 1, 1, B.w, opt.workspace_allocator);
            else // if (B.w == A.w)
            {
                B2.dims = 4;
                B2.w = B.w * B.elempack;
                B2.elempack = 1;
                B2.elemsize = B.elemsize / B.elempack;
                B2.cstep = B2.w;
            }
        }
        if (outdims == 4 && B.dims == 2)
            B2 = B.reshape(1, 1, B.w, B.h, opt.workspace_allocator);
        if (outdims == 4 && B.dims == 3)
            B2 = B.reshape(1, B.w, B.h, B.c, opt.workspace_allocator);
    }

    const int outw = std::max(A2.w, B2.w);
    const int outh = std::max(A2.h, B2.h);
    const int outd = std::max(A2.d, B2.d);
    const int outc = std::max(A2.c, B2.c);
    const size_t out_elemsize = std::max(A2.elemsize, B2.elemsize);
    const int out_elempack = std::max(A2.elempack, B2.elempack);

    Mat& top_blob = top_blobs[0];
    if (outdims == 1)
    {
        top_blob.create(outw, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (outdims == 2)
    {
        top_blob.create(outw, outh, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (outdims == 3)
    {
        top_blob.create(outw, outh, outc, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (outdims == 4)
    {
        top_blob.create(outw, outh, outd, outc, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    const bool a_pack_is_lower = A2.elempack < B2.elempack;
    const bool a_pack_is_equal = A2.elempack == B2.elempack;
    const bool a_size_is_lower = A2.w * A2.h * A2.d * A2.c * A2.elempack < B2.w * B2.h * B2.d * B2.c * B2.elempack;
    if (a_pack_is_lower || (a_pack_is_equal && a_size_is_lower))
    {
        binary_op_broadcast_fp16s(B2, A2, top_blob, get_reverse_op_type(op_type), opt);
    }
    else
    {
        binary_op_broadcast_fp16s(A2, B2, top_blob, op_type, opt);
    }

    return 0;
}

int BinaryOp_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    binary_op_scalar_inplace_fp16s(bottom_top_blob, b, op_type, opt);

    return 0;
}
#endif // __riscv_vector && __riscv_zfh

} // namespace ncnn
