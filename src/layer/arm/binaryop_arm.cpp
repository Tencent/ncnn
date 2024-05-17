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

#include "binaryop_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#endif // __ARM_NEON

#include "arm_usability.h"
#include "cpu.h"

namespace ncnn {

BinaryOp_arm::BinaryOp_arm()
{
#if __ARM_NEON
    support_packing = true;
#if NCNN_ARM82
    support_fp16_storage = cpu_support_arm_asimdhp();
#endif
#endif // __ARM_NEON

#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

template<typename Op>
static void binary_op_vector_no_broadcast(const float* ptr, const float* ptr1, float* outptr, int size)
{
    const Op op;

    int i = 0;
#if __ARM_NEON
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _p = vld1q_f32(ptr);
        float32x4_t _b = vld1q_f32(ptr1);
        float32x4_t _outp = op(_p, _b);
        vst1q_f32(outptr, _outp);
        ptr += 4;
        ptr1 += 4;
        outptr += 4;
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        *outptr = op(*ptr, *ptr1);
        ptr += 1;
        ptr1 += 1;
        outptr += 1;
    }
}

template<typename Op>
static void binary_op_vector_broadcast_b(const float* ptr, const float* ptr1, float* outptr, int size, int elempack)
{
    const Op op;

    const float b = *ptr1;

    int i = 0;
#if __ARM_NEON
    float32x4_t _b_128 = (elempack == 4) ? vld1q_f32(ptr1) : vdupq_n_f32(b);
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _p = vld1q_f32(ptr);
        float32x4_t _outp = op(_p, _b_128);
        vst1q_f32(outptr, _outp);
        ptr += 4;
        outptr += 4;
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        *outptr = op(*ptr, b);
        ptr += 1;
        outptr += 1;
    }
}

template<typename Op>
static void binary_op_vector_broadcast_a(const float* ptr, const float* ptr1, float* outptr, int size, int elempack)
{
    const Op op;

    const float a = *ptr;

    int i = 0;
#if __ARM_NEON
    float32x4_t _a_128 = (elempack == 4) ? vld1q_f32(ptr) : vdupq_n_f32(a);
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _b = vld1q_f32(ptr1);
        float32x4_t _outp = op(_a_128, _b);
        vst1q_f32(outptr, _outp);
        ptr1 += 4;
        outptr += 4;
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        *outptr = op(a, *ptr1);
        ptr1 += 1;
        outptr += 1;
    }
}

template<typename Op>
static void binary_op_vector_broadcast_pb(const float* ptr, const float* ptr1, float* outptr, int w, int elempack)
{
    const Op op;

#if __ARM_NEON
    if (elempack == 4)
    {
        int i = 0;
        for (; i < w; i++)
        {
            float32x4_t _p = vld1q_f32(ptr);
            float32x4_t _b = vdupq_n_f32(*ptr1);
            float32x4_t _outp = op(_p, _b);
            vst1q_f32(outptr, _outp);
            ptr += 4;
            ptr1 += 1;
            outptr += 4;
        }
    }
#endif // __ARM_NEON
}

template<typename Op>
static void binary_op_vector_broadcast_pb_b(const float* ptr, const float* ptr1, float* outptr, int w, int elempack)
{
    const Op op;

    const int size = w * elempack;

    int i = 0;
#if __ARM_NEON
    float32x4_t _b = vdupq_n_f32(*ptr1);
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _p = vld1q_f32(ptr);
        float32x4_t _outp = op(_p, _b);
        vst1q_f32(outptr, _outp);
        ptr += 4;
        outptr += 4;
    }
#endif // __ARM_NEON
}

template<typename Op>
static void binary_op_vector_broadcast_pb_a(const float* ptr, const float* ptr1, float* outptr, int w, int elempack)
{
    const Op op;

#if __ARM_NEON
    if (elempack == 4)
    {
        int i = 0;
        float32x4_t _p = vld1q_f32(ptr);
        for (; i < w; i++)
        {
            float32x4_t _b = vdupq_n_f32(*ptr1);
            float32x4_t _outp = op(_p, _b);
            vst1q_f32(outptr, _outp);
            ptr1 += 1;
            outptr += 4;
        }
    }
#endif // __ARM_NEON
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

namespace BinaryOp_arm_functor {

#if __ARM_NEON
#define MAKE_FUNCTION(NAME, IMPL, IMPL4)                                         \
    struct NAME                                                                  \
    {                                                                            \
        float operator()(const float& x, const float& y) const                   \
        {                                                                        \
            return IMPL;                                                         \
        }                                                                        \
        float32x4_t operator()(const float32x4_t& x, const float32x4_t& y) const \
        {                                                                        \
            return IMPL4;                                                        \
        }                                                                        \
    };
#else
#define MAKE_FUNCTION(NAME, IMPL, IMPL4)                       \
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
MAKE_FUNCTION(binary_op_add, x + y, vaddq_f32(x, y))
MAKE_FUNCTION(binary_op_sub, x - y, vsubq_f32(x, y))
MAKE_FUNCTION(binary_op_mul, x * y, vmulq_f32(x, y))
#if __aarch64__
MAKE_FUNCTION(binary_op_div, x / y, vdivq_f32(x, y))
#else
MAKE_FUNCTION(binary_op_div, x / y, div_ps(x, y))
#endif
MAKE_FUNCTION(binary_op_max, std::max(x, y), vmaxq_f32(x, y))
MAKE_FUNCTION(binary_op_min, std::min(x, y), vminq_f32(x, y))
MAKE_FUNCTION(binary_op_pow, (float)powf(x, y), pow_ps(x, y))
MAKE_FUNCTION(binary_op_rsub, y - x, vsubq_f32(y, x))
#if __aarch64__
MAKE_FUNCTION(binary_op_rdiv, y / x, vdivq_f32(y, x))
#else
MAKE_FUNCTION(binary_op_rdiv, y / x, div_ps(y, x))
#endif
MAKE_FUNCTION(binary_op_rpow, (float)powf(y, x), pow_ps(y, x))
MAKE_FUNCTION(binary_op_atan2, (float)atan2f(x, y), atan2_ps(x, y))
MAKE_FUNCTION(binary_op_ratan2, (float)atan2f(y, x), atan2_ps(y, x))
// *INDENT-ON*
// clang-format on

#undef MAKE_FUNCTION

} // namespace BinaryOp_arm_functor

static void binary_op_vector(const float* ptr, const float* ptr1, float* outptr, int aw, int bw, int ap, int bp, int op_type)
{
    using namespace BinaryOp_arm_functor;

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

int BinaryOp_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    int elembits = std::max(bottom_blobs[0].elembits(), bottom_blobs[1].elembits());

#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
        return forward_fp16s(bottom_blobs, top_blobs, opt);
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s(bottom_blobs, top_blobs, opt);
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

int BinaryOp_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elembits = bottom_top_blob.elembits();

#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
        return forward_inplace_fp16s(bottom_top_blob, opt);
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    binary_op_scalar_inplace(bottom_top_blob, b, op_type, opt);

    return 0;
}

#if NCNN_BF16
template<typename Op>
static void binary_op_vector_no_broadcast_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int size)
{
    const Op op;

    int i = 0;
#if __ARM_NEON
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _p = bfloat2float(vld1_u16(ptr));
        float32x4_t _b = bfloat2float(vld1_u16(ptr1));
        float32x4_t _outp = op(_p, _b);
        vst1_u16(outptr, float2bfloat(_outp));
        ptr += 4;
        ptr1 += 4;
        outptr += 4;
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        *outptr = float32_to_bfloat16(op(bfloat16_to_float32(*ptr), bfloat16_to_float32(*ptr1)));
        ptr += 1;
        ptr1 += 1;
        outptr += 1;
    }
}

template<typename Op>
static void binary_op_vector_broadcast_b_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int size, int elempack)
{
    const Op op;

    const float b = bfloat16_to_float32(*ptr1);

    int i = 0;
#if __ARM_NEON
    float32x4_t _b_128 = (elempack == 4) ? bfloat2float(vld1_u16(ptr1)) : vdupq_n_f32(b);
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _p = bfloat2float(vld1_u16(ptr));
        float32x4_t _outp = op(_p, _b_128);
        vst1_u16(outptr, float2bfloat(_outp));
        ptr += 4;
        outptr += 4;
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        *outptr = float32_to_bfloat16(op(bfloat16_to_float32(*ptr), b));
        ptr += 1;
        outptr += 1;
    }
}

template<typename Op>
static void binary_op_vector_broadcast_a_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int size, int elempack)
{
    const Op op;

    const float a = bfloat16_to_float32(*ptr);

    int i = 0;
#if __ARM_NEON
    float32x4_t _a_128 = (elempack == 4) ? bfloat2float(vld1_u16(ptr)) : vdupq_n_f32(a);
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _b = bfloat2float(vld1_u16(ptr1));
        float32x4_t _outp = op(_a_128, _b);
        vst1_u16(outptr, float2bfloat(_outp));
        ptr1 += 4;
        outptr += 4;
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        *outptr = float32_to_bfloat16(op(a, bfloat16_to_float32(*ptr1)));
        ptr1 += 1;
        outptr += 1;
    }
}

template<typename Op>
static void binary_op_vector_broadcast_pb_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int w, int elempack)
{
    const Op op;

#if __ARM_NEON
    if (elempack == 4)
    {
        int i = 0;
        for (; i < w; i++)
        {
            float32x4_t _p = bfloat2float(vld1_u16(ptr));
            float32x4_t _b = bfloat2float(vdup_n_u16(*ptr1));
            float32x4_t _outp = op(_p, _b);
            vst1_u16(outptr, float2bfloat(_outp));
            ptr += 4;
            ptr1 += 1;
            outptr += 4;
        }
    }
#endif // __ARM_NEON
}

template<typename Op>
static void binary_op_vector_broadcast_pb_b_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int w, int elempack)
{
    const Op op;

    const int size = w * elempack;

    int i = 0;
#if __ARM_NEON
    float32x4_t _b = bfloat2float(vdup_n_u16(*ptr1));
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _p = bfloat2float(vld1_u16(ptr));
        float32x4_t _outp = op(_p, _b);
        vst1_u16(outptr, float2bfloat(_outp));
        ptr += 4;
        outptr += 4;
    }
#endif // __ARM_NEON
}

template<typename Op>
static void binary_op_vector_broadcast_pb_a_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int w, int elempack)
{
    const Op op;

#if __ARM_NEON
    if (elempack == 4)
    {
        int i = 0;
        float32x4_t _p = bfloat2float(vld1_u16(ptr));
        for (; i < w; i++)
        {
            float32x4_t _b = bfloat2float(vdup_n_u16(*ptr1));
            float32x4_t _outp = op(_p, _b);
            vst1_u16(outptr, float2bfloat(_outp));
            ptr1 += 1;
            outptr += 4;
        }
    }
#endif // __ARM_NEON
}

template<typename Op>
static void binary_op_vector_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int aw, int bw, int ap, int bp)
{
    const int w = std::max(aw, bw);
    const int elempack = std::max(ap, bp);
    const int size = w * elempack;

    if (ap == bp)
    {
        if (aw == bw)
        {
            // no broadcast
            return binary_op_vector_no_broadcast_bf16s<Op>(ptr, ptr1, outptr, size);
        }

        if (bw == 1)
        {
            // broadcast single b
            return binary_op_vector_broadcast_b_bf16s<Op>(ptr, ptr1, outptr, size, elempack);
        }

        if (aw == 1)
        {
            // broadcast single a
            return binary_op_vector_broadcast_a_bf16s<Op>(ptr, ptr1, outptr, size, elempack);
        }
    }

    if (bp == 1)
    {
        if (aw == bw)
        {
            // broadcast pack1 b
            return binary_op_vector_broadcast_pb_bf16s<Op>(ptr, ptr1, outptr, w, elempack);
        }

        if (bw == 1)
        {
            // broadcast pack1 single b
            return binary_op_vector_broadcast_pb_b_bf16s<Op>(ptr, ptr1, outptr, w, elempack);
        }

        if (aw == 1)
        {
            // broadcast single a and pack1 b
            return binary_op_vector_broadcast_pb_a_bf16s<Op>(ptr, ptr1, outptr, w, elempack);
        }
    }

    // shall never reach here
}

static void binary_op_vector_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int aw, int bw, int ap, int bp, int op_type)
{
    using namespace BinaryOp_arm_functor;

    if (op_type == BinaryOp::Operation_ADD) return binary_op_vector_bf16s<binary_op_add>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_SUB) return binary_op_vector_bf16s<binary_op_sub>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_MUL) return binary_op_vector_bf16s<binary_op_mul>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_DIV) return binary_op_vector_bf16s<binary_op_div>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_MAX) return binary_op_vector_bf16s<binary_op_max>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_MIN) return binary_op_vector_bf16s<binary_op_min>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_POW) return binary_op_vector_bf16s<binary_op_pow>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RSUB) return binary_op_vector_bf16s<binary_op_rsub>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RDIV) return binary_op_vector_bf16s<binary_op_rdiv>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RPOW) return binary_op_vector_bf16s<binary_op_rpow>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_ATAN2) return binary_op_vector_bf16s<binary_op_atan2>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RATAN2) return binary_op_vector_bf16s<binary_op_ratan2>(ptr, ptr1, outptr, aw, bw, ap, bp);

    // should never reach here
}

template<typename Op>
static void binary_op_vector_scalar_b_bf16s(const unsigned short* ptr, float b, unsigned short* outptr, int size)
{
    const Op op;

    int i = 0;
#if __ARM_NEON
    float32x4_t _b_128 = vdupq_n_f32(b);
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _p = bfloat2float(vld1_u16(ptr));
        float32x4_t _outp = op(_p, _b_128);
        vst1_u16(outptr, float2bfloat(_outp));
        ptr += 4;
        outptr += 4;
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        *outptr = float32_to_bfloat16(op(bfloat16_to_float32(*ptr), b));
        ptr += 1;
        outptr += 1;
    }
}

static void binary_op_vector_scalar_b_bf16s(const unsigned short* ptr, float b, unsigned short* outptr, int size, int op_type)
{
    using namespace BinaryOp_arm_functor;

    if (op_type == BinaryOp::Operation_ADD) return binary_op_vector_scalar_b_bf16s<binary_op_add>(ptr, b, outptr, size);
    if (op_type == BinaryOp::Operation_SUB) return binary_op_vector_scalar_b_bf16s<binary_op_sub>(ptr, b, outptr, size);
    if (op_type == BinaryOp::Operation_MUL) return binary_op_vector_scalar_b_bf16s<binary_op_mul>(ptr, b, outptr, size);
    if (op_type == BinaryOp::Operation_DIV) return binary_op_vector_scalar_b_bf16s<binary_op_div>(ptr, b, outptr, size);
    if (op_type == BinaryOp::Operation_MAX) return binary_op_vector_scalar_b_bf16s<binary_op_max>(ptr, b, outptr, size);
    if (op_type == BinaryOp::Operation_MIN) return binary_op_vector_scalar_b_bf16s<binary_op_min>(ptr, b, outptr, size);
    if (op_type == BinaryOp::Operation_POW) return binary_op_vector_scalar_b_bf16s<binary_op_pow>(ptr, b, outptr, size);
    if (op_type == BinaryOp::Operation_RSUB) return binary_op_vector_scalar_b_bf16s<binary_op_rsub>(ptr, b, outptr, size);
    if (op_type == BinaryOp::Operation_RDIV) return binary_op_vector_scalar_b_bf16s<binary_op_rdiv>(ptr, b, outptr, size);
    if (op_type == BinaryOp::Operation_RPOW) return binary_op_vector_scalar_b_bf16s<binary_op_rpow>(ptr, b, outptr, size);
    if (op_type == BinaryOp::Operation_ATAN2) return binary_op_vector_scalar_b_bf16s<binary_op_atan2>(ptr, b, outptr, size);
    if (op_type == BinaryOp::Operation_RATAN2) return binary_op_vector_scalar_b_bf16s<binary_op_ratan2>(ptr, b, outptr, size);

    // should never reach here
}

static void binary_op_scalar_bf16s(const Mat& a, float b, Mat& c, int op_type, const Option& opt)
{
    const int channels = a.c;
    const int size = a.w * a.h * a.d * a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const unsigned short* ptr = a.channel(q);
        unsigned short* outptr = c.channel(q);

        binary_op_vector_scalar_b_bf16s(ptr, b, outptr, size, op_type);
    }
}

static void binary_op_no_broadcast_bf16s(const Mat& a, const Mat& b, Mat& c, int op_type, const Option& opt)
{
    const int channels = a.c;
    const int size = a.w * a.h * a.d * a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const unsigned short* ptr = a.channel(q);
        const unsigned short* ptr1 = b.channel(q);
        unsigned short* outptr = c.channel(q);

        binary_op_vector_bf16s(ptr, ptr1, outptr, size, size, 1, 1, op_type);
    }
}

static void binary_op_broadcast_bf16s(const Mat& a, const Mat& b, Mat& c, int op_type, const Option& opt)
{
    if (b.w * b.h * b.d * b.c * b.elempack == 1)
    {
        return binary_op_scalar_bf16s(a, bfloat16_to_float32(((const unsigned short*)b)[0]), c, op_type, opt);
    }

    if (a.dims == b.dims && a.w == b.w && a.h == b.h && a.d == b.d && a.c == b.c && a.elempack == b.elempack)
    {
        return binary_op_no_broadcast_bf16s(a, b, c, op_type, opt);
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

            const unsigned short* ptr = a.row<const unsigned short>(y0);
            const unsigned short* ptr1 = b.row<const unsigned short>(y1);
            unsigned short* outptr = c.row<unsigned short>(y);

            binary_op_vector_bf16s(ptr, ptr1, outptr, a.w, b.w, a.elempack, b.elempack, op_type);
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
                const unsigned short* ptr = a.channel(q0);
                const unsigned short* ptr1 = b.channel(q1);
                unsigned short* outptr = c.channel(q);

                binary_op_vector_bf16s(ptr, ptr1, outptr, a.w * a.h * a.d, 1, a.elempack, b.elempack, op_type);
                continue;
            }

            if (b.h * b.w == 1)
            {
                for (int z = 0; z < c.d; z++)
                {
                    const int z0 = std::min(z, a.d - 1);
                    const int z1 = std::min(z, b.d - 1);

                    const unsigned short* ptr = a.channel(q0).depth(z0);
                    const unsigned short* ptr1 = b.channel(q1).depth(z1);
                    unsigned short* outptr = c.channel(q).depth(z);

                    binary_op_vector_bf16s(ptr, ptr1, outptr, a.w * a.h, 1, a.elempack, b.elempack, op_type);
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

                    const unsigned short* ptr = a.channel(q0).depth(z0).row<const unsigned short>(y0);
                    const unsigned short* ptr1 = b.channel(q1).depth(z1).row<const unsigned short>(y1);
                    unsigned short* outptr = c.channel(q).depth(z).row<unsigned short>(y);

                    binary_op_vector_bf16s(ptr, ptr1, outptr, a.w, b.w, a.elempack, b.elempack, op_type);
                }
            }
        }
    }
}

static void binary_op_scalar_inplace_bf16s(Mat& a, float b, int op_type, const Option& opt)
{
    const int channels = a.c;
    const int size = a.w * a.h * a.d * a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = a.channel(q);

        binary_op_vector_scalar_b_bf16s(ptr, b, ptr, size, op_type);
    }
}

int BinaryOp_arm::forward_bf16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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
        binary_op_broadcast_bf16s(B2, A2, top_blob, get_reverse_op_type(op_type), opt);
    }
    else
    {
        binary_op_broadcast_bf16s(A2, B2, top_blob, op_type, opt);
    }

    return 0;
}

int BinaryOp_arm::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    binary_op_scalar_inplace_bf16s(bottom_top_blob, b, op_type, opt);

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
