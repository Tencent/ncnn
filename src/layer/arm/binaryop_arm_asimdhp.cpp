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

#include "binaryop_arm.h"

#include <math.h>

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#endif // __ARM_NEON

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
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

        int i = 0;
        float16x8_t _b = vdupq_n_f16(b);
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            _p = op(_p, _b);
            vst1q_f16(outptr, _p);
            ptr += 8;
            outptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float16x4_t _p = vld1_f16(ptr);
            _p = op(_p, vget_low_f16(_b));
            vst1_f16(outptr, _p);
            ptr += 4;
            outptr += 4;
        }
        for (; i < size; i++)
        {
            *outptr = op(*ptr, b);
            ptr++;
            outptr++;
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

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            float16x8_t _p1 = vld1q_f16(ptr1);
            float16x8_t _outp = op(_p, _p1);
            vst1q_f16(outptr, _outp);
            ptr += 8;
            ptr1 += 8;
            outptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float16x4_t _p = vld1_f16(ptr);
            float16x4_t _p1 = vld1_f16(ptr1);
            float16x4_t _outp = op(_p, _p1);
            vst1_f16(outptr, _outp);
            ptr += 4;
            ptr1 += 4;
            outptr += 4;
        }
        for (; i < size; i++)
        {
            *outptr = op(*ptr, *ptr1);
            ptr += 1;
            ptr1 += 1;
            outptr += 1;
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
            float16x4_t _b_128 = (elempack == 4) ? vld1_f16((const __fp16*)b + y * 4) : vdup_n_f16(_b);
            float16x8_t _b_256 = (elempack == 8) ? vld1q_f16((const __fp16*)b + y * 8) : vcombine_f16(_b_128, _b_128);

            const int size = w * elempack;

            int i = 0;
            for (; i + 7 < size; i += 8)
            {
                float16x8_t _p = vld1q_f16(ptr);
                float16x8_t _outp = op(_p, _b_256);
                vst1q_f16(outptr, _outp);
                ptr += 8;
                outptr += 8;
            }
            for (; i + 3 < size; i += 4)
            {
                float16x4_t _p = vld1_f16(ptr);
                float16x4_t _outp = op(_p, _b_128);
                vst1_f16(outptr, _outp);
                ptr += 4;
                outptr += 4;
            }
            for (; i < size; i++)
            {
                *outptr = op(*ptr, _b);
                ptr += 1;
                outptr += 1;
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
            float16x4_t _b_128 = (elempack == 4) ? vld1_f16((const __fp16*)b + q * 4) : vdup_n_f16(_b);
            float16x8_t _b_256 = (elempack == 8) ? vld1q_f16((const __fp16*)b + q * 8) : vcombine_f16(_b_128, _b_128);

            const int size = w * h * d * elempack;

            int i = 0;
            for (; i + 7 < size; i += 8)
            {
                float16x8_t _p = vld1q_f16(ptr);
                float16x8_t _outp = op(_p, _b_256);
                vst1q_f16(outptr, _outp);
                ptr += 8;
                outptr += 8;
            }
            for (; i + 3 < size; i += 4)
            {
                float16x4_t _p = vld1_f16(ptr);
                float16x4_t _outp = op(_p, _b_128);
                vst1_f16(outptr, _outp);
                ptr += 4;
                outptr += 4;
            }
            for (; i < size; i++)
            {
                *outptr = op(*ptr, _b);
                ptr += 1;
                outptr += 1;
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
                float16x4_t _b_128 = (elempack == 4) ? vld1_f16((const __fp16*)ptr1 + y * 4) : vdup_n_f16(_b);
                float16x8_t _b_256 = (elempack == 8) ? vld1q_f16((const __fp16*)ptr1 + y * 8) : vcombine_f16(_b_128, _b_128);

                int i = 0;
                for (; i + 7 < size; i += 8)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    float16x8_t _outp = op(_p, _b_256);
                    vst1q_f16(outptr, _outp);
                    ptr += 8;
                    outptr += 8;
                }
                for (; i + 3 < size; i += 4)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    float16x4_t _outp = op(_p, _b_128);
                    vst1_f16(outptr, _outp);
                    ptr += 4;
                    outptr += 4;
                }
                for (; i < size; i++)
                {
                    *outptr = op(*ptr, _b);
                    ptr += 1;
                    outptr += 1;
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
                float16x4_t _b_128 = (elempack == 4) ? vld1_f16((const __fp16*)ptr1 + z * 4) : vdup_n_f16(_b);
                float16x8_t _b_256 = (elempack == 8) ? vld1q_f16((const __fp16*)ptr1 + z * 8) : vcombine_f16(_b_128, _b_128);

                int i = 0;
                for (; i + 7 < size; i += 8)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    float16x8_t _outp = op(_p, _b_256);
                    vst1q_f16(outptr, _outp);
                    ptr += 8;
                    outptr += 8;
                }
                for (; i + 3 < size; i += 4)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    float16x4_t _outp = op(_p, _b_128);
                    vst1_f16(outptr, _outp);
                    ptr += 4;
                    outptr += 4;
                }
                for (; i < size; i++)
                {
                    *outptr = op(*ptr, _b);
                    ptr += 1;
                    outptr += 1;
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
                    float16x4_t _b_128 = (elempack == 4) ? vld1_f16((const __fp16*)ptr1 + y * 4) : vdup_n_f16(_b);
                    float16x8_t _b_256 = (elempack == 8) ? vld1q_f16((const __fp16*)ptr1 + y * 8) : vcombine_f16(_b_128, _b_128);

                    int i = 0;
                    for (; i + 7 < size; i += 8)
                    {
                        float16x8_t _p = vld1q_f16(ptr);
                        float16x8_t _outp = op(_p, _b_256);
                        vst1q_f16(outptr, _outp);
                        ptr += 8;
                        outptr += 8;
                    }
                    for (; i + 3 < size; i += 4)
                    {
                        float16x4_t _p = vld1_f16(ptr);
                        float16x4_t _outp = op(_p, _b_128);
                        vst1_f16(outptr, _outp);
                        ptr += 4;
                        outptr += 4;
                    }
                    for (; i < size; i++)
                    {
                        *outptr = op(*ptr, _b);
                        ptr += 1;
                        outptr += 1;
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

            if (elempack == 8)
            {
                for (int x = 0; x < w; x++)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    float16x8_t _b = vdupq_n_f16(*ptr1);
                    float16x8_t _outp = op(_p, _b);
                    vst1q_f16(outptr, _outp);
                    ptr += 8;
                    ptr1 += 1;
                    outptr += 8;
                }
            }
            if (elempack == 4)
            {
                for (int x = 0; x < w; x++)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    float16x4_t _b = vdup_n_f16(*ptr1);
                    float16x4_t _outp = op(_p, _b);
                    vst1_f16(outptr, _outp);
                    ptr += 4;
                    ptr1 += 1;
                    outptr += 4;
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

                    if (elempack == 8)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            float16x8_t _p = vld1q_f16(ptr);
                            float16x8_t _b = vdupq_n_f16(*ptr1);
                            float16x8_t _outp = op(_p, _b);
                            vst1q_f16(outptr, _outp);
                            ptr += 8;
                            ptr1 += 1;
                            outptr += 8;
                        }
                    }
                    if (elempack == 4)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            float16x4_t _p = vld1_f16(ptr);
                            float16x4_t _b = vdup_n_f16(*ptr1);
                            float16x4_t _outp = op(_p, _b);
                            vst1_f16(outptr, _outp);
                            ptr += 4;
                            ptr1 += 1;
                            outptr += 4;
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

            int i = 0;
            for (; i + 7 < size; i += 8)
            {
                float16x8_t _p = vld1q_f16(ptr);
                float16x8_t _p1 = vld1q_f16(ptr1);
                float16x8_t _outp = op(_p, _p1);
                vst1q_f16(outptr, _outp);
                ptr += 8;
                ptr1 += 8;
                outptr += 8;
            }
            for (; i + 3 < size; i += 4)
            {
                float16x4_t _p = vld1_f16(ptr);
                float16x4_t _p1 = vld1_f16(ptr1);
                float16x4_t _outp = op(_p, _p1);
                vst1_f16(outptr, _outp);
                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }
            for (; i < size; i++)
            {
                *outptr = op(*ptr, *ptr1);
                ptr += 1;
                ptr1 += 1;
                outptr += 1;
            }
        }
    }

    return 0;
}

template<typename Op>
static int binary_op_scalar_inplace_fp16s(Mat& a, __fp16 b, const Option& opt)
{
    Op op;

    const int channels = a.c;
    const int size = a.w * a.h * a.d * a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = a.channel(q);

        int i = 0;
        float16x8_t _b = vdupq_n_f16(b);
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            _p = op(_p, _b);
            vst1q_f16(ptr, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float16x4_t _p = vld1_f16(ptr);
            _p = op(_p, vget_low_f16(_b));
            vst1_f16(ptr, _p);
            ptr += 4;
        }
        for (; i < size; i++)
        {
            *ptr = op(*ptr, b);
            ptr++;
        }
    }

    return 0;
}

namespace BinaryOp_arm_functor {

#define MAKE_FUNCTION(NAME, IMPL, IMPL4, IMPL8)                                  \
    struct NAME                                                                  \
    {                                                                            \
        __fp16 operator()(const __fp16& x, const __fp16& y) const                \
        {                                                                        \
            return IMPL;                                                         \
        }                                                                        \
        float16x4_t operator()(const float16x4_t& x, const float16x4_t& y) const \
        {                                                                        \
            return IMPL4;                                                        \
        }                                                                        \
        float16x8_t operator()(const float16x8_t& x, const float16x8_t& y) const \
        {                                                                        \
            return IMPL8;                                                        \
        }                                                                        \
    };

// clang-format off
// *INDENT-OFF*
MAKE_FUNCTION(binary_op_add_fp16s, x + y, vadd_f16(x, y), vaddq_f16(x, y))
MAKE_FUNCTION(binary_op_sub_fp16s, x - y, vsub_f16(x, y), vsubq_f16(x, y))
MAKE_FUNCTION(binary_op_mul_fp16s, x * y, vmul_f16(x, y), vmulq_f16(x, y))
MAKE_FUNCTION(binary_op_div_fp16s, x / y, vdiv_f16(x, y), vdivq_f16(x, y))
MAKE_FUNCTION(binary_op_max_fp16s, std::max(x, y), vmax_f16(x, y), vmaxq_f16(x, y))
MAKE_FUNCTION(binary_op_min_fp16s, std::min(x, y), vmin_f16(x, y), vminq_f16(x, y))
MAKE_FUNCTION(binary_op_pow_fp16s, (__fp16)pow(x, y), vcvt_f16_f32(pow_ps(vcvt_f32_f16(x), vcvt_f32_f16(y))), vcombine_f16(vcvt_f16_f32(pow_ps(vcvt_f32_f16(vget_low_f16(x)), vcvt_f32_f16(vget_low_f16(y)))), vcvt_f16_f32(pow_ps(vcvt_f32_f16(vget_high_f16(x)), vcvt_f32_f16(vget_high_f16(y))))))
MAKE_FUNCTION(binary_op_rsub_fp16s, y - x, vsub_f16(y, x), vsubq_f16(y, x))
MAKE_FUNCTION(binary_op_rdiv_fp16s, y / x, vdiv_f16(y, x), vdivq_f16(y, x))
MAKE_FUNCTION(binary_op_rpow_fp16s, (__fp16)pow(y, x), vcvt_f16_f32(pow_ps(vcvt_f32_f16(y), vcvt_f32_f16(x))), vcombine_f16(vcvt_f16_f32(pow_ps(vcvt_f32_f16(vget_low_f16(y)), vcvt_f32_f16(vget_low_f16(x)))), vcvt_f16_f32(pow_ps(vcvt_f32_f16(vget_high_f16(y)), vcvt_f32_f16(vget_high_f16(x))))))
MAKE_FUNCTION(binary_op_atan2_fp16s, (__fp16)atan2(x, y), vcvt_f16_f32(atan2_ps(vcvt_f32_f16(x), vcvt_f32_f16(y))), vcombine_f16(vcvt_f16_f32(atan2_ps(vcvt_f32_f16(vget_low_f16(x)), vcvt_f32_f16(vget_low_f16(y)))), vcvt_f16_f32(atan2_ps(vcvt_f32_f16(vget_high_f16(x)), vcvt_f32_f16(vget_high_f16(y))))))
MAKE_FUNCTION(binary_op_ratan2_fp16s, (__fp16)atan2(y, x), vcvt_f16_f32(atan2_ps(vcvt_f32_f16(y), vcvt_f32_f16(x))), vcombine_f16(vcvt_f16_f32(atan2_ps(vcvt_f32_f16(vget_low_f16(y)), vcvt_f32_f16(vget_low_f16(x)))), vcvt_f16_f32(atan2_ps(vcvt_f32_f16(vget_high_f16(y)), vcvt_f32_f16(vget_high_f16(x))))))
// *INDENT-ON*
// clang-format on

#undef MAKE_FUNCTION

} // namespace BinaryOp_arm_functor

static int binary_op_scalar_fp16s(const Mat& a, float b, Mat& c, int op_type, const Option& opt)
{
    using namespace BinaryOp_arm_functor;

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
    using namespace BinaryOp_arm_functor;

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

    using namespace BinaryOp_arm_functor;

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
    using namespace BinaryOp_arm_functor;

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
    using namespace BinaryOp_arm_functor;

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

int BinaryOp_arm::forward_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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

int BinaryOp_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    using namespace BinaryOp_arm_functor;

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
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn
