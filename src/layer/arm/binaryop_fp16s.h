// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static inline float16x4_t fmod_f16(const float16x4_t& x, const float16x4_t& y)
{
    float32x4_t fx = vcvt_f32_f16(x);
    float32x4_t fy = vcvt_f32_f16(y);
    return vcvt_f16_f32(fmod_ps(fx, fy));
}

static inline float16x8_t fmodq_f16(const float16x8_t& x, const float16x8_t& y)
{
    float16x4_t xl = vget_low_f16(x);
    float16x4_t xh = vget_high_f16(x);
    float16x4_t yl = vget_low_f16(y);
    float16x4_t yh = vget_high_f16(y);

    float16x4_t rl = fmod_f16(xl, yl);
    float16x4_t rh = fmod_f16(xh, yh);
    return vcombine_f16(rl, rh);
}

static inline float16x4_t round_f16(const float16x4_t& x)
{
    return vcvt_f16_f32(round_ps(vcvt_f32_f16(x)));
}

static inline float16x8_t roundq_f16(const float16x8_t& x)
{
    float16x4_t xl = vget_low_f16(x);
    float16x4_t xh = vget_high_f16(x);
    float16x4_t rl = round_f16(xl);
    float16x4_t rh = round_f16(xh);
    return vcombine_f16(rl, rh);
}

static inline float16x4_t logaddexp_f16(const float16x4_t& x, const float16x4_t& y)
{
    return vcvt_f16_f32(logaddexp_ps(vcvt_f32_f16(x), vcvt_f32_f16(y)));
}

static inline float16x8_t logaddexpq_f16(const float16x8_t& x, const float16x8_t& y)
{
    float16x4_t xl = vget_low_f16(x);
    float16x4_t xh = vget_high_f16(x);
    float16x4_t yl = vget_low_f16(y);
    float16x4_t yh = vget_high_f16(y);
    float16x4_t rl = logaddexp_f16(xl, yl);
    float16x4_t rh = logaddexp_f16(xh, yh);
    return vcombine_f16(rl, rh);
}

static inline float16x4_t floor_divide_f16(const float16x4_t& x, const float16x4_t& y)
{
    return vcvt_f16_f32(floor_divide_ps(vcvt_f32_f16(x), vcvt_f32_f16(y)));
}

static inline float16x8_t floor_divideq_f16(const float16x8_t& x, const float16x8_t& y)
{
    float16x4_t xl = vget_low_f16(x);
    float16x4_t xh = vget_high_f16(x);
    float16x4_t yl = vget_low_f16(y);
    float16x4_t yh = vget_high_f16(y);
    float16x4_t rl = floor_divide_f16(xl, yl);
    float16x4_t rh = floor_divide_f16(xh, yh);
    return vcombine_f16(rl, rh);
}

static inline float16x4_t remainder_f16(const float16x4_t& x, const float16x4_t& y)
{
    return vcvt_f16_f32(remainder_ps(vcvt_f32_f16(x), vcvt_f32_f16(y)));
}

static inline float16x8_t remainderq_f16(const float16x8_t& x, const float16x8_t& y)
{
    float16x4_t xl = vget_low_f16(x);
    float16x4_t xh = vget_high_f16(x);
    float16x4_t yl = vget_low_f16(y);
    float16x4_t yh = vget_high_f16(y);
    float16x4_t rl = remainder_f16(xl, yl);
    float16x4_t rh = remainder_f16(xh, yh);
    return vcombine_f16(rl, rh);
}

template<typename Op>
static void binary_op_vector_no_broadcast_fp16s(const __fp16* ptr, const __fp16* ptr1, __fp16* outptr, int size)
{
    const Op op;

    int i = 0;
    for (; i + 7 < size; i += 8)
    {
        float16x8_t _p = vld1q_f16(ptr);
        float16x8_t _b = vld1q_f16(ptr1);
        float16x8_t _outp = op(_p, _b);
        vst1q_f16(outptr, _outp);
        ptr += 8;
        ptr1 += 8;
        outptr += 8;
    }
    for (; i + 3 < size; i += 4)
    {
        float16x4_t _p = vld1_f16(ptr);
        float16x4_t _b = vld1_f16(ptr1);
        float16x4_t _outp = op(_p, _b);
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

template<typename Op>
static void binary_op_vector_broadcast_b_fp16s(const __fp16* ptr, const __fp16* ptr1, __fp16* outptr, int size, int elempack)
{
    const Op op;

    const __fp16 b = *ptr1;

    int i = 0;
    float16x4_t _b_128 = (elempack == 4) ? vld1_f16(ptr1) : vdup_n_f16(b);
    float16x8_t _b_256 = (elempack == 8) ? vld1q_f16(ptr1) : vcombine_f16(_b_128, _b_128);
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
        *outptr = op(*ptr, b);
        ptr += 1;
        outptr += 1;
    }
}

template<typename Op>
static void binary_op_vector_broadcast_a_fp16s(const __fp16* ptr, const __fp16* ptr1, __fp16* outptr, int size, int elempack)
{
    const Op op;

    const __fp16 a = *ptr;

    int i = 0;
    float16x4_t _a_128 = (elempack == 4) ? vld1_f16(ptr) : vdup_n_f16(a);
    float16x8_t _a_256 = (elempack == 8) ? vld1q_f16(ptr) : vcombine_f16(_a_128, _a_128);
    for (; i + 7 < size; i += 8)
    {
        float16x8_t _b = vld1q_f16(ptr1);
        float16x8_t _outp = op(_a_256, _b);
        vst1q_f16(outptr, _outp);
        ptr1 += 8;
        outptr += 8;
    }
    for (; i + 3 < size; i += 4)
    {
        float16x4_t _b = vld1_f16(ptr1);
        float16x4_t _outp = op(_a_128, _b);
        vst1_f16(outptr, _outp);
        ptr1 += 4;
        outptr += 4;
    }
    for (; i < size; i++)
    {
        *outptr = op(a, *ptr1);
        ptr1 += 1;
        outptr += 1;
    }
}

template<typename Op>
static void binary_op_vector_broadcast_pb_fp16s(const __fp16* ptr, const __fp16* ptr1, __fp16* outptr, int w, int elempack)
{
    const Op op;

    if (elempack == 8)
    {
        int i = 0;
        for (; i < w; i++)
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
        int i = 0;
        for (; i + 1 < w; i += 2)
        {
            float16x8_t _p = vld1q_f16(ptr);
            float16x4_t _b0 = vdup_n_f16(ptr1[0]);
            float16x4_t _b1 = vdup_n_f16(ptr1[1]);
            float16x8_t _b = vcombine_f16(_b0, _b1);
            float16x8_t _outp = op(_p, _b);
            vst1q_f16(outptr, _outp);
            ptr += 8;
            ptr1 += 2;
            outptr += 8;
        }
        for (; i < w; i++)
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
}

template<typename Op>
static void binary_op_vector_broadcast_pb_b_fp16s(const __fp16* ptr, const __fp16* ptr1, __fp16* outptr, int w, int elempack)
{
    const Op op;

    const int size = w * elempack;

    int i = 0;
    float16x8_t _b = vdupq_n_f16(*ptr1);
    for (; i + 7 < size; i += 8)
    {
        float16x8_t _p = vld1q_f16(ptr);
        float16x8_t _outp = op(_p, _b);
        vst1q_f16(outptr, _outp);
        ptr += 8;
        outptr += 8;
    }
    for (; i + 3 < size; i += 4)
    {
        float16x4_t _p = vld1_f16(ptr);
        float16x4_t _outp = op(_p, vget_low_f16(_b));
        vst1_f16(outptr, _outp);
        ptr += 4;
        outptr += 4;
    }
}

template<typename Op>
static void binary_op_vector_broadcast_pb_a_fp16s(const __fp16* ptr, const __fp16* ptr1, __fp16* outptr, int w, int elempack)
{
    const Op op;

    if (elempack == 8)
    {
        int i = 0;
        float16x8_t _p = vld1q_f16(ptr);
        for (; i < w; i++)
        {
            float16x8_t _b = vdupq_n_f16(*ptr1);
            float16x8_t _outp = op(_p, _b);
            vst1q_f16(outptr, _outp);
            ptr1 += 1;
            outptr += 8;
        }
    }
    if (elempack == 4)
    {
        int i = 0;
        float16x4_t _p0 = vld1_f16(ptr);
        float16x8_t _p = vcombine_f16(_p0, _p0);
        for (; i + 1 < w; i += 2)
        {
            float16x4_t _b0 = vdup_n_f16(ptr1[0]);
            float16x4_t _b1 = vdup_n_f16(ptr1[1]);
            float16x8_t _b = vcombine_f16(_b0, _b1);
            float16x8_t _outp = op(_p, _b);
            vst1q_f16(outptr, _outp);
            ptr1 += 2;
            outptr += 8;
        }
        for (; i < w; i++)
        {
            float16x4_t _b = vdup_n_f16(*ptr1);
            float16x4_t _outp = op(_p0, _b);
            vst1_f16(outptr, _outp);
            ptr1 += 1;
            outptr += 4;
        }
    }
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

namespace BinaryOp_arm_fp16s_functor {

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
MAKE_FUNCTION(binary_op_pow_fp16s, (__fp16)powf(x, y), vcvt_f16_f32(pow_ps(vcvt_f32_f16(x), vcvt_f32_f16(y))), vcombine_f16(vcvt_f16_f32(pow_ps(vcvt_f32_f16(vget_low_f16(x)), vcvt_f32_f16(vget_low_f16(y)))), vcvt_f16_f32(pow_ps(vcvt_f32_f16(vget_high_f16(x)), vcvt_f32_f16(vget_high_f16(y))))))
MAKE_FUNCTION(binary_op_rsub_fp16s, y - x, vsub_f16(y, x), vsubq_f16(y, x))
MAKE_FUNCTION(binary_op_rdiv_fp16s, y / x, vdiv_f16(y, x), vdivq_f16(y, x))
MAKE_FUNCTION(binary_op_rpow_fp16s, (__fp16)powf(y, x), vcvt_f16_f32(pow_ps(vcvt_f32_f16(y), vcvt_f32_f16(x))), vcombine_f16(vcvt_f16_f32(pow_ps(vcvt_f32_f16(vget_low_f16(y)), vcvt_f32_f16(vget_low_f16(x)))), vcvt_f16_f32(pow_ps(vcvt_f32_f16(vget_high_f16(y)), vcvt_f32_f16(vget_high_f16(x))))))
MAKE_FUNCTION(binary_op_atan2_fp16s, (__fp16)atan2f(x, y), vcvt_f16_f32(atan2_ps(vcvt_f32_f16(x), vcvt_f32_f16(y))), vcombine_f16(vcvt_f16_f32(atan2_ps(vcvt_f32_f16(vget_low_f16(x)), vcvt_f32_f16(vget_low_f16(y)))), vcvt_f16_f32(atan2_ps(vcvt_f32_f16(vget_high_f16(x)), vcvt_f32_f16(vget_high_f16(y))))))
MAKE_FUNCTION(binary_op_ratan2_fp16s, (__fp16)atan2f(y, x), vcvt_f16_f32(atan2_ps(vcvt_f32_f16(y), vcvt_f32_f16(x))), vcombine_f16(vcvt_f16_f32(atan2_ps(vcvt_f32_f16(vget_low_f16(y)), vcvt_f32_f16(vget_low_f16(x)))), vcvt_f16_f32(atan2_ps(vcvt_f32_f16(vget_high_f16(y)), vcvt_f32_f16(vget_high_f16(x))))))
MAKE_FUNCTION(binary_op_fmod_fp16s, (__fp16)fmodf((float)x, (float)y), fmod_f16(x, y), fmodq_f16(x, y))
MAKE_FUNCTION(binary_op_rfmod_fp16s, (__fp16)fmodf((float)y, (float)x), fmod_f16(y, x), fmodq_f16(y, x))
MAKE_FUNCTION(binary_op_logaddexp_fp16s, (__fp16)(std::max((float)x, (float)y) + log1pf(expf(std::min((float)x, (float)y) - std::max((float)x, (float)y)))), logaddexp_f16(x, y), logaddexpq_f16(x, y))
MAKE_FUNCTION(binary_op_floor_divide_fp16s, (__fp16)floorf((float)x / (float)y), floor_divide_f16(x, y), floor_divideq_f16(x, y))
MAKE_FUNCTION(binary_op_rfloor_divide_fp16s, (__fp16)floorf((float)y / (float)x), floor_divide_f16(y, x), floor_divideq_f16(y, x))
MAKE_FUNCTION(binary_op_remainder_fp16s, (__fp16)remainderf((float)x, (float)y), remainder_f16(x, y), remainderq_f16(x, y))
MAKE_FUNCTION(binary_op_rremainder_fp16s, (__fp16)remainderf((float)y, (float)x), remainder_f16(y, x), remainderq_f16(y, x))
// *INDENT-ON*
// clang-format on

#undef MAKE_FUNCTION

} // namespace BinaryOp_arm_fp16s_functor

static void binary_op_vector_fp16s(const __fp16* ptr, const __fp16* ptr1, __fp16* outptr, int aw, int bw, int ap, int bp, int op_type)
{
    using namespace BinaryOp_arm_fp16s_functor;

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
    if (op_type == BinaryOp::Operation_FMOD) return binary_op_vector_fp16s<binary_op_fmod_fp16s>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RFMOD) return binary_op_vector_fp16s<binary_op_rfmod_fp16s>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_LOGADDEXP) return binary_op_vector_fp16s<binary_op_logaddexp_fp16s>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_FLOOR_DIVIDE) return binary_op_vector_fp16s<binary_op_floor_divide_fp16s>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RFLOOR_DIVIDE) return binary_op_vector_fp16s<binary_op_rfloor_divide_fp16s>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_REMAINDER) return binary_op_vector_fp16s<binary_op_remainder_fp16s>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RREMAINDER) return binary_op_vector_fp16s<binary_op_rremainder_fp16s>(ptr, ptr1, outptr, aw, bw, ap, bp);

    // should never reach here
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
