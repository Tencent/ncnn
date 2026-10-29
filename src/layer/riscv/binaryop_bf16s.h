// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
void binaryop_bf16s_zvfbfmin(const Mat& a, const Mat& b, Mat& c, int op_type, const Option& opt);
void binaryop_scalar_inplace_bf16s_zvfbfmin(Mat& a, unsigned short b, int op_type, const Option& opt);
#endif

namespace BinaryOp_riscv_bf16s_functor {

#if __riscv_vector
#define MAKE_FUNCTION(NAME, IMPL, IMPLVV, IMPLVS, IMPLSV)                                            \
    struct NAME                                                                                      \
    {                                                                                                \
        float operator()(const float& x, const float& y) const                                       \
        {                                                                                            \
            return IMPL;                                                                             \
        }                                                                                            \
        vfloat32m4_t operator()(const vfloat32m4_t& x, const vfloat32m4_t& y, const size_t vl) const \
        {                                                                                            \
            return IMPLVV;                                                                           \
        }                                                                                            \
        vfloat32m4_t operator()(const vfloat32m4_t& x, const float& y, const size_t vl) const        \
        {                                                                                            \
            return IMPLVS;                                                                           \
        }                                                                                            \
        vfloat32m4_t operator()(const float& x, const vfloat32m4_t& y, const size_t vl) const        \
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
MAKE_FUNCTION(binary_op_add, x + y, __riscv_vfadd_vv_f32m4(x, y, vl), __riscv_vfadd_vf_f32m4(x, y, vl), __riscv_vfadd_vf_f32m4(y, x, vl))
MAKE_FUNCTION(binary_op_sub, x - y, __riscv_vfsub_vv_f32m4(x, y, vl), __riscv_vfsub_vf_f32m4(x, y, vl), __riscv_vfrsub_vf_f32m4(y, x, vl))
MAKE_FUNCTION(binary_op_mul, x * y, __riscv_vfmul_vv_f32m4(x, y, vl), __riscv_vfmul_vf_f32m4(x, y, vl), __riscv_vfmul_vf_f32m4(y, x, vl))
MAKE_FUNCTION(binary_op_div, x / y, __riscv_vfdiv_vv_f32m4(x, y, vl), __riscv_vfdiv_vf_f32m4(x, y, vl), __riscv_vfrdiv_vf_f32m4(y, x, vl))
MAKE_FUNCTION(binary_op_max, std::max(x, y), __riscv_vfmax_vv_f32m4(x, y, vl), __riscv_vfmax_vf_f32m4(x, y, vl), __riscv_vfmax_vf_f32m4(y, x, vl))
MAKE_FUNCTION(binary_op_min, std::min(x, y), __riscv_vfmin_vv_f32m4(x, y, vl), __riscv_vfmin_vf_f32m4(x, y, vl), __riscv_vfmin_vf_f32m4(y, x, vl))
MAKE_FUNCTION(binary_op_pow, (float)powf(x, y), pow_ps(x, y, vl), pow_ps(x, __riscv_vfmv_v_f_f32m4(y, vl), vl), pow_ps(__riscv_vfmv_v_f_f32m4(x, vl), y, vl))
MAKE_FUNCTION(binary_op_rsub, y - x, __riscv_vfsub_vv_f32m4(y, x, vl), __riscv_vfrsub_vf_f32m4(x, y, vl), __riscv_vfsub_vf_f32m4(y, x, vl))
MAKE_FUNCTION(binary_op_rdiv, y / x, __riscv_vfdiv_vv_f32m4(y, x, vl), __riscv_vfrdiv_vf_f32m4(x, y, vl), __riscv_vfdiv_vf_f32m4(y, x, vl))
MAKE_FUNCTION(binary_op_rpow, (float)powf(y, x), pow_ps(y, x, vl), pow_ps(__riscv_vfmv_v_f_f32m4(y, vl), x, vl), pow_ps(y, __riscv_vfmv_v_f_f32m4(x, vl), vl))
MAKE_FUNCTION(binary_op_atan2, (float)atan2f(x, y), atan2_ps(x, y, vl), atan2_ps(x, __riscv_vfmv_v_f_f32m4(y, vl), vl), atan2_ps(__riscv_vfmv_v_f_f32m4(x, vl), y, vl))
MAKE_FUNCTION(binary_op_ratan2, (float)atan2f(y, x), atan2_ps(y, x, vl), atan2_ps(__riscv_vfmv_v_f_f32m4(y, vl), x, vl), atan2_ps(y, __riscv_vfmv_v_f_f32m4(x, vl), vl))
MAKE_FUNCTION(binary_op_fmod, (float)fmodf(x, y), fmod_ps(x, y, vl), fmod_ps(x, __riscv_vfmv_v_f_f32m4(y, vl), vl), fmod_ps(__riscv_vfmv_v_f_f32m4(x, vl), y, vl))
MAKE_FUNCTION(binary_op_rfmod, (float)fmodf(y, x), fmod_ps(y, x, vl), fmod_ps(__riscv_vfmv_v_f_f32m4(y, vl), x, vl), fmod_ps(y, __riscv_vfmv_v_f_f32m4(x, vl), vl))
MAKE_FUNCTION(binary_op_logaddexp, (float)(std::max(x, y) + log1pf(expf(std::min(x, y) - std::max(x, y)))), logaddexp_ps(x, y, vl), logaddexp_ps(x, __riscv_vfmv_v_f_f32m4(y, vl), vl), logaddexp_ps(__riscv_vfmv_v_f_f32m4(x, vl), y, vl))
MAKE_FUNCTION(binary_op_floor_divide, (float)floorf(x / y), floor_divide_ps(x, y, vl), floor_divide_ps(x, __riscv_vfmv_v_f_f32m4(y, vl), vl), floor_divide_ps(__riscv_vfmv_v_f_f32m4(x, vl), y, vl))
MAKE_FUNCTION(binary_op_rfloor_divide, (float)floorf(y / x), floor_divide_ps(y, x, vl), floor_divide_ps(__riscv_vfmv_v_f_f32m4(y, vl), x, vl), floor_divide_ps(y, __riscv_vfmv_v_f_f32m4(x, vl), vl))
MAKE_FUNCTION(binary_op_remainder, (float)remainderf(x, y), remainder_ps(x, y, vl), remainder_ps(x, __riscv_vfmv_v_f_f32m4(y, vl), vl), remainder_ps(__riscv_vfmv_v_f_f32m4(x, vl), y, vl))
MAKE_FUNCTION(binary_op_rremainder, (float)remainderf(y, x), remainder_ps(y, x, vl), remainder_ps(__riscv_vfmv_v_f_f32m4(y, vl), x, vl), remainder_ps(y, __riscv_vfmv_v_f_f32m4(x, vl), vl))
// *INDENT-ON*
// clang-format on

#undef MAKE_FUNCTION

} // namespace BinaryOp_riscv_bf16s_functor

template<typename Op>
static void binary_op_vector_no_broadcast_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int size)
{
    const Op op;

#if __riscv_vector
    int n = size;
    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e16m2(n);
        vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
        vfloat32m4_t _p1 = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr1, vl), vl);
        vfloat32m4_t _outp = op(_p, _p1, vl);
        __riscv_vse16_v_u16m2(outptr, float2bfloat_rvv(_outp, vl), vl);
        n -= vl;
        ptr += vl;
        ptr1 += vl;
        outptr += vl;
    }
#else
    for (int i = 0; i < size; i++)
    {
        *outptr = float32_to_bfloat16(op(bfloat16_to_float32(*ptr), bfloat16_to_float32(*ptr1)));
        ptr += 1;
        ptr1 += 1;
        outptr += 1;
    }
#endif
}

template<typename Op>
static void binary_op_vector_broadcast_b_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int size, int elempack)
{
    const Op op;

#if __riscv_vector
    if (elempack == 1)
    {
        // broadcast single b
        const float b = bfloat16_to_float32(*ptr1);

        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m2(n);
            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
            vfloat32m4_t _outp = op(_p, b, vl);
            __riscv_vse16_v_u16m2(outptr, float2bfloat_rvv(_outp, vl), vl);
            n -= vl;
            ptr += vl;
            outptr += vl;
        }
        return;
    }

    // b holds one packed vector, repeated for every w position
    size_t vl = __riscv_vsetvl_e16m2(elempack);
    vfloat32m4_t _bx = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr1, vl), vl);

    int n = size;
    while (n > 0)
    {
        vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
        vfloat32m4_t _outp = op(_p, _bx, vl);
        __riscv_vse16_v_u16m2(outptr, float2bfloat_rvv(_outp, vl), vl);
        n -= (int)vl;
        ptr += vl;
        outptr += vl;
    }
#else
    const float b = bfloat16_to_float32(*ptr1);

    for (int i = 0; i < size; i++)
    {
        *outptr = float32_to_bfloat16(op(bfloat16_to_float32(*ptr), b));
        ptr += 1;
        outptr += 1;
    }
#endif
}

template<typename Op>
static void binary_op_vector_broadcast_a_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int size, int elempack)
{
    const Op op;

#if __riscv_vector
    if (elempack == 1)
    {
        // broadcast single a
        const float a = bfloat16_to_float32(*ptr);

        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m2(n);
            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr1, vl), vl);
            vfloat32m4_t _outp = op(a, _p, vl);
            __riscv_vse16_v_u16m2(outptr, float2bfloat_rvv(_outp, vl), vl);
            n -= vl;
            ptr1 += vl;
            outptr += vl;
        }
        return;
    }

    // a holds one packed vector, repeated for every w position
    size_t vl = __riscv_vsetvl_e16m2(elempack);
    vfloat32m4_t _ax = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);

    int n = size;
    while (n > 0)
    {
        vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr1, vl), vl);
        vfloat32m4_t _outp = op(_ax, _p, vl);
        __riscv_vse16_v_u16m2(outptr, float2bfloat_rvv(_outp, vl), vl);
        n -= (int)vl;
        ptr1 += vl;
        outptr += vl;
    }
#else
    const float a = bfloat16_to_float32(*ptr);

    for (int i = 0; i < size; i++)
    {
        *outptr = float32_to_bfloat16(op(a, bfloat16_to_float32(*ptr1)));
        ptr1 += 1;
        outptr += 1;
    }
#endif
}

template<typename Op>
static void binary_op_vector_broadcast_pb_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int w, int elempack)
{
    const Op op;

#if __riscv_vector
    size_t vl = __riscv_vsetvl_e16m2(elempack);
    int i = 0;
    for (; i < w; i++)
    {
        vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
        vfloat32m4_t _outp = op(_p, bfloat16_to_float32(*ptr1), vl);
        __riscv_vse16_v_u16m2(outptr, float2bfloat_rvv(_outp, vl), vl);
        ptr += vl;
        ptr1 += 1;
        outptr += vl;
    }
#else
    for (int i = 0; i < w; i++)
    {
        for (int lane = 0; lane < elempack; lane++)
        {
            outptr[lane] = float32_to_bfloat16(op(bfloat16_to_float32(ptr[lane]), bfloat16_to_float32(*ptr1)));
        }
        ptr += elempack;
        ptr1 += 1;
        outptr += elempack;
    }
#endif
}

template<typename Op>
static void binary_op_vector_broadcast_pb_b_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int w, int elempack)
{
    const Op op;

#if __riscv_vector
    int n = w * elempack;
    const float b = bfloat16_to_float32(*ptr1);

    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e16m2(n);
        vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
        vfloat32m4_t _outp = op(_p, b, vl);
        __riscv_vse16_v_u16m2(outptr, float2bfloat_rvv(_outp, vl), vl);
        n -= (int)vl;
        ptr += vl;
        outptr += vl;
    }
#else
    const float b = bfloat16_to_float32(*ptr1);
    const int size = w * elempack;

    for (int i = 0; i < size; i++)
    {
        *outptr = float32_to_bfloat16(op(bfloat16_to_float32(*ptr), b));
        ptr += 1;
        outptr += 1;
    }
#endif
}

template<typename Op>
static void binary_op_vector_broadcast_pb_a_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int w, int elempack)
{
    const Op op;

#if __riscv_vector
    size_t vl = __riscv_vsetvl_e16m2(elempack);
    vfloat32m4_t _ax = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);

    for (int i = 0; i < w; i++)
    {
        vfloat32m4_t _outp = op(_ax, bfloat16_to_float32(*ptr1), vl);
        __riscv_vse16_v_u16m2(outptr, float2bfloat_rvv(_outp, vl), vl);
        ptr1 += 1;
        outptr += vl;
    }
#else
    for (int i = 0; i < w; i++)
    {
        for (int lane = 0; lane < elempack; lane++)
        {
            outptr[lane] = float32_to_bfloat16(op(bfloat16_to_float32(ptr[lane]), bfloat16_to_float32(*ptr1)));
        }
        ptr1 += 1;
        outptr += elempack;
    }
#endif
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
    using namespace BinaryOp_riscv_bf16s_functor;

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
    if (op_type == BinaryOp::Operation_FMOD) return binary_op_vector_bf16s<binary_op_fmod>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RFMOD) return binary_op_vector_bf16s<binary_op_rfmod>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_LOGADDEXP) return binary_op_vector_bf16s<binary_op_logaddexp>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_FLOOR_DIVIDE) return binary_op_vector_bf16s<binary_op_floor_divide>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RFLOOR_DIVIDE) return binary_op_vector_bf16s<binary_op_rfloor_divide>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_REMAINDER) return binary_op_vector_bf16s<binary_op_remainder>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RREMAINDER) return binary_op_vector_bf16s<binary_op_rremainder>(ptr, ptr1, outptr, aw, bw, ap, bp);

    // should never reach here
}

static void binary_op_scalar_bf16s(const Mat& a, unsigned short b, Mat& c, int op_type, const Option& opt)
{
    const int channels = a.c;
    const int size = a.w * a.h * a.d * a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const unsigned short* ptr = a.channel(q);
        unsigned short* outptr = c.channel(q);

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        unsigned long old_fxcr = csrr_fxcr();
        old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
        binary_op_vector_bf16s(ptr, &b, outptr, size, 1, 1, 1, op_type);
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
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

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        unsigned long old_fxcr = csrr_fxcr();
        old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
        binary_op_vector_bf16s(ptr, ptr1, outptr, size, size, 1, 1, op_type);
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
    }
}

static void binary_op_broadcast_bf16s(const Mat& a, const Mat& b, Mat& c, int op_type, const Option& opt)
{
    if (b.w * b.h * b.d * b.c * b.elempack == 1)
    {
        return binary_op_scalar_bf16s(a, ((const unsigned short*)b)[0], c, op_type, opt);
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

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
            unsigned long old_fxcr = csrr_fxcr();
            old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
            binary_op_vector_bf16s(ptr, ptr1, outptr, a.w, b.w, a.elempack, b.elempack, op_type);
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
            csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
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

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
            unsigned long old_fxcr = csrr_fxcr();
            old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
            if (b.d * b.h * b.w == 1)
            {
                const unsigned short* ptr = a.channel(q0);
                const unsigned short* ptr1 = b.channel(q1);
                unsigned short* outptr = c.channel(q);

                binary_op_vector_bf16s(ptr, ptr1, outptr, a.w * a.h * a.d, 1, a.elempack, b.elempack, op_type);
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
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
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
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
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
            csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
        }
    }
}

static void binary_op_scalar_inplace_bf16s(Mat& a, unsigned short b, int op_type, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (cpu_support_riscv_zvfbfmin())
    {
        binaryop_scalar_inplace_bf16s_zvfbfmin(a, b, op_type, opt);
        return;
    }
#endif

    const int channels = a.c;
    const int size = a.w * a.h * a.d * a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = a.channel(q);

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        unsigned long old_fxcr = csrr_fxcr();
        old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
        binary_op_vector_bf16s(ptr, &b, ptr, size, 1, 1, 1, op_type);
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
    }
}

static void binaryop_bf16s(const Mat& a, const Mat& b, Mat& c, int op_type, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (cpu_support_riscv_zvfbfmin())
    {
        binaryop_bf16s_zvfbfmin(a, b, c, op_type, opt);
        return;
    }
#endif

    binary_op_broadcast_bf16s(a, b, c, op_type, opt);
}
