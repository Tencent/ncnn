// Copyright 2021 Xavier Hsinyuan <me@lstlx.com>
// SPDX-License-Identifier: BSD-3-Clause

#if __riscv_vector
#define MAKE_FUNCTION(NAME, IMPL, IMPLVV, IMPLVS, IMPLSV)                                                             \
    struct NAME                                                                                                       \
    {                                                                                                                 \
        NCNN_FORCEINLINE float operator()(const float& x, const float& y) const                                       \
        {                                                                                                             \
            return IMPL;                                                                                              \
        }                                                                                                             \
        NCNN_FORCEINLINE vfloat32m8_t operator()(const vfloat32m8_t& x, const vfloat32m8_t& y, const size_t vl) const \
        {                                                                                                             \
            return IMPLVV;                                                                                            \
        }                                                                                                             \
        NCNN_FORCEINLINE vfloat32m8_t operator()(const vfloat32m8_t& x, const float& y, const size_t vl) const        \
        {                                                                                                             \
            return IMPLVS;                                                                                            \
        }                                                                                                             \
        NCNN_FORCEINLINE vfloat32m8_t operator()(const float& x, const vfloat32m8_t& y, const size_t vl) const        \
        {                                                                                                             \
            return IMPLSV;                                                                                            \
        }                                                                                                             \
    };
#else
#define MAKE_FUNCTION(NAME, IMPL, IMPLVV, IMPLVS, IMPLSV)                       \
    struct NAME                                                                 \
    {                                                                           \
        NCNN_FORCEINLINE float operator()(const float& x, const float& y) const \
        {                                                                       \
            return IMPL;                                                        \
        }                                                                       \
    };
#endif

// clang-format off
// *INDENT-OFF*
MAKE_FUNCTION(binary_op_add, x + y, __riscv_vfadd_vv_f32m8(x, y, vl), __riscv_vfadd_vf_f32m8(x, y, vl), __riscv_vfadd_vf_f32m8(y, x, vl))
MAKE_FUNCTION(binary_op_sub, x - y, __riscv_vfsub_vv_f32m8(x, y, vl), __riscv_vfsub_vf_f32m8(x, y, vl), __riscv_vfrsub_vf_f32m8(y, x, vl))
MAKE_FUNCTION(binary_op_mul, x * y, __riscv_vfmul_vv_f32m8(x, y, vl), __riscv_vfmul_vf_f32m8(x, y, vl), __riscv_vfmul_vf_f32m8(y, x, vl))
MAKE_FUNCTION(binary_op_div, x / y, __riscv_vfdiv_vv_f32m8(x, y, vl), __riscv_vfdiv_vf_f32m8(x, y, vl), __riscv_vfrdiv_vf_f32m8(y, x, vl))
MAKE_FUNCTION(binary_op_max, std::max(x, y), __riscv_vfmax_vv_f32m8(x, y, vl), __riscv_vfmax_vf_f32m8(x, y, vl), __riscv_vfmax_vf_f32m8(y, x, vl))
MAKE_FUNCTION(binary_op_min, std::min(x, y), __riscv_vfmin_vv_f32m8(x, y, vl), __riscv_vfmin_vf_f32m8(x, y, vl), __riscv_vfmin_vf_f32m8(y, x, vl))
MAKE_FUNCTION(binary_op_pow, (float)powf(x, y), pow_ps(x, y, vl), pow_ps(x, __riscv_vfmv_v_f_f32m8(y, vl), vl), pow_ps(__riscv_vfmv_v_f_f32m8(x, vl), y, vl))
MAKE_FUNCTION(binary_op_rsub, y - x, __riscv_vfsub_vv_f32m8(y, x, vl), __riscv_vfrsub_vf_f32m8(x, y, vl), __riscv_vfsub_vf_f32m8(y, x, vl))
MAKE_FUNCTION(binary_op_rdiv, y / x, __riscv_vfdiv_vv_f32m8(y, x, vl), __riscv_vfrdiv_vf_f32m8(x, y, vl), __riscv_vfdiv_vf_f32m8(y, x, vl))
MAKE_FUNCTION(binary_op_rpow, (float)powf(y, x), pow_ps(y, x, vl), pow_ps(__riscv_vfmv_v_f_f32m8(y, vl), x, vl), pow_ps(y, __riscv_vfmv_v_f_f32m8(x, vl), vl))
MAKE_FUNCTION(binary_op_atan2, (float)atan2f(x, y), atan2_ps(x, y, vl), atan2_ps(x, __riscv_vfmv_v_f_f32m8(y, vl), vl), atan2_ps(__riscv_vfmv_v_f_f32m8(x, vl), y, vl))
MAKE_FUNCTION(binary_op_ratan2, (float)atan2f(y, x), atan2_ps(y, x, vl), atan2_ps(__riscv_vfmv_v_f_f32m8(y, vl), x, vl), atan2_ps(y, __riscv_vfmv_v_f_f32m8(x, vl), vl))
MAKE_FUNCTION(binary_op_fmod, (float)fmodf(x, y), fmod_ps(x, y, vl), fmod_ps(x, __riscv_vfmv_v_f_f32m8(y, vl), vl), fmod_ps(__riscv_vfmv_v_f_f32m8(x, vl), y, vl))
MAKE_FUNCTION(binary_op_rfmod, (float)fmodf(y, x), fmod_ps(y, x, vl), fmod_ps(__riscv_vfmv_v_f_f32m8(y, vl), x, vl), fmod_ps(y, __riscv_vfmv_v_f_f32m8(x, vl), vl))
MAKE_FUNCTION(binary_op_logaddexp, (float)(std::max(x, y) + log1pf(expf(std::min(x, y) - std::max(x, y)))), logaddexp_ps(x, y, vl), logaddexp_ps(x, __riscv_vfmv_v_f_f32m8(y, vl), vl), logaddexp_ps(__riscv_vfmv_v_f_f32m8(x, vl), y, vl))
MAKE_FUNCTION(binary_op_floor_divide, (float)floorf(x / y), floor_divide_ps(x, y, vl), floor_divide_ps(x, __riscv_vfmv_v_f_f32m8(y, vl), vl), floor_divide_ps(__riscv_vfmv_v_f_f32m8(x, vl), y, vl))
MAKE_FUNCTION(binary_op_rfloor_divide, (float)floorf(y / x), floor_divide_ps(y, x, vl), floor_divide_ps(__riscv_vfmv_v_f_f32m8(y, vl), x, vl), floor_divide_ps(y, __riscv_vfmv_v_f_f32m8(x, vl), vl))
MAKE_FUNCTION(binary_op_remainder, (float)remainderf(x, y), remainder_ps(x, y, vl), remainder_ps(x, __riscv_vfmv_v_f_f32m8(y, vl), vl), remainder_ps(__riscv_vfmv_v_f_f32m8(x, vl), y, vl))
MAKE_FUNCTION(binary_op_rremainder, (float)remainderf(y, x), remainder_ps(y, x, vl), remainder_ps(__riscv_vfmv_v_f_f32m8(y, vl), x, vl), remainder_ps(y, __riscv_vfmv_v_f_f32m8(x, vl), vl))
// *INDENT-ON*
// clang-format on

#undef MAKE_FUNCTION
