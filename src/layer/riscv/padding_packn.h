// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#define _PADDING_PACKN_RVV(SEW, TSEW, LMUL, T, VT)                                                                                          \
    static void padding_constant_packn_##VT##_rvv(const Mat& src, Mat& dst, int top, int bottom, int left, int right, v##VT##m##LMUL##_t v) \
    {                                                                                                                                       \
        const int packn = csrr_vlenb() / sizeof(T);                                                                                         \
        const size_t vl = __riscv_vsetvl_e##SEW##m##LMUL(packn);                                                                            \
                                                                                                                                            \
        const T* ptr = src;                                                                                                                 \
        T* outptr = dst;                                                                                                                    \
                                                                                                                                            \
        /* fill top */                                                                                                                      \
        for (int y = 0; y < top; y++)                                                                                                       \
        {                                                                                                                                   \
            for (int x = 0; x < dst.w; x++)                                                                                                 \
            {                                                                                                                               \
                __riscv_vse##SEW##_v_##TSEW##m##LMUL(outptr, v, vl);                                                                        \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
        }                                                                                                                                   \
        /* fill center */                                                                                                                   \
        for (int y = 0; y < src.h; y++)                                                                                                     \
        {                                                                                                                                   \
            for (int x = 0; x < left; x++)                                                                                                  \
            {                                                                                                                               \
                __riscv_vse##SEW##_v_##TSEW##m##LMUL(outptr, v, vl);                                                                        \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            for (int x = 0; x < src.w; x++)                                                                                                 \
            {                                                                                                                               \
                v##VT##m##LMUL##_t _p = __riscv_vle##SEW##_v_##TSEW##m##LMUL(ptr, vl);                                                      \
                __riscv_vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                       \
                ptr += packn;                                                                                                               \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            for (int x = 0; x < right; x++)                                                                                                 \
            {                                                                                                                               \
                __riscv_vse##SEW##_v_##TSEW##m##LMUL(outptr, v, vl);                                                                        \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
        }                                                                                                                                   \
        /* fill bottom */                                                                                                                   \
        for (int y = 0; y < bottom; y++)                                                                                                    \
        {                                                                                                                                   \
            for (int x = 0; x < dst.w; x++)                                                                                                 \
            {                                                                                                                               \
                __riscv_vse##SEW##_v_##TSEW##m##LMUL(outptr, v, vl);                                                                        \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
        }                                                                                                                                   \
    }                                                                                                                                       \
                                                                                                                                            \
    static void padding_replicate_packn_##VT##_rvv(const Mat& src, Mat& dst, int top, int bottom, int left, int right)                      \
    {                                                                                                                                       \
        const int packn = csrr_vlenb() / sizeof(T);                                                                                         \
        const size_t vl = __riscv_vsetvl_e##SEW##m##LMUL(packn);                                                                            \
                                                                                                                                            \
        const T* ptr = src;                                                                                                                 \
        T* outptr = dst;                                                                                                                    \
                                                                                                                                            \
        /* fill top */                                                                                                                      \
        for (int y = 0; y < top; y++)                                                                                                       \
        {                                                                                                                                   \
            const T* ptr0 = ptr;                                                                                                            \
            v##VT##m##LMUL##_t _p = __riscv_vle##SEW##_v_##TSEW##m##LMUL(ptr0, vl);                                                         \
            for (int x = 0; x < left; x++)                                                                                                  \
            {                                                                                                                               \
                __riscv_vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                       \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            for (int x = 0; x < src.w; x++)                                                                                                 \
            {                                                                                                                               \
                _p = __riscv_vle##SEW##_v_##TSEW##m##LMUL(ptr0, vl);                                                                        \
                __riscv_vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                       \
                ptr0 += packn;                                                                                                              \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            for (int x = 0; x < right; x++)                                                                                                 \
            {                                                                                                                               \
                __riscv_vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                       \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
        }                                                                                                                                   \
        /* fill center */                                                                                                                   \
        for (int y = 0; y < src.h; y++)                                                                                                     \
        {                                                                                                                                   \
            v##VT##m##LMUL##_t _p = __riscv_vle##SEW##_v_##TSEW##m##LMUL(ptr, vl);                                                          \
            for (int x = 0; x < left; x++)                                                                                                  \
            {                                                                                                                               \
                __riscv_vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                       \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            for (int x = 0; x < src.w; x++)                                                                                                 \
            {                                                                                                                               \
                _p = __riscv_vle##SEW##_v_##TSEW##m##LMUL(ptr, vl);                                                                         \
                __riscv_vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                       \
                ptr += packn;                                                                                                               \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            for (int x = 0; x < right; x++)                                                                                                 \
            {                                                                                                                               \
                __riscv_vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                       \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
        }                                                                                                                                   \
        /* fill bottom */                                                                                                                   \
        ptr -= src.w * packn;                                                                                                               \
        for (int y = 0; y < bottom; y++)                                                                                                    \
        {                                                                                                                                   \
            const T* ptr0 = ptr;                                                                                                            \
            v##VT##m##LMUL##_t _p = __riscv_vle##SEW##_v_##TSEW##m##LMUL(ptr0, vl);                                                         \
            for (int x = 0; x < left; x++)                                                                                                  \
            {                                                                                                                               \
                __riscv_vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                       \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            for (int x = 0; x < src.w; x++)                                                                                                 \
            {                                                                                                                               \
                _p = __riscv_vle##SEW##_v_##TSEW##m##LMUL(ptr0, vl);                                                                        \
                __riscv_vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                       \
                ptr0 += packn;                                                                                                              \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            for (int x = 0; x < right; x++)                                                                                                 \
            {                                                                                                                               \
                __riscv_vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                       \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
        }                                                                                                                                   \
    }                                                                                                                                       \
                                                                                                                                            \
    static void padding_reflect_packn_##VT##_rvv(const Mat& src, Mat& dst, int top, int bottom, int left, int right)                        \
    {                                                                                                                                       \
        const int packn = csrr_vlenb() / sizeof(T);                                                                                         \
        const size_t vl = __riscv_vsetvl_e##SEW##m##LMUL(packn);                                                                            \
                                                                                                                                            \
        const T* ptr = src;                                                                                                                 \
        T* outptr = dst;                                                                                                                    \
                                                                                                                                            \
        /* fill top */                                                                                                                      \
        ptr += top * src.w * packn;                                                                                                         \
        for (int y = 0; y < top; y++)                                                                                                       \
        {                                                                                                                                   \
            const T* ptr0 = ptr;                                                                                                            \
            for (int x = 0; x < left; x++)                                                                                                  \
            {                                                                                                                               \
                v##VT##m##LMUL##_t _p = __riscv_vle##SEW##_v_##TSEW##m##LMUL(ptr0 + (left - x) * packn, vl);                                \
                __riscv_vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                       \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            for (int x = 0; x < src.w; x++)                                                                                                 \
            {                                                                                                                               \
                v##VT##m##LMUL##_t _p = __riscv_vle##SEW##_v_##TSEW##m##LMUL(ptr0, vl);                                                     \
                __riscv_vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                       \
                ptr0 += packn;                                                                                                              \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            for (int x = 0; x < right; x++)                                                                                                 \
            {                                                                                                                               \
                v##VT##m##LMUL##_t _p = __riscv_vle##SEW##_v_##TSEW##m##LMUL(ptr0 - packn * 2 - x * packn, vl);                             \
                __riscv_vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                       \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            ptr -= src.w * packn;                                                                                                           \
        }                                                                                                                                   \
        /* fill center */                                                                                                                   \
        for (int y = 0; y < src.h; y++)                                                                                                     \
        {                                                                                                                                   \
            for (int x = 0; x < left; x++)                                                                                                  \
            {                                                                                                                               \
                v##VT##m##LMUL##_t _p = __riscv_vle##SEW##_v_##TSEW##m##LMUL(ptr + (left - x) * packn, vl);                                 \
                __riscv_vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                       \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            for (int x = 0; x < src.w; x++)                                                                                                 \
            {                                                                                                                               \
                v##VT##m##LMUL##_t _p = __riscv_vle##SEW##_v_##TSEW##m##LMUL(ptr, vl);                                                      \
                __riscv_vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                       \
                ptr += packn;                                                                                                               \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            for (int x = 0; x < right; x++)                                                                                                 \
            {                                                                                                                               \
                v##VT##m##LMUL##_t _p = __riscv_vle##SEW##_v_##TSEW##m##LMUL(ptr - packn * 2 - x * packn, vl);                              \
                __riscv_vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                       \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
        }                                                                                                                                   \
        /* fill bottom */                                                                                                                   \
        ptr -= 2 * src.w * packn;                                                                                                           \
        for (int y = 0; y < bottom; y++)                                                                                                    \
        {                                                                                                                                   \
            const T* ptr0 = ptr;                                                                                                            \
            for (int x = 0; x < left; x++)                                                                                                  \
            {                                                                                                                               \
                v##VT##m##LMUL##_t _p = __riscv_vle##SEW##_v_##TSEW##m##LMUL(ptr0 + (left - x) * packn, vl);                                \
                __riscv_vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                       \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            for (int x = 0; x < src.w; x++)                                                                                                 \
            {                                                                                                                               \
                v##VT##m##LMUL##_t _p = __riscv_vle##SEW##_v_##TSEW##m##LMUL(ptr0, vl);                                                     \
                __riscv_vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                       \
                ptr0 += packn;                                                                                                              \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            for (int x = 0; x < right; x++)                                                                                                 \
            {                                                                                                                               \
                v##VT##m##LMUL##_t _p = __riscv_vle##SEW##_v_##TSEW##m##LMUL(ptr0 - packn * 2 - x * packn, vl);                             \
                __riscv_vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                       \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            ptr -= src.w * packn;                                                                                                           \
        }                                                                                                                                   \
    }

_PADDING_PACKN_RVV(32, f32, 1, float, float32)
_PADDING_PACKN_RVV(16, u16, 1, unsigned short, uint16)
_PADDING_PACKN_RVV(8, i8, 1, signed char, int8)
