// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#define _PADDING_PACKN_RVV(SEW, TSEW, LMUL, T, VT)                                                                                          \
    static void padding_constant_packn_##VT##_rvv(const Mat& src, Mat& dst, int top, int bottom, int left, int right, v##VT##m##LMUL##_t v) \
    {                                                                                                                                       \
        const int packn = csrr_vlenb() / sizeof(T);                                                                                         \
        const size_t vl = vsetvl_e##SEW##m##LMUL(packn);                                                                                    \
                                                                                                                                            \
        const T* ptr = src;                                                                                                                 \
        T* outptr = dst;                                                                                                                    \
                                                                                                                                            \
        /* fill top */                                                                                                                      \
        for (int y = 0; y < top; y++)                                                                                                       \
        {                                                                                                                                   \
            for (int x = 0; x < dst.w; x++)                                                                                                 \
            {                                                                                                                               \
                vse##SEW##_v_##TSEW##m##LMUL(outptr, v, vl);                                                                                \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
        }                                                                                                                                   \
        /* fill center */                                                                                                                   \
        for (int y = 0; y < src.h; y++)                                                                                                     \
        {                                                                                                                                   \
            for (int x = 0; x < left; x++)                                                                                                  \
            {                                                                                                                               \
                vse##SEW##_v_##TSEW##m##LMUL(outptr, v, vl);                                                                                \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            for (int x = 0; x < src.w; x++)                                                                                                 \
            {                                                                                                                               \
                v##VT##m##LMUL##_t _p = vle##SEW##_v_##TSEW##m##LMUL(ptr, vl);                                                              \
                vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                               \
                ptr += packn;                                                                                                               \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            for (int x = 0; x < right; x++)                                                                                                 \
            {                                                                                                                               \
                vse##SEW##_v_##TSEW##m##LMUL(outptr, v, vl);                                                                                \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
        }                                                                                                                                   \
        /* fill bottom */                                                                                                                   \
        for (int y = 0; y < bottom; y++)                                                                                                    \
        {                                                                                                                                   \
            for (int x = 0; x < dst.w; x++)                                                                                                 \
            {                                                                                                                               \
                vse##SEW##_v_##TSEW##m##LMUL(outptr, v, vl);                                                                                \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
        }                                                                                                                                   \
    }                                                                                                                                       \
                                                                                                                                            \
    static void padding_replicate_packn_##VT##_rvv(const Mat& src, Mat& dst, int top, int bottom, int left, int right)                      \
    {                                                                                                                                       \
        const int packn = csrr_vlenb() / sizeof(T);                                                                                         \
        const size_t vl = vsetvl_e##SEW##m##LMUL(packn);                                                                                    \
                                                                                                                                            \
        const T* ptr = src;                                                                                                                 \
        T* outptr = dst;                                                                                                                    \
                                                                                                                                            \
        /* fill top */                                                                                                                      \
        for (int y = 0; y < top; y++)                                                                                                       \
        {                                                                                                                                   \
            const T* ptr0 = ptr;                                                                                                            \
            v##VT##m##LMUL##_t _p = vle##SEW##_v_##TSEW##m##LMUL(ptr0, vl);                                                                 \
            for (int x = 0; x < left; x++)                                                                                                  \
            {                                                                                                                               \
                vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                               \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            for (int x = 0; x < src.w; x++)                                                                                                 \
            {                                                                                                                               \
                _p = vle##SEW##_v_##TSEW##m##LMUL(ptr0, vl);                                                                                \
                vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                               \
                ptr0 += packn;                                                                                                              \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            for (int x = 0; x < right; x++)                                                                                                 \
            {                                                                                                                               \
                vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                               \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
        }                                                                                                                                   \
        /* fill center */                                                                                                                   \
        for (int y = 0; y < src.h; y++)                                                                                                     \
        {                                                                                                                                   \
            v##VT##m##LMUL##_t _p = vle##SEW##_v_##TSEW##m##LMUL(ptr, vl);                                                                  \
            for (int x = 0; x < left; x++)                                                                                                  \
            {                                                                                                                               \
                vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                               \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            for (int x = 0; x < src.w; x++)                                                                                                 \
            {                                                                                                                               \
                _p = vle##SEW##_v_##TSEW##m##LMUL(ptr, vl);                                                                                 \
                vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                               \
                ptr += packn;                                                                                                               \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            for (int x = 0; x < right; x++)                                                                                                 \
            {                                                                                                                               \
                vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                               \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
        }                                                                                                                                   \
        /* fill bottom */                                                                                                                   \
        ptr -= src.w * packn;                                                                                                               \
        for (int y = 0; y < bottom; y++)                                                                                                    \
        {                                                                                                                                   \
            const T* ptr0 = ptr;                                                                                                            \
            v##VT##m##LMUL##_t _p = vle##SEW##_v_##TSEW##m##LMUL(ptr0, vl);                                                                 \
            for (int x = 0; x < left; x++)                                                                                                  \
            {                                                                                                                               \
                vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                               \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            for (int x = 0; x < src.w; x++)                                                                                                 \
            {                                                                                                                               \
                _p = vle##SEW##_v_##TSEW##m##LMUL(ptr0, vl);                                                                                \
                vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                               \
                ptr0 += packn;                                                                                                              \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            for (int x = 0; x < right; x++)                                                                                                 \
            {                                                                                                                               \
                vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                               \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
        }                                                                                                                                   \
    }                                                                                                                                       \
                                                                                                                                            \
    static void padding_reflect_packn_##VT##_rvv(const Mat& src, Mat& dst, int top, int bottom, int left, int right)                        \
    {                                                                                                                                       \
        const int packn = csrr_vlenb() / sizeof(T);                                                                                         \
        const size_t vl = vsetvl_e##SEW##m##LMUL(packn);                                                                                    \
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
                v##VT##m##LMUL##_t _p = vle##SEW##_v_##TSEW##m##LMUL(ptr0 + (left - x) * packn, vl);                                        \
                vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                               \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            for (int x = 0; x < src.w; x++)                                                                                                 \
            {                                                                                                                               \
                v##VT##m##LMUL##_t _p = vle##SEW##_v_##TSEW##m##LMUL(ptr0, vl);                                                             \
                vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                               \
                ptr0 += packn;                                                                                                              \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            for (int x = 0; x < right; x++)                                                                                                 \
            {                                                                                                                               \
                v##VT##m##LMUL##_t _p = vle##SEW##_v_##TSEW##m##LMUL(ptr0 - packn * 2 - x * packn, vl);                                     \
                vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                               \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            ptr -= src.w * packn;                                                                                                           \
        }                                                                                                                                   \
        /* fill center */                                                                                                                   \
        for (int y = 0; y < src.h; y++)                                                                                                     \
        {                                                                                                                                   \
            for (int x = 0; x < left; x++)                                                                                                  \
            {                                                                                                                               \
                v##VT##m##LMUL##_t _p = vle##SEW##_v_##TSEW##m##LMUL(ptr + (left - x) * packn, vl);                                         \
                vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                               \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            for (int x = 0; x < src.w; x++)                                                                                                 \
            {                                                                                                                               \
                v##VT##m##LMUL##_t _p = vle##SEW##_v_##TSEW##m##LMUL(ptr, vl);                                                              \
                vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                               \
                ptr += packn;                                                                                                               \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            for (int x = 0; x < right; x++)                                                                                                 \
            {                                                                                                                               \
                v##VT##m##LMUL##_t _p = vle##SEW##_v_##TSEW##m##LMUL(ptr - packn * 2 - x * packn, vl);                                      \
                vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                               \
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
                v##VT##m##LMUL##_t _p = vle##SEW##_v_##TSEW##m##LMUL(ptr0 + (left - x) * packn, vl);                                        \
                vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                               \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            for (int x = 0; x < src.w; x++)                                                                                                 \
            {                                                                                                                               \
                v##VT##m##LMUL##_t _p = vle##SEW##_v_##TSEW##m##LMUL(ptr0, vl);                                                             \
                vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                               \
                ptr0 += packn;                                                                                                              \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            for (int x = 0; x < right; x++)                                                                                                 \
            {                                                                                                                               \
                v##VT##m##LMUL##_t _p = vle##SEW##_v_##TSEW##m##LMUL(ptr0 - packn * 2 - x * packn, vl);                                     \
                vse##SEW##_v_##TSEW##m##LMUL(outptr, _p, vl);                                                                               \
                outptr += packn;                                                                                                            \
            }                                                                                                                               \
            ptr -= src.w * packn;                                                                                                           \
        }                                                                                                                                   \
    }

_PADDING_PACKN_RVV(32, f32, 1, float, float32)
_PADDING_PACKN_RVV(16, u16, 1, unsigned short, uint16)
_PADDING_PACKN_RVV(8, i8, 1, signed char, int8)
