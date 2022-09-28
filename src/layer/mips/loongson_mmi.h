// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright(C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License(the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef _MIPS_LOONGSON_MMI_H
#define _MIPS_LOONGSON_MMI_H

#if __mips_loongson_mmi

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

typedef uint8_t uint8x8_t __attribute__((vector_size(8)));
typedef uint16_t uint16x4_t __attribute__((vector_size(8)));
typedef uint32_t uint32x2_t __attribute__((vector_size(8)));

typedef int8_t int8x8_t __attribute__((vector_size(8)));
typedef int16_t int16x4_t __attribute__((vector_size(8)));
typedef int32_t int32x2_t __attribute__((vector_size(8)));

static inline __attribute__((__always_inline__)) uint32x2_t __mmi_pzerow_u()
{
    return (uint32x2_t) {
        0, 0
    };
}
static inline __attribute__((__always_inline__)) uint16x4_t __mmi_pzeroh_u()
{
    return (uint16x4_t) {
        0, 0, 0, 0
    };
}
static inline __attribute__((__always_inline__)) uint8x8_t __mmi_pzerob_u()
{
    return (uint8x8_t) {
        0, 0, 0, 0, 0, 0, 0, 0
    };
}

static inline __attribute__((__always_inline__)) int32x2_t __mmi_pzerow_s()
{
    return (int32x2_t) {
        0, 0
    };
}
static inline __attribute__((__always_inline__)) int16x4_t __mmi_pzeroh_s()
{
    return (int16x4_t) {
        0, 0, 0, 0
    };
}
static inline __attribute__((__always_inline__)) int8x8_t __mmi_pzerob_s()
{
    return (int8x8_t) {
        0, 0, 0, 0, 0, 0, 0, 0
    };
}

static inline __attribute__((__always_inline__)) uint32x2_t __mmi_pfillw_u(uint32_t v)
{
    return (uint32x2_t) {
        v, v
    };
}
static inline __attribute__((__always_inline__)) uint16x4_t __mmi_pfillh_u(uint16_t v)
{
    return (uint16x4_t) {
        v, v, v, v
    };
}
static inline __attribute__((__always_inline__)) uint8x8_t __mmi_pfillb_u(uint8_t v)
{
    return (uint8x8_t) {
        v, v, v, v, v, v, v, v
    };
}

static inline __attribute__((__always_inline__)) int32x2_t __mmi_pfillw_s(int32_t v)
{
    return (int32x2_t) {
        v, v
    };
}
static inline __attribute__((__always_inline__)) int16x4_t __mmi_pfillh_s(int16_t v)
{
    return (int16x4_t) {
        v, v, v, v
    };
}
static inline __attribute__((__always_inline__)) int8x8_t __mmi_pfillb_s(int8_t v)
{
    return (int8x8_t) {
        v, v, v, v, v, v, v, v
    };
}

static inline __attribute__((__always_inline__)) uint32x2_t __mmi_pldw_u(const void* p)
{
    return *(const uint32x2_t*)p;
}
static inline __attribute__((__always_inline__)) uint16x4_t __mmi_pldh_u(const void* p)
{
    return *(const uint16x4_t*)p;
}
static inline __attribute__((__always_inline__)) uint8x8_t __mmi_pldb_u(const void* p)
{
    return *(const uint8x8_t*)p;
}

static inline __attribute__((__always_inline__)) int32x2_t __mmi_pldw_s(const void* p)
{
    return *(const int32x2_t*)p;
}
static inline __attribute__((__always_inline__)) int16x4_t __mmi_pldh_s(const void* p)
{
    return *(const int16x4_t*)p;
}
static inline __attribute__((__always_inline__)) int8x8_t __mmi_pldb_s(const void* p)
{
    return *(const int8x8_t*)p;
}

static inline __attribute__((__always_inline__)) void __mmi_pstw_u(void* p, uint32x2_t v)
{
    *(uint32x2_t*)p = v;
}
static inline __attribute__((__always_inline__)) void __mmi_psth_u(void* p, uint16x4_t v)
{
    *(uint16x4_t*)p = v;
}
static inline __attribute__((__always_inline__)) void __mmi_pstb_u(void* p, uint8x8_t v)
{
    *(uint8x8_t*)p = v;
}

static inline __attribute__((__always_inline__)) void __mmi_pstw_s(void* p, int32x2_t v)
{
    *(int32x2_t*)p = v;
}
static inline __attribute__((__always_inline__)) void __mmi_psth_s(void* p, int16x4_t v)
{
    *(int16x4_t*)p = v;
}
static inline __attribute__((__always_inline__)) void __mmi_pstb_s(void* p, int8x8_t v)
{
    *(int8x8_t*)p = v;
}

#define __mmi_packsswh    __builtin_loongson_packsswh
#define __mmi_packsshb    __builtin_loongson_packsshb
#define __mmi_packushb    __builtin_loongson_packushb
#define __mmi_paddw_u     __builtin_loongson_paddw_u
#define __mmi_paddh_u     __builtin_loongson_paddh_u
#define __mmi_paddb_u     __builtin_loongson_paddb_u
#define __mmi_paddw_s     __builtin_loongson_paddw_s
#define __mmi_paddh_s     __builtin_loongson_paddh_s
#define __mmi_paddb_s     __builtin_loongson_paddb_s
#define __mmi_paddd_u     __builtin_loongson_paddd_u
#define __mmi_paddd_s     __builtin_loongson_paddd_s
#define __mmi_paddsh      __builtin_loongson_paddsh
#define __mmi_paddsb      __builtin_loongson_paddsb
#define __mmi_paddush     __builtin_loongson_paddush
#define __mmi_paddusb     __builtin_loongson_paddusb
#define __mmi_pandn_ud    __builtin_loongson_pandn_ud
#define __mmi_pandn_uw    __builtin_loongson_pandn_uw
#define __mmi_pandn_uh    __builtin_loongson_pandn_uh
#define __mmi_pandn_ub    __builtin_loongson_pandn_ub
#define __mmi_pandn_sd    __builtin_loongson_pandn_sd
#define __mmi_pandn_sw    __builtin_loongson_pandn_sw
#define __mmi_pandn_sh    __builtin_loongson_pandn_sh
#define __mmi_pandn_sb    __builtin_loongson_pandn_sb
#define __mmi_pavgh       __builtin_loongson_pavgh
#define __mmi_pavgb       __builtin_loongson_pavgb
#define __mmi_pcmpeqw_u   __builtin_loongson_pcmpeqw_u
#define __mmi_pcmpeqh_u   __builtin_loongson_pcmpeqh_u
#define __mmi_pcmpeqb_u   __builtin_loongson_pcmpeqb_u
#define __mmi_pcmpeqw_s   __builtin_loongson_pcmpeqw_s
#define __mmi_pcmpeqh_s   __builtin_loongson_pcmpeqh_s
#define __mmi_pcmpeqb_s   __builtin_loongson_pcmpeqb_s
#define __mmi_pcmpgtw_u   __builtin_loongson_pcmpgtw_u
#define __mmi_pcmpgth_u   __builtin_loongson_pcmpgth_u
#define __mmi_pcmpgtb_u   __builtin_loongson_pcmpgtb_u
#define __mmi_pcmpgtw_s   __builtin_loongson_pcmpgtw_s
#define __mmi_pcmpgth_s   __builtin_loongson_pcmpgth_s
#define __mmi_pcmpgtb_s   __builtin_loongson_pcmpgtb_s
#define __mmi_pextrh_u    __builtin_loongson_pextrh_u
#define __mmi_pextrh_s    __builtin_loongson_pextrh_s
#define __mmi_pinsrh_0_u  __builtin_loongson_pinsrh_0_u
#define __mmi_pinsrh_1_u  __builtin_loongson_pinsrh_1_u
#define __mmi_pinsrh_2_u  __builtin_loongson_pinsrh_2_u
#define __mmi_pinsrh_3_u  __builtin_loongson_pinsrh_3_u
#define __mmi_pinsrh_0_s  __builtin_loongson_pinsrh_0_s
#define __mmi_pinsrh_1_s  __builtin_loongson_pinsrh_1_s
#define __mmi_pinsrh_2_s  __builtin_loongson_pinsrh_2_s
#define __mmi_pinsrh_3_s  __builtin_loongson_pinsrh_3_s
#define __mmi_pmaddhw     __builtin_loongson_pmaddhw
#define __mmi_pmaxsh      __builtin_loongson_pmaxsh
#define __mmi_pmaxub      __builtin_loongson_pmaxub
#define __mmi_pminsh      __builtin_loongson_pminsh
#define __mmi_pminub      __builtin_loongson_pminub
#define __mmi_pmovmskb_u  __builtin_loongson_pmovmskb_u
#define __mmi_pmovmskb_s  __builtin_loongson_pmovmskb_s
#define __mmi_pmulhuh     __builtin_loongson_pmulhuh
#define __mmi_pmulhh      __builtin_loongson_pmulhh
#define __mmi_pmullh      __builtin_loongson_pmullh
#define __mmi_pmuluw      __builtin_loongson_pmuluw
#define __mmi_pasubub     __builtin_loongson_pasubub
#define __mmi_biadd       __builtin_loongson_biadd
#define __mmi_psadbh      __builtin_loongson_psadbh
#define __mmi_pshufh_u    __builtin_loongson_pshufh_u
#define __mmi_pshufh_s    __builtin_loongson_pshufh_s
#define __mmi_psllh_u     __builtin_loongson_psllh_u
#define __mmi_psllh_s     __builtin_loongson_psllh_s
#define __mmi_psllw_u     __builtin_loongson_psllw_u
#define __mmi_psllw_s     __builtin_loongson_psllw_s
#define __mmi_psrlh_u     __builtin_loongson_psrlh_u
#define __mmi_psrlh_s     __builtin_loongson_psrlh_s
#define __mmi_psrlw_u     __builtin_loongson_psrlw_u
#define __mmi_psrlw_s     __builtin_loongson_psrlw_s
#define __mmi_psrah_u     __builtin_loongson_psrah_u
#define __mmi_psrah_s     __builtin_loongson_psrah_s
#define __mmi_psraw_u     __builtin_loongson_psraw_u
#define __mmi_psraw_s     __builtin_loongson_psraw_s
#define __mmi_psubw_u     __builtin_loongson_psubw_u
#define __mmi_psubh_u     __builtin_loongson_psubh_u
#define __mmi_psubb_u     __builtin_loongson_psubb_u
#define __mmi_psubw_s     __builtin_loongson_psubw_s
#define __mmi_psubh_s     __builtin_loongson_psubh_s
#define __mmi_psubb_s     __builtin_loongson_psubb_s
#define __mmi_psubd_u     __builtin_loongson_psubd_u
#define __mmi_psubd_s     __builtin_loongson_psubd_s
#define __mmi_psubsh      __builtin_loongson_psubsh
#define __mmi_psubsb      __builtin_loongson_psubsb
#define __mmi_psubush     __builtin_loongson_psubush
#define __mmi_psubusb     __builtin_loongson_psubusb
#define __mmi_punpckhwd_u __builtin_loongson_punpckhwd_u
#define __mmi_punpckhhw_u __builtin_loongson_punpckhhw_u
#define __mmi_punpckhbh_u __builtin_loongson_punpckhbh_u
#define __mmi_punpckhwd_s __builtin_loongson_punpckhwd_s
#define __mmi_punpckhhw_s __builtin_loongson_punpckhhw_s
#define __mmi_punpckhbh_s __builtin_loongson_punpckhbh_s
#define __mmi_punpcklwd_u __builtin_loongson_punpcklwd_u
#define __mmi_punpcklhw_u __builtin_loongson_punpcklhw_u
#define __mmi_punpcklbh_u __builtin_loongson_punpcklbh_u
#define __mmi_punpcklwd_s __builtin_loongson_punpcklwd_s
#define __mmi_punpcklhw_s __builtin_loongson_punpcklhw_s
#define __mmi_punpcklbh_s __builtin_loongson_punpcklbh_s

#ifdef __cplusplus
}
#endif

#endif // __mips_loongson_mmi

#endif // _MIPS_LOONGSON_MMI_H
