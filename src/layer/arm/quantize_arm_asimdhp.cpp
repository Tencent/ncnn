// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "quantize_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#endif // __ARM_NEON

namespace ncnn {

#include "quantize_fp16s.h"

void quantize_fp16s_asimdhp(const unsigned short* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int elempack)
{
    quantize_fp16s((const __fp16*)ptr, s8ptr, scale_data, elemcount, elempack);
}

void quantize_pack4to8_fp16s_asimdhp(const unsigned short* ptr0, const unsigned short* ptr1, signed char* s8ptr, const Mat& scale_data, int elemcount)
{
    quantize_pack4to8_fp16s((const __fp16*)ptr0, (const __fp16*)ptr1, s8ptr, scale_data, elemcount);
}

void quantize_pack4to1_fp16s_asimdhp(const unsigned short* ptr, signed char* s8ptr0, signed char* s8ptr1, signed char* s8ptr2, signed char* s8ptr3, const Mat& scale_data, int elemcount)
{
    quantize_pack4to1_fp16s((const __fp16*)ptr, s8ptr0, s8ptr1, s8ptr2, s8ptr3, scale_data, elemcount);
}

void quantize_fp16sa_asimdhp(const unsigned short* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int elempack)
{
    quantize_fp16sa((const __fp16*)ptr, s8ptr, scale_data, elemcount, elempack);
}

void quantize_pack4to1_fp16sa_asimdhp(const unsigned short* ptr, signed char* s8ptr0, signed char* s8ptr1, signed char* s8ptr2, signed char* s8ptr3, const Mat& scale_data, int elemcount)
{
    quantize_pack4to1_fp16sa((const __fp16*)ptr, s8ptr0, s8ptr1, s8ptr2, s8ptr3, scale_data, elemcount);
}

} // namespace ncnn
