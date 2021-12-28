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

#ifndef RISCV_USABILITY_H
#define RISCV_USABILITY_H

#if __riscv_vector
static inline int csrr_vl()
{
    int a = 0;
    asm volatile("csrr %0, vl"
                 : "=r"(a)
                 :
                 : "memory");
    return a;
}

static inline int csrr_vtype()
{
    int a = 0;
    asm volatile("csrr %0, vtype"
                 : "=r"(a)
                 :
                 : "memory");
    return a;
}

static inline int csrr_vlenb()
{
    int a = 0;
    asm volatile("csrr %0, vlenb"
                 : "=r"(a)
                 :
                 : "memory");
    return a;
}

static inline vfloat32m8_t vle32_v_f32m8_f32m1(const float* ptr)
{
    // 128bit
    static const uint8_t index_128bit[32] = {
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3
    };

    // 256bit
    static const uint8_t index_256bit[64] = {
        0, 1, 2, 3, 4, 5, 6, 7,
        0, 1, 2, 3, 4, 5, 6, 7,
        0, 1, 2, 3, 4, 5, 6, 7,
        0, 1, 2, 3, 4, 5, 6, 7,
        0, 1, 2, 3, 4, 5, 6, 7,
        0, 1, 2, 3, 4, 5, 6, 7,
        0, 1, 2, 3, 4, 5, 6, 7,
        0, 1, 2, 3, 4, 5, 6, 7
    };

    const int packn = csrr_vlenb() / 4;
    const word_type vl = vsetvl_e32m1(packn);

    const uint8_t* index = packn == 4 ? index_128bit : index_256bit;

    vuint8m2_t bindex = vle8_v_u8m2(index, vl * 4);

    return vloxei8_v_f32m8(ptr, bindex, vl);
}

#if __riscv_zfh
static inline vfloat16m8_t vle16_v_f16m8_f16m1(const __fp16* ptr)
{
    // 128bit
    static const uint8_t index_128bit[64] = {
        0, 1, 2, 3, 4, 5, 6, 7,
        0, 1, 2, 3, 4, 5, 6, 7,
        0, 1, 2, 3, 4, 5, 6, 7,
        0, 1, 2, 3, 4, 5, 6, 7,
        0, 1, 2, 3, 4, 5, 6, 7,
        0, 1, 2, 3, 4, 5, 6, 7,
        0, 1, 2, 3, 4, 5, 6, 7,
        0, 1, 2, 3, 4, 5, 6, 7
    };

    // 256bit
    static const uint8_t index_256bit[128] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    };

    const int packn = csrr_vlenb() / 2;
    const word_type vl = vsetvl_e16m1(packn);

    const uint8_t* index = packn == 8 ? index_128bit : index_256bit;

    vuint8m4_t bindex = vle8_v_u8m4(index, vl * 2);

    return vloxei8_v_f16m8(ptr, bindex, vl);
}
#endif // __riscv_zfh
#endif // __riscv_vector

#endif // RISCV_USABILITY_H
