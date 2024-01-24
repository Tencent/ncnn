#include <iostream>
#include <cstdio>
#include <riscv_vector.h>

static inline void transpose4x4_f16(vfloat16mf2_t & _r0, vfloat16mf2_t & _r1, vfloat16mf2_t & _r2, vfloat16mf2_t & _r3, size_t vl)
{
    __fp16 tmp[4][4];
    vsse16_v_f16m1(&tmp[0][0], sizeof(__fp16) * 4, _r0, vl);
    vsse16_v_f16m1(&tmp[0][1], sizeof(__fp16) * 4, _r1, vl);
    vsse16_v_f16m1(&tmp[0][2], sizeof(__fp16) * 4, _r2, vl);
    vsse16_v_f16m1(&tmp[0][3], sizeof(__fp16) * 4, _r3, vl);
    __fp16* ptr = (__fp16*)tmp;
    _r0 = vle16_v_f16m1(ptr + 0 * 4, vl);
    _r1 = vle16_v_f16m1(ptr + 1 * 4, vl);
    _r2 = vle16_v_f16m1(ptr + 2 * 4, vl);
    _r3 = vle16_v_f16m1(ptr + 3 * 4, vl);
}

int main()
{

    __fp16 a[8] = {1,2,3,4,5,6,7,8};
    float a_float[8] = {1,2,3,4,5,6,7,8};
    __fp16 b[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float c[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    auto v0 = vle16_v_f16m1(a, 4);
    vse16_v_f16m1(b, v0, 8);
    for (int i = 0; i < 8; i++)
    {
       printf("%f\n", b[i]); 
    }
    auto vec_a = vle32_v_f32m2(a_float, 4);
    vse32_v_f32m2(c, vec_a, 8);
    for (int i = 0; i < 8; i++)
    {
       printf("%f\n", c[i]); 
    }
    return 0;
}