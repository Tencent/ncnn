// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

static void absval_bf16s(Mat& bottom_top_blob, const Option& opt)
{
    const int size = bottom_top_blob.w * bottom_top_blob.h * bottom_top_blob.d * bottom_top_blob.elempack;
    const int channels = bottom_top_blob.c;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = bottom_top_blob.channel(q);
        // abs only clears the bf16 sign bit
#if __riscv_vector
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m8(n);
            vuint16m8_t _p = __riscv_vle16_v_u16m8(ptr, vl);
            _p = __riscv_vand_vx_u16m8(_p, 0x7fff, vl);
            __riscv_vse16_v_u16m8(ptr, _p, vl);
            ptr += vl;
            n -= vl;
        }
#else
        for (int i = 0; i < size; i++)
        {
            ptr[i] &= 0x7fff;
        }
#endif
    }
}
