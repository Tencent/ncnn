// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// Compute output channels [p, p + count) for one unpacked input vector.
static void innerproduct_bf16s(const unsigned short* input, unsigned short* output, const Mat& weight_data_tm, const Mat& bias_data, int num_input, int p, int count, int activation_type, const Mat& activation_params)
{
    const int weight_elempack = weight_data_tm.elempack;
    const unsigned short* weights = (const unsigned short*)weight_data_tm + p / weight_elempack * num_input * weight_elempack;

#if __riscv_vector
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
    unsigned long old_fxcr = csrr_fxcr();
    old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
    const size_t vl = __riscv_vsetvl_e16m1(count);
    vfloat32m2_t _sum = __riscv_vfmv_v_f_f32m2(0.f, vl);
    if (!bias_data.empty())
        _sum = __riscv_vle32_v_f32m2((const float*)bias_data + p, vl);

    for (int i = 0; i < num_input; i++)
    {
        vuint16m1_t _w;
        if (weight_elempack == 1)
            _w = __riscv_vlse16_v_u16m1(weights + i, num_input * sizeof(unsigned short), vl);
        else
            _w = __riscv_vle16_v_u16m1(weights + i * weight_elempack, vl);

        _sum = __riscv_vfmacc_vf_f32m2(_sum, bfloat16_to_float32(input[i]), bfloat2float_rvv(_w, vl), vl);
    }

    _sum = activation_ps(_sum, activation_type, activation_params, vl);
    __riscv_vse16_v_u16m1(output + p, float2bfloat_rvv(_sum, vl), vl);
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
    csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else
    for (int q = 0; q < count; q++)
    {
        float sum = bias_data.empty() ? 0.f : bias_data[p + q];
        const unsigned short* kptr = weights + q * num_input;
        for (int i = 0; i < num_input; i++)
            sum += bfloat16_to_float32(input[i]) * bfloat16_to_float32(kptr[i]);

        output[p + q] = float32_to_bfloat16(activation_ss(sum, activation_type, activation_params));
    }
#endif
}
