// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
void eltwise_bf16s_zvfbfmin(const std::vector<Mat>& bottom_blobs, Mat& top_blob, int op_type, const Mat& coeffs, const Option& opt);
#endif

static void eltwise_bf16s(const std::vector<Mat>& bottom_blobs, Mat& top_blob, int op_type, const Mat& coeffs, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (cpu_support_riscv_zvfbfmin())
    {
        eltwise_bf16s_zvfbfmin(bottom_blobs, top_blob, op_type, coeffs, opt);
        return;
    }
#endif

    const Mat& bottom_blob = bottom_blobs[0];
    const int channels = bottom_blob.c;
    const int size = bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const unsigned short* ptr = bottom_blob.channel(q);
        unsigned short* outptr = top_blob.channel(q);
#if __riscv_vector
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        unsigned long old_fxcr = csrr_fxcr();
        old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
        int i = 0;
        while (i < size)
        {
            size_t vl = __riscv_vsetvl_e16m2(size - i);
            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr + i, vl), vl);
            if (op_type == 0) // Operation_PROD
            {
                for (size_t b = 1; b < bottom_blobs.size(); b++)
                {
                    const unsigned short* p = bottom_blobs[b].channel(q);
                    vfloat32m4_t _v = bfloat2float_rvv(__riscv_vle16_v_u16m2(p + i, vl), vl);
                    _p = __riscv_vfmul_vv_f32m4(_p, _v, vl);
                }
            }
            else if (op_type == 1) // Operation_SUM
            {
                if (coeffs.empty())
                {
                    for (size_t b = 1; b < bottom_blobs.size(); b++)
                    {
                        const unsigned short* p = bottom_blobs[b].channel(q);
                        vfloat32m4_t _v = bfloat2float_rvv(__riscv_vle16_v_u16m2(p + i, vl), vl);
                        _p = __riscv_vfadd_vv_f32m4(_p, _v, vl);
                    }
                }
                else
                {
                    _p = __riscv_vfmul_vf_f32m4(_p, coeffs[0], vl);
                    for (size_t b = 1; b < bottom_blobs.size(); b++)
                    {
                        const unsigned short* p = bottom_blobs[b].channel(q);
                        vfloat32m4_t _v = bfloat2float_rvv(__riscv_vle16_v_u16m2(p + i, vl), vl);
                        _v = __riscv_vfmul_vf_f32m4(_v, coeffs[b], vl);
                        _p = __riscv_vfadd_vv_f32m4(_p, _v, vl);
                    }
                }
            }
            else // Operation_MAX
            {
                for (size_t b = 1; b < bottom_blobs.size(); b++)
                {
                    const unsigned short* p = bottom_blobs[b].channel(q);
                    vfloat32m4_t _v = bfloat2float_rvv(__riscv_vle16_v_u16m2(p + i, vl), vl);
                    _p = __riscv_vfmax_vv_f32m4(_p, _v, vl);
                }
            }
            // Keep the reduction in FP32 until every input has been consumed.
            __riscv_vse16_v_u16m2(outptr + i, float2bfloat_rvv(_p, vl), vl);
            i += vl;
        }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else
        for (int i = 0; i < size; i++)
        {
            float v = bfloat16_to_float32(ptr[i]);
            if (op_type == 0)
            {
                for (size_t b = 1; b < bottom_blobs.size(); b++)
                {
                    const unsigned short* p = bottom_blobs[b].channel(q);
                    v *= bfloat16_to_float32(p[i]);
                }
            }
            else if (op_type == 1)
            {
                if (coeffs.empty())
                {
                    for (size_t b = 1; b < bottom_blobs.size(); b++)
                    {
                        const unsigned short* p = bottom_blobs[b].channel(q);
                        v += bfloat16_to_float32(p[i]);
                    }
                }
                else
                {
                    v *= coeffs[0];
                    for (size_t b = 1; b < bottom_blobs.size(); b++)
                    {
                        const unsigned short* p = bottom_blobs[b].channel(q);
                        v += bfloat16_to_float32(p[i]) * coeffs[b];
                    }
                }
            }
            else
            {
                for (size_t b = 1; b < bottom_blobs.size(); b++)
                {
                    const unsigned short* p = bottom_blobs[b].channel(q);
                    const float v1 = bfloat16_to_float32(p[i]);
                    if (v < v1)
                        v = v1;
                }
            }
            outptr[i] = float32_to_bfloat16(v);
        }
#endif
    }
}
