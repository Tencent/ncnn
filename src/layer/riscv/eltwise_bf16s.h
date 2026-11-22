// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
void eltwise_bf16s_zvfbfmin(const std::vector<Mat>& bottom_blobs, Mat& top_blob, Mat& top_blob_fp32, int op_type, const Mat& coeffs, const Option& opt);
#endif

// top_blob_fp32 is the fp32 accumulator for more than two inputs
static void eltwise_bf16s(const std::vector<Mat>& bottom_blobs, Mat& top_blob, Mat& top_blob_fp32, int op_type, const Mat& coeffs, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (cpu_support_riscv_zvfbfmin())
    {
        eltwise_bf16s_zvfbfmin(bottom_blobs, top_blob, top_blob_fp32, op_type, coeffs, opt);
        return;
    }
#endif

    const Mat& bottom_blob = bottom_blobs[0];
    const int channels = bottom_blob.c;
    const int size = bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.elempack;

    if (bottom_blobs.size() == 2)
    {
        // fast path without fp32 accumulator
        const Mat& bottom_blob1 = bottom_blobs[1];

        if (op_type == 0) // Operation_PROD
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob1.channel(q);
                unsigned short* outptr = top_blob.channel(q);
#if __riscv_vector
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                unsigned long old_fxcr = csrr_fxcr();
                old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m2(n);

                    vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
                    vfloat32m4_t _p1 = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr1, vl), vl);
                    _p = __riscv_vfmul_vv_f32m4(_p, _p1, vl);
                    __riscv_vse16_v_u16m2(outptr, float2bfloat_rvv(_p, vl), vl);

                    ptr += vl;
                    ptr1 += vl;
                    outptr += vl;
                    n -= vl;
                }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else  // __riscv_vector
                for (int i = 0; i < size; i++)
                {
                    outptr[i] = float32_to_bfloat16(bfloat16_to_float32(ptr[i]) * bfloat16_to_float32(ptr1[i]));
                }
#endif // __riscv_vector
            }
        }
        if (op_type == 1 && coeffs.empty()) // Operation_SUM
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob1.channel(q);
                unsigned short* outptr = top_blob.channel(q);
#if __riscv_vector
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                unsigned long old_fxcr = csrr_fxcr();
                old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m2(n);

                    vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
                    vfloat32m4_t _p1 = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr1, vl), vl);
                    _p = __riscv_vfadd_vv_f32m4(_p, _p1, vl);
                    __riscv_vse16_v_u16m2(outptr, float2bfloat_rvv(_p, vl), vl);

                    ptr += vl;
                    ptr1 += vl;
                    outptr += vl;
                    n -= vl;
                }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else  // __riscv_vector
                for (int i = 0; i < size; i++)
                {
                    outptr[i] = float32_to_bfloat16(bfloat16_to_float32(ptr[i]) + bfloat16_to_float32(ptr1[i]));
                }
#endif // __riscv_vector
            }
        }
        if (op_type == 1 && !coeffs.empty()) // Operation_SUM with coeffs
        {
            const float coeff0 = coeffs[0];
            const float coeff1 = coeffs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob1.channel(q);
                unsigned short* outptr = top_blob.channel(q);
#if __riscv_vector
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                unsigned long old_fxcr = csrr_fxcr();
                old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m2(n);

                    vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
                    vfloat32m4_t _p1 = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr1, vl), vl);
                    _p = __riscv_vfmul_vf_f32m4(_p, coeff0, vl);
                    _p = __riscv_vfmacc_vf_f32m4(_p, coeff1, _p1, vl);
                    __riscv_vse16_v_u16m2(outptr, float2bfloat_rvv(_p, vl), vl);

                    ptr += vl;
                    ptr1 += vl;
                    outptr += vl;
                    n -= vl;
                }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else  // __riscv_vector
                for (int i = 0; i < size; i++)
                {
                    outptr[i] = float32_to_bfloat16(bfloat16_to_float32(ptr[i]) * coeff0 + bfloat16_to_float32(ptr1[i]) * coeff1);
                }
#endif // __riscv_vector
            }
        }
        if (op_type == 2) // Operation_MAX
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob1.channel(q);
                unsigned short* outptr = top_blob.channel(q);
#if __riscv_vector
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                unsigned long old_fxcr = csrr_fxcr();
                old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m2(n);

                    vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
                    vfloat32m4_t _p1 = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr1, vl), vl);
                    _p = __riscv_vfmax_vv_f32m4(_p, _p1, vl);
                    __riscv_vse16_v_u16m2(outptr, float2bfloat_rvv(_p, vl), vl);

                    ptr += vl;
                    ptr1 += vl;
                    outptr += vl;
                    n -= vl;
                }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else  // __riscv_vector
                for (int i = 0; i < size; i++)
                {
                    outptr[i] = float32_to_bfloat16(std::max(bfloat16_to_float32(ptr[i]), bfloat16_to_float32(ptr1[i])));
                }
#endif // __riscv_vector
            }
        }

        return;
    }

    // keep the reduction in fp32 until every input has been consumed
    if (op_type == 0) // Operation_PROD
    {
        // top_blob_fp32 = bottom_blob * bottom_blobs[1]
        {
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob_fp32.channel(q);
#if __riscv_vector
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                unsigned long old_fxcr = csrr_fxcr();
                old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m2(n);

                    vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
                    vfloat32m4_t _p1 = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr1, vl), vl);
                    _p = __riscv_vfmul_vv_f32m4(_p, _p1, vl);
                    __riscv_vse32_v_f32m4(outptr, _p, vl);

                    ptr += vl;
                    ptr1 += vl;
                    outptr += vl;
                    n -= vl;
                }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else  // __riscv_vector
                for (int i = 0; i < size; i++)
                {
                    outptr[i] = bfloat16_to_float32(ptr[i]) * bfloat16_to_float32(ptr1[i]);
                }
#endif // __riscv_vector
            }
        }

        // top_blob_fp32 *= bottom_blobs[i] for i = 2, 3, ...
        for (size_t b = 2; b < bottom_blobs.size() - 1; b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob_fp32.channel(q);
#if __riscv_vector
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                unsigned long old_fxcr = csrr_fxcr();
                old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m2(n);

                    vfloat32m4_t _p = __riscv_vle32_v_f32m4(outptr, vl);
                    vfloat32m4_t _p1 = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr1, vl), vl);
                    _p = __riscv_vfmul_vv_f32m4(_p, _p1, vl);
                    __riscv_vse32_v_f32m4(outptr, _p, vl);

                    ptr1 += vl;
                    outptr += vl;
                    n -= vl;
                }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else  // __riscv_vector
                for (int i = 0; i < size; i++)
                {
                    outptr[i] *= bfloat16_to_float32(ptr1[i]);
                }
#endif // __riscv_vector
            }
        }

        // top_blob = top_blob_fp32 * bottom_blobs[last]
        {
            const Mat& bottom_blob1 = bottom_blobs[bottom_blobs.size() - 1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = top_blob_fp32.channel(q);
                const unsigned short* ptr1 = bottom_blob1.channel(q);
                unsigned short* outptr = top_blob.channel(q);
#if __riscv_vector
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                unsigned long old_fxcr = csrr_fxcr();
                old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m2(n);

                    vfloat32m4_t _p = __riscv_vle32_v_f32m4(ptr, vl);
                    vfloat32m4_t _p1 = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr1, vl), vl);
                    _p = __riscv_vfmul_vv_f32m4(_p, _p1, vl);
                    __riscv_vse16_v_u16m2(outptr, float2bfloat_rvv(_p, vl), vl);

                    ptr += vl;
                    ptr1 += vl;
                    outptr += vl;
                    n -= vl;
                }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else  // __riscv_vector
                for (int i = 0; i < size; i++)
                {
                    outptr[i] = float32_to_bfloat16(ptr[i] * bfloat16_to_float32(ptr1[i]));
                }
#endif // __riscv_vector
            }
        }
    }
    if (op_type == 1 && coeffs.empty()) // Operation_SUM
    {
        // top_blob_fp32 = bottom_blob + bottom_blobs[1]
        {
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob_fp32.channel(q);
#if __riscv_vector
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                unsigned long old_fxcr = csrr_fxcr();
                old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m2(n);

                    vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
                    vfloat32m4_t _p1 = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr1, vl), vl);
                    _p = __riscv_vfadd_vv_f32m4(_p, _p1, vl);
                    __riscv_vse32_v_f32m4(outptr, _p, vl);

                    ptr += vl;
                    ptr1 += vl;
                    outptr += vl;
                    n -= vl;
                }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else  // __riscv_vector
                for (int i = 0; i < size; i++)
                {
                    outptr[i] = bfloat16_to_float32(ptr[i]) + bfloat16_to_float32(ptr1[i]);
                }
#endif // __riscv_vector
            }
        }

        // top_blob_fp32 += bottom_blobs[i] for i = 2, 3, ...
        for (size_t b = 2; b < bottom_blobs.size() - 1; b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob_fp32.channel(q);
#if __riscv_vector
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                unsigned long old_fxcr = csrr_fxcr();
                old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m2(n);

                    vfloat32m4_t _p = __riscv_vle32_v_f32m4(outptr, vl);
                    vfloat32m4_t _p1 = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr1, vl), vl);
                    _p = __riscv_vfadd_vv_f32m4(_p, _p1, vl);
                    __riscv_vse32_v_f32m4(outptr, _p, vl);

                    ptr1 += vl;
                    outptr += vl;
                    n -= vl;
                }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else  // __riscv_vector
                for (int i = 0; i < size; i++)
                {
                    outptr[i] += bfloat16_to_float32(ptr1[i]);
                }
#endif // __riscv_vector
            }
        }

        // top_blob = top_blob_fp32 + bottom_blobs[last]
        {
            const Mat& bottom_blob1 = bottom_blobs[bottom_blobs.size() - 1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = top_blob_fp32.channel(q);
                const unsigned short* ptr1 = bottom_blob1.channel(q);
                unsigned short* outptr = top_blob.channel(q);
#if __riscv_vector
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                unsigned long old_fxcr = csrr_fxcr();
                old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m2(n);

                    vfloat32m4_t _p = __riscv_vle32_v_f32m4(ptr, vl);
                    vfloat32m4_t _p1 = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr1, vl), vl);
                    _p = __riscv_vfadd_vv_f32m4(_p, _p1, vl);
                    __riscv_vse16_v_u16m2(outptr, float2bfloat_rvv(_p, vl), vl);

                    ptr += vl;
                    ptr1 += vl;
                    outptr += vl;
                    n -= vl;
                }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else  // __riscv_vector
                for (int i = 0; i < size; i++)
                {
                    outptr[i] = float32_to_bfloat16(ptr[i] + bfloat16_to_float32(ptr1[i]));
                }
#endif // __riscv_vector
            }
        }
    }
    if (op_type == 1 && !coeffs.empty()) // Operation_SUM with coeffs
    {
        // top_blob_fp32 = bottom_blob * coeffs[0] + bottom_blobs[1] * coeffs[1]
        {
            const Mat& bottom_blob1 = bottom_blobs[1];
            const float coeff0 = coeffs[0];
            const float coeff1 = coeffs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob_fp32.channel(q);
#if __riscv_vector
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                unsigned long old_fxcr = csrr_fxcr();
                old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m2(n);

                    vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
                    vfloat32m4_t _p1 = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr1, vl), vl);
                    _p = __riscv_vfmul_vf_f32m4(_p, coeff0, vl);
                    _p = __riscv_vfmacc_vf_f32m4(_p, coeff1, _p1, vl);
                    __riscv_vse32_v_f32m4(outptr, _p, vl);

                    ptr += vl;
                    ptr1 += vl;
                    outptr += vl;
                    n -= vl;
                }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else  // __riscv_vector
                for (int i = 0; i < size; i++)
                {
                    outptr[i] = bfloat16_to_float32(ptr[i]) * coeff0 + bfloat16_to_float32(ptr1[i]) * coeff1;
                }
#endif // __riscv_vector
            }
        }

        // top_blob_fp32 += bottom_blobs[i] * coeffs[i] for i = 2, 3, ...
        for (size_t b = 2; b < bottom_blobs.size() - 1; b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
            const float coeff = coeffs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob_fp32.channel(q);
#if __riscv_vector
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                unsigned long old_fxcr = csrr_fxcr();
                old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m2(n);

                    vfloat32m4_t _p = __riscv_vle32_v_f32m4(outptr, vl);
                    vfloat32m4_t _p1 = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr1, vl), vl);
                    _p = __riscv_vfmacc_vf_f32m4(_p, coeff, _p1, vl);
                    __riscv_vse32_v_f32m4(outptr, _p, vl);

                    ptr1 += vl;
                    outptr += vl;
                    n -= vl;
                }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else  // __riscv_vector
                for (int i = 0; i < size; i++)
                {
                    outptr[i] += bfloat16_to_float32(ptr1[i]) * coeff;
                }
#endif // __riscv_vector
            }
        }

        // top_blob = top_blob_fp32 + bottom_blobs[last] * coeffs[last]
        {
            const Mat& bottom_blob1 = bottom_blobs[bottom_blobs.size() - 1];
            const float coeff = coeffs[bottom_blobs.size() - 1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = top_blob_fp32.channel(q);
                const unsigned short* ptr1 = bottom_blob1.channel(q);
                unsigned short* outptr = top_blob.channel(q);
#if __riscv_vector
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                unsigned long old_fxcr = csrr_fxcr();
                old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m2(n);

                    vfloat32m4_t _p = __riscv_vle32_v_f32m4(ptr, vl);
                    vfloat32m4_t _p1 = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr1, vl), vl);
                    _p = __riscv_vfmacc_vf_f32m4(_p, coeff, _p1, vl);
                    __riscv_vse16_v_u16m2(outptr, float2bfloat_rvv(_p, vl), vl);

                    ptr += vl;
                    ptr1 += vl;
                    outptr += vl;
                    n -= vl;
                }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else  // __riscv_vector
                for (int i = 0; i < size; i++)
                {
                    outptr[i] = float32_to_bfloat16(ptr[i] + bfloat16_to_float32(ptr1[i]) * coeff);
                }
#endif // __riscv_vector
            }
        }
    }
    if (op_type == 2) // Operation_MAX
    {
        // top_blob_fp32 = max(bottom_blob, bottom_blobs[1])
        {
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob_fp32.channel(q);
#if __riscv_vector
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                unsigned long old_fxcr = csrr_fxcr();
                old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m2(n);

                    vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
                    vfloat32m4_t _p1 = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr1, vl), vl);
                    _p = __riscv_vfmax_vv_f32m4(_p, _p1, vl);
                    __riscv_vse32_v_f32m4(outptr, _p, vl);

                    ptr += vl;
                    ptr1 += vl;
                    outptr += vl;
                    n -= vl;
                }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else  // __riscv_vector
                for (int i = 0; i < size; i++)
                {
                    outptr[i] = std::max(bfloat16_to_float32(ptr[i]), bfloat16_to_float32(ptr1[i]));
                }
#endif // __riscv_vector
            }
        }

        // top_blob_fp32 = max(top_blob_fp32, bottom_blobs[i]) for i = 2, 3, ...
        for (size_t b = 2; b < bottom_blobs.size() - 1; b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob_fp32.channel(q);
#if __riscv_vector
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                unsigned long old_fxcr = csrr_fxcr();
                old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m2(n);

                    vfloat32m4_t _p = __riscv_vle32_v_f32m4(outptr, vl);
                    vfloat32m4_t _p1 = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr1, vl), vl);
                    _p = __riscv_vfmax_vv_f32m4(_p, _p1, vl);
                    __riscv_vse32_v_f32m4(outptr, _p, vl);

                    ptr1 += vl;
                    outptr += vl;
                    n -= vl;
                }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else  // __riscv_vector
                for (int i = 0; i < size; i++)
                {
                    outptr[i] = std::max(outptr[i], bfloat16_to_float32(ptr1[i]));
                }
#endif // __riscv_vector
            }
        }

        // top_blob = max(top_blob_fp32, bottom_blobs[last])
        {
            const Mat& bottom_blob1 = bottom_blobs[bottom_blobs.size() - 1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = top_blob_fp32.channel(q);
                const unsigned short* ptr1 = bottom_blob1.channel(q);
                unsigned short* outptr = top_blob.channel(q);
#if __riscv_vector
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                unsigned long old_fxcr = csrr_fxcr();
                old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m2(n);

                    vfloat32m4_t _p = __riscv_vle32_v_f32m4(ptr, vl);
                    vfloat32m4_t _p1 = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr1, vl), vl);
                    _p = __riscv_vfmax_vv_f32m4(_p, _p1, vl);
                    __riscv_vse16_v_u16m2(outptr, float2bfloat_rvv(_p, vl), vl);

                    ptr += vl;
                    ptr1 += vl;
                    outptr += vl;
                    n -= vl;
                }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else  // __riscv_vector
                for (int i = 0; i < size; i++)
                {
                    outptr[i] = float32_to_bfloat16(std::max(ptr[i], bfloat16_to_float32(ptr1[i])));
                }
#endif // __riscv_vector
            }
        }
    }
}
