// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
void eltwise_bf16s_zvfbfmin(const std::vector<Mat>& bottom_blobs, Mat& top_blob, Mat& top_blob_fp32, int op_type, const Mat& coeffs, const Option& opt);
#endif

// op 0 = PROD, 1 = SUM, 2 = weighted SUM, 3 = accumulate weighted SUM, 4 = MAX
// a * c0 + b * c1 for the first weighted pair, a + b * c1 for every following input
template<int op>
static inline float eltwise_bf16s_op(float a, float b, float c0, float c1)
{
    if (op == 0) return a * b;
    if (op == 1) return a + b;
    if (op == 2) return a * c0 + b * c1;
    if (op == 3) return a + b * c1;
    return std::max(a, b);
}

#if __riscv_vector
template<int op>
static inline vfloat32m4_t eltwise_bf16s_op(vfloat32m4_t _a, vfloat32m4_t _b, float c0, float c1, size_t vl)
{
    if (op == 0) return __riscv_vfmul_vv_f32m4(_a, _b, vl);
    if (op == 1) return __riscv_vfadd_vv_f32m4(_a, _b, vl);
    if (op == 2) return __riscv_vfmacc_vf_f32m4(__riscv_vfmul_vf_f32m4(_a, c0, vl), c1, _b, vl);
    if (op == 3) return __riscv_vfmacc_vf_f32m4(_a, c1, _b, vl);
    return __riscv_vfmax_vv_f32m4(_a, _b, vl);
}
#endif // __riscv_vector

// outptr = op(ptr, ptr1)
template<int op>
static void eltwise_bf16s_pass(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int size, float c0, float c1)
{
#if __riscv_vector
    int n = size;
    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e16m2(n);
        vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
        vfloat32m4_t _p1 = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr1, vl), vl);
        _p = eltwise_bf16s_op<op>(_p, _p1, c0, c1, vl);
        __riscv_vse16_v_u16m2(outptr, float2bfloat_rvv(_p, vl), vl);
        ptr += vl;
        ptr1 += vl;
        outptr += vl;
        n -= vl;
    }
#else  // __riscv_vector
    for (int i = 0; i < size; i++)
    {
        outptr[i] = float32_to_bfloat16(eltwise_bf16s_op<op>(bfloat16_to_float32(ptr[i]), bfloat16_to_float32(ptr1[i]), c0, c1));
    }
#endif // __riscv_vector
}

// outptr = op(ptr, ptr1) kept in fp32
template<int op>
static void eltwise_bf16s_pass_fp32(const unsigned short* ptr, const unsigned short* ptr1, float* outptr, int size, float c0, float c1)
{
#if __riscv_vector
    int n = size;
    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e16m2(n);
        vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
        vfloat32m4_t _p1 = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr1, vl), vl);
        _p = eltwise_bf16s_op<op>(_p, _p1, c0, c1, vl);
        __riscv_vse32_v_f32m4(outptr, _p, vl);
        ptr += vl;
        ptr1 += vl;
        outptr += vl;
        n -= vl;
    }
#else  // __riscv_vector
    for (int i = 0; i < size; i++)
    {
        outptr[i] = eltwise_bf16s_op<op>(bfloat16_to_float32(ptr[i]), bfloat16_to_float32(ptr1[i]), c0, c1);
    }
#endif // __riscv_vector
}

// outptr = op(outptr, ptr1) kept in fp32
template<int op>
static void eltwise_bf16s_pass_acc(float* outptr, const unsigned short* ptr1, int size, float c1)
{
#if __riscv_vector
    int n = size;
    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e16m2(n);
        vfloat32m4_t _p = __riscv_vle32_v_f32m4(outptr, vl);
        vfloat32m4_t _p1 = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr1, vl), vl);
        _p = eltwise_bf16s_op<op>(_p, _p1, 1.f, c1, vl);
        __riscv_vse32_v_f32m4(outptr, _p, vl);
        ptr1 += vl;
        outptr += vl;
        n -= vl;
    }
#else  // __riscv_vector
    for (int i = 0; i < size; i++)
    {
        outptr[i] = eltwise_bf16s_op<op>(outptr[i], bfloat16_to_float32(ptr1[i]), 1.f, c1);
    }
#endif // __riscv_vector
}

// outptr = op(ptr, ptr1) with fp32 ptr, narrowed to bf16
template<int op>
static void eltwise_bf16s_pass_last(const float* ptr, const unsigned short* ptr1, unsigned short* outptr, int size, float c1)
{
#if __riscv_vector
    int n = size;
    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e16m2(n);
        vfloat32m4_t _p = __riscv_vle32_v_f32m4(ptr, vl);
        vfloat32m4_t _p1 = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr1, vl), vl);
        _p = eltwise_bf16s_op<op>(_p, _p1, 1.f, c1, vl);
        __riscv_vse16_v_u16m2(outptr, float2bfloat_rvv(_p, vl), vl);
        ptr += vl;
        ptr1 += vl;
        outptr += vl;
        n -= vl;
    }
#else  // __riscv_vector
    for (int i = 0; i < size; i++)
    {
        outptr[i] = float32_to_bfloat16(eltwise_bf16s_op<op>(ptr[i], bfloat16_to_float32(ptr1[i]), 1.f, c1));
    }
#endif // __riscv_vector
}

template<int op_first, int op_acc>
static void eltwise_bf16s_impl(const std::vector<Mat>& bottom_blobs, Mat& top_blob, Mat& top_blob_fp32, const Mat& coeffs, const Option& opt)
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& bottom_blob1 = bottom_blobs[1];
    const int channels = bottom_blob.c;
    const int size = bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.elempack;
    const int blob_count = (int)bottom_blobs.size();

    const float coeff0 = coeffs.empty() ? 1.f : coeffs[0];
    const float coeff1 = coeffs.empty() ? 1.f : coeffs[1];

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const unsigned short* ptr = bottom_blob.channel(q);
        const unsigned short* ptr1 = bottom_blob1.channel(q);
        unsigned short* outptr = top_blob.channel(q);

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        unsigned long old_fxcr = csrr_fxcr();
        old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
        if (blob_count == 2)
        {
            eltwise_bf16s_pass<op_first>(ptr, ptr1, outptr, size, coeff0, coeff1);
        }
        else
        {
            // keep the reduction in fp32 until every input has been consumed
            float* accptr = top_blob_fp32.row(get_omp_thread_num());

            eltwise_bf16s_pass_fp32<op_first>(ptr, ptr1, accptr, size, coeff0, coeff1);

            for (int b = 2; b < blob_count - 1; b++)
            {
                const float coeff = coeffs.empty() ? 1.f : coeffs[b];
                eltwise_bf16s_pass_acc<op_acc>(accptr, bottom_blobs[b].channel(q), size, coeff);
            }

            const float coeff = coeffs.empty() ? 1.f : coeffs[blob_count - 1];
            eltwise_bf16s_pass_last<op_acc>(accptr, bottom_blobs[blob_count - 1].channel(q), outptr, size, coeff);
        }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
    }
}

// top_blob_fp32 holds one fp32 accumulator row per thread, only used for more than two inputs
static void eltwise_bf16s(const std::vector<Mat>& bottom_blobs, Mat& top_blob, Mat& top_blob_fp32, int op_type, const Mat& coeffs, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (cpu_support_riscv_zvfbfmin())
    {
        eltwise_bf16s_zvfbfmin(bottom_blobs, top_blob, top_blob_fp32, op_type, coeffs, opt);
        return;
    }
#endif

    if (op_type == 0) // Operation_PROD
        eltwise_bf16s_impl<0, 0>(bottom_blobs, top_blob, top_blob_fp32, coeffs, opt);
    else if (op_type == 1 && coeffs.empty()) // Operation_SUM
        eltwise_bf16s_impl<1, 1>(bottom_blobs, top_blob, top_blob_fp32, coeffs, opt);
    else if (op_type == 1) // Operation_SUM with coeffs
        eltwise_bf16s_impl<2, 3>(bottom_blobs, top_blob, top_blob_fp32, coeffs, opt);
    else // Operation_MAX
        eltwise_bf16s_impl<4, 4>(bottom_blobs, top_blob, top_blob_fp32, coeffs, opt);
}
