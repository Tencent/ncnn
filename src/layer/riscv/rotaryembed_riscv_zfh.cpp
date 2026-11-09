// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "rotaryembed_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

namespace ncnn {

#if NCNN_ZFH
int RotaryEmbed_riscv::forward_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& cos_cache = bottom_blobs[1];
    const Mat& sin_cache = bottom_blobs[2];

    const int embed_dim = bottom_blob.w;
    const int seqlen = bottom_blob.h;
    const int num_heads = bottom_blob.c;
    const int half = embed_dim / 2;

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_heads; q++)
    {
        const Mat head = bottom_blob.channel(q);
        Mat out_head = top_blob.channel(q);

        for (int i = 0; i < seqlen; i++)
        {
            const __fp16* cos_ptr = cos_cache.row<const __fp16>(i);
            const __fp16* sin_ptr = sin_cache.row<const __fp16>(i);

            if (interleaved)
            {
                const __fp16* ptr = head.row<const __fp16>(i);
                __fp16* outptr = out_head.row<__fp16>(i);

#if __riscv_vector
                int n = half;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m2(n);

                    vfloat16m2x2_t _p = __riscv_vlseg2e16_v_f16m2x2(ptr, vl);
                    vfloat16m2_t _x0 = __riscv_vget_v_f16m2x2_f16m2(_p, 0);
                    vfloat16m2_t _x1 = __riscv_vget_v_f16m2x2_f16m2(_p, 1);
                    vfloat16m2_t _c = __riscv_vle16_v_f16m2(cos_ptr, vl);
                    vfloat16m2_t _s = __riscv_vle16_v_f16m2(sin_ptr, vl);

                    vfloat32m4_t _y0 = __riscv_vfwnmsac_vv_f32m4(__riscv_vfwmul_vv_f32m4(_x0, _c, vl), _x1, _s, vl); // x0*c - x1*s
                    vfloat32m4_t _y1 = __riscv_vfwmacc_vv_f32m4(__riscv_vfwmul_vv_f32m4(_x1, _c, vl), _x0, _s, vl);  // x0*s + x1*c
                    __riscv_vsseg2e16_v_f16m2x2(outptr, __riscv_vcreate_v_f16m2x2(__riscv_vfncvt_f_f_w_f16m2(_y0, vl), __riscv_vfncvt_f_f_w_f16m2(_y1, vl)), vl);

                    ptr += vl * 2;
                    outptr += vl * 2;
                    cos_ptr += vl;
                    sin_ptr += vl;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int j = 0; j < half; j++)
                {
                    const float x0 = (float)ptr[0];
                    const float x1 = (float)ptr[1];
                    const float cos_val = (float)*cos_ptr++;
                    const float sin_val = (float)*sin_ptr++;
                    outptr[0] = (__fp16)(x0 * cos_val - x1 * sin_val);
                    outptr[1] = (__fp16)(x0 * sin_val + x1 * cos_val);
                    ptr += 2;
                    outptr += 2;
                }
#endif // __riscv_vector
            }
            else
            {
                // full-width cache => independent cos/sin per half (2D rope); half-width => reuse
                const int cw = cos_cache.w == embed_dim ? half : 0;
                const __fp16* ptr0 = head.row<const __fp16>(i);
                const __fp16* ptr1 = ptr0 + half;
                const __fp16* cos_ptr1 = cos_ptr + cw;
                const __fp16* sin_ptr1 = sin_ptr + cw;
                __fp16* outptr0 = out_head.row<__fp16>(i);
                __fp16* outptr1 = outptr0 + half;

#if __riscv_vector
                int n = half;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m2(n);

                    vfloat16m2_t _x0 = __riscv_vle16_v_f16m2(ptr0, vl);
                    vfloat16m2_t _x1 = __riscv_vle16_v_f16m2(ptr1, vl);
                    vfloat16m2_t _c0 = __riscv_vle16_v_f16m2(cos_ptr, vl);
                    vfloat16m2_t _s0 = __riscv_vle16_v_f16m2(sin_ptr, vl);
                    vfloat16m2_t _c1 = __riscv_vle16_v_f16m2(cos_ptr1, vl);
                    vfloat16m2_t _s1 = __riscv_vle16_v_f16m2(sin_ptr1, vl);

                    vfloat32m4_t _y0 = __riscv_vfwnmsac_vv_f32m4(__riscv_vfwmul_vv_f32m4(_x0, _c0, vl), _x1, _s0, vl); // x0*c0 - x1*s0
                    vfloat32m4_t _y1 = __riscv_vfwmacc_vv_f32m4(__riscv_vfwmul_vv_f32m4(_x1, _c1, vl), _x0, _s1, vl);  // x1*c1 + x0*s1
                    __riscv_vse16_v_f16m2(outptr0, __riscv_vfncvt_f_f_w_f16m2(_y0, vl), vl);
                    __riscv_vse16_v_f16m2(outptr1, __riscv_vfncvt_f_f_w_f16m2(_y1, vl), vl);

                    ptr0 += vl;
                    ptr1 += vl;
                    cos_ptr += vl;
                    sin_ptr += vl;
                    cos_ptr1 += vl;
                    sin_ptr1 += vl;
                    outptr0 += vl;
                    outptr1 += vl;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int j = 0; j < half; j++)
                {
                    const float x0 = (float)*ptr0++;
                    const float x1 = (float)*ptr1++;
                    *outptr0++ = (__fp16)(x0 * (float)*cos_ptr++ - x1 * (float)*sin_ptr++);
                    *outptr1++ = (__fp16)(x0 * (float)*sin_ptr1++ + x1 * (float)*cos_ptr1++);
                }
#endif // __riscv_vector
            }
        }
    }

    return 0;
}

int RotaryEmbed_riscv::forward_fp16sa(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& cos_cache = bottom_blobs[1];
    const Mat& sin_cache = bottom_blobs[2];

    const int embed_dim = bottom_blob.w;
    const int seqlen = bottom_blob.h;
    const int num_heads = bottom_blob.c;
    const int half = embed_dim / 2;

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_heads; q++)
    {
        const Mat head = bottom_blob.channel(q);
        Mat out_head = top_blob.channel(q);

        for (int i = 0; i < seqlen; i++)
        {
            const __fp16* cos_ptr = cos_cache.row<const __fp16>(i);
            const __fp16* sin_ptr = sin_cache.row<const __fp16>(i);

            if (interleaved)
            {
                const __fp16* ptr = head.row<const __fp16>(i);
                __fp16* outptr = out_head.row<__fp16>(i);

#if __riscv_vector
                int n = half;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m4(n);

                    vfloat16m4x2_t _p = __riscv_vlseg2e16_v_f16m4x2(ptr, vl);
                    vfloat16m4_t _x0 = __riscv_vget_v_f16m4x2_f16m4(_p, 0);
                    vfloat16m4_t _x1 = __riscv_vget_v_f16m4x2_f16m4(_p, 1);
                    vfloat16m4_t _c = __riscv_vle16_v_f16m4(cos_ptr, vl);
                    vfloat16m4_t _s = __riscv_vle16_v_f16m4(sin_ptr, vl);

                    vfloat16m4_t _y0 = __riscv_vfnmsac_vv_f16m4(__riscv_vfmul_vv_f16m4(_x0, _c, vl), _x1, _s, vl); // x0*c - x1*s
                    vfloat16m4_t _y1 = __riscv_vfmacc_vv_f16m4(__riscv_vfmul_vv_f16m4(_x1, _c, vl), _x0, _s, vl);  // x0*s + x1*c
                    __riscv_vsseg2e16_v_f16m4x2(outptr, __riscv_vcreate_v_f16m4x2(_y0, _y1), vl);

                    ptr += vl * 2;
                    outptr += vl * 2;
                    cos_ptr += vl;
                    sin_ptr += vl;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int j = 0; j < half; j++)
                {
                    const __fp16 x0 = ptr[0];
                    const __fp16 x1 = ptr[1];
                    const __fp16 cos_val = *cos_ptr++;
                    const __fp16 sin_val = *sin_ptr++;
                    outptr[0] = x0 * cos_val - x1 * sin_val;
                    outptr[1] = x0 * sin_val + x1 * cos_val;
                    ptr += 2;
                    outptr += 2;
                }
#endif // __riscv_vector
            }
            else
            {
                // full-width cache => independent cos/sin per half (2D rope); half-width => reuse
                const int cw = cos_cache.w == embed_dim ? half : 0;
                const __fp16* ptr0 = head.row<const __fp16>(i);
                const __fp16* ptr1 = ptr0 + half;
                const __fp16* cos_ptr1 = cos_ptr + cw;
                const __fp16* sin_ptr1 = sin_ptr + cw;
                __fp16* outptr0 = out_head.row<__fp16>(i);
                __fp16* outptr1 = outptr0 + half;

#if __riscv_vector
                int n = half;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m4(n);

                    vfloat16m4_t _x0 = __riscv_vle16_v_f16m4(ptr0, vl);
                    vfloat16m4_t _x1 = __riscv_vle16_v_f16m4(ptr1, vl);
                    vfloat16m4_t _c0 = __riscv_vle16_v_f16m4(cos_ptr, vl);
                    vfloat16m4_t _s0 = __riscv_vle16_v_f16m4(sin_ptr, vl);
                    vfloat16m4_t _c1 = __riscv_vle16_v_f16m4(cos_ptr1, vl);
                    vfloat16m4_t _s1 = __riscv_vle16_v_f16m4(sin_ptr1, vl);

                    vfloat16m4_t _y0 = __riscv_vfnmsac_vv_f16m4(__riscv_vfmul_vv_f16m4(_x0, _c0, vl), _x1, _s0, vl); // x0*c0 - x1*s0
                    vfloat16m4_t _y1 = __riscv_vfmacc_vv_f16m4(__riscv_vfmul_vv_f16m4(_x1, _c1, vl), _x0, _s1, vl);  // x1*c1 + x0*s1
                    __riscv_vse16_v_f16m4(outptr0, _y0, vl);
                    __riscv_vse16_v_f16m4(outptr1, _y1, vl);

                    ptr0 += vl;
                    ptr1 += vl;
                    cos_ptr += vl;
                    sin_ptr += vl;
                    cos_ptr1 += vl;
                    sin_ptr1 += vl;
                    outptr0 += vl;
                    outptr1 += vl;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int j = 0; j < half; j++)
                {
                    const __fp16 x0 = *ptr0++;
                    const __fp16 x1 = *ptr1++;
                    *outptr0++ = x0 * *cos_ptr++ - x1 * *sin_ptr++;
                    *outptr1++ = x0 * *sin_ptr1++ + x1 * *cos_ptr1++;
                }
#endif // __riscv_vector
            }
        }
    }

    return 0;
}
#endif // NCNN_ZFH

} // namespace ncnn
