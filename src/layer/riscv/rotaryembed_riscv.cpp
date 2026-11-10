// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "rotaryembed_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "cpu.h"
#include "riscv_usability.h"

namespace ncnn {

#if NCNN_BF16
#include "rotaryembed_bf16s.h"
#endif

RotaryEmbed_riscv::RotaryEmbed_riscv()
{
#if NCNN_ZFH
#if __riscv_vector
    support_fp16_storage = cpu_support_riscv_zvfh();
#else
    support_fp16_storage = cpu_support_riscv_zfh();
#endif
#endif
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int RotaryEmbed_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if NCNN_ZFH
    if (support_fp16_storage && opt.use_fp16_storage && bottom_blobs[0].elembits() == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_fp16sa(bottom_blobs, top_blobs, opt);
        else
            return forward_fp16s(bottom_blobs, top_blobs, opt);
    }
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_blobs[0].elembits() == 16)
        return forward_bf16s(bottom_blobs, top_blobs, opt);
#endif

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
            const float* cos_ptr = cos_cache.row(i);
            const float* sin_ptr = sin_cache.row(i);

            if (interleaved)
            {
                const float* ptr = head.row(i);
                float* outptr = out_head.row(i);

#if __riscv_vector
                int n = half;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m4(n);

                    vfloat32m4x2_t _p = __riscv_vlseg2e32_v_f32m4x2(ptr, vl);
                    vfloat32m4_t _x0 = __riscv_vget_v_f32m4x2_f32m4(_p, 0);
                    vfloat32m4_t _x1 = __riscv_vget_v_f32m4x2_f32m4(_p, 1);
                    vfloat32m4_t _c = __riscv_vle32_v_f32m4(cos_ptr, vl);
                    vfloat32m4_t _s = __riscv_vle32_v_f32m4(sin_ptr, vl);

                    vfloat32m4_t _y0 = __riscv_vfnmsac_vv_f32m4(__riscv_vfmul_vv_f32m4(_x0, _c, vl), _x1, _s, vl); // x0*c - x1*s
                    vfloat32m4_t _y1 = __riscv_vfmacc_vv_f32m4(__riscv_vfmul_vv_f32m4(_x1, _c, vl), _x0, _s, vl);  // x0*s + x1*c
                    __riscv_vsseg2e32_v_f32m4x2(outptr, __riscv_vcreate_v_f32m4x2(_y0, _y1), vl);

                    ptr += vl * 2;
                    outptr += vl * 2;
                    cos_ptr += vl;
                    sin_ptr += vl;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int j = 0; j < half; j++)
                {
                    const float x0 = ptr[0];
                    const float x1 = ptr[1];
                    const float cos_val = *cos_ptr++;
                    const float sin_val = *sin_ptr++;
                    outptr[0] = x0 * cos_val - x1 * sin_val;
                    outptr[1] = x0 * sin_val + x1 * cos_val;
                    ptr += 2;
                    outptr += 2;
                }
#endif // __riscv_vector
            }
            else
            {
                // A full-width (embed_dim) cache carries independent cos/sin for the two halves
                // (2D / vision rope); a half-width (embed_dim/2) cache is the standard-rope form
                // whose halves are identical, so the second half reuses the first.
                const int cw = cos_cache.w == embed_dim ? half : 0;
                const float* ptr0 = head.row(i);
                const float* ptr1 = ptr0 + half;
                const float* cos_ptr1 = cos_ptr + cw;
                const float* sin_ptr1 = sin_ptr + cw;
                float* outptr0 = out_head.row(i);
                float* outptr1 = outptr0 + half;

#if __riscv_vector
                int n = half;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m4(n);

                    vfloat32m4_t _x0 = __riscv_vle32_v_f32m4(ptr0, vl);
                    vfloat32m4_t _x1 = __riscv_vle32_v_f32m4(ptr1, vl);
                    vfloat32m4_t _c0 = __riscv_vle32_v_f32m4(cos_ptr, vl);
                    vfloat32m4_t _s0 = __riscv_vle32_v_f32m4(sin_ptr, vl);
                    vfloat32m4_t _c1 = __riscv_vle32_v_f32m4(cos_ptr1, vl);
                    vfloat32m4_t _s1 = __riscv_vle32_v_f32m4(sin_ptr1, vl);

                    vfloat32m4_t _y0 = __riscv_vfnmsac_vv_f32m4(__riscv_vfmul_vv_f32m4(_x0, _c0, vl), _x1, _s0, vl); // x0*c0 - x1*s0
                    vfloat32m4_t _y1 = __riscv_vfmacc_vv_f32m4(__riscv_vfmul_vv_f32m4(_x1, _c1, vl), _x0, _s1, vl);  // x1*c1 + x0*s1
                    __riscv_vse32_v_f32m4(outptr0, _y0, vl);
                    __riscv_vse32_v_f32m4(outptr1, _y1, vl);

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
                    const float x0 = *ptr0++;
                    const float x1 = *ptr1++;
                    *outptr0++ = x0 * *cos_ptr++ - x1 * *sin_ptr++;
                    *outptr1++ = x0 * *sin_ptr1++ + x1 * *cos_ptr1++;
                }
#endif // __riscv_vector
            }
        }
    }

    return 0;
}

#if NCNN_BF16
int RotaryEmbed_riscv::forward_bf16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& cos_cache = bottom_blobs[1];
    const Mat& sin_cache = bottom_blobs[2];

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    rotaryembed_bf16s(bottom_blob, cos_cache, sin_cache, top_blob, interleaved, opt);

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
