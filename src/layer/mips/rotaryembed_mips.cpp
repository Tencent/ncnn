// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "rotaryembed_mips.h"

#if __mips_msa
#include <msa.h>
#endif // __mips_msa

#include "mips_usability.h"

namespace ncnn {

RotaryEmbed_mips::RotaryEmbed_mips()
{
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int RotaryEmbed_mips::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_blobs[0].elembits() == 16)
        return forward_bf16s(bottom_blobs, top_blobs, opt);
#endif

    return RotaryEmbed::forward(bottom_blobs, top_blobs, opt);
}

#if NCNN_BF16
int RotaryEmbed_mips::forward_bf16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& cos_cache = bottom_blobs[1];
    const Mat& sin_cache = bottom_blobs[2];

    const int embed_dim = bottom_blob.w;
    const int seqlen = bottom_blob.h;
    const int num_heads = bottom_blob.c;

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
            if (interleaved)
            {
                const unsigned short* ptr = head.row<const unsigned short>(i);
                const unsigned short* cos_ptr = cos_cache.row<const unsigned short>(i);
                const unsigned short* sin_ptr = sin_cache.row<const unsigned short>(i);
                unsigned short* outptr = out_head.row<unsigned short>(i);

                int j = 0;
#if __mips_msa
                v4i32 _signmask = __msa_fill_w(0);
                _signmask = __msa_insert_w(_signmask, 1, -1);
                _signmask = __msa_insert_w(_signmask, 3, -1);

                for (; j + 3 < embed_dim / 2; j += 4)
                {
                    v4f32 _a0 = bfloat2float_msa(ptr);
                    v4f32 _a1 = bfloat2float_msa(ptr + 4);

                    v4f32 _c4 = bfloat2float_msa(cos_ptr);
                    v4f32 _s4 = bfloat2float_msa(sin_ptr);

                    v4f32 _clo = (v4f32)__msa_ilvr_w((v4i32)_c4, (v4i32)_c4);
                    v4f32 _chi = (v4f32)__msa_ilvl_w((v4i32)_c4, (v4i32)_c4);
                    v4f32 _slo = (v4f32)__msa_ilvr_w((v4i32)_s4, (v4i32)_s4);
                    v4f32 _shi = (v4f32)__msa_ilvl_w((v4i32)_s4, (v4i32)_s4);

                    v4f32 _swap0 = (v4f32)__msa_shf_w((v4i32)_a0, _MSA_SHUFFLE(2, 3, 0, 1));
                    v4f32 _swap1 = (v4f32)__msa_shf_w((v4i32)_a1, _MSA_SHUFFLE(2, 3, 0, 1));

                    v4f32 _ac0 = __msa_fmul_w(_a0, _clo);
                    v4f32 _ac1 = __msa_fmul_w(_a1, _chi);
                    v4f32 _y0sub = __ncnn_msa_fmsub_w(_ac0, _swap0, _slo);
                    v4f32 _y0add = __ncnn_msa_fmadd_w(_ac0, _swap0, _slo);
                    v4f32 _y1sub = __ncnn_msa_fmsub_w(_ac1, _swap1, _shi);
                    v4f32 _y1add = __ncnn_msa_fmadd_w(_ac1, _swap1, _shi);

                    v4f32 _y0 = (v4f32)__msa_bsel_v((v16u8)_signmask, (v16u8)_y0sub, (v16u8)_y0add);
                    v4f32 _y1 = (v4f32)__msa_bsel_v((v16u8)_signmask, (v16u8)_y1sub, (v16u8)_y1add);

                    v4i32 _y01_bf16 = float2bfloat_msa(_y0, _y1);
                    __msa_st_h((v8i16)_y01_bf16, outptr, 0);

                    ptr += 8;
                    outptr += 8;
                    cos_ptr += 4;
                    sin_ptr += 4;
                }
#endif // __mips_msa
                for (; j < embed_dim / 2; j++)
                {
                    const float x0 = bfloat16_to_float32(ptr[0]);
                    const float x1 = bfloat16_to_float32(ptr[1]);
                    const float cos_val = bfloat16_to_float32(*cos_ptr++);
                    const float sin_val = bfloat16_to_float32(*sin_ptr++);

                    outptr[0] = float32_to_bfloat16(x0 * cos_val - x1 * sin_val);
                    outptr[1] = float32_to_bfloat16(x0 * sin_val + x1 * cos_val);

                    ptr += 2;
                    outptr += 2;
                }
            }
            else
            {
                const unsigned short* ptr0 = head.row<const unsigned short>(i);
                const unsigned short* ptr1 = ptr0 + embed_dim / 2;
                const unsigned short* cos_ptr = cos_cache.row<const unsigned short>(i);
                const unsigned short* sin_ptr = sin_cache.row<const unsigned short>(i);
                unsigned short* outptr0 = out_head.row<unsigned short>(i);
                unsigned short* outptr1 = outptr0 + embed_dim / 2;

                int j = 0;
#if __mips_msa
                for (; j + 3 < embed_dim / 2; j += 4)
                {
                    v4f32 _x0 = bfloat2float_msa(ptr0);
                    v4f32 _x1 = bfloat2float_msa(ptr1);
                    v4f32 _c = bfloat2float_msa(cos_ptr);
                    v4f32 _s = bfloat2float_msa(sin_ptr);

                    v4f32 _y0 = __ncnn_msa_fmsub_w(__msa_fmul_w(_x0, _c), _x1, _s);
                    v4f32 _y1 = __ncnn_msa_fmadd_w(__msa_fmul_w(_x1, _c), _x0, _s);

                    __msa_storel_d(float2bfloat_msa(_y0), outptr0);
                    __msa_storel_d(float2bfloat_msa(_y1), outptr1);

                    ptr0 += 4;
                    ptr1 += 4;
                    cos_ptr += 4;
                    sin_ptr += 4;
                    outptr0 += 4;
                    outptr1 += 4;
                }
#endif // __mips_msa
                for (; j < embed_dim / 2; j++)
                {
                    const float x0 = bfloat16_to_float32(*ptr0++);
                    const float x1 = bfloat16_to_float32(*ptr1++);
                    const float cos_val = bfloat16_to_float32(*cos_ptr++);
                    const float sin_val = bfloat16_to_float32(*sin_ptr++);

                    *outptr0++ = float32_to_bfloat16(x0 * cos_val - x1 * sin_val);
                    *outptr1++ = float32_to_bfloat16(x0 * sin_val + x1 * cos_val);
                }
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
