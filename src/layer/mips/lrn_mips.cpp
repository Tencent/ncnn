// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "lrn_mips.h"

#if __mips_msa
#include <msa.h>
#include "msa_mathfun.h"
#endif // __mips_msa

namespace ncnn {

LRN_mips::LRN_mips()
{
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int LRN_mips::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    size_t elemsize = bottom_top_blob.elemsize;
    int size = w * h;

    Mat square_blob;
    square_blob.create(w, h, channels, elemsize, opt.workspace_allocator);
    if (square_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = bottom_top_blob.channel(q);
        float* outptr = square_blob.channel(q);

        int i = 0;
#if __mips_msa
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            v4f32 _outp = __msa_fmul_w(_p, _p);
            __msa_st_w((v4i32)_outp, outptr, 0);

            ptr += 4;
            outptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            *outptr = *ptr * *ptr;
            ptr++;
            outptr++;
        }
    }

    if (region_type == NormRegion_ACROSS_CHANNELS)
    {
        Mat square_sum;
        square_sum.create(w, h, channels, elemsize, opt.workspace_allocator);
        if (square_sum.empty())
            return -100;
        square_sum.fill(0.f);

        const float alpha_div_size = alpha / local_size;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            for (int p = q - local_size / 2; p <= q + local_size / 2; p++)
            {
                if (p < 0 || p >= channels)
                    continue;

                const float* sptr = square_blob.channel(p);
                float* ssptr = square_sum.channel(q);

                int i = 0;
#if __mips_msa
                for (; i + 3 < size; i += 4)
                {
                    v4f32 _sp = (v4f32)__msa_ld_w(sptr, 0);
                    v4f32 _ssp = (v4f32)__msa_ld_w(ssptr, 0);
                    _ssp = __msa_fadd_w(_ssp, _sp);
                    __msa_st_w((v4i32)_ssp, ssptr, 0);

                    sptr += 4;
                    ssptr += 4;
                }
#endif // __mips_msa
                for (; i < size; i++)
                {
                    *ssptr += *sptr;
                    sptr++;
                    ssptr++;
                }
            }

            float* ptr = bottom_top_blob.channel(q);
            float* ssptr = square_sum.channel(q);

            int i = 0;
#if __mips_msa
            v4f32 _bias = (v4f32)__msa_fill_w_f32(bias);
            v4f32 _ads = (v4f32)__msa_fill_w_f32(alpha_div_size);
            v4f32 _mb = (v4f32)__msa_fill_w_f32(-beta);
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                v4f32 _ssp = (v4f32)__msa_ld_w(ssptr, 0);
                _ssp = __ncnn_msa_fmadd_w(_bias, _ssp, _ads);
                _ssp = pow_ps(_ssp, _mb);
                _p = __msa_fmul_w(_p, _ssp);
                __msa_st_w((v4i32)_p, ptr, 0);

                ssptr += 4;
                ptr += 4;
            }
#endif // __mips_msa
            for (; i < size; i++)
            {
                *ptr = *ptr * powf(bias + alpha_div_size * *ssptr, -beta);
                ssptr++;
                ptr++;
            }
        }
    }
    else if (region_type == NormRegion_WITHIN_CHANNEL)
    {
        int outw = w;
        int outh = h;

        Mat square_blob_bordered = square_blob;
        int pad = local_size / 2;
        if (pad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            opt_b.use_packing_layout = false;
            copy_make_border(square_blob, square_blob_bordered, pad, local_size - pad - 1, pad, local_size - pad - 1, BORDER_CONSTANT, 0.f, opt_b);
            if (square_blob_bordered.empty())
                return -100;

            w = square_blob_bordered.w;
            h = square_blob_bordered.h;
        }

        const int maxk = local_size * local_size;
        const float alpha_div_size = alpha / maxk;

        std::vector<int> _space_ofs(maxk);
        int* space_ofs = &_space_ofs[0];
        {
            int p1 = 0;
            int p2 = 0;
            int gap = w - local_size;
            for (int i = 0; i < local_size; i++)
            {
                for (int j = 0; j < local_size; j++)
                {
                    space_ofs[p1] = p2;
                    p1++;
                    p2++;
                }
                p2 += gap;
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            const Mat m = square_blob_bordered.channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const float* sptr = m.row(i) + j;

                    float ss = 0.f;
                    for (int k = 0; k < maxk; k++)
                    {
                        ss += sptr[space_ofs[k]];
                    }

                    ptr[j] = ptr[j] * powf(bias + alpha_div_size * ss, -beta);
                }

                ptr += outw;
            }
        }
    }

    return 0;
}

#if NCNN_BF16
int LRN_mips::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    Option opt_cast = opt;
    opt_cast.blob_allocator = opt.workspace_allocator;

    Mat bottom_top_blob_fp32;
    cast_bfloat16_to_float32(bottom_top_blob, bottom_top_blob_fp32, opt_cast);
    if (bottom_top_blob_fp32.empty())
        return -100;

    int ret = forward_inplace(bottom_top_blob_fp32, opt);
    if (ret != 0)
        return ret;

    cast_float32_to_bfloat16(bottom_top_blob_fp32, bottom_top_blob, opt);
    if (bottom_top_blob.empty())
        return -100;

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
