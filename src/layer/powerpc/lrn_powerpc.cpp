// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "lrn_powerpc.h"

#if __ALTIVEC__
#include <altivec.h>
#include "powerpc_usability.h"
#include "powerpc_mathfun.h"
#endif // __ALTIVEC__

#include <vector>

namespace ncnn {

LRN_powerpc::LRN_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
#endif // __ALTIVEC__
}

int LRN_powerpc::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if __ALTIVEC__
    if (bottom_top_blob.elembits() == 32 && bottom_top_blob.elempack == 1)
    {
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
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = __ncnn_altivec_load(ptr);
                v4f32 _outp = __ncnn_altivec_mul(_p, _p);
                __ncnn_altivec_store(outptr, _outp);

                ptr += 4;
                outptr += 4;
            }
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
                for (int p = q - local_size / 2; p <= q + (local_size - 1) / 2; p++)
                {
                    if (p < 0 || p >= channels)
                        continue;

                    const float* sptr = square_blob.channel(p);
                    float* ssptr = square_sum.channel(q);

                    int i = 0;
                    for (; i + 3 < size; i += 4)
                    {
                        v4f32 _sp = __ncnn_altivec_load(sptr);
                        v4f32 _ssp = __ncnn_altivec_load(ssptr);
                        _ssp = vec_add(_ssp, _sp);
                        __ncnn_altivec_store(ssptr, _ssp);

                        sptr += 4;
                        ssptr += 4;
                    }
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
                v4f32 _bias = vec_splats(bias);
                v4f32 _ads = vec_splats(alpha_div_size);
                v4f32 _mb = vec_splats(-beta);
                for (; i + 3 < size; i += 4)
                {
                    v4f32 _p = __ncnn_altivec_load(ptr);
                    v4f32 _ssp = __ncnn_altivec_load(ssptr);
                    _ssp = __ncnn_altivec_fmadd(_bias, _ssp, _ads);
                    _ssp = pow_ps(_ssp, _mb);
                    _p = __ncnn_altivec_mul(_p, _ssp);
                    __ncnn_altivec_store(ptr, _p);

                    ssptr += 4;
                    ptr += 4;
                }
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
#endif // __ALTIVEC__

    // non-fp32 / packed blobs fall back to the generic implementation
    if (bottom_top_blob.elempack != 1)
    {
        Mat bottom_top_blob_unpacked;
        Option opt_unpacked = opt;
        opt_unpacked.blob_allocator = opt.workspace_allocator;

        convert_packing(bottom_top_blob, bottom_top_blob_unpacked, 1, opt_unpacked);
        if (bottom_top_blob_unpacked.empty())
            return -100;

        int ret = LRN::forward_inplace(bottom_top_blob_unpacked, opt_unpacked);
        if (ret != 0)
            return ret;

        if (bottom_top_blob_unpacked.c % 4 == 0 && opt.use_packing_layout)
        {
            convert_packing(bottom_top_blob_unpacked, bottom_top_blob, 4, opt);
            if (bottom_top_blob.empty())
                return -100;
        }

        return 0;
    }

    return LRN::forward_inplace(bottom_top_blob, opt);
}

} // namespace ncnn
