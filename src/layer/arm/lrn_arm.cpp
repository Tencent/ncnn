// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "lrn_arm.h"

#include "cpu.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#include "neon_mathfun.h"
#endif // __ARM_NEON

namespace ncnn {

#if NCNN_BF16
#if __ARM_NEON
static inline float32x4_t load_bf16s(const unsigned short* ptr)
{
    return bfloat2float(vld1_u16(ptr));
}

static inline void store_bf16s(unsigned short* ptr, float32x4_t v)
{
    vst1_u16(ptr, float2bfloat(v));
}
#endif // __ARM_NEON

static inline float load_bf16s1(const unsigned short* ptr)
{
    return bfloat16_to_float32(*ptr);
}

static inline void store_bf16s1(unsigned short* ptr, float v)
{
    *ptr = float32_to_bfloat16(v);
}

int LRN_arm::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int channels = bottom_top_blob.c;
    const int size = w * h;

    // squared values are always accumulated in fp32
    Mat square_blob;
    square_blob.create(w, h, channels, 4u, opt.workspace_allocator);
    if (square_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const unsigned short* ptr = bottom_top_blob.channel(q);
        float* outptr = square_blob.channel(q);

        int i = 0;
#if __ARM_NEON
        for (; i + 7 < size; i += 8)
        {
            uint16x8_t _p = vld1q_u16(ptr);
            float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
            float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
            vst1q_f32(outptr, vmulq_f32(_p0, _p0));
            vst1q_f32(outptr + 4, vmulq_f32(_p1, _p1));
            ptr += 8;
            outptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = load_bf16s(ptr);
            vst1q_f32(outptr, vmulq_f32(_p, _p));
            ptr += 4;
            outptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            float v = load_bf16s1(ptr);
            *outptr = v * v;
            ptr++;
            outptr++;
        }
    }

    if (region_type == LRN::NormRegion_ACROSS_CHANNELS)
    {
        Mat square_sum;
        square_sum.create(w, h, channels, 4u, opt.workspace_allocator);
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
#if __ARM_NEON
                for (; i + 7 < size; i += 8)
                {
                    float32x4_t _s0 = vld1q_f32(sptr);
                    float32x4_t _s1 = vld1q_f32(sptr + 4);
                    float32x4_t _ss0 = vld1q_f32(ssptr);
                    float32x4_t _ss1 = vld1q_f32(ssptr + 4);
                    vst1q_f32(ssptr, vaddq_f32(_ss0, _s0));
                    vst1q_f32(ssptr + 4, vaddq_f32(_ss1, _s1));
                    sptr += 8;
                    ssptr += 8;
                }
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _s = vld1q_f32(sptr);
                    float32x4_t _ss = vld1q_f32(ssptr);
                    vst1q_f32(ssptr, vaddq_f32(_ss, _s));
                    sptr += 4;
                    ssptr += 4;
                }
#endif // __ARM_NEON
                for (; i < size; i++)
                {
                    *ssptr += *sptr;
                    sptr++;
                    ssptr++;
                }
            }

            unsigned short* ptr = bottom_top_blob.channel(q);
            float* ssptr = square_sum.channel(q);

            int i = 0;
#if __ARM_NEON
            float32x4_t _bias = vdupq_n_f32(bias);
            float32x4_t _ads = vdupq_n_f32(alpha_div_size);
            float32x4_t _mb = vdupq_n_f32(-beta);
            for (; i + 7 < size; i += 8)
            {
                uint16x8_t _p = vld1q_u16(ptr);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _ss0 = vld1q_f32(ssptr);
                float32x4_t _ss1 = vld1q_f32(ssptr + 4);
                _ss0 = vmulq_f32(_ss0, _ads);
                _ss1 = vmulq_f32(_ss1, _ads);
                _ss0 = vaddq_f32(_ss0, _bias);
                _ss1 = vaddq_f32(_ss1, _bias);
                _ss0 = pow_ps(_ss0, _mb);
                _ss1 = pow_ps(_ss1, _mb);
                vst1q_u16(ptr, vcombine_u16(float2bfloat(vmulq_f32(_p0, _ss0)), float2bfloat(vmulq_f32(_p1, _ss1))));
                ssptr += 8;
                ptr += 8;
            }
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = load_bf16s(ptr);
                float32x4_t _ss = vld1q_f32(ssptr);
                _ss = vmulq_f32(_ss, _ads);
                _ss = vaddq_f32(_ss, _bias);
                _ss = pow_ps(_ss, _mb);
                store_bf16s(ptr, vmulq_f32(_p, _ss));
                ssptr += 4;
                ptr += 4;
            }
#endif // __ARM_NEON
            for (; i < size; i++)
            {
                store_bf16s1(ptr, load_bf16s1(ptr) * powf(bias + alpha_div_size * *ssptr, -beta));
                ssptr++;
                ptr++;
            }
        }
    }
    else if (region_type == LRN::NormRegion_WITHIN_CHANNEL)
    {
        int outw = w;

        Mat square_blob_bordered = square_blob;
        int pad = local_size / 2;
        if (pad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(square_blob, square_blob_bordered, pad, local_size - pad - 1, pad, local_size - pad - 1, BORDER_CONSTANT, 0.f, opt_b);
            if (square_blob_bordered.empty())
                return -100;

            outw = square_blob_bordered.w;
        }

        const int maxk = local_size * local_size;
        const float alpha_div_size = alpha / maxk;

        std::vector<int> _space_ofs(maxk);
        int* space_ofs = &_space_ofs[0];
        {
            int p1 = 0;
            int p2 = 0;
            int gap = outw - local_size;
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
            unsigned short* ptr = bottom_top_blob.channel(q);
            const Mat m = square_blob_bordered.channel(q);

            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    const float* sptr = m.row(i) + j;

                    float ss = 0.f;
                    for (int k = 0; k < maxk; k++)
                    {
                        ss += sptr[space_ofs[k]];
                    }

                    store_bf16s1(ptr + j, load_bf16s1(ptr + j) * powf(bias + alpha_div_size * ss, -beta));
                }

                ptr += w;
            }
        }
    }

    return 0;
}

#endif // NCNN_BF16

LRN_arm::LRN_arm()
{
#if __ARM_NEON
#if NCNN_ARM82
    support_fp16_storage = cpu_support_arm_asimdhp();
#endif
#endif // __ARM_NEON
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int LRN_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    const int elembits = bottom_top_blob.elembits();

#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
        return forward_inplace_fp16s(bottom_top_blob, opt);
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    size_t elemsize = bottom_top_blob.elemsize;
    int size = w * h;

    // squared values with local_size padding
    Mat square_blob;
    square_blob.create(w, h, channels, elemsize, opt.workspace_allocator);
    if (square_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = bottom_top_blob.channel(q);
        float* outptr = square_blob.channel(q);

#if __ARM_NEON
        int nn = size >> 2;
        int remain = size - (nn << 2);
#else
        int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
        for (; nn > 0; nn--)
        {
            float32x4_t _p = vld1q_f32(ptr);
            float32x4_t _outp = vmulq_f32(_p, _p);
            vst1q_f32(outptr, _outp);

            ptr += 4;
            outptr += 4;
        }
#endif // __ARM_NEON
        for (; remain > 0; remain--)
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
            // square sum
            for (int p = q - local_size / 2; p <= q + (local_size - 1) / 2; p++)
            {
                if (p < 0 || p >= channels)
                    continue;

                const float* sptr = square_blob.channel(p);
                float* ssptr = square_sum.channel(q);

#if __ARM_NEON
                int nn = size >> 2;
                int remain = size - (nn << 2);
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
                for (; nn > 0; nn--)
                {
                    float32x4_t _sp = vld1q_f32(sptr);
                    float32x4_t _ssp = vld1q_f32(ssptr);
                    _ssp = vaddq_f32(_ssp, _sp);
                    vst1q_f32(ssptr, _ssp);

                    sptr += 4;
                    ssptr += 4;
                }
#endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
                    *ssptr += *sptr;
                    sptr++;
                    ssptr++;
                }
            }

            float* ptr = bottom_top_blob.channel(q);
            float* ssptr = square_sum.channel(q);

#if __ARM_NEON
            int nn = size >> 2;
            int remain = size - (nn << 2);
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
            float32x4_t _bias = vdupq_n_f32(bias);
            float32x4_t _ads = vdupq_n_f32(alpha_div_size);
            float32x4_t _mb = vdupq_n_f32(-beta);
            for (; nn > 0; nn--)
            {
                float32x4_t _p = vld1q_f32(ptr);
                float32x4_t _ssp = vld1q_f32(ssptr);
                _ssp = vmulq_f32(_ssp, _ads);
                _ssp = vaddq_f32(_ssp, _bias);
                _ssp = pow_ps(_ssp, _mb);
                _p = vmulq_f32(_p, _ssp);
                vst1q_f32(ptr, _p);

                ssptr += 4;
                ptr += 4;
            }
#endif // __ARM_NEON
            for (; remain > 0; remain--)
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
            copy_make_border(square_blob, square_blob_bordered, pad, local_size - pad - 1, pad, local_size - pad - 1, BORDER_CONSTANT, 0.f, opt_b);
            if (square_blob_bordered.empty())
                return -100;

            w = square_blob_bordered.w;
            h = square_blob_bordered.h;
        }

        const int maxk = local_size * local_size;

        const float alpha_div_size = alpha / maxk;

        // norm window offsets
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
                        float val = sptr[space_ofs[k]];
                        ss += val;
                    }

                    ptr[j] = ptr[j] * powf(bias + alpha_div_size * ss, -beta);
                }

                ptr += outw;
            }
        }
    }

    return 0;
}

} // namespace ncnn
