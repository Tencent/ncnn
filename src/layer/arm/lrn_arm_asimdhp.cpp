// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "lrn_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#include "neon_mathfun.h"
#endif // __ARM_NEON

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static inline float32x4_t load_fp16s(const __fp16* ptr)
{
    return vcvt_f32_f16(vld1_f16(ptr));
}

static inline void store_fp16s(__fp16* ptr, float32x4_t v)
{
    vst1_f16(ptr, vcvt_f16_f32(v));
}

static inline float load_fp16s1(const __fp16* ptr)
{
    return (float)*ptr;
}

static inline void store_fp16s1(__fp16* ptr, float v)
{
    *ptr = (__fp16)v;
}

int LRN_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
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
        const __fp16* ptr = bottom_top_blob.channel(q);
        float* outptr = square_blob.channel(q);

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            float32x4_t _p0 = vcvt_f32_f16(vget_low_f16(_p));
            float32x4_t _p1 = vcvt_f32_f16(vget_high_f16(_p));
            vst1q_f32(outptr, vmulq_f32(_p0, _p0));
            vst1q_f32(outptr + 4, vmulq_f32(_p1, _p1));
            ptr += 8;
            outptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = load_fp16s(ptr);
            vst1q_f32(outptr, vmulq_f32(_p, _p));
            ptr += 4;
            outptr += 4;
        }
        for (; i < size; i++)
        {
            float v = load_fp16s1(ptr);
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
                for (; i < size; i++)
                {
                    *ssptr += *sptr;
                    sptr++;
                    ssptr++;
                }
            }

            __fp16* ptr = bottom_top_blob.channel(q);
            float* ssptr = square_sum.channel(q);

            int i = 0;
            float32x4_t _bias = vdupq_n_f32(bias);
            float32x4_t _ads = vdupq_n_f32(alpha_div_size);
            float32x4_t _mb = vdupq_n_f32(-beta);
            for (; i + 7 < size; i += 8)
            {
                float16x8_t _p = vld1q_f16(ptr);
                float32x4_t _p0 = vcvt_f32_f16(vget_low_f16(_p));
                float32x4_t _p1 = vcvt_f32_f16(vget_high_f16(_p));
                float32x4_t _ss0 = vld1q_f32(ssptr);
                float32x4_t _ss1 = vld1q_f32(ssptr + 4);
                _ss0 = vmulq_f32(_ss0, _ads);
                _ss1 = vmulq_f32(_ss1, _ads);
                _ss0 = vaddq_f32(_ss0, _bias);
                _ss1 = vaddq_f32(_ss1, _bias);
                _ss0 = pow_ps(_ss0, _mb);
                _ss1 = pow_ps(_ss1, _mb);
                vst1q_f16(ptr, vcombine_f16(vcvt_f16_f32(vmulq_f32(_p0, _ss0)), vcvt_f16_f32(vmulq_f32(_p1, _ss1))));
                ssptr += 8;
                ptr += 8;
            }
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = load_fp16s(ptr);
                float32x4_t _ss = vld1q_f32(ssptr);
                _ss = vmulq_f32(_ss, _ads);
                _ss = vaddq_f32(_ss, _bias);
                _ss = pow_ps(_ss, _mb);
                store_fp16s(ptr, vmulq_f32(_p, _ss));
                ssptr += 4;
                ptr += 4;
            }
            for (; i < size; i++)
            {
                store_fp16s1(ptr, load_fp16s1(ptr) * powf(bias + alpha_div_size * *ssptr, -beta));
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
            __fp16* ptr = bottom_top_blob.channel(q);
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

                    store_fp16s1(ptr + j, load_fp16s1(ptr + j) * powf(bias + alpha_div_size * ss, -beta));
                }

                ptr += w;
            }
        }
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn
