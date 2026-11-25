// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

template<typename Conv>
static int lrn_forward_inplace_16bit(Mat& bottom_top_blob, const Option& opt, int region_type, int local_size, float alpha, float beta, float bias)
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
            float32x4_t _p0 = Conv::load4(ptr);
            float32x4_t _p1 = Conv::load4(ptr + 4);
            vst1q_f32(outptr, vmulq_f32(_p0, _p0));
            vst1q_f32(outptr + 4, vmulq_f32(_p1, _p1));
            ptr += 8;
            outptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = Conv::load4(ptr);
            vst1q_f32(outptr, vmulq_f32(_p, _p));
            ptr += 4;
            outptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            float v = Conv::load1(ptr);
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
                float32x4_t _p0 = Conv::load4(ptr);
                float32x4_t _p1 = Conv::load4(ptr + 4);
                float32x4_t _ss0 = vld1q_f32(ssptr);
                float32x4_t _ss1 = vld1q_f32(ssptr + 4);
                _ss0 = vmulq_f32(_ss0, _ads);
                _ss1 = vmulq_f32(_ss1, _ads);
                _ss0 = vaddq_f32(_ss0, _bias);
                _ss1 = vaddq_f32(_ss1, _bias);
                _ss0 = pow_ps(_ss0, _mb);
                _ss1 = pow_ps(_ss1, _mb);
                Conv::store4(ptr, vmulq_f32(_p0, _ss0));
                Conv::store4(ptr + 4, vmulq_f32(_p1, _ss1));
                ssptr += 8;
                ptr += 8;
            }
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = Conv::load4(ptr);
                float32x4_t _ss = vld1q_f32(ssptr);
                _ss = vmulq_f32(_ss, _ads);
                _ss = vaddq_f32(_ss, _bias);
                _ss = pow_ps(_ss, _mb);
                Conv::store4(ptr, vmulq_f32(_p, _ss));
                ssptr += 4;
                ptr += 4;
            }
#endif // __ARM_NEON
            for (; i < size; i++)
            {
                Conv::store1(ptr, Conv::load1(ptr) * powf(bias + alpha_div_size * *ssptr, -beta));
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

                    Conv::store1(ptr + j, Conv::load1(ptr + j) * powf(bias + alpha_div_size * ss, -beta));
                }

                ptr += w;
            }
        }
    }

    return 0;
}
