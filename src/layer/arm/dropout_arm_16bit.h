// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

template<typename Conv>
static int dropout_forward_inplace_16bit(Mat& bottom_top_blob, float scale, const Option& opt)
{
    if (scale == 1.f)
        return 0;

    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;
    const int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __ARM_NEON
        float32x4_t _scale = vdupq_n_f32(scale);
        for (; i + 15 < size; i += 16)
        {
            float32x4_t _p0 = Conv::load4(ptr);
            float32x4_t _p1 = Conv::load4(ptr + 4);
            float32x4_t _p2 = Conv::load4(ptr + 8);
            float32x4_t _p3 = Conv::load4(ptr + 12);
            Conv::store4(ptr, vmulq_f32(_p0, _scale));
            Conv::store4(ptr + 4, vmulq_f32(_p1, _scale));
            Conv::store4(ptr + 8, vmulq_f32(_p2, _scale));
            Conv::store4(ptr + 12, vmulq_f32(_p3, _scale));
            ptr += 16;
        }
        for (; i + 7 < size; i += 8)
        {
            float32x4_t _p0 = Conv::load4(ptr);
            float32x4_t _p1 = Conv::load4(ptr + 4);
            Conv::store4(ptr, vmulq_f32(_p0, _scale));
            Conv::store4(ptr + 4, vmulq_f32(_p1, _scale));
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = Conv::load4(ptr);
            Conv::store4(ptr, vmulq_f32(_p, _scale));
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            Conv::store1(ptr, Conv::load1(ptr) * scale);
            ptr++;
        }
    }

    return 0;
}
