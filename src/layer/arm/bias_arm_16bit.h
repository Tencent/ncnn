// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

template<typename Conv>
static int bias_forward_inplace_16bit(Mat& bottom_top_blob, const float* bias_data, const Option& opt)
{
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int size = w * h * d;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = bottom_top_blob.channel(q);

        const float bias = bias_data[q];

        int i = 0;
#if __ARM_NEON
        float32x4_t _bias = vdupq_n_f32(bias);
        for (; i + 7 < size; i += 8)
        {
            float32x4_t _p0 = Conv::load4(ptr);
            float32x4_t _p1 = Conv::load4(ptr + 4);
            Conv::store4(ptr, vaddq_f32(_p0, _bias));
            Conv::store4(ptr + 4, vaddq_f32(_p1, _bias));
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = Conv::load4(ptr);
            Conv::store4(ptr, vaddq_f32(_p, _bias));
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            Conv::store1(ptr, Conv::load1(ptr) + bias);
            ptr++;
        }
    }

    return 0;
}
