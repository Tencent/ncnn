#include "linearint8_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

int LinearInt8_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (bottom_blob.dims != 2 || bottom_blob.w != in_dim)
        return -1;

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;

    top_blob.create(out_dim, h, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int8_t* wt = (const int8_t*)weight;

#if (__ARM_NEON && __aarch64__)

    float zero = 0.0f;

    if (!(w % group_size) && !(group_size % 8))
    {
        for (int j = 0; j < h; j++)
        {
            const float* m = bottom_blob.row(j);
            float* out = top_blob.row(j);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < out_dim; p++)
            {
                int base = w * p;
                float32x4_t acc_p0 = vld1q_dup_f32(&zero), acc_p1 = vld1q_dup_f32(&zero);
                for (int k = 0; k < w; k += group_size)
                {
                    int scales_index = (base + k) / group_size;
                    int index = base + k;
                    const float* sc = (const float*)scales + scales_index;
                    for (int i = 0, ind = index; i < group_size; i += 8, ind += 8)
                    {
                        int8x8_t i8x8 = vld1_s8(wt + ind);
                        int16x8_t i16x8 = vmovl_s8(i8x8);
                        int32x4_t i32_0 = vmovl_s16(vget_low_s16(i16x8));
                        int32x4_t i32_1 = vmovl_s16(vget_high_s16(i16x8));
                        float32x4_t wt_p0 = vcvtq_f32_s32(i32_0);
                        float32x4_t wt_p1 = vcvtq_f32_s32(i32_1);
                        float32x4_t m_p0 = vld1q_f32(m + k + i);
                        float32x4_t m_p1 = vld1q_f32(m + k + i + 4);
                        float32x4_t sc_p = vld1q_dup_f32(sc);
                        float32x4_t acc_real0 = vmulq_f32(wt_p0, sc_p);
                        float32x4_t acc_real1 = vmulq_f32(wt_p1, sc_p);
                        acc_p0 = vmlaq_f32(acc_p0, m_p0, acc_real0);
                        acc_p1 = vmlaq_f32(acc_p1, m_p1, acc_real1);
                    }
                }
                out[p] = vaddvq_f32(acc_p0) + vaddvq_f32(acc_p1);
            }
        }
        return 0;
    }
#endif

    for (int j = 0; j < h; j++)
    {
        const float* m = bottom_blob.row(j);
        float* out = top_blob.row(j);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < out_dim; p++)
        {
            int base = w * p;
            float acc = 0.0f;
            for (int i = 0, index = base, scales_index = index / group_size; i < w; i++, index++)
            {
                acc += m[i] * wt[index] * scales[scales_index];
                if (index % group_size == group_size - 1) scales_index++;
            }
            out[p] = acc;
        }
    }

    return 0;
}

} // namespace ncnn
