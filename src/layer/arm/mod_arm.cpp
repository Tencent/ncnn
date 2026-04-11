// ARM NEON optimized implementation for Mod
// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "mod_arm.h"
#include <cmath>

#if __ARM_NEON
#include <arm_neon.h>
#endif

namespace ncnn {

#if __ARM_NEON
int Mod_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (bottom_blobs.size() < 2)
        return -1;

    const Mat& a_blob = bottom_blobs[0];
    const Mat& b_blob = bottom_blobs[1];

    Mat& top_blob = top_blobs[0];
    top_blob.create(a_blob.w, a_blob.h, a_blob.c, a_blob.elemsize, a_blob.elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const float* a = a_blob;
    const float* b = b_blob;
    float* out = top_blob;

    const int total = (int)top_blob.total();

    // ARM NEON optimized path
    if (opt.num_threads > 1)
    {
        const int nn = total >> 2;
        const int remain = total - (nn << 2);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < nn; i++)
        {
            int idx = i << 2;
            
            // Load 4 values
            float32x4_t a_vec = vld1q_f32(a + idx);
            float32x4_t b_vec = vld1q_f32(b + idx);
            
            // Check for zero divisor
            uint32x4_t zero_mask = vceqq_f32(b_vec, vdupq_n_f32(0.0f));
            
            float32x4_t out_vec;
            float out_arr[4];
            
            if (fmod == 0)
            {
                // Python-style modulo: result has same sign as divisor
                // Use fmodf and adjust sign
                for (int j = 0; j < 4; j++)
                {
                    if (b_vec[j] == 0.0f)
                    {
                        out_arr[j] = 0.0f;
                    }
                    else
                    {
                        float result = std::fmod(a_vec[j], b_vec[j]);
                        if ((result != 0.0f) && ((b_vec[j] < 0.0f) != (result < 0.0f)))
                        {
                            result += b_vec[j];
                        }
                        out_arr[j] = result;
                    }
                }
                out_vec = vld1q_f32(out_arr);
            }
            else
            {
                // C-style fmod: result has same sign as dividend
                for (int j = 0; j < 4; j++)
                {
                    out_arr[j] = (b_vec[j] == 0.0f) ? 0.0f : std::fmod(a_vec[j], b_vec[j]);
                }
                out_vec = vld1q_f32(out_arr);
            }
            
            // Apply zero mask
            out_vec = vbslq_f32(vmvnq_u32(zero_mask), out_vec, vdupq_n_f32(0.0f));
            
            vst1q_f32(out + idx, out_vec);
        }

        // Handle remaining elements
        for (int i = nn << 2; i < total; i++)
        {
            if (b[i] == 0.0f)
            {
                out[i] = 0.0f;
            }
            else if (fmod == 0)
            {
                float result = std::fmod(a[i], b[i]);
                if ((result != 0.0f) && ((b[i] < 0.0f) != (result < 0.0f)))
                {
                    result += b[i];
                }
                out[i] = result;
            }
            else
            {
                out[i] = std::fmod(a[i], b[i]);
            }
        }

        return 0;
    }

    // Scalar path
    if (fmod == 0)
    {
        for (int i = 0; i < total; i++)
        {
            if (b[i] == 0.0f)
            {
                out[i] = 0.0f;
            }
            else
            {
                float result = std::fmod(a[i], b[i]);
                if ((result != 0.0f) && ((b[i] < 0.0f) != (result < 0.0f)))
                {
                    result += b[i];
                }
                out[i] = result;
            }
        }
    }
    else
    {
        for (int i = 0; i < total; i++)
        {
            out[i] = (b[i] == 0.0f) ? 0.0f : std::fmod(a[i], b[i]);
        }
    }

    return 0;
}
#else
int Mod_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    return Mod::forward(bottom_blobs, top_blobs, opt);
}
#endif

} // namespace ncnn
