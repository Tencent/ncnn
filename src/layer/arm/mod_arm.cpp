// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "mod_arm.h"
#include <cmath>

#if __ARM_NEON
#include <arm_neon.h>
#endif

namespace ncnn {

int Mod_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (bottom_blobs.size() < 2)
        return -1;

    const Mat& a_blob = bottom_blobs[0];
    const Mat& b_blob = bottom_blobs[1];

    // Output has same shape as a_blob
    const Mat& out_shape = a_blob;

    Mat& top_blob = top_blobs[0];
    top_blob.create(out_shape.w, out_shape.h, out_shape.c, a_blob.elemsize, a_blob.elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const float* a = a_blob;
    const float* b = b_blob;
    float* out = top_blob;

    const int total = (int)top_blob.total();

#if __ARM_NEON
    // ARM NEON optimized path - process 4 elements at a time
    const int nn = total >> 2;
    const int remain = total - (nn << 2);

    if (fmod == 0)
    {
        // Python-style modulo
        for (int i = 0; i < nn; i++)
        {
            int idx = i << 2;
            
            float32x4_t a_vec = vld1q_f32(a + idx);
            float32x4_t b_vec = vld1q_f32(b + idx);
            
            // Check for zero divisor
            uint32x4_t zero_mask = vceqq_f32(b_vec, vdupq_n_f32(0.0f));
            
            // Compute fmod
            float result[4];
            for (int j = 0; j < 4; j++)
            {
                if (b_vec[j] == 0.0f)
                {
                    result[j] = 0.0f;
                }
                else
                {
                    float res = std::fmod(a_vec[j], b_vec[j]);
                    // Python-style: result has same sign as divisor
                    if ((res != 0.0f) && ((b_vec[j] < 0.0f) != (res < 0.0f)))
                    {
                        res += b_vec[j];
                    }
                    result[j] = res;
                }
            }
            
            vst1q_f32(out + idx, vld1q_f32(result));
        }
    }
    else
    {
        // C-style fmod
        for (int i = 0; i < nn; i++)
        {
            int idx = i << 2;
            
            float32x4_t a_vec = vld1q_f32(a + idx);
            float32x4_t b_vec = vld1q_f32(b + idx);
            
            // Check for zero divisor
            uint32x4_t zero_mask = vceqq_f32(b_vec, vdupq_n_f32(0.0f));
            
            // Compute fmod
            float result[4];
            for (int j = 0; j < 4; j++)
            {
                if (b_vec[j] == 0.0f)
                {
                    result[j] = 0.0f;
                }
                else
                {
                    result[j] = std::fmod(a_vec[j], b_vec[j]);
                }
            }
            
            vst1q_f32(out + idx, vld1q_f32(result));
        }
    }

    // Handle remaining elements
    for (int i = 0; i < remain; i++)
    {
        int idx = (nn << 2) + i;
        float val_a = a[idx];
        float val_b = b[idx];
        
        if (val_b == 0.0f)
        {
            out[idx] = 0.0f;
        }
        else if (fmod == 0)
        {
            float result = std::fmod(val_a, val_b);
            if ((result != 0.0f) && ((val_b < 0.0f) != (result < 0.0f)))
            {
                result += val_b;
            }
            out[idx] = result;
        }
        else
        {
            out[idx] = std::fmod(val_a, val_b);
        }
    }
#else
    // Scalar fallback with OpenMP
    if (fmod == 0)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < total; i++)
        {
            float val_a = a[i];
            float val_b = b[i];
            
            if (val_b == 0.0f)
            {
                out[i] = 0.0f;
            }
            else
            {
                float result = std::fmod(val_a, val_b);
                if ((result != 0.0f) && ((val_b < 0.0f) != (result < 0.0f)))
                {
                    result += val_b;
                }
                out[i] = result;
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < total; i++)
        {
            float val_a = a[i];
            float val_b = b[i];
            
            if (val_b == 0.0f)
            {
                out[i] = 0.0f;
            }
            else
            {
                out[i] = std::fmod(val_a, val_b);
            }
        }
    }
#endif // __ARM_NEON

    return 0;
}

} // namespace ncnn
