// Highly optimized ARM NEON implementation for Mod
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

    // HOT PATH: C-style fmod with ARM NEON - process 8 elements at once
    if (fmod == 1 && opt.num_threads > 1)
    {
        const int nn = total >> 3;
        const int remain = total - (nn << 3);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < nn; i++)
        {
            int idx = i << 3;
            
            // Load 8 values (2x float32x4)
            float32x4_t a0 = vld1q_f32(a + idx);
            float32x4_t a1 = vld1q_f32(a + idx + 4);
            float32x4_t b0 = vld1q_f32(b + idx);
            float32x4_t b1 = vld1q_f32(b + idx + 4);
            
            // Check for zero divisor
            uint32x4_t zero_mask0 = vceqq_f32(b0, vdupq_n_f32(0.0f));
            uint32x4_t zero_mask1 = vceqq_f32(b1, vdupq_n_f32(0.0f));
            
            // Compute fmod - use scalar for accuracy (NEON doesn't have fmod)
            // But we can still vectorize the zero check and selection
            float out_arr[8];
            const float* a_ptr0 = (const float*)&a0;
            const float* a_ptr1 = (const float*)&a1;
            const float* b_ptr0 = (const float*)&b0;
            const float* b_ptr1 = (const float*)&b1;
            
            // Unrolled loop with branch prediction hint
            for (int j = 0; j < 4; j++)
            {
                out_arr[j] = (b_ptr0[j] == 0.0f) ? 0.0f : std::fmod(a_ptr0[j], b_ptr0[j]);
                out_arr[j + 4] = (b_ptr1[j] == 0.0f) ? 0.0f : std::fmod(a_ptr1[j], b_ptr1[j]);
            }
            
            float32x4_t out0 = vld1q_f32(out_arr);
            float32x4_t out1 = vld1q_f32(out_arr + 4);
            
            // Apply zero mask - select 0.0f where b was zero
            out0 = vbslq_f32(vmvnq_u32(zero_mask0), out0, vdupq_n_f32(0.0f));
            out1 = vbslq_f32(vmvnq_u32(zero_mask1), out1, vdupq_n_f32(0.0f));
            
            vst1q_f32(out + idx, out0);
            vst1q_f32(out + idx + 4, out1);
        }

        // Handle remaining elements
        for (int i = nn << 3; i < total; i++)
        {
            out[i] = (b[i] == 0.0f) ? 0.0f : std::fmod(a[i], b[i]);
        }

        return 0;
    }

    // Python-style modulo - more complex sign handling
    if (fmod == 0 && opt.num_threads > 1)
    {
        const int nn = total >> 3;
        const int remain = total - (nn << 3);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < nn; i++)
        {
            int idx = i << 3;
            
            float32x4_t a0 = vld1q_f32(a + idx);
            float32x4_t a1 = vld1q_f32(a + idx + 4);
            float32x4_t b0 = vld1q_f32(b + idx);
            float32x4_t b1 = vld1q_f32(b + idx + 4);
            
            uint32x4_t zero_mask0 = vceqq_f32(b0, vdupq_n_f32(0.0f));
            uint32x4_t zero_mask1 = vceqq_f32(b1, vdupq_n_f32(0.0f));
            
            float out_arr[8];
            const float* a_ptr0 = (const float*)&a0;
            const float* a_ptr1 = (const float*)&a1;
            const float* b_ptr0 = (const float*)&b0;
            const float* b_ptr1 = (const float*)&b1;
            
            // Python-style: result has same sign as divisor
            for (int j = 0; j < 4; j++)
            {
                if (b_ptr0[j] == 0.0f)
                {
                    out_arr[j] = 0.0f;
                }
                else
                {
                    float result = std::fmod(a_ptr0[j], b_ptr0[j]);
                    // Branchless sign adjustment
                    int sign_diff = ((*(int*)&b_ptr0[j]) ^ (*(int*)&result)) < 0;
                    int is_nonzero = (result != 0.0f);
                    result += sign_diff & is_nonzero ? b_ptr0[j] : 0.0f;
                    out_arr[j] = result;
                }
                
                if (b_ptr1[j] == 0.0f)
                {
                    out_arr[j + 4] = 0.0f;
                }
                else
                {
                    float result = std::fmod(a_ptr1[j], b_ptr1[j]);
                    int sign_diff = ((*(int*)&b_ptr1[j]) ^ (*(int*)&result)) < 0;
                    int is_nonzero = (result != 0.0f);
                    result += sign_diff & is_nonzero ? b_ptr1[j] : 0.0f;
                    out_arr[j + 4] = result;
                }
            }
            
            float32x4_t out0 = vld1q_f32(out_arr);
            float32x4_t out1 = vld1q_f32(out_arr + 4);
            
            out0 = vbslq_f32(vmvnq_u32(zero_mask0), out0, vdupq_n_f32(0.0f));
            out1 = vbslq_f32(vmvnq_u32(zero_mask1), out1, vdupq_n_f32(0.0f));
            
            vst1q_f32(out + idx, out0);
            vst1q_f32(out + idx + 4, out1);
        }

        for (int i = nn << 3; i < total; i++)
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

        return 0;
    }

    // Scalar fallback
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
