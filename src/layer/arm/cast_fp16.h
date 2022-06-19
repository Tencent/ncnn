// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#if NCNN_RUNTIME_CPU && NCNN_VFPV4 && __ARM_NEON && !(__ARM_FP & 2)
void cast_fp32_to_fp16_neon_vfpv4(const Mat& bottom_blob, Mat& top_blob, const Option& opt);
void cast_fp16_to_fp32_neon_vfpv4(const Mat& bottom_blob, Mat& top_blob, const Option& opt);
#endif

static void cast_fp32_to_fp16_neon(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_VFPV4 && __ARM_NEON && !(__ARM_FP & 2)
    if (ncnn::cpu_support_arm_vfpv4())
    {
        cast_fp32_to_fp16_neon_vfpv4(bottom_blob, top_blob, opt);
        return;
    }
#endif

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    const int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);
        unsigned short* outptr = top_blob.channel(q);

        int i = 0;
#if (__ARM_FP & 2)
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _v_fp32 = vld1q_f32(ptr);
            float16x4_t _v_fp16 = vcvt_f16_f32(_v_fp32);
            vst1_u16(outptr, vreinterpret_u16_f16(_v_fp16));

            ptr += 4;
            outptr += 4;
        }
#endif
        for (; i < size; i++)
        {
            *outptr++ = float32_to_float16(*ptr++);
        }
    }
}

static void cast_fp16_to_fp32_neon(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
#if NCNN_VFPV4 && __ARM_NEON && !(__ARM_FP & 2)
    if (ncnn::cpu_support_arm_vfpv4())
    {
        cast_fp16_to_fp32_neon_vfpv4(bottom_blob, top_blob, opt);
        return;
    }
#endif

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    const int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const unsigned short* ptr = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);

        int i = 0;
#if (__ARM_FP & 2)
        for (; i + 3 < size; i += 4)
        {
            float16x4_t _v_fp16 = vreinterpret_f16_u16(vld1_u16(ptr));
            float32x4_t _v_fp32 = vcvt_f32_f16(_v_fp16);
            vst1q_f32(outptr, _v_fp32);

            ptr += 4;
            outptr += 4;
        }
#endif
        for (; i < size; i++)
        {
            *outptr++ = float16_to_float32(*ptr++);
        }
    }
}
