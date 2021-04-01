// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

static void requantize_relu_pack4_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& scale_in_data, const Mat& scale_out_data, const Mat& bias_data, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int size = w * h;
    int outc = top_blob.c;
    int out_elempack = top_blob.elempack;

    int scale_in_data_size = scale_in_data.w;
    int scale_out_data_size = scale_out_data.w;
    int bias_data_size = bias_data.w;

    // int8(relu(v * scale_in) * scale_out)
    // int8_relu(v * (scale_in * scale_out))

    // int8(relu(v * scale_in + bias) * scale_out)
    // int8_relu(v * (scale_in * scale_out) + (bias * scale_out))

    if (out_elempack == 8)
    {
        if (bias_data_size == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const int* intptr0 = bottom_blob.channel(q * 2);
                const int* intptr1 = bottom_blob.channel(q * 2 + 1);
                signed char* ptr = top_blob.channel(q);

                float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + q * 8);
                float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + q * 8 + 4);
                float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + q * 8);
                float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + q * 8 + 4);

                float32x4_t _scale0 = vmulq_f32(_scale_in0, _scale_out0);
                float32x4_t _scale1 = vmulq_f32(_scale_in1, _scale_out1);

                int i = 0;
#if __aarch64__
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _v00 = vcvtq_f32_s32(vld1q_s32(intptr0));
                    float32x4_t _v01 = vcvtq_f32_s32(vld1q_s32(intptr0 + 4));
                    float32x4_t _v02 = vcvtq_f32_s32(vld1q_s32(intptr0 + 8));
                    float32x4_t _v03 = vcvtq_f32_s32(vld1q_s32(intptr0 + 12));
                    float32x4_t _v10 = vcvtq_f32_s32(vld1q_s32(intptr1));
                    float32x4_t _v11 = vcvtq_f32_s32(vld1q_s32(intptr1 + 4));
                    float32x4_t _v12 = vcvtq_f32_s32(vld1q_s32(intptr1 + 8));
                    float32x4_t _v13 = vcvtq_f32_s32(vld1q_s32(intptr1 + 12));
                    _v00 = vmulq_f32(_v00, _scale0);
                    _v01 = vmulq_f32(_v01, _scale0);
                    _v02 = vmulq_f32(_v02, _scale0);
                    _v03 = vmulq_f32(_v03, _scale0);
                    _v10 = vmulq_f32(_v10, _scale1);
                    _v11 = vmulq_f32(_v11, _scale1);
                    _v12 = vmulq_f32(_v12, _scale1);
                    _v13 = vmulq_f32(_v13, _scale1);
                    vst1_s8(ptr, float2int8relu(_v00, _v10));
                    vst1_s8(ptr + 8, float2int8relu(_v01, _v11));
                    vst1_s8(ptr + 16, float2int8relu(_v02, _v12));
                    vst1_s8(ptr + 24, float2int8relu(_v03, _v13));

                    intptr0 += 16;
                    intptr1 += 16;
                    ptr += 32;
                }
#endif // __aarch64__
                for (; i < size; i++)
                {
                    float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr0));
                    float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr1));
                    _v0 = vmulq_f32(_v0, _scale0);
                    _v1 = vmulq_f32(_v1, _scale1);
                    vst1_s8(ptr, float2int8relu(_v0, _v1));

                    intptr0 += 4;
                    intptr1 += 4;
                    ptr += 8;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const int* intptr0 = bottom_blob.channel(q * 2);
                const int* intptr1 = bottom_blob.channel(q * 2 + 1);
                signed char* ptr = top_blob.channel(q);

                float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + q * 8);
                float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + q * 8 + 4);
                float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + q * 8);
                float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + q * 8 + 4);
                float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + q * 8);
                float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + q * 8 + 4);

                float32x4_t _scale0 = vmulq_f32(_scale_in0, _scale_out0);
                float32x4_t _scale1 = vmulq_f32(_scale_in1, _scale_out1);
                _bias0 = vmulq_f32(_bias0, _scale_out0);
                _bias1 = vmulq_f32(_bias1, _scale_out1);

                int i = 0;
#if __aarch64__
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _v00 = vcvtq_f32_s32(vld1q_s32(intptr0));
                    float32x4_t _v01 = vcvtq_f32_s32(vld1q_s32(intptr0 + 4));
                    float32x4_t _v02 = vcvtq_f32_s32(vld1q_s32(intptr0 + 8));
                    float32x4_t _v03 = vcvtq_f32_s32(vld1q_s32(intptr0 + 12));
                    float32x4_t _v10 = vcvtq_f32_s32(vld1q_s32(intptr1));
                    float32x4_t _v11 = vcvtq_f32_s32(vld1q_s32(intptr1 + 4));
                    float32x4_t _v12 = vcvtq_f32_s32(vld1q_s32(intptr1 + 8));
                    float32x4_t _v13 = vcvtq_f32_s32(vld1q_s32(intptr1 + 12));
                    _v00 = vfmaq_f32(_bias0, _v00, _scale0);
                    _v01 = vfmaq_f32(_bias0, _v01, _scale0);
                    _v02 = vfmaq_f32(_bias0, _v02, _scale0);
                    _v03 = vfmaq_f32(_bias0, _v03, _scale0);
                    _v10 = vfmaq_f32(_bias1, _v10, _scale1);
                    _v11 = vfmaq_f32(_bias1, _v11, _scale1);
                    _v12 = vfmaq_f32(_bias1, _v12, _scale1);
                    _v13 = vfmaq_f32(_bias1, _v13, _scale1);
                    vst1_s8(ptr, float2int8relu(_v00, _v10));
                    vst1_s8(ptr + 8, float2int8relu(_v01, _v11));
                    vst1_s8(ptr + 16, float2int8relu(_v02, _v12));
                    vst1_s8(ptr + 24, float2int8relu(_v03, _v13));

                    intptr0 += 16;
                    intptr1 += 16;
                    ptr += 32;
                }
#endif // __aarch64__
                for (; i + 1 < size; i += 2)
                {
#if __aarch64__
                    float32x4_t _v00 = vcvtq_f32_s32(vld1q_s32(intptr0));
                    float32x4_t _v01 = vcvtq_f32_s32(vld1q_s32(intptr0 + 4));
                    float32x4_t _v10 = vcvtq_f32_s32(vld1q_s32(intptr1));
                    float32x4_t _v11 = vcvtq_f32_s32(vld1q_s32(intptr1 + 4));
                    _v00 = vfmaq_f32(_bias0, _v00, _scale0);
                    _v01 = vfmaq_f32(_bias0, _v01, _scale0);
                    _v10 = vfmaq_f32(_bias1, _v10, _scale1);
                    _v11 = vfmaq_f32(_bias1, _v11, _scale1);
                    vst1_s8(ptr, float2int8relu(_v00, _v10));
                    vst1_s8(ptr + 8, float2int8relu(_v01, _v11));

                    intptr0 += 8;
                    intptr1 += 8;
                    ptr += 16;
#else  // __aarch64__
                    asm volatile(
                        "pld            [%0, #256]      \n"
                        "vld1.s32       {d8-d11}, [%0 :128]! \n"
                        "pld            [%1, #256]      \n"
                        "vld1.s32       {d12-d15}, [%1 :128]! \n"

                        "vmov           q0, %q8         \n"
                        "vmov           q1, %q9         \n"
                        "vmov           q2, %q8         \n"
                        "vmov           q3, %q9         \n"

                        "vcvt.f32.s32   q4, q4          \n"
                        "vcvt.f32.s32   q5, q5          \n"
                        "vcvt.f32.s32   q6, q6          \n"
                        "vcvt.f32.s32   q7, q7          \n"

                        "veor           q8, q8          \n" // _zero

                        "vmla.f32       q0, q4, %q6     \n"
                        "vmla.f32       q1, q5, %q7     \n"
                        "vmla.f32       q2, q6, %q6     \n"
                        "vmla.f32       q3, q7, %q7     \n"

                        "vcvtr.s32.f32  s0, s0          \n"
                        "vcvtr.s32.f32  s1, s1          \n"
                        "vcvtr.s32.f32  s2, s2          \n"
                        "vcvtr.s32.f32  s3, s3          \n"
                        "vcvtr.s32.f32  s4, s4          \n"
                        "vcvtr.s32.f32  s5, s5          \n"
                        "vcvtr.s32.f32  s6, s6          \n"
                        "vcvtr.s32.f32  s7, s7          \n"
                        "vcvtr.s32.f32  s8, s8          \n"
                        "vcvtr.s32.f32  s9, s9          \n"
                        "vcvtr.s32.f32  s10, s10        \n"
                        "vcvtr.s32.f32  s11, s11        \n"
                        "vcvtr.s32.f32  s12, s12        \n"
                        "vcvtr.s32.f32  s13, s13        \n"
                        "vcvtr.s32.f32  s14, s14        \n"
                        "vcvtr.s32.f32  s15, s15        \n"

                        "vqmovn.s32     d8, q0          \n"
                        "vqmovn.s32     d10, q1         \n"
                        "vqmovn.s32     d9, q2          \n"
                        "vqmovn.s32     d11, q3         \n"
                        "vqmovn.s16     d8, q4          \n"
                        "vqmovn.s16     d9, q5          \n"

                        "vmax.s8        q4, q4, q8      \n"

                        "vst1.s8        {d8-d9}, [%2 :64]! \n"

                        : "=r"(intptr0),
                        "=r"(intptr1),
                        "=r"(ptr)
                        : "0"(intptr0),
                        "1"(intptr1),
                        "2"(ptr),
                        "w"(_scale0), // %6
                        "w"(_scale1), // %7
                        "w"(_bias0),  // %8
                        "w"(_bias1)   // %9
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8");
#endif // __aarch64__
                }
                for (; i < size; i++)
                {
#if __aarch64__
                    float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr0));
                    float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr1));
                    _v0 = vmlaq_f32(_bias0, _v0, _scale0);
                    _v1 = vmlaq_f32(_bias1, _v1, _scale1);
                    vst1_s8(ptr, float2int8relu(_v0, _v1));

                    intptr0 += 4;
                    intptr1 += 4;
                    ptr += 8;
#else  // __aarch64__
                    asm volatile(
                        "pld            [%0, #128]      \n"
                        "vld1.s32       {d4-d5}, [%0 :128]! \n"
                        "pld            [%1, #128]      \n"
                        "vld1.s32       {d6-d7}, [%1 :128]! \n"

                        "vmov           q0, %q8         \n"
                        "vmov           q1, %q9         \n"

                        "vcvt.f32.s32   q2, q2          \n"
                        "vcvt.f32.s32   q3, q3          \n"

                        "veor           d8, d8          \n" // _zero

                        "vmla.f32       q0, q2, %q6     \n"
                        "vmla.f32       q1, q3, %q7     \n"

                        "vcvtr.s32.f32  s0, s0          \n"
                        "vcvtr.s32.f32  s1, s1          \n"
                        "vcvtr.s32.f32  s2, s2          \n"
                        "vcvtr.s32.f32  s3, s3          \n"
                        "vcvtr.s32.f32  s4, s4          \n"
                        "vcvtr.s32.f32  s5, s5          \n"
                        "vcvtr.s32.f32  s6, s6          \n"
                        "vcvtr.s32.f32  s7, s7          \n"

                        "vqmovn.s32     d4, q0          \n"
                        "vqmovn.s32     d5, q1          \n"
                        "vqmovn.s16     d4, q2          \n"

                        "vmax.s8        d4, d4, d8      \n"

                        "vst1.s8        {d4}, [%2 :64]! \n"

                        : "=r"(intptr0),
                        "=r"(intptr1),
                        "=r"(ptr)
                        : "0"(intptr0),
                        "1"(intptr1),
                        "2"(ptr),
                        "w"(_scale0), // %6
                        "w"(_scale1), // %7
                        "w"(_bias0),  // %8
                        "w"(_bias1)   // %9
                        : "memory", "q0", "q1", "q2", "q3", "q4");
#endif // __aarch64__
                }
            }
        }
    }
    if (out_elempack == 1)
    {
        if (bias_data_size == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                signed char* ptr0 = top_blob.channel(q * 4);
                signed char* ptr1 = top_blob.channel(q * 4 + 1);
                signed char* ptr2 = top_blob.channel(q * 4 + 2);
                signed char* ptr3 = top_blob.channel(q * 4 + 3);

                float32x4_t _scale_in = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + q * 4);
                float32x4_t _scale_out = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + q * 4);

                float32x4_t _scale = vmulq_f32(_scale_in, _scale_out);

                int i = 0;
                for (; i < size; i++)
                {
                    float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                    _v = vmulq_f32(_v, _scale);
                    int8x8_t v = float2int8relu(_v, _v);
                    ptr0[0] = vget_lane_s8(v, 0);
                    ptr1[0] = vget_lane_s8(v, 1);
                    ptr2[0] = vget_lane_s8(v, 2);
                    ptr3[0] = vget_lane_s8(v, 3);

                    intptr += 4;
                    ptr0 += 1;
                    ptr1 += 1;
                    ptr2 += 1;
                    ptr3 += 1;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                signed char* ptr0 = top_blob.channel(q * 4);
                signed char* ptr1 = top_blob.channel(q * 4 + 1);
                signed char* ptr2 = top_blob.channel(q * 4 + 2);
                signed char* ptr3 = top_blob.channel(q * 4 + 3);

                float32x4_t _scale_in = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + q * 4);
                float32x4_t _scale_out = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + q * 4);
                float32x4_t _bias = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + q * 4);

                float32x4_t _scale = vmulq_f32(_scale_in, _scale_out);
                _bias = vmulq_f32(_bias, _scale_out);

                int i = 0;
                for (; i < size; i++)
                {
                    float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
#if __aarch64__
                    _v = vfmaq_f32(_bias, _v, _scale);
#else
                    _v = vmlaq_f32(_bias, _v, _scale);
#endif
                    int8x8_t v = float2int8relu(_v, _v);
                    ptr0[0] = vget_lane_s8(v, 0);
                    ptr1[0] = vget_lane_s8(v, 1);
                    ptr2[0] = vget_lane_s8(v, 2);
                    ptr3[0] = vget_lane_s8(v, 3);

                    intptr += 4;
                    ptr0 += 1;
                    ptr1 += 1;
                    ptr2 += 1;
                    ptr3 += 1;
                }
            }
        }
    }
}
