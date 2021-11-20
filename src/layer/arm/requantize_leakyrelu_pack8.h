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

static void requantize_leakyrelu_pack8_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& scale_in_data, const Mat& scale_out_data, const Mat& bias_data, float slope, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int size = w * h;

    int scale_in_data_size = scale_in_data.w;
    int scale_out_data_size = scale_out_data.w;
    int bias_data_size = bias_data.w;

    // int8(leakyrelu(v * scale_in, slope) * scale_out)
    // int8_leakyrelu(v * (scale_in * scale_out), slope)

    // int8(leakyrelu(v * scale_in + bias, slope) * scale_out)
    // int8_leakyrelu(v * (scale_in * scale_out) + (bias * scale_out), slope)

    if (bias_data_size == 0)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const int* intptr = bottom_blob.channel(q);
            signed char* ptr = top_blob.channel(q);

            float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + q * 8);
            float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + q * 8 + 4);
            float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + q * 8);
            float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + q * 8 + 4);

            float32x4_t _scale0 = vmulq_f32(_scale_in0, _scale_out0);
            float32x4_t _scale1 = vmulq_f32(_scale_in1, _scale_out1);
            float32x4_t _slope = vdupq_n_f32(slope);

            int i = 0;
#if __aarch64__
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                float32x4_t _v2 = vcvtq_f32_s32(vld1q_s32(intptr + 8));
                float32x4_t _v3 = vcvtq_f32_s32(vld1q_s32(intptr + 12));
                float32x4_t _v4 = vcvtq_f32_s32(vld1q_s32(intptr + 16));
                float32x4_t _v5 = vcvtq_f32_s32(vld1q_s32(intptr + 20));
                float32x4_t _v6 = vcvtq_f32_s32(vld1q_s32(intptr + 24));
                float32x4_t _v7 = vcvtq_f32_s32(vld1q_s32(intptr + 28));
                _v0 = vmulq_f32(_v0, _scale0);
                _v1 = vmulq_f32(_v1, _scale1);
                _v2 = vmulq_f32(_v2, _scale0);
                _v3 = vmulq_f32(_v3, _scale1);
                _v4 = vmulq_f32(_v4, _scale0);
                _v5 = vmulq_f32(_v5, _scale1);
                _v6 = vmulq_f32(_v6, _scale0);
                _v7 = vmulq_f32(_v7, _scale1);
                vst1_s8(ptr, float2int8leakyrelu(_v0, _v1, _slope));
                vst1_s8(ptr + 8, float2int8leakyrelu(_v2, _v3, _slope));
                vst1_s8(ptr + 16, float2int8leakyrelu(_v4, _v5, _slope));
                vst1_s8(ptr + 24, float2int8leakyrelu(_v6, _v7, _slope));

                intptr += 32;
                ptr += 32;
            }
#endif // __aarch64__
            for (; i + 1 < size; i += 2)
            {
                float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                float32x4_t _v2 = vcvtq_f32_s32(vld1q_s32(intptr + 8));
                float32x4_t _v3 = vcvtq_f32_s32(vld1q_s32(intptr + 12));
                _v0 = vmulq_f32(_v0, _scale0);
                _v1 = vmulq_f32(_v1, _scale1);
                _v2 = vmulq_f32(_v2, _scale0);
                _v3 = vmulq_f32(_v3, _scale1);
                vst1_s8(ptr, float2int8leakyrelu(_v0, _v1, _slope));
                vst1_s8(ptr + 8, float2int8leakyrelu(_v2, _v3, _slope));

                intptr += 16;
                ptr += 16;
            }
            for (; i < size; i++)
            {
                float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                _v0 = vmulq_f32(_v0, _scale0);
                _v1 = vmulq_f32(_v1, _scale1);
                vst1_s8(ptr, float2int8leakyrelu(_v0, _v1, _slope));

                intptr += 8;
                ptr += 8;
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const int* intptr = bottom_blob.channel(q);
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
            float32x4_t _slope = vdupq_n_f32(slope);

            int i = 0;
#if __aarch64__
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                float32x4_t _v2 = vcvtq_f32_s32(vld1q_s32(intptr + 8));
                float32x4_t _v3 = vcvtq_f32_s32(vld1q_s32(intptr + 12));
                float32x4_t _v4 = vcvtq_f32_s32(vld1q_s32(intptr + 16));
                float32x4_t _v5 = vcvtq_f32_s32(vld1q_s32(intptr + 20));
                float32x4_t _v6 = vcvtq_f32_s32(vld1q_s32(intptr + 24));
                float32x4_t _v7 = vcvtq_f32_s32(vld1q_s32(intptr + 28));

                _v0 = vfmaq_f32(_bias0, _v0, _scale0);
                _v1 = vfmaq_f32(_bias1, _v1, _scale1);
                _v2 = vfmaq_f32(_bias0, _v2, _scale0);
                _v3 = vfmaq_f32(_bias1, _v3, _scale1);
                _v4 = vfmaq_f32(_bias0, _v4, _scale0);
                _v5 = vfmaq_f32(_bias1, _v5, _scale1);
                _v6 = vfmaq_f32(_bias0, _v6, _scale0);
                _v7 = vfmaq_f32(_bias1, _v7, _scale1);

                vst1_s8(ptr, float2int8leakyrelu(_v0, _v1, _slope));
                vst1_s8(ptr + 8, float2int8leakyrelu(_v2, _v3, _slope));
                vst1_s8(ptr + 16, float2int8leakyrelu(_v4, _v5, _slope));
                vst1_s8(ptr + 24, float2int8leakyrelu(_v6, _v7, _slope));

                intptr += 32;
                ptr += 32;
            }
#endif // __aarch64__
            for (; i + 1 < size; i += 2)
            {
                float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                float32x4_t _v2 = vcvtq_f32_s32(vld1q_s32(intptr + 8));
                float32x4_t _v3 = vcvtq_f32_s32(vld1q_s32(intptr + 12));

#if __aarch64__
                _v0 = vfmaq_f32(_bias0, _v0, _scale0);
                _v1 = vfmaq_f32(_bias1, _v1, _scale1);
                _v2 = vfmaq_f32(_bias0, _v2, _scale0);
                _v3 = vfmaq_f32(_bias1, _v3, _scale1);
#else  // __aarch64__
                _v0 = vmlaq_f32(_bias0, _v0, _scale0);
                _v1 = vmlaq_f32(_bias1, _v1, _scale1);
                _v2 = vmlaq_f32(_bias0, _v2, _scale0);
                _v3 = vmlaq_f32(_bias1, _v3, _scale1);
#endif // __aarch64__

                vst1_s8(ptr, float2int8leakyrelu(_v0, _v1, _slope));
                vst1_s8(ptr + 8, float2int8leakyrelu(_v2, _v3, _slope));

                intptr += 16;
                ptr += 16;
            }
            for (; i < size; i++)
            {
                float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
#if __aarch64__
                _v0 = vfmaq_f32(_bias0, _v0, _scale0);
                _v1 = vfmaq_f32(_bias1, _v1, _scale1);
#else  // __aarch64__
                _v0 = vmlaq_f32(_bias0, _v0, _scale0);
                _v1 = vmlaq_f32(_bias1, _v1, _scale1);
#endif // __aarch64__
                vst1_s8(ptr, float2int8leakyrelu(_v0, _v1, _slope));

                intptr += 8;
                ptr += 8;
            }
        }
    }
}
