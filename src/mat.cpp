// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "mat.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON
#include "cpu.h"
#include "layer.h"
#include "layer_type.h"

#include <math.h>

#if NCNN_VULKAN
#if __ANDROID_API__ >= 26
#include <android/hardware_buffer.h>
#endif // __ANDROID_API__ >= 26
#endif // NCNN_VULKAN

namespace ncnn {

void Mat::substract_mean_normalize(const float* mean_vals, const float* norm_vals)
{
    Layer* op;

    if (mean_vals && !norm_vals)
    {
        // substract mean only
        op = create_layer(LayerType::Bias);

        ParamDict pd;
        pd.set(0, c);

        op->load_param(pd);

        Mat weights[1];
        weights[0] = Mat(c);
        for (int q = 0; q < c; q++)
        {
            weights[0][q] = -mean_vals[q];
        }

        op->load_model(ModelBinFromMatArray(weights));
    }
    else if (!mean_vals && norm_vals)
    {
        // normalize only
        op = create_layer(LayerType::Scale);

        ParamDict pd;
        pd.set(0, c);

        op->load_param(pd);

        Mat weights[1];
        weights[0] = Mat(c);
        for (int q = 0; q < c; q++)
        {
            weights[0][q] = norm_vals[q];
        }

        op->load_model(ModelBinFromMatArray(weights));
    }
    else if (mean_vals && norm_vals)
    {
        // substract mean and normalize
        op = create_layer(LayerType::Scale);

        ParamDict pd;
        pd.set(0, c);
        pd.set(1, 1);

        op->load_param(pd);

        Mat weights[2];
        weights[0] = Mat(c);
        weights[1] = Mat(c);
        for (int q = 0; q < c; q++)
        {
            weights[0][q] = norm_vals[q];
            weights[1][q] = -mean_vals[q] * norm_vals[q];
        }

        op->load_model(ModelBinFromMatArray(weights));
    }
    else // if (!mean_vals && !norm_vals)
    {
        return;
    }

    Option opt;
    opt.num_threads = 1; // TODO

    op->create_pipeline(opt);

    op->forward_inplace(*this, opt);

    op->destroy_pipeline(opt);

    delete op;
}

Mat Mat::from_float16(const unsigned short* data, int size)
{
    Mat m(size);
    if (m.empty())
        return m;

    float* ptr = m; //.data;

#if __ARM_NEON && (__ARM_FP & 2)
    int nn = cpu_support_arm_vfpv4() ? size >> 2 : 0;
    int remain = size - (nn << 2);
#else
    int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON && (__ARM_FP & 2)
#if __aarch64__
    if (nn > 0)
    {
        asm volatile(
            "0:                             \n"
            "ld1    {v0.4h}, [%1], #8       \n"
            "fcvtl  v1.4s, v0.4h            \n"
            "subs   %w0, %w0, #1            \n"
            "st1    {v1.4s}, [%2], #16      \n"
            "bne    0b                      \n"
            : "=r"(nn),   // %0
            "=r"(data), // %1
            "=r"(ptr)   // %2
            : "0"(nn),
            "1"(data),
            "2"(ptr)
            : "cc", "memory", "v0", "v1");
    }
#else
    if (nn > 0)
    {
        asm volatile(
            "0:                             \n"
            "pld        [%1, #64]           \n"
            "vld1.s16   {d0}, [%1 :64]!     \n"
            "vcvt.f32.f16 q1, d0            \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d2-d3}, [%2 :128]! \n"
            "bne        0b                  \n"
            : "=r"(nn),   // %0
            "=r"(data), // %1
            "=r"(ptr)   // %2
            : "0"(nn),
            "1"(data),
            "2"(ptr)
            : "cc", "memory", "q0", "q1");
    }
#endif // __aarch64__
#endif // __ARM_NEON
    for (; remain > 0; remain--)
    {
        *ptr = float16_to_float32(*data);

        data++;
        ptr++;
    }

    return m;
}

#if NCNN_VULKAN
#if __ANDROID_API__ >= 26
VkImageMat VkImageMat::from_android_hardware_buffer(VkAndroidHardwareBufferImageAllocator* allocator)
{
    int width = allocator->width();
    int height = allocator->height();

    return VkImageMat(width, height, allocator);
}
#endif // __ANDROID_API__ >= 26
#endif // NCNN_VULKAN

unsigned short float32_to_float16(float value)
{
    // 1 : 8 : 23
    union
    {
        unsigned int u;
        float f;
    } tmp;

    tmp.f = value;

    // 1 : 8 : 23
    unsigned short sign = (tmp.u & 0x80000000) >> 31;
    unsigned short exponent = (tmp.u & 0x7F800000) >> 23;
    unsigned int significand = tmp.u & 0x7FFFFF;

    //     NCNN_LOGE("%d %d %d", sign, exponent, significand);

    // 1 : 5 : 10
    unsigned short fp16;
    if (exponent == 0)
    {
        // zero or denormal, always underflow
        fp16 = (sign << 15) | (0x00 << 10) | 0x00;
    }
    else if (exponent == 0xFF)
    {
        // infinity or NaN
        fp16 = (sign << 15) | (0x1F << 10) | (significand ? 0x200 : 0x00);
    }
    else
    {
        // normalized
        short newexp = exponent + (-127 + 15);
        if (newexp >= 31)
        {
            // overflow, return infinity
            fp16 = (sign << 15) | (0x1F << 10) | 0x00;
        }
        else if (newexp <= 0)
        {
            // underflow
            if (newexp >= -10)
            {
                // denormal half-precision
                unsigned short sig = (significand | 0x800000) >> (14 - newexp);
                fp16 = (sign << 15) | (0x00 << 10) | sig;
            }
            else
            {
                // underflow
                fp16 = (sign << 15) | (0x00 << 10) | 0x00;
            }
        }
        else
        {
            fp16 = (sign << 15) | (newexp << 10) | (significand >> 13);
        }
    }

    return fp16;
}

float float16_to_float32(unsigned short value)
{
    // 1 : 5 : 10
    unsigned short sign = (value & 0x8000) >> 15;
    unsigned short exponent = (value & 0x7c00) >> 10;
    unsigned short significand = value & 0x03FF;

    //     NCNN_LOGE("%d %d %d", sign, exponent, significand);

    // 1 : 8 : 23
    union
    {
        unsigned int u;
        float f;
    } tmp;
    if (exponent == 0)
    {
        if (significand == 0)
        {
            // zero
            tmp.u = (sign << 31);
        }
        else
        {
            // denormal
            exponent = 0;
            // find non-zero bit
            while ((significand & 0x200) == 0)
            {
                significand <<= 1;
                exponent++;
            }
            significand <<= 1;
            significand &= 0x3FF;
            tmp.u = (sign << 31) | ((-exponent + (-15 + 127)) << 23) | (significand << 13);
        }
    }
    else if (exponent == 0x1F)
    {
        // infinity or NaN
        tmp.u = (sign << 31) | (0xFF << 23) | (significand << 13);
    }
    else
    {
        // normalized
        tmp.u = (sign << 31) | ((exponent + (-15 + 127)) << 23) | (significand << 13);
    }

    return tmp.f;
}

void copy_make_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int type, float v, const Option& opt)
{
    Layer* padding = create_layer(LayerType::Padding);

    ParamDict pd;
    pd.set(0, top);
    pd.set(1, bottom);
    pd.set(2, left);
    pd.set(3, right);
    pd.set(4, type);
    pd.set(5, v);

    padding->load_param(pd);

    padding->create_pipeline(opt);

    padding->forward(src, dst, opt);

    padding->destroy_pipeline(opt);

    delete padding;
}

void copy_cut_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right, const Option& opt)
{
    if (left + right > src.w || top + bottom > src.h)
    {
        NCNN_LOGE("copy_cut_border parameter error, top: %d, bottom: %d, left: %d, right: %d, src.w: %d, src.h: %d", top, bottom, left, right, src.w, src.h);
        return;
    }
    Layer* crop = create_layer(LayerType::Crop);

    ParamDict pd;
    pd.set(0, left);
    pd.set(1, top);
    pd.set(2, 0);
    pd.set(3, src.w - left - right);
    pd.set(4, src.h - top - bottom);
    pd.set(5, -233);

    crop->load_param(pd);

    crop->create_pipeline(opt);

    crop->forward(src, dst, opt);

    crop->destroy_pipeline(opt);

    delete crop;
}

void resize_nearest(const Mat& src, Mat& dst, int w, int h, const Option& opt)
{
    Layer* interp = create_layer(LayerType::Interp);

    ParamDict pd;
    pd.set(0, 1);
    pd.set(3, h);
    pd.set(4, w);

    interp->load_param(pd);

    interp->create_pipeline(opt);

    interp->forward(src, dst, opt);

    interp->destroy_pipeline(opt);

    delete interp;
}

void resize_bilinear(const Mat& src, Mat& dst, int w, int h, const Option& opt)
{
    Layer* interp = create_layer(LayerType::Interp);

    ParamDict pd;
    pd.set(0, 2);
    pd.set(3, h);
    pd.set(4, w);

    interp->load_param(pd);

    interp->create_pipeline(opt);

    interp->forward(src, dst, opt);

    interp->destroy_pipeline(opt);

    delete interp;
}

void resize_bicubic(const Mat& src, Mat& dst, int w, int h, const Option& opt)
{
    Layer* interp = create_layer(LayerType::Interp);

    ParamDict pd;
    pd.set(0, 3);
    pd.set(3, h);
    pd.set(4, w);

    interp->load_param(pd);

    interp->create_pipeline(opt);

    interp->forward(src, dst, opt);

    interp->destroy_pipeline(opt);

    delete interp;
}

void convert_packing(const Mat& src, Mat& dst, int _elempack, const Option& opt)
{
    Layer* packing = create_layer(LayerType::Packing);

    ParamDict pd;
    pd.set(0, _elempack);

    packing->load_param(pd);

    packing->create_pipeline(opt);

    packing->forward(src, dst, opt);

    packing->destroy_pipeline(opt);

    delete packing;
}

void flatten(const Mat& src, Mat& dst, const Option& opt)
{
    Layer* flatten = create_layer(LayerType::Flatten);

    ParamDict pd;

    flatten->load_param(pd);

    flatten->create_pipeline(opt);

    flatten->forward(src, dst, opt);

    flatten->destroy_pipeline(opt);

    delete flatten;
}

void cast_float32_to_float16(const Mat& src, Mat& dst, const Option& opt)
{
    Layer* cast = create_layer(LayerType::Cast);

    ParamDict pd;
    pd.set(0, 1);
    pd.set(1, 2);

    cast->load_param(pd);

    cast->create_pipeline(opt);

    cast->forward(src, dst, opt);

    cast->destroy_pipeline(opt);

    delete cast;
}

void cast_float16_to_float32(const Mat& src, Mat& dst, const Option& opt)
{
    Layer* cast = create_layer(LayerType::Cast);

    ParamDict pd;
    pd.set(0, 2);
    pd.set(1, 1);

    cast->load_param(pd);

    cast->create_pipeline(opt);

    cast->forward(src, dst, opt);

    cast->destroy_pipeline(opt);

    delete cast;
}

void cast_int8_to_float32(const Mat& src, Mat& dst, const Option& opt)
{
    Layer* cast = create_layer(LayerType::Cast);

    ParamDict pd;
    pd.set(0, 3);
    pd.set(1, 1);

    cast->load_param(pd);

    cast->create_pipeline(opt);

    cast->forward(src, dst, opt);

    cast->destroy_pipeline(opt);

    delete cast;
}

void cast_float32_to_bfloat16(const Mat& src, Mat& dst, const Option& opt)
{
    Layer* cast = create_layer(LayerType::Cast);

    ParamDict pd;
    pd.set(0, 1);
    pd.set(1, 4);

    cast->load_param(pd);

    cast->create_pipeline(opt);

    cast->forward(src, dst, opt);

    cast->destroy_pipeline(opt);

    delete cast;
}

void cast_bfloat16_to_float32(const Mat& src, Mat& dst, const Option& opt)
{
    Layer* cast = create_layer(LayerType::Cast);

    ParamDict pd;
    pd.set(0, 4);
    pd.set(1, 1);

    cast->load_param(pd);

    cast->create_pipeline(opt);

    cast->forward(src, dst, opt);

    cast->destroy_pipeline(opt);

    delete cast;
}

void quantize_float32_to_int8(const Mat& src, Mat& dst, float scale, const Option& opt)
{
    Layer* quantize = create_layer(LayerType::Quantize);

    ParamDict pd;
    pd.set(0, scale);

    quantize->load_param(pd);

    quantize->create_pipeline(opt);

    quantize->forward(src, dst, opt);

    quantize->destroy_pipeline(opt);

    delete quantize;
}

void dequantize_int32_to_float32(Mat& m, float scale, const float* bias, int bias_data_size, const Option& opt)
{
    Layer* dequantize = create_layer(LayerType::Dequantize);

    ParamDict pd;
    pd.set(0, scale);
    pd.set(1, bias ? 1 : 0);
    pd.set(2, bias_data_size);

    dequantize->load_param(pd);

    Mat weights[1];
    weights[0] = Mat(bias_data_size, (void*)bias);

    dequantize->load_model(ModelBinFromMatArray(weights));

    dequantize->create_pipeline(opt);

    dequantize->forward_inplace(m, opt);

    dequantize->destroy_pipeline(opt);

    delete dequantize;
}

void requantize_int8_to_int8(const Mat& src, Mat& dst, float scale_in, float scale_out, const float* bias, int bias_data_size, int fusion_relu, const Option& opt)
{
    Layer* requantize = create_layer(LayerType::Requantize);

    ParamDict pd;
    pd.set(0, scale_in);
    pd.set(1, scale_out);
    pd.set(2, bias ? 1 : 0);
    pd.set(3, bias_data_size);
    pd.set(4, fusion_relu);

    requantize->load_param(pd);

    Mat weights[1];
    weights[0] = Mat(bias_data_size, (void*)bias);

    requantize->load_model(ModelBinFromMatArray(weights));

    requantize->create_pipeline(opt);

    requantize->forward(src, dst, opt);

    requantize->destroy_pipeline(opt);

    delete requantize;
}

} // namespace ncnn
