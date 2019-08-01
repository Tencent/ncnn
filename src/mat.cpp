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
#include <math.h>

#include "cpu.h"

#include "layer_type.h"
#include "layer.h"

namespace ncnn {

void Mat::substract_mean_normalize(const float* mean_vals, const float* norm_vals)
{
    ncnn::Layer* op;

    if (mean_vals && !norm_vals)
    {
        // substract mean only
        op = ncnn::create_layer(ncnn::LayerType::Bias);

        ncnn::ParamDict pd;
        pd.set(0, c);

        op->load_param(pd);

        ncnn::Mat weights[1];
        weights[0] = Mat(c);
        for (int q=0; q<c; q++)
        {
            weights[0][q] = -mean_vals[q];
        }

        op->load_model(ncnn::ModelBinFromMatArray(weights));
    }
    else if (!mean_vals && norm_vals)
    {
        // normalize only
        op = ncnn::create_layer(ncnn::LayerType::Scale);

        ncnn::ParamDict pd;
        pd.set(0, c);

        op->load_param(pd);

        ncnn::Mat weights[1];
        weights[0] = Mat(c);
        for (int q=0; q<c; q++)
        {
            weights[0][q] = norm_vals[q];
        }

        op->load_model(ncnn::ModelBinFromMatArray(weights));
    }
    else if (mean_vals && norm_vals)
    {
        // substract mean and normalize
        op = ncnn::create_layer(ncnn::LayerType::Scale);

        ncnn::ParamDict pd;
        pd.set(0, c);
        pd.set(1, 1);

        op->load_param(pd);

        ncnn::Mat weights[2];
        weights[0] = Mat(c);
        weights[1] = Mat(c);
        for (int q=0; q<c; q++)
        {
            weights[0][q] = norm_vals[q];
            weights[1][q] = - mean_vals[q] * norm_vals[q];
        }

        op->load_model(ncnn::ModelBinFromMatArray(weights));
    }
    else // if (!mean_vals && !norm_vals)
    {
        return;
    }

    op->forward_inplace(*this);

    delete op;
}

// convert half precision floating point to float
static float half2float(unsigned short value)
{
    // 1 : 5 : 10
    unsigned short sign = (value & 0x8000) >> 15;
    unsigned short exponent = (value & 0x7c00) >> 10;
    unsigned short significand = value & 0x03FF;

//     fprintf(stderr, "%d %d %d\n", sign, exponent, significand);

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

Mat Mat::from_float16(const unsigned short* data, int size)
{
    Mat m(size);
    if (m.empty())
        return m;

    float* ptr = m;//.data;

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
        : "=r"(nn),     // %0
          "=r"(data),   // %1
          "=r"(ptr)     // %2
        : "0"(nn),
          "1"(data),
          "2"(ptr)
        : "cc", "memory", "v0", "v1"
    );
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
        : "=r"(nn),     // %0
          "=r"(data),   // %1
          "=r"(ptr)     // %2
        : "0"(nn),
          "1"(data),
          "2"(ptr)
        : "cc", "memory", "q0", "q1"
    );
    }
#endif // __aarch64__
#endif // __ARM_NEON
    for (; remain>0; remain--)
    {
        *ptr = half2float(*data);

        data++;
        ptr++;
    }

    return m;
}

void copy_make_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int type, float v, const Option& opt)
{
    ncnn::Layer* padding = ncnn::create_layer(ncnn::LayerType::Padding);

    ncnn::ParamDict pd;
    pd.set(0, top);
    pd.set(1, bottom);
    pd.set(2, left);
    pd.set(3, right);
    pd.set(4, type);
    pd.set(5, v);

    padding->load_param(pd);

    padding->forward(src, dst, opt);

    delete padding;
}

void copy_cut_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right, const Option& opt)
{
    ncnn::Layer* crop = ncnn::create_layer(ncnn::LayerType::Crop);

    ncnn::ParamDict pd;
    pd.set(0, left);
    pd.set(1, top);
    pd.set(2, 0);
    pd.set(3, src.w - left - right);
    pd.set(4, src.h - top - bottom);
    pd.set(5, -233);

    crop->load_param(pd);

    crop->forward(src, dst, opt);

    delete crop;
}

void resize_bilinear(const Mat& src, Mat& dst, int w, int h, const Option& opt)
{
    ncnn::Layer* interp = ncnn::create_layer(ncnn::LayerType::Interp);

    ncnn::ParamDict pd;
    pd.set(0, 2);
    pd.set(3, h);
    pd.set(4, w);

    interp->load_param(pd);

    interp->forward(src, dst, opt);

    delete interp;
}

void resize_bicubic(const Mat& src, Mat& dst, int w, int h, const Option& opt)
{
    ncnn::Layer* interp = ncnn::create_layer(ncnn::LayerType::Interp);

    ncnn::ParamDict pd;
    pd.set(0, 3);
    pd.set(3, h);
    pd.set(4, w);

    interp->load_param(pd);

    interp->forward(src, dst, opt);

    delete interp;
}

void convert_packing(const Mat& src, Mat& dst, int _elempack, const Option& opt)
{
    ncnn::Layer* packing = ncnn::create_layer(ncnn::LayerType::Packing);

    ncnn::ParamDict pd;
    pd.set(0, _elempack);

    packing->load_param(pd);

    packing->forward(src, dst, opt);

    delete packing;
}

void cast_float32_to_float16(const Mat& src, Mat& dst, const Option& opt)
{
    ncnn::Layer* cast = ncnn::create_layer(ncnn::LayerType::Cast);

    ncnn::ParamDict pd;
    pd.set(0, 1);
    pd.set(1, 2);

    cast->load_param(pd);

    cast->forward(src, dst, opt);

    delete cast;
}

void cast_float16_to_float32(const Mat& src, Mat& dst, const Option& opt)
{
    ncnn::Layer* cast = ncnn::create_layer(ncnn::LayerType::Cast);

    ncnn::ParamDict pd;
    pd.set(0, 2);
    pd.set(1, 1);

    cast->load_param(pd);

    cast->forward(src, dst, opt);

    delete cast;
}

} // namespace ncnn
