// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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
#include <limits.h>

#include "platform.h"

namespace ncnn {

#if NCNN_PIXEL_AFFINE
void get_rotation_matrix(float angle, float scale, float dx, float dy, float* tm)
{
    angle *= (float)(3.14159265358979323846 / 180);
    float alpha = cosf(angle) * scale;
    float beta = sinf(angle) * scale;

    tm[0] = alpha;
    tm[1] = beta;
    tm[2] = (1.f - alpha) * dx - beta * dy;
    tm[3] = -beta;
    tm[4] = alpha;
    tm[5] = beta * dx + (1.f - alpha) * dy;
}

void get_affine_transform(const float* points_from, const float* points_to, int num_point, float* tm)
{
    float ma[4][4] = {{0.f}};
    float mb[4] = {0.f};
    float mm[4];

    for (int i = 0; i < num_point; i++)
    {
        ma[0][0] += points_from[0] * points_from[0] + points_from[1] * points_from[1];
        ma[0][2] += points_from[0];
        ma[0][3] += points_from[1];

        mb[0] += points_from[0] * points_to[0] + points_from[1] * points_to[1];
        mb[1] += points_from[0] * points_to[1] - points_from[1] * points_to[0];
        mb[2] += points_to[0];
        mb[3] += points_to[1];

        points_from += 2;
        points_to += 2;
    }

    ma[1][1] = ma[0][0];
    ma[2][1] = ma[1][2] = -ma[0][3];
    ma[3][1] = ma[1][3] = ma[2][0] = ma[0][2];
    ma[2][2] = ma[3][3] = (float)num_point;
    ma[3][0] = ma[0][3];

    // MM = inv(A) * B
    // matrix 4x4 invert by https://github.com/willnode/N-Matrix-Programmer
    // suppose the user provide valid points combination
    // I have not taken det == zero into account here   :>  --- nihui
    float mai[4][4];
    float det;
    // clang-format off
    // *INDENT-OFF*
    {
        float A2323 = ma[2][2] * ma[3][3] - ma[2][3] * ma[3][2];
        float A1323 = ma[2][1] * ma[3][3] - ma[2][3] * ma[3][1];
        float A1223 = ma[2][1] * ma[3][2] - ma[2][2] * ma[3][1];
        float A0323 = ma[2][0] * ma[3][3] - ma[2][3] * ma[3][0];
        float A0223 = ma[2][0] * ma[3][2] - ma[2][2] * ma[3][0];
        float A0123 = ma[2][0] * ma[3][1] - ma[2][1] * ma[3][0];
        float A2313 = ma[1][2] * ma[3][3] - ma[1][3] * ma[3][2];
        float A1313 = ma[1][1] * ma[3][3] - ma[1][3] * ma[3][1];
        float A1213 = ma[1][1] * ma[3][2] - ma[1][2] * ma[3][1];
        float A2312 = ma[1][2] * ma[2][3] - ma[1][3] * ma[2][2];
        float A1312 = ma[1][1] * ma[2][3] - ma[1][3] * ma[2][1];
        float A1212 = ma[1][1] * ma[2][2] - ma[1][2] * ma[2][1];
        float A0313 = ma[1][0] * ma[3][3] - ma[1][3] * ma[3][0];
        float A0213 = ma[1][0] * ma[3][2] - ma[1][2] * ma[3][0];
        float A0312 = ma[1][0] * ma[2][3] - ma[1][3] * ma[2][0];
        float A0212 = ma[1][0] * ma[2][2] - ma[1][2] * ma[2][0];
        float A0113 = ma[1][0] * ma[3][1] - ma[1][1] * ma[3][0];
        float A0112 = ma[1][0] * ma[2][1] - ma[1][1] * ma[2][0];

        det = ma[0][0] * (ma[1][1] * A2323 - ma[1][2] * A1323 + ma[1][3] * A1223)
            - ma[0][1] * (ma[1][0] * A2323 - ma[1][2] * A0323 + ma[1][3] * A0223)
            + ma[0][2] * (ma[1][0] * A1323 - ma[1][1] * A0323 + ma[1][3] * A0123)
            - ma[0][3] * (ma[1][0] * A1223 - ma[1][1] * A0223 + ma[1][2] * A0123);

        det = 1.f / det;

        mai[0][0] =   (ma[1][1] * A2323 - ma[1][2] * A1323 + ma[1][3] * A1223);
        mai[0][1] = - (ma[0][1] * A2323 - ma[0][2] * A1323 + ma[0][3] * A1223);
        mai[0][2] =   (ma[0][1] * A2313 - ma[0][2] * A1313 + ma[0][3] * A1213);
        mai[0][3] = - (ma[0][1] * A2312 - ma[0][2] * A1312 + ma[0][3] * A1212);
        mai[1][0] = - (ma[1][0] * A2323 - ma[1][2] * A0323 + ma[1][3] * A0223);
        mai[1][1] =   (ma[0][0] * A2323 - ma[0][2] * A0323 + ma[0][3] * A0223);
        mai[1][2] = - (ma[0][0] * A2313 - ma[0][2] * A0313 + ma[0][3] * A0213);
        mai[1][3] =   (ma[0][0] * A2312 - ma[0][2] * A0312 + ma[0][3] * A0212);
        mai[2][0] =   (ma[1][0] * A1323 - ma[1][1] * A0323 + ma[1][3] * A0123);
        mai[2][1] = - (ma[0][0] * A1323 - ma[0][1] * A0323 + ma[0][3] * A0123);
        mai[2][2] =   (ma[0][0] * A1313 - ma[0][1] * A0313 + ma[0][3] * A0113);
        mai[2][3] = - (ma[0][0] * A1312 - ma[0][1] * A0312 + ma[0][3] * A0112);
        mai[3][0] = - (ma[1][0] * A1223 - ma[1][1] * A0223 + ma[1][2] * A0123);
        mai[3][1] =   (ma[0][0] * A1223 - ma[0][1] * A0223 + ma[0][2] * A0123);
        mai[3][2] = - (ma[0][0] * A1213 - ma[0][1] * A0213 + ma[0][2] * A0113);
        mai[3][3] =   (ma[0][0] * A1212 - ma[0][1] * A0212 + ma[0][2] * A0112);
    }
    // *INDENT-ON*
    // clang-format on

    mm[0] = det * (mai[0][0] * mb[0] + mai[0][1] * mb[1] + mai[0][2] * mb[2] + mai[0][3] * mb[3]);
    mm[1] = det * (mai[1][0] * mb[0] + mai[1][1] * mb[1] + mai[1][2] * mb[2] + mai[1][3] * mb[3]);
    mm[2] = det * (mai[2][0] * mb[0] + mai[2][1] * mb[1] + mai[2][2] * mb[2] + mai[2][3] * mb[3]);
    mm[3] = det * (mai[3][0] * mb[0] + mai[3][1] * mb[1] + mai[3][2] * mb[2] + mai[3][3] * mb[3]);

    tm[0] = tm[4] = mm[0];
    tm[1] = -mm[1];
    tm[3] = mm[1];
    tm[2] = mm[2];
    tm[5] = mm[3];
}

void invert_affine_transform(const float* tm, float* tm_inv)
{
    float D = tm[0] * tm[4] - tm[1] * tm[3];
    D = D != 0.f ? 1.f / D : 0.f;

    float A11 = tm[4] * D;
    float A22 = tm[0] * D;
    float A12 = -tm[1] * D;
    float A21 = -tm[3] * D;
    float b1 = -A11 * tm[2] - A12 * tm[5];
    float b2 = -A21 * tm[2] - A22 * tm[5];

    tm_inv[0] = A11;
    tm_inv[1] = A12;
    tm_inv[2] = b1;
    tm_inv[3] = A21;
    tm_inv[4] = A22;
    tm_inv[5] = b2;
}

void warpaffine_bilinear_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int type, unsigned int v)
{
    return warpaffine_bilinear_c1(src, srcw, srch, srcw, dst, w, h, w, tm, type, v);
}

void warpaffine_bilinear_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int type, unsigned int v)
{
    return warpaffine_bilinear_c2(src, srcw, srch, srcw * 2, dst, w, h, w * 2, tm, type, v);
}

void warpaffine_bilinear_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int type, unsigned int v)
{
    return warpaffine_bilinear_c3(src, srcw, srch, srcw * 3, dst, w, h, w * 3, tm, type, v);
}

void warpaffine_bilinear_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int type, unsigned int v)
{
    return warpaffine_bilinear_c4(src, srcw, srch, srcw * 4, dst, w, h, w * 4, tm, type, v);
}

void warpaffine_bilinear_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, const float* tm, int type, unsigned int v)
{
    const unsigned char* border_color = (const unsigned char*)&v;
    const int wgap = stride - w;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dst;

#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X), SHRT_MIN), SHRT_MAX)
#define SATURATE_CAST_INT(X)   (int)::std::min(::std::max((int)((X) + ((X) >= 0.f ? 0.5f : -0.5f)), INT_MIN), INT_MAX)

    std::vector<int> adelta(w);
    std::vector<int> bdelta(w);
    for (int x = 0; x < w; x++)
    {
        adelta[x] = SATURATE_CAST_INT(tm[0] * x * (1 << 10));
        bdelta[x] = SATURATE_CAST_INT(tm[3] * x * (1 << 10));
    }

    int y = 0;
    for (; y < h; y++)
    {
        int X0 = SATURATE_CAST_INT((tm[1] * y + tm[2]) * (1 << 10));
        int Y0 = SATURATE_CAST_INT((tm[4] * y + tm[5]) * (1 << 10));

        int x = 0;
        for (; x + 7 < w; x += 8)
        {
            int sxy_inout = 0;
            {
                int X_0 = X0 + adelta[x];
                int Y_0 = Y0 + bdelta[x];
                int X_7 = X0 + adelta[x + 7];
                int Y_7 = Y0 + bdelta[x + 7];

                short sx_0 = SATURATE_CAST_SHORT((X_0 >> 10));
                short sy_0 = SATURATE_CAST_SHORT((Y_0 >> 10));
                short sx_7 = SATURATE_CAST_SHORT((X_7 >> 10));
                short sy_7 = SATURATE_CAST_SHORT((Y_7 >> 10));

                if (((unsigned short)sx_0 < srcw - 1 && (unsigned short)sy_0 < srch - 1) && ((unsigned short)sx_7 < srcw - 1 && (unsigned short)sy_7 < srch - 1))
                {
                    // all inside
                    sxy_inout = 1;
                }
                else if ((sx_0 < -1 && sx_7 < -1) || (sx_0 >= srcw && sx_7 >= srcw) || (sy_0 < -1 && sy_7 < -1) || (sy_0 >= srch && sy_7 >= srch))
                {
                    // all outside
                    sxy_inout = 2;
                }
            }

            if (sxy_inout == 1)
            {
                // all inside
#if __ARM_NEON
                int32x4_t _Xl = vaddq_s32(vdupq_n_s32(X0), vld1q_s32(adelta.data() + x));
                int32x4_t _Xh = vaddq_s32(vdupq_n_s32(X0), vld1q_s32(adelta.data() + x + 4));
                int32x4_t _Yl = vaddq_s32(vdupq_n_s32(Y0), vld1q_s32(bdelta.data() + x));
                int32x4_t _Yh = vaddq_s32(vdupq_n_s32(Y0), vld1q_s32(bdelta.data() + x + 4));

                int16x4_t _sxl = vqshrn_n_s32(_Xl, 10);
                int16x4_t _sxh = vqshrn_n_s32(_Xh, 10);
                int16x4_t _syl = vqshrn_n_s32(_Yl, 10);
                int16x4_t _syh = vqshrn_n_s32(_Yh, 10);

                uint32x4_t _v1024m1 = vdupq_n_u32((1 << 10) - 1);
                uint16x8_t _fx = vcombine_u16(vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Xl), _v1024m1)), vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Xh), _v1024m1)));
                uint16x8_t _fy = vcombine_u16(vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Yl), _v1024m1)), vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Yh), _v1024m1)));

                uint16x8_t _alpha0 = vsubq_u16(vdupq_n_u16(1 << 10), _fx);
                uint16x8_t _alpha1 = _fx;
                uint16x8_t _beta0 = vsubq_u16(vdupq_n_u16(1 << 10), _fy);
                uint16x8_t _beta1 = _fy;

                int16x4_t _srcstride = vdup_n_s16(srcstride);

                int32x4_t _a0l = vaddw_s16(vmull_s16(_srcstride, _syl), _sxl);
                int32x4_t _a0h = vaddw_s16(vmull_s16(_srcstride, _syh), _sxh);
                int32x4_t _b0l = vaddw_s16(_a0l, _srcstride);
                int32x4_t _b0h = vaddw_s16(_a0h, _srcstride);

                uint8x8x2_t _a0a1 = uint8x8x2_t();
                uint8x8x2_t _b0b1 = uint8x8x2_t();
                {
                    _a0a1 = vld2_lane_u8(src0 + vgetq_lane_s32(_a0l, 0), _a0a1, 0);
                    _b0b1 = vld2_lane_u8(src0 + vgetq_lane_s32(_b0l, 0), _b0b1, 0);

                    _a0a1 = vld2_lane_u8(src0 + vgetq_lane_s32(_a0l, 1), _a0a1, 1);
                    _b0b1 = vld2_lane_u8(src0 + vgetq_lane_s32(_b0l, 1), _b0b1, 1);

                    _a0a1 = vld2_lane_u8(src0 + vgetq_lane_s32(_a0l, 2), _a0a1, 2);
                    _b0b1 = vld2_lane_u8(src0 + vgetq_lane_s32(_b0l, 2), _b0b1, 2);

                    _a0a1 = vld2_lane_u8(src0 + vgetq_lane_s32(_a0l, 3), _a0a1, 3);
                    _b0b1 = vld2_lane_u8(src0 + vgetq_lane_s32(_b0l, 3), _b0b1, 3);

                    _a0a1 = vld2_lane_u8(src0 + vgetq_lane_s32(_a0h, 0), _a0a1, 4);
                    _b0b1 = vld2_lane_u8(src0 + vgetq_lane_s32(_b0h, 0), _b0b1, 4);

                    _a0a1 = vld2_lane_u8(src0 + vgetq_lane_s32(_a0h, 1), _a0a1, 5);
                    _b0b1 = vld2_lane_u8(src0 + vgetq_lane_s32(_b0h, 1), _b0b1, 5);

                    _a0a1 = vld2_lane_u8(src0 + vgetq_lane_s32(_a0h, 2), _a0a1, 6);
                    _b0b1 = vld2_lane_u8(src0 + vgetq_lane_s32(_b0h, 2), _b0b1, 6);

                    _a0a1 = vld2_lane_u8(src0 + vgetq_lane_s32(_a0h, 3), _a0a1, 7);
                    _b0b1 = vld2_lane_u8(src0 + vgetq_lane_s32(_b0h, 3), _b0b1, 7);
                }

                uint16x8_t _a0_0 = vmovl_u8(_a0a1.val[0]);
                uint16x8_t _a1_0 = vmovl_u8(_a0a1.val[1]);
                uint16x8_t _b0_0 = vmovl_u8(_b0b1.val[0]);
                uint16x8_t _b1_0 = vmovl_u8(_b0b1.val[1]);

                uint16x4_t _a00_0l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_a0_0), vget_low_u16(_alpha0)), vget_low_u16(_a1_0), vget_low_u16(_alpha1)), 5);
                uint16x4_t _a00_0h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_a0_0), vget_high_u16(_alpha0)), vget_high_u16(_a1_0), vget_high_u16(_alpha1)), 5);
                uint16x4_t _b00_0l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_b0_0), vget_low_u16(_alpha0)), vget_low_u16(_b1_0), vget_low_u16(_alpha1)), 5);
                uint16x4_t _b00_0h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_b0_0), vget_high_u16(_alpha0)), vget_high_u16(_b1_0), vget_high_u16(_alpha1)), 5);

                uint16x4_t _dst_0l = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_0l, vget_low_u16(_beta0)), _b00_0l, vget_low_u16(_beta1)), 15);
                uint16x4_t _dst_0h = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_0h, vget_high_u16(_beta0)), _b00_0h, vget_high_u16(_beta1)), 15);

                uint8x8_t _dst = vqmovn_u16(vcombine_u16(_dst_0l, _dst_0h));

                vst1_u8(dst0, _dst);

                dst0 += 8;
#else
                for (int xi = 0; xi < 8; xi++)
                {
                    int X = X0 + adelta[x + xi];
                    int Y = Y0 + bdelta[x + xi];

                    short sx = SATURATE_CAST_SHORT((X >> 10));
                    short sy = SATURATE_CAST_SHORT((Y >> 10));

                    short fx = X & ((1 << 10) - 1);
                    short fy = Y & ((1 << 10) - 1);

                    short alpha0 = (1 << 10) - fx;
                    short alpha1 = fx;

                    short beta0 = (1 << 10) - fy;
                    short beta1 = fy;

                    const unsigned char* a0 = src0 + srcstride * sy + sx;
                    const unsigned char* a1 = src0 + srcstride * sy + sx + 1;
                    const unsigned char* b0 = src0 + srcstride * (sy + 1) + sx;
                    const unsigned char* b1 = src0 + srcstride * (sy + 1) + sx + 1;

                    dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);

                    dst0 += 1;
                }
#endif // __ARM_NEON
            }
            else if (sxy_inout == 2)
            {
                // all outside
                if (type != -233)
                {
#if __ARM_NEON
                    uint8x8_t _border_color = vdup_n_u8(border_color[0]);
                    vst1_u8(dst0, _border_color);
#else
                    for (int xi = 0; xi < 8; xi++)
                    {
                        dst0[xi] = border_color[0];
                    }
#endif // __ARM_NEON
                }
                else
                {
                    // skip
                }

                dst0 += 8;
            }
            else // if (sxy_inout == 0)
            {
                for (int xi = 0; xi < 8; xi++)
                {
                    int X = X0 + adelta[x + xi];
                    int Y = Y0 + bdelta[x + xi];

                    short sx = SATURATE_CAST_SHORT((X >> 10));
                    short sy = SATURATE_CAST_SHORT((Y >> 10));

                    if (type != -233 && (sx < -1 || sx >= srcw || sy < -1 || sy >= srch))
                    {
                        dst0[0] = border_color[0];
                    }
                    else if (type == -233 && ((unsigned short)sx >= srcw - 1 || (unsigned short)sy >= srch - 1))
                    {
                        // skip
                    }
                    else
                    {
                        short fx = X & ((1 << 10) - 1);
                        short fy = Y & ((1 << 10) - 1);

                        short alpha0 = (1 << 10) - fx;
                        short alpha1 = fx;

                        short beta0 = (1 << 10) - fy;
                        short beta1 = fy;

                        short sx1 = sx + 1;
                        short sy1 = sy + 1;

                        const unsigned char* a0 = src0 + srcstride * sy + sx;
                        const unsigned char* a1 = src0 + srcstride * sy + sx + 1;
                        const unsigned char* b0 = src0 + srcstride * (sy + 1) + sx;
                        const unsigned char* b1 = src0 + srcstride * (sy + 1) + sx + 1;

                        if ((unsigned short)sx >= srcw || (unsigned short)sy >= srch)
                        {
                            a0 = type != -233 ? border_color : dst0;
                        }
                        if ((unsigned short)sx1 >= srcw || (unsigned short)sy >= srch)
                        {
                            a1 = type != -233 ? border_color : dst0;
                        }
                        if ((unsigned short)sx >= srcw || (unsigned short)sy1 >= srch)
                        {
                            b0 = type != -233 ? border_color : dst0;
                        }
                        if ((unsigned short)sx1 >= srcw || (unsigned short)sy1 >= srch)
                        {
                            b1 = type != -233 ? border_color : dst0;
                        }

                        dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
                    }

                    dst0 += 1;
                }
            }
        }
        for (; x < w; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));

            if (type != -233 && (sx < -1 || sx >= srcw || sy < -1 || sy >= srch))
            {
                dst0[0] = border_color[0];
            }
            else if (type == -233 && ((unsigned short)sx >= srcw - 1 || (unsigned short)sy >= srch - 1))
            {
                // skip
            }
            else
            {
                short fx = X & ((1 << 10) - 1);
                short fy = Y & ((1 << 10) - 1);

                short alpha0 = (1 << 10) - fx;
                short alpha1 = fx;

                short beta0 = (1 << 10) - fy;
                short beta1 = fy;

                short sx1 = sx + 1;
                short sy1 = sy + 1;

                const unsigned char* a0 = src0 + srcstride * sy + sx;
                const unsigned char* a1 = src0 + srcstride * sy + sx + 1;
                const unsigned char* b0 = src0 + srcstride * (sy + 1) + sx;
                const unsigned char* b1 = src0 + srcstride * (sy + 1) + sx + 1;

                if ((unsigned short)sx >= srcw || (unsigned short)sy >= srch)
                {
                    a0 = type != -233 ? border_color : dst0;
                }
                if ((unsigned short)sx1 >= srcw || (unsigned short)sy >= srch)
                {
                    a1 = type != -233 ? border_color : dst0;
                }
                if ((unsigned short)sx >= srcw || (unsigned short)sy1 >= srch)
                {
                    b0 = type != -233 ? border_color : dst0;
                }
                if ((unsigned short)sx1 >= srcw || (unsigned short)sy1 >= srch)
                {
                    b1 = type != -233 ? border_color : dst0;
                }

                dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
            }

            dst0 += 1;
        }

        dst0 += wgap;
    }

#undef SATURATE_CAST_SHORT
#undef SATURATE_CAST_INT
}

void warpaffine_bilinear_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, const float* tm, int type, unsigned int v)
{
    const unsigned char* border_color = (const unsigned char*)&v;
    const int wgap = stride - w * 2;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dst;

#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X), SHRT_MIN), SHRT_MAX)
#define SATURATE_CAST_INT(X)   (int)::std::min(::std::max((int)((X) + ((X) >= 0.f ? 0.5f : -0.5f)), INT_MIN), INT_MAX)

    std::vector<int> adelta(w);
    std::vector<int> bdelta(w);
    for (int x = 0; x < w; x++)
    {
        adelta[x] = SATURATE_CAST_INT(tm[0] * x * (1 << 10));
        bdelta[x] = SATURATE_CAST_INT(tm[3] * x * (1 << 10));
    }

    int y = 0;
    for (; y < h; y++)
    {
        int X0 = SATURATE_CAST_INT((tm[1] * y + tm[2]) * (1 << 10));
        int Y0 = SATURATE_CAST_INT((tm[4] * y + tm[5]) * (1 << 10));

        int x = 0;
        for (; x + 7 < w; x += 8)
        {
            int sxy_inout = 0;
            {
                int X_0 = X0 + adelta[x];
                int Y_0 = Y0 + bdelta[x];
                int X_7 = X0 + adelta[x + 7];
                int Y_7 = Y0 + bdelta[x + 7];

                short sx_0 = SATURATE_CAST_SHORT((X_0 >> 10));
                short sy_0 = SATURATE_CAST_SHORT((Y_0 >> 10));
                short sx_7 = SATURATE_CAST_SHORT((X_7 >> 10));
                short sy_7 = SATURATE_CAST_SHORT((Y_7 >> 10));

                if (((unsigned short)sx_0 < srcw - 1 && (unsigned short)sy_0 < srch - 1) && ((unsigned short)sx_7 < srcw - 1 && (unsigned short)sy_7 < srch - 1))
                {
                    // all inside
                    sxy_inout = 1;
                }
                else if ((sx_0 < -1 && sx_7 < -1) || (sx_0 >= srcw && sx_7 >= srcw) || (sy_0 < -1 && sy_7 < -1) || (sy_0 >= srch && sy_7 >= srch))
                {
                    // all outside
                    sxy_inout = 2;
                }
            }

            if (sxy_inout == 1)
            {
                // all inside
#if __ARM_NEON
                int32x4_t _Xl = vaddq_s32(vdupq_n_s32(X0), vld1q_s32(adelta.data() + x));
                int32x4_t _Xh = vaddq_s32(vdupq_n_s32(X0), vld1q_s32(adelta.data() + x + 4));
                int32x4_t _Yl = vaddq_s32(vdupq_n_s32(Y0), vld1q_s32(bdelta.data() + x));
                int32x4_t _Yh = vaddq_s32(vdupq_n_s32(Y0), vld1q_s32(bdelta.data() + x + 4));

                int16x4_t _sxl = vqshrn_n_s32(_Xl, 10);
                int16x4_t _sxh = vqshrn_n_s32(_Xh, 10);
                int16x4_t _syl = vqshrn_n_s32(_Yl, 10);
                int16x4_t _syh = vqshrn_n_s32(_Yh, 10);

                uint32x4_t _v1024m1 = vdupq_n_u32((1 << 10) - 1);
                uint16x8_t _fx = vcombine_u16(vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Xl), _v1024m1)), vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Xh), _v1024m1)));
                uint16x8_t _fy = vcombine_u16(vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Yl), _v1024m1)), vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Yh), _v1024m1)));

                uint16x8_t _alpha0 = vsubq_u16(vdupq_n_u16(1 << 10), _fx);
                uint16x8_t _alpha1 = _fx;
                uint16x8_t _beta0 = vsubq_u16(vdupq_n_u16(1 << 10), _fy);
                uint16x8_t _beta1 = _fy;

                int16x4_t _srcstride = vdup_n_s16(srcstride);
                int16x4_t _v2 = vdup_n_s16(2);

                int32x4_t _a0l = vmlal_s16(vmull_s16(_srcstride, _syl), _sxl, _v2);
                int32x4_t _a0h = vmlal_s16(vmull_s16(_srcstride, _syh), _sxh, _v2);
                int32x4_t _b0l = vaddw_s16(_a0l, _srcstride);
                int32x4_t _b0h = vaddw_s16(_a0h, _srcstride);

                uint8x8x4_t _a0a1 = uint8x8x4_t();
                uint8x8x4_t _b0b1 = uint8x8x4_t();
                {
                    _a0a1 = vld4_lane_u8(src0 + vgetq_lane_s32(_a0l, 0), _a0a1, 0);
                    _b0b1 = vld4_lane_u8(src0 + vgetq_lane_s32(_b0l, 0), _b0b1, 0);

                    _a0a1 = vld4_lane_u8(src0 + vgetq_lane_s32(_a0l, 1), _a0a1, 1);
                    _b0b1 = vld4_lane_u8(src0 + vgetq_lane_s32(_b0l, 1), _b0b1, 1);

                    _a0a1 = vld4_lane_u8(src0 + vgetq_lane_s32(_a0l, 2), _a0a1, 2);
                    _b0b1 = vld4_lane_u8(src0 + vgetq_lane_s32(_b0l, 2), _b0b1, 2);

                    _a0a1 = vld4_lane_u8(src0 + vgetq_lane_s32(_a0l, 3), _a0a1, 3);
                    _b0b1 = vld4_lane_u8(src0 + vgetq_lane_s32(_b0l, 3), _b0b1, 3);

                    _a0a1 = vld4_lane_u8(src0 + vgetq_lane_s32(_a0h, 0), _a0a1, 4);
                    _b0b1 = vld4_lane_u8(src0 + vgetq_lane_s32(_b0h, 0), _b0b1, 4);

                    _a0a1 = vld4_lane_u8(src0 + vgetq_lane_s32(_a0h, 1), _a0a1, 5);
                    _b0b1 = vld4_lane_u8(src0 + vgetq_lane_s32(_b0h, 1), _b0b1, 5);

                    _a0a1 = vld4_lane_u8(src0 + vgetq_lane_s32(_a0h, 2), _a0a1, 6);
                    _b0b1 = vld4_lane_u8(src0 + vgetq_lane_s32(_b0h, 2), _b0b1, 6);

                    _a0a1 = vld4_lane_u8(src0 + vgetq_lane_s32(_a0h, 3), _a0a1, 7);
                    _b0b1 = vld4_lane_u8(src0 + vgetq_lane_s32(_b0h, 3), _b0b1, 7);
                }

                uint16x8_t _a0_0 = vmovl_u8(_a0a1.val[0]);
                uint16x8_t _a0_1 = vmovl_u8(_a0a1.val[1]);
                uint16x8_t _a1_0 = vmovl_u8(_a0a1.val[2]);
                uint16x8_t _a1_1 = vmovl_u8(_a0a1.val[3]);
                uint16x8_t _b0_0 = vmovl_u8(_b0b1.val[0]);
                uint16x8_t _b0_1 = vmovl_u8(_b0b1.val[1]);
                uint16x8_t _b1_0 = vmovl_u8(_b0b1.val[2]);
                uint16x8_t _b1_1 = vmovl_u8(_b0b1.val[3]);

                uint16x4_t _a00_0l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_a0_0), vget_low_u16(_alpha0)), vget_low_u16(_a1_0), vget_low_u16(_alpha1)), 5);
                uint16x4_t _a00_1l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_a0_1), vget_low_u16(_alpha0)), vget_low_u16(_a1_1), vget_low_u16(_alpha1)), 5);
                uint16x4_t _a00_0h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_a0_0), vget_high_u16(_alpha0)), vget_high_u16(_a1_0), vget_high_u16(_alpha1)), 5);
                uint16x4_t _a00_1h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_a0_1), vget_high_u16(_alpha0)), vget_high_u16(_a1_1), vget_high_u16(_alpha1)), 5);
                uint16x4_t _b00_0l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_b0_0), vget_low_u16(_alpha0)), vget_low_u16(_b1_0), vget_low_u16(_alpha1)), 5);
                uint16x4_t _b00_1l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_b0_1), vget_low_u16(_alpha0)), vget_low_u16(_b1_1), vget_low_u16(_alpha1)), 5);
                uint16x4_t _b00_0h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_b0_0), vget_high_u16(_alpha0)), vget_high_u16(_b1_0), vget_high_u16(_alpha1)), 5);
                uint16x4_t _b00_1h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_b0_1), vget_high_u16(_alpha0)), vget_high_u16(_b1_1), vget_high_u16(_alpha1)), 5);

                uint16x4_t _dst_0l = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_0l, vget_low_u16(_beta0)), _b00_0l, vget_low_u16(_beta1)), 15);
                uint16x4_t _dst_1l = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_1l, vget_low_u16(_beta0)), _b00_1l, vget_low_u16(_beta1)), 15);
                uint16x4_t _dst_0h = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_0h, vget_high_u16(_beta0)), _b00_0h, vget_high_u16(_beta1)), 15);
                uint16x4_t _dst_1h = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_1h, vget_high_u16(_beta0)), _b00_1h, vget_high_u16(_beta1)), 15);

                uint8x8x2_t _dst;
                _dst.val[0] = vqmovn_u16(vcombine_u16(_dst_0l, _dst_0h));
                _dst.val[1] = vqmovn_u16(vcombine_u16(_dst_1l, _dst_1h));

                vst2_u8(dst0, _dst);

                dst0 += 2 * 8;
#else
                for (int xi = 0; xi < 8; xi++)
                {
                    int X = X0 + adelta[x + xi];
                    int Y = Y0 + bdelta[x + xi];

                    short sx = SATURATE_CAST_SHORT((X >> 10));
                    short sy = SATURATE_CAST_SHORT((Y >> 10));

                    short fx = X & ((1 << 10) - 1);
                    short fy = Y & ((1 << 10) - 1);

                    short alpha0 = (1 << 10) - fx;
                    short alpha1 = fx;

                    short beta0 = (1 << 10) - fy;
                    short beta1 = fy;

                    const unsigned char* a0 = src0 + srcstride * sy + sx * 2;
                    const unsigned char* a1 = src0 + srcstride * sy + sx * 2 + 2;
                    const unsigned char* b0 = src0 + srcstride * (sy + 1) + sx * 2;
                    const unsigned char* b1 = src0 + srcstride * (sy + 1) + sx * 2 + 2;

                    dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
                    dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);

                    dst0 += 2;
                }
#endif // __ARM_NEON
            }
            else if (sxy_inout == 2)
            {
                // all outside
                if (type != -233)
                {
#if __ARM_NEON
                    uint8x8x2_t _border_color;
                    _border_color.val[0] = vdup_n_u8(border_color[0]);
                    _border_color.val[1] = vdup_n_u8(border_color[1]);

                    vst2_u8(dst0, _border_color);
#else
                    for (int xi = 0; xi < 8; xi++)
                    {
                        dst0[xi * 2] = border_color[0];
                        dst0[xi * 2 + 1] = border_color[1];
                    }
#endif
                }
                else
                {
                    // skip
                }

                dst0 += 16;
            }
            else // if (sxy_inout == 0)
            {
                for (int xi = 0; xi < 8; xi++)
                {
                    int X = X0 + adelta[x + xi];
                    int Y = Y0 + bdelta[x + xi];

                    short sx = SATURATE_CAST_SHORT((X >> 10));
                    short sy = SATURATE_CAST_SHORT((Y >> 10));

                    if (type != -233 && (sx < -1 || sx >= srcw || sy < -1 || sy >= srch))
                    {
                        dst0[0] = border_color[0];
                        dst0[1] = border_color[1];
                    }
                    else if (type == -233 && ((unsigned short)sx >= srcw - 1 || (unsigned short)sy >= srch - 1))
                    {
                        // skip
                    }
                    else
                    {
                        short fx = X & ((1 << 10) - 1);
                        short fy = Y & ((1 << 10) - 1);

                        short alpha0 = (1 << 10) - fx;
                        short alpha1 = fx;

                        short beta0 = (1 << 10) - fy;
                        short beta1 = fy;

                        short sx1 = sx + 1;
                        short sy1 = sy + 1;

                        const unsigned char* a0 = src0 + srcstride * sy + sx * 2;
                        const unsigned char* a1 = src0 + srcstride * sy + sx * 2 + 2;
                        const unsigned char* b0 = src0 + srcstride * (sy + 1) + sx * 2;
                        const unsigned char* b1 = src0 + srcstride * (sy + 1) + sx * 2 + 2;

                        if ((unsigned short)sx >= srcw || (unsigned short)sy >= srch)
                        {
                            a0 = type != -233 ? border_color : dst0;
                        }
                        if ((unsigned short)sx1 >= srcw || (unsigned short)sy >= srch)
                        {
                            a1 = type != -233 ? border_color : dst0;
                        }
                        if ((unsigned short)sx >= srcw || (unsigned short)sy1 >= srch)
                        {
                            b0 = type != -233 ? border_color : dst0;
                        }
                        if ((unsigned short)sx1 >= srcw || (unsigned short)sy1 >= srch)
                        {
                            b1 = type != -233 ? border_color : dst0;
                        }

                        dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
                        dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);
                    }

                    dst0 += 2;
                }
            }
        }
        for (; x < w; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));

            if (type != -233 && (sx < -1 || sx >= srcw || sy < -1 || sy >= srch))
            {
                dst0[0] = border_color[0];
                dst0[1] = border_color[1];
            }
            else if (type == -233 && ((unsigned short)sx >= srcw - 1 || (unsigned short)sy >= srch - 1))
            {
                // skip
            }
            else
            {
                short fx = X & ((1 << 10) - 1);
                short fy = Y & ((1 << 10) - 1);

                short alpha0 = (1 << 10) - fx;
                short alpha1 = fx;

                short beta0 = (1 << 10) - fy;
                short beta1 = fy;

                short sx1 = sx + 1;
                short sy1 = sy + 1;

                const unsigned char* a0 = src0 + srcstride * sy + sx * 2;
                const unsigned char* a1 = src0 + srcstride * sy + sx * 2 + 2;
                const unsigned char* b0 = src0 + srcstride * (sy + 1) + sx * 2;
                const unsigned char* b1 = src0 + srcstride * (sy + 1) + sx * 2 + 2;

                if ((unsigned short)sx >= srcw || (unsigned short)sy >= srch)
                {
                    a0 = type != -233 ? border_color : dst0;
                }
                if ((unsigned short)sx1 >= srcw || (unsigned short)sy >= srch)
                {
                    a1 = type != -233 ? border_color : dst0;
                }
                if ((unsigned short)sx >= srcw || (unsigned short)sy1 >= srch)
                {
                    b0 = type != -233 ? border_color : dst0;
                }
                if ((unsigned short)sx1 >= srcw || (unsigned short)sy1 >= srch)
                {
                    b1 = type != -233 ? border_color : dst0;
                }

                dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
                dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);
            }

            dst0 += 2;
        }

        dst0 += wgap;
    }

#undef SATURATE_CAST_SHORT
#undef SATURATE_CAST_INT
}

void warpaffine_bilinear_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, const float* tm, int type, unsigned int v)
{
    const unsigned char* border_color = (const unsigned char*)&v;
    const int wgap = stride - w * 3;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dst;

#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X), SHRT_MIN), SHRT_MAX)
#define SATURATE_CAST_INT(X)   (int)::std::min(::std::max((int)((X) + ((X) >= 0.f ? 0.5f : -0.5f)), INT_MIN), INT_MAX)

    std::vector<int> adelta(w);
    std::vector<int> bdelta(w);
    for (int x = 0; x < w; x++)
    {
        adelta[x] = SATURATE_CAST_INT(tm[0] * x * (1 << 10));
        bdelta[x] = SATURATE_CAST_INT(tm[3] * x * (1 << 10));
    }

    int y = 0;
    for (; y < h; y++)
    {
        int X0 = SATURATE_CAST_INT((tm[1] * y + tm[2]) * (1 << 10));
        int Y0 = SATURATE_CAST_INT((tm[4] * y + tm[5]) * (1 << 10));

        int x = 0;
        for (; x + 7 < w; x += 8)
        {
            int sxy_inout = 0;
            {
                int X_0 = X0 + adelta[x];
                int Y_0 = Y0 + bdelta[x];
                int X_7 = X0 + adelta[x + 7];
                int Y_7 = Y0 + bdelta[x + 7];

                short sx_0 = SATURATE_CAST_SHORT((X_0 >> 10));
                short sy_0 = SATURATE_CAST_SHORT((Y_0 >> 10));
                short sx_7 = SATURATE_CAST_SHORT((X_7 >> 10));
                short sy_7 = SATURATE_CAST_SHORT((Y_7 >> 10));

                if (((unsigned short)sx_0 < srcw - 1 && (unsigned short)sy_0 < srch - 1) && ((unsigned short)sx_7 < srcw - 1 && (unsigned short)sy_7 < srch - 1))
                {
                    // all inside
                    sxy_inout = 1;
                }
                else if ((sx_0 < -1 && sx_7 < -1) || (sx_0 >= srcw && sx_7 >= srcw) || (sy_0 < -1 && sy_7 < -1) || (sy_0 >= srch && sy_7 >= srch))
                {
                    // all outside
                    sxy_inout = 2;
                }
            }

            if (sxy_inout == 1)
            {
                // all inside
#if __ARM_NEON
                int32x4_t _Xl = vaddq_s32(vdupq_n_s32(X0), vld1q_s32(adelta.data() + x));
                int32x4_t _Xh = vaddq_s32(vdupq_n_s32(X0), vld1q_s32(adelta.data() + x + 4));
                int32x4_t _Yl = vaddq_s32(vdupq_n_s32(Y0), vld1q_s32(bdelta.data() + x));
                int32x4_t _Yh = vaddq_s32(vdupq_n_s32(Y0), vld1q_s32(bdelta.data() + x + 4));

                int16x4_t _sxl = vqshrn_n_s32(_Xl, 10);
                int16x4_t _sxh = vqshrn_n_s32(_Xh, 10);
                int16x4_t _syl = vqshrn_n_s32(_Yl, 10);
                int16x4_t _syh = vqshrn_n_s32(_Yh, 10);

                uint32x4_t _v1024m1 = vdupq_n_u32((1 << 10) - 1);
                uint16x8_t _fx = vcombine_u16(vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Xl), _v1024m1)), vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Xh), _v1024m1)));
                uint16x8_t _fy = vcombine_u16(vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Yl), _v1024m1)), vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Yh), _v1024m1)));

                uint16x8_t _alpha0 = vsubq_u16(vdupq_n_u16(1 << 10), _fx);
                uint16x8_t _alpha1 = _fx;
                uint16x8_t _beta0 = vsubq_u16(vdupq_n_u16(1 << 10), _fy);
                uint16x8_t _beta1 = _fy;

                int16x4_t _srcstride = vdup_n_s16(srcstride);
                int16x4_t _v3 = vdup_n_s16(3);

                int32x4_t _a0l = vmlal_s16(vmull_s16(_srcstride, _syl), _sxl, _v3);
                int32x4_t _a0h = vmlal_s16(vmull_s16(_srcstride, _syh), _sxh, _v3);
                int32x4_t _b0l = vaddw_s16(_a0l, _srcstride);
                int32x4_t _b0h = vaddw_s16(_a0h, _srcstride);
                int32x4_t _a1l = vaddw_s16(_a0l, _v3);
                int32x4_t _a1h = vaddw_s16(_a0h, _v3);
                int32x4_t _b1l = vaddw_s16(_b0l, _v3);
                int32x4_t _b1h = vaddw_s16(_b0h, _v3);

                uint8x8x3_t _a0 = uint8x8x3_t();
                uint8x8x3_t _a1 = uint8x8x3_t();
                uint8x8x3_t _b0 = uint8x8x3_t();
                uint8x8x3_t _b1 = uint8x8x3_t();
                {
                    _a0 = vld3_lane_u8(src0 + vgetq_lane_s32(_a0l, 0), _a0, 0);
                    _a1 = vld3_lane_u8(src0 + vgetq_lane_s32(_a1l, 0), _a1, 0);
                    _b0 = vld3_lane_u8(src0 + vgetq_lane_s32(_b0l, 0), _b0, 0);
                    _b1 = vld3_lane_u8(src0 + vgetq_lane_s32(_b1l, 0), _b1, 0);

                    _a0 = vld3_lane_u8(src0 + vgetq_lane_s32(_a0l, 1), _a0, 1);
                    _a1 = vld3_lane_u8(src0 + vgetq_lane_s32(_a1l, 1), _a1, 1);
                    _b0 = vld3_lane_u8(src0 + vgetq_lane_s32(_b0l, 1), _b0, 1);
                    _b1 = vld3_lane_u8(src0 + vgetq_lane_s32(_b1l, 1), _b1, 1);

                    _a0 = vld3_lane_u8(src0 + vgetq_lane_s32(_a0l, 2), _a0, 2);
                    _a1 = vld3_lane_u8(src0 + vgetq_lane_s32(_a1l, 2), _a1, 2);
                    _b0 = vld3_lane_u8(src0 + vgetq_lane_s32(_b0l, 2), _b0, 2);
                    _b1 = vld3_lane_u8(src0 + vgetq_lane_s32(_b1l, 2), _b1, 2);

                    _a0 = vld3_lane_u8(src0 + vgetq_lane_s32(_a0l, 3), _a0, 3);
                    _a1 = vld3_lane_u8(src0 + vgetq_lane_s32(_a1l, 3), _a1, 3);
                    _b0 = vld3_lane_u8(src0 + vgetq_lane_s32(_b0l, 3), _b0, 3);
                    _b1 = vld3_lane_u8(src0 + vgetq_lane_s32(_b1l, 3), _b1, 3);

                    _a0 = vld3_lane_u8(src0 + vgetq_lane_s32(_a0h, 0), _a0, 4);
                    _a1 = vld3_lane_u8(src0 + vgetq_lane_s32(_a1h, 0), _a1, 4);
                    _b0 = vld3_lane_u8(src0 + vgetq_lane_s32(_b0h, 0), _b0, 4);
                    _b1 = vld3_lane_u8(src0 + vgetq_lane_s32(_b1h, 0), _b1, 4);

                    _a0 = vld3_lane_u8(src0 + vgetq_lane_s32(_a0h, 1), _a0, 5);
                    _a1 = vld3_lane_u8(src0 + vgetq_lane_s32(_a1h, 1), _a1, 5);
                    _b0 = vld3_lane_u8(src0 + vgetq_lane_s32(_b0h, 1), _b0, 5);
                    _b1 = vld3_lane_u8(src0 + vgetq_lane_s32(_b1h, 1), _b1, 5);

                    _a0 = vld3_lane_u8(src0 + vgetq_lane_s32(_a0h, 2), _a0, 6);
                    _a1 = vld3_lane_u8(src0 + vgetq_lane_s32(_a1h, 2), _a1, 6);
                    _b0 = vld3_lane_u8(src0 + vgetq_lane_s32(_b0h, 2), _b0, 6);
                    _b1 = vld3_lane_u8(src0 + vgetq_lane_s32(_b1h, 2), _b1, 6);

                    _a0 = vld3_lane_u8(src0 + vgetq_lane_s32(_a0h, 3), _a0, 7);
                    _a1 = vld3_lane_u8(src0 + vgetq_lane_s32(_a1h, 3), _a1, 7);
                    _b0 = vld3_lane_u8(src0 + vgetq_lane_s32(_b0h, 3), _b0, 7);
                    _b1 = vld3_lane_u8(src0 + vgetq_lane_s32(_b1h, 3), _b1, 7);
                }

                uint16x8_t _a0_0 = vmovl_u8(_a0.val[0]);
                uint16x8_t _a0_1 = vmovl_u8(_a0.val[1]);
                uint16x8_t _a0_2 = vmovl_u8(_a0.val[2]);
                uint16x8_t _a1_0 = vmovl_u8(_a1.val[0]);
                uint16x8_t _a1_1 = vmovl_u8(_a1.val[1]);
                uint16x8_t _a1_2 = vmovl_u8(_a1.val[2]);
                uint16x8_t _b0_0 = vmovl_u8(_b0.val[0]);
                uint16x8_t _b0_1 = vmovl_u8(_b0.val[1]);
                uint16x8_t _b0_2 = vmovl_u8(_b0.val[2]);
                uint16x8_t _b1_0 = vmovl_u8(_b1.val[0]);
                uint16x8_t _b1_1 = vmovl_u8(_b1.val[1]);
                uint16x8_t _b1_2 = vmovl_u8(_b1.val[2]);

                uint16x4_t _a00_0l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_a0_0), vget_low_u16(_alpha0)), vget_low_u16(_a1_0), vget_low_u16(_alpha1)), 5);
                uint16x4_t _a00_1l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_a0_1), vget_low_u16(_alpha0)), vget_low_u16(_a1_1), vget_low_u16(_alpha1)), 5);
                uint16x4_t _a00_2l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_a0_2), vget_low_u16(_alpha0)), vget_low_u16(_a1_2), vget_low_u16(_alpha1)), 5);
                uint16x4_t _a00_0h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_a0_0), vget_high_u16(_alpha0)), vget_high_u16(_a1_0), vget_high_u16(_alpha1)), 5);
                uint16x4_t _a00_1h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_a0_1), vget_high_u16(_alpha0)), vget_high_u16(_a1_1), vget_high_u16(_alpha1)), 5);
                uint16x4_t _a00_2h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_a0_2), vget_high_u16(_alpha0)), vget_high_u16(_a1_2), vget_high_u16(_alpha1)), 5);
                uint16x4_t _b00_0l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_b0_0), vget_low_u16(_alpha0)), vget_low_u16(_b1_0), vget_low_u16(_alpha1)), 5);
                uint16x4_t _b00_1l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_b0_1), vget_low_u16(_alpha0)), vget_low_u16(_b1_1), vget_low_u16(_alpha1)), 5);
                uint16x4_t _b00_2l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_b0_2), vget_low_u16(_alpha0)), vget_low_u16(_b1_2), vget_low_u16(_alpha1)), 5);
                uint16x4_t _b00_0h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_b0_0), vget_high_u16(_alpha0)), vget_high_u16(_b1_0), vget_high_u16(_alpha1)), 5);
                uint16x4_t _b00_1h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_b0_1), vget_high_u16(_alpha0)), vget_high_u16(_b1_1), vget_high_u16(_alpha1)), 5);
                uint16x4_t _b00_2h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_b0_2), vget_high_u16(_alpha0)), vget_high_u16(_b1_2), vget_high_u16(_alpha1)), 5);

                uint16x4_t _dst_0l = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_0l, vget_low_u16(_beta0)), _b00_0l, vget_low_u16(_beta1)), 15);
                uint16x4_t _dst_1l = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_1l, vget_low_u16(_beta0)), _b00_1l, vget_low_u16(_beta1)), 15);
                uint16x4_t _dst_2l = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_2l, vget_low_u16(_beta0)), _b00_2l, vget_low_u16(_beta1)), 15);
                uint16x4_t _dst_0h = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_0h, vget_high_u16(_beta0)), _b00_0h, vget_high_u16(_beta1)), 15);
                uint16x4_t _dst_1h = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_1h, vget_high_u16(_beta0)), _b00_1h, vget_high_u16(_beta1)), 15);
                uint16x4_t _dst_2h = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_2h, vget_high_u16(_beta0)), _b00_2h, vget_high_u16(_beta1)), 15);

                uint8x8x3_t _dst;
                _dst.val[0] = vqmovn_u16(vcombine_u16(_dst_0l, _dst_0h));
                _dst.val[1] = vqmovn_u16(vcombine_u16(_dst_1l, _dst_1h));
                _dst.val[2] = vqmovn_u16(vcombine_u16(_dst_2l, _dst_2h));

                vst3_u8(dst0, _dst);

                dst0 += 3 * 8;
#else
                for (int xi = 0; xi < 8; xi++)
                {
                    int X = X0 + adelta[x + xi];
                    int Y = Y0 + bdelta[x + xi];

                    short sx = SATURATE_CAST_SHORT((X >> 10));
                    short sy = SATURATE_CAST_SHORT((Y >> 10));

                    short fx = X & ((1 << 10) - 1);
                    short fy = Y & ((1 << 10) - 1);

                    short alpha0 = (1 << 10) - fx;
                    short alpha1 = fx;

                    short beta0 = (1 << 10) - fy;
                    short beta1 = fy;

                    const unsigned char* a0 = src0 + srcstride * sy + sx * 3;
                    const unsigned char* a1 = src0 + srcstride * sy + sx * 3 + 3;
                    const unsigned char* b0 = src0 + srcstride * (sy + 1) + sx * 3;
                    const unsigned char* b1 = src0 + srcstride * (sy + 1) + sx * 3 + 3;

                    dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
                    dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);
                    dst0[2] = (unsigned char)(((((unsigned short)((a0[2] * alpha0 + a1[2] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[2] * alpha0 + b1[2] * alpha1) >> 5) * beta1))) >> 15);

                    dst0 += 3;
                }
#endif // __ARM_NEON
            }
            else if (sxy_inout == 2)
            {
                // all outside
                if (type != -233)
                {
#if __ARM_NEON
                    uint8x8x3_t _border_color;
                    _border_color.val[0] = vdup_n_u8(border_color[0]);
                    _border_color.val[1] = vdup_n_u8(border_color[1]);
                    _border_color.val[2] = vdup_n_u8(border_color[2]);

                    vst3_u8(dst0, _border_color);
#else
                    for (int xi = 0; xi < 8; xi++)
                    {
                        dst0[xi * 3] = border_color[0];
                        dst0[xi * 3 + 1] = border_color[1];
                        dst0[xi * 3 + 2] = border_color[2];
                    }
#endif // __ARM_NEON
                }
                else
                {
                    // skip
                }

                dst0 += 24;
            }
            else // if (sxy_inout == 0)
            {
                for (int xi = 0; xi < 8; xi++)
                {
                    int X = X0 + adelta[x + xi];
                    int Y = Y0 + bdelta[x + xi];

                    short sx = SATURATE_CAST_SHORT((X >> 10));
                    short sy = SATURATE_CAST_SHORT((Y >> 10));

                    if (type != -233 && (sx < -1 || sx >= srcw || sy < -1 || sy >= srch))
                    {
                        dst0[0] = border_color[0];
                        dst0[1] = border_color[1];
                        dst0[2] = border_color[2];
                    }
                    else if (type == -233 && ((unsigned short)sx >= srcw - 1 || (unsigned short)sy >= srch - 1))
                    {
                        // skip
                    }
                    else
                    {
                        short fx = X & ((1 << 10) - 1);
                        short fy = Y & ((1 << 10) - 1);

                        short alpha0 = (1 << 10) - fx;
                        short alpha1 = fx;

                        short beta0 = (1 << 10) - fy;
                        short beta1 = fy;

                        short sx1 = sx + 1;
                        short sy1 = sy + 1;

                        const unsigned char* a0 = src0 + srcstride * sy + sx * 3;
                        const unsigned char* a1 = src0 + srcstride * sy + sx * 3 + 3;
                        const unsigned char* b0 = src0 + srcstride * (sy + 1) + sx * 3;
                        const unsigned char* b1 = src0 + srcstride * (sy + 1) + sx * 3 + 3;

                        if ((unsigned short)sx >= srcw || (unsigned short)sy >= srch)
                        {
                            a0 = type != -233 ? border_color : dst0;
                        }
                        if ((unsigned short)sx1 >= srcw || (unsigned short)sy >= srch)
                        {
                            a1 = type != -233 ? border_color : dst0;
                        }
                        if ((unsigned short)sx >= srcw || (unsigned short)sy1 >= srch)
                        {
                            b0 = type != -233 ? border_color : dst0;
                        }
                        if ((unsigned short)sx1 >= srcw || (unsigned short)sy1 >= srch)
                        {
                            b1 = type != -233 ? border_color : dst0;
                        }

                        dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
                        dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);
                        dst0[2] = (unsigned char)(((((unsigned short)((a0[2] * alpha0 + a1[2] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[2] * alpha0 + b1[2] * alpha1) >> 5) * beta1))) >> 15);
                    }

                    dst0 += 3;
                }
            }
        }
        for (; x < w; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));

            if (type != -233 && (sx < -1 || sx >= srcw || sy < -1 || sy >= srch))
            {
                dst0[0] = border_color[0];
                dst0[1] = border_color[1];
                dst0[2] = border_color[2];
            }
            else if (type == -233 && ((unsigned short)sx >= srcw - 1 || (unsigned short)sy >= srch - 1))
            {
                // skip
            }
            else
            {
                short fx = X & ((1 << 10) - 1);
                short fy = Y & ((1 << 10) - 1);

                short alpha0 = (1 << 10) - fx;
                short alpha1 = fx;

                short beta0 = (1 << 10) - fy;
                short beta1 = fy;

                short sx1 = sx + 1;
                short sy1 = sy + 1;

                const unsigned char* a0 = src0 + srcstride * sy + sx * 3;
                const unsigned char* a1 = src0 + srcstride * sy + sx * 3 + 3;
                const unsigned char* b0 = src0 + srcstride * (sy + 1) + sx * 3;
                const unsigned char* b1 = src0 + srcstride * (sy + 1) + sx * 3 + 3;

                if ((unsigned short)sx >= srcw || (unsigned short)sy >= srch)
                {
                    a0 = type != -233 ? border_color : dst0;
                }
                if ((unsigned short)sx1 >= srcw || (unsigned short)sy >= srch)
                {
                    a1 = type != -233 ? border_color : dst0;
                }
                if ((unsigned short)sx >= srcw || (unsigned short)sy1 >= srch)
                {
                    b0 = type != -233 ? border_color : dst0;
                }
                if ((unsigned short)sx1 >= srcw || (unsigned short)sy1 >= srch)
                {
                    b1 = type != -233 ? border_color : dst0;
                }

                dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
                dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);
                dst0[2] = (unsigned char)(((((unsigned short)((a0[2] * alpha0 + a1[2] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[2] * alpha0 + b1[2] * alpha1) >> 5) * beta1))) >> 15);
            }

            dst0 += 3;
        }

        dst0 += wgap;
    }

#undef SATURATE_CAST_SHORT
#undef SATURATE_CAST_INT
}

void warpaffine_bilinear_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, const float* tm, int type, unsigned int v)
{
    const unsigned char* border_color = (const unsigned char*)&v;
    const int wgap = stride - w * 4;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dst;

#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X), SHRT_MIN), SHRT_MAX)
#define SATURATE_CAST_INT(X)   (int)::std::min(::std::max((int)((X) + ((X) >= 0.f ? 0.5f : -0.5f)), INT_MIN), INT_MAX)

    std::vector<int> adelta(w);
    std::vector<int> bdelta(w);
    for (int x = 0; x < w; x++)
    {
        adelta[x] = SATURATE_CAST_INT(tm[0] * x * (1 << 10));
        bdelta[x] = SATURATE_CAST_INT(tm[3] * x * (1 << 10));
    }

    int y = 0;
    for (; y < h; y++)
    {
        int X0 = SATURATE_CAST_INT((tm[1] * y + tm[2]) * (1 << 10));
        int Y0 = SATURATE_CAST_INT((tm[4] * y + tm[5]) * (1 << 10));

        int x = 0;
        for (; x + 7 < w; x += 8)
        {
            int sxy_inout = 0;
            {
                int X_0 = X0 + adelta[x];
                int Y_0 = Y0 + bdelta[x];
                int X_7 = X0 + adelta[x + 7];
                int Y_7 = Y0 + bdelta[x + 7];

                short sx_0 = SATURATE_CAST_SHORT((X_0 >> 10));
                short sy_0 = SATURATE_CAST_SHORT((Y_0 >> 10));
                short sx_7 = SATURATE_CAST_SHORT((X_7 >> 10));
                short sy_7 = SATURATE_CAST_SHORT((Y_7 >> 10));

                if (((unsigned short)sx_0 < srcw - 1 && (unsigned short)sy_0 < srch - 1) && ((unsigned short)sx_7 < srcw - 1 && (unsigned short)sy_7 < srch - 1))
                {
                    // all inside
                    sxy_inout = 1;
                }
                else if ((sx_0 < -1 && sx_7 < -1) || (sx_0 >= srcw && sx_7 >= srcw) || (sy_0 < -1 && sy_7 < -1) || (sy_0 >= srch && sy_7 >= srch))
                {
                    // all outside
                    sxy_inout = 2;
                }
            }

            if (sxy_inout == 1)
            {
                // all inside
#if __ARM_NEON
                int32x4_t _Xl = vaddq_s32(vdupq_n_s32(X0), vld1q_s32(adelta.data() + x));
                int32x4_t _Xh = vaddq_s32(vdupq_n_s32(X0), vld1q_s32(adelta.data() + x + 4));
                int32x4_t _Yl = vaddq_s32(vdupq_n_s32(Y0), vld1q_s32(bdelta.data() + x));
                int32x4_t _Yh = vaddq_s32(vdupq_n_s32(Y0), vld1q_s32(bdelta.data() + x + 4));

                int16x4_t _sxl = vqshrn_n_s32(_Xl, 10);
                int16x4_t _sxh = vqshrn_n_s32(_Xh, 10);
                int16x4_t _syl = vqshrn_n_s32(_Yl, 10);
                int16x4_t _syh = vqshrn_n_s32(_Yh, 10);

                uint32x4_t _v1024m1 = vdupq_n_u32((1 << 10) - 1);
                uint16x8_t _fx = vcombine_u16(vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Xl), _v1024m1)), vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Xh), _v1024m1)));
                uint16x8_t _fy = vcombine_u16(vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Yl), _v1024m1)), vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Yh), _v1024m1)));

                uint16x8_t _alpha0 = vsubq_u16(vdupq_n_u16(1 << 10), _fx);
                uint16x8_t _alpha1 = _fx;
                uint16x8_t _beta0 = vsubq_u16(vdupq_n_u16(1 << 10), _fy);
                uint16x8_t _beta1 = _fy;

                int16x4_t _srcstride = vdup_n_s16(srcstride);
                int16x4_t _v4 = vdup_n_s16(4);

                int32x4_t _a0l = vmlal_s16(vmull_s16(_srcstride, _syl), _sxl, _v4);
                int32x4_t _a0h = vmlal_s16(vmull_s16(_srcstride, _syh), _sxh, _v4);
                int32x4_t _b0l = vaddw_s16(_a0l, _srcstride);
                int32x4_t _b0h = vaddw_s16(_a0h, _srcstride);

                uint8x8x4_t _a0 = uint8x8x4_t();
                uint8x8x4_t _a1 = uint8x8x4_t();
                uint8x8x4_t _b0 = uint8x8x4_t();
                uint8x8x4_t _b1 = uint8x8x4_t();
                {
                    uint8x8_t _a0a1_0 = vld1_u8(src0 + vgetq_lane_s32(_a0l, 0));
                    uint8x8_t _a0a1_1 = vld1_u8(src0 + vgetq_lane_s32(_a0l, 1));
                    uint8x8_t _a0a1_2 = vld1_u8(src0 + vgetq_lane_s32(_a0l, 2));
                    uint8x8_t _a0a1_3 = vld1_u8(src0 + vgetq_lane_s32(_a0l, 3));
                    uint8x8_t _a0a1_4 = vld1_u8(src0 + vgetq_lane_s32(_a0h, 0));
                    uint8x8_t _a0a1_5 = vld1_u8(src0 + vgetq_lane_s32(_a0h, 1));
                    uint8x8_t _a0a1_6 = vld1_u8(src0 + vgetq_lane_s32(_a0h, 2));
                    uint8x8_t _a0a1_7 = vld1_u8(src0 + vgetq_lane_s32(_a0h, 3));

                    // transpose 8x8
                    uint8x8x2_t _a0a101t_r = vtrn_u8(_a0a1_0, _a0a1_1);
                    uint8x8x2_t _a0a123t_r = vtrn_u8(_a0a1_2, _a0a1_3);
                    uint8x8x2_t _a0a145t_r = vtrn_u8(_a0a1_4, _a0a1_5);
                    uint8x8x2_t _a0a167t_r = vtrn_u8(_a0a1_6, _a0a1_7);

                    uint16x4x2_t _a0a102tt_r = vtrn_u16(vreinterpret_u16_u8(_a0a101t_r.val[0]), vreinterpret_u16_u8(_a0a123t_r.val[0]));
                    uint16x4x2_t _a0a113tt_r = vtrn_u16(vreinterpret_u16_u8(_a0a101t_r.val[1]), vreinterpret_u16_u8(_a0a123t_r.val[1]));
                    uint16x4x2_t _a0a146tt_r = vtrn_u16(vreinterpret_u16_u8(_a0a145t_r.val[0]), vreinterpret_u16_u8(_a0a167t_r.val[0]));
                    uint16x4x2_t _a0a157tt_r = vtrn_u16(vreinterpret_u16_u8(_a0a145t_r.val[1]), vreinterpret_u16_u8(_a0a167t_r.val[1]));

                    uint32x2x2_t _a0a104ttt_r = vtrn_u32(vreinterpret_u32_u16(_a0a102tt_r.val[0]), vreinterpret_u32_u16(_a0a146tt_r.val[0]));
                    uint32x2x2_t _a0a115ttt_r = vtrn_u32(vreinterpret_u32_u16(_a0a113tt_r.val[0]), vreinterpret_u32_u16(_a0a157tt_r.val[0]));
                    uint32x2x2_t _a0a126ttt_r = vtrn_u32(vreinterpret_u32_u16(_a0a102tt_r.val[1]), vreinterpret_u32_u16(_a0a146tt_r.val[1]));
                    uint32x2x2_t _a0a137ttt_r = vtrn_u32(vreinterpret_u32_u16(_a0a113tt_r.val[1]), vreinterpret_u32_u16(_a0a157tt_r.val[1]));

                    _a0.val[0] = vreinterpret_u8_u32(_a0a104ttt_r.val[0]);
                    _a0.val[1] = vreinterpret_u8_u32(_a0a115ttt_r.val[0]);
                    _a0.val[2] = vreinterpret_u8_u32(_a0a126ttt_r.val[0]);
                    _a0.val[3] = vreinterpret_u8_u32(_a0a137ttt_r.val[0]);
                    _a1.val[0] = vreinterpret_u8_u32(_a0a104ttt_r.val[1]);
                    _a1.val[1] = vreinterpret_u8_u32(_a0a115ttt_r.val[1]);
                    _a1.val[2] = vreinterpret_u8_u32(_a0a126ttt_r.val[1]);
                    _a1.val[3] = vreinterpret_u8_u32(_a0a137ttt_r.val[1]);

                    uint8x8_t _b0b1_0 = vld1_u8(src0 + vgetq_lane_s32(_b0l, 0));
                    uint8x8_t _b0b1_1 = vld1_u8(src0 + vgetq_lane_s32(_b0l, 1));
                    uint8x8_t _b0b1_2 = vld1_u8(src0 + vgetq_lane_s32(_b0l, 2));
                    uint8x8_t _b0b1_3 = vld1_u8(src0 + vgetq_lane_s32(_b0l, 3));
                    uint8x8_t _b0b1_4 = vld1_u8(src0 + vgetq_lane_s32(_b0h, 0));
                    uint8x8_t _b0b1_5 = vld1_u8(src0 + vgetq_lane_s32(_b0h, 1));
                    uint8x8_t _b0b1_6 = vld1_u8(src0 + vgetq_lane_s32(_b0h, 2));
                    uint8x8_t _b0b1_7 = vld1_u8(src0 + vgetq_lane_s32(_b0h, 3));

                    // transpose 8x8
                    uint8x8x2_t _b0b101t_r = vtrn_u8(_b0b1_0, _b0b1_1);
                    uint8x8x2_t _b0b123t_r = vtrn_u8(_b0b1_2, _b0b1_3);
                    uint8x8x2_t _b0b145t_r = vtrn_u8(_b0b1_4, _b0b1_5);
                    uint8x8x2_t _b0b167t_r = vtrn_u8(_b0b1_6, _b0b1_7);

                    uint16x4x2_t _b0b102tt_r = vtrn_u16(vreinterpret_u16_u8(_b0b101t_r.val[0]), vreinterpret_u16_u8(_b0b123t_r.val[0]));
                    uint16x4x2_t _b0b113tt_r = vtrn_u16(vreinterpret_u16_u8(_b0b101t_r.val[1]), vreinterpret_u16_u8(_b0b123t_r.val[1]));
                    uint16x4x2_t _b0b146tt_r = vtrn_u16(vreinterpret_u16_u8(_b0b145t_r.val[0]), vreinterpret_u16_u8(_b0b167t_r.val[0]));
                    uint16x4x2_t _b0b157tt_r = vtrn_u16(vreinterpret_u16_u8(_b0b145t_r.val[1]), vreinterpret_u16_u8(_b0b167t_r.val[1]));

                    uint32x2x2_t _b0b104ttt_r = vtrn_u32(vreinterpret_u32_u16(_b0b102tt_r.val[0]), vreinterpret_u32_u16(_b0b146tt_r.val[0]));
                    uint32x2x2_t _b0b115ttt_r = vtrn_u32(vreinterpret_u32_u16(_b0b113tt_r.val[0]), vreinterpret_u32_u16(_b0b157tt_r.val[0]));
                    uint32x2x2_t _b0b126ttt_r = vtrn_u32(vreinterpret_u32_u16(_b0b102tt_r.val[1]), vreinterpret_u32_u16(_b0b146tt_r.val[1]));
                    uint32x2x2_t _b0b137ttt_r = vtrn_u32(vreinterpret_u32_u16(_b0b113tt_r.val[1]), vreinterpret_u32_u16(_b0b157tt_r.val[1]));

                    _b0.val[0] = vreinterpret_u8_u32(_b0b104ttt_r.val[0]);
                    _b0.val[1] = vreinterpret_u8_u32(_b0b115ttt_r.val[0]);
                    _b0.val[2] = vreinterpret_u8_u32(_b0b126ttt_r.val[0]);
                    _b0.val[3] = vreinterpret_u8_u32(_b0b137ttt_r.val[0]);
                    _b1.val[0] = vreinterpret_u8_u32(_b0b104ttt_r.val[1]);
                    _b1.val[1] = vreinterpret_u8_u32(_b0b115ttt_r.val[1]);
                    _b1.val[2] = vreinterpret_u8_u32(_b0b126ttt_r.val[1]);
                    _b1.val[3] = vreinterpret_u8_u32(_b0b137ttt_r.val[1]);
                }

                uint16x8_t _a0_0 = vmovl_u8(_a0.val[0]);
                uint16x8_t _a0_1 = vmovl_u8(_a0.val[1]);
                uint16x8_t _a0_2 = vmovl_u8(_a0.val[2]);
                uint16x8_t _a0_3 = vmovl_u8(_a0.val[3]);
                uint16x8_t _a1_0 = vmovl_u8(_a1.val[0]);
                uint16x8_t _a1_1 = vmovl_u8(_a1.val[1]);
                uint16x8_t _a1_2 = vmovl_u8(_a1.val[2]);
                uint16x8_t _a1_3 = vmovl_u8(_a1.val[3]);
                uint16x8_t _b0_0 = vmovl_u8(_b0.val[0]);
                uint16x8_t _b0_1 = vmovl_u8(_b0.val[1]);
                uint16x8_t _b0_2 = vmovl_u8(_b0.val[2]);
                uint16x8_t _b0_3 = vmovl_u8(_b0.val[3]);
                uint16x8_t _b1_0 = vmovl_u8(_b1.val[0]);
                uint16x8_t _b1_1 = vmovl_u8(_b1.val[1]);
                uint16x8_t _b1_2 = vmovl_u8(_b1.val[2]);
                uint16x8_t _b1_3 = vmovl_u8(_b1.val[3]);

                uint16x4_t _a00_0l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_a0_0), vget_low_u16(_alpha0)), vget_low_u16(_a1_0), vget_low_u16(_alpha1)), 5);
                uint16x4_t _a00_1l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_a0_1), vget_low_u16(_alpha0)), vget_low_u16(_a1_1), vget_low_u16(_alpha1)), 5);
                uint16x4_t _a00_2l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_a0_2), vget_low_u16(_alpha0)), vget_low_u16(_a1_2), vget_low_u16(_alpha1)), 5);
                uint16x4_t _a00_3l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_a0_3), vget_low_u16(_alpha0)), vget_low_u16(_a1_3), vget_low_u16(_alpha1)), 5);
                uint16x4_t _a00_0h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_a0_0), vget_high_u16(_alpha0)), vget_high_u16(_a1_0), vget_high_u16(_alpha1)), 5);
                uint16x4_t _a00_1h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_a0_1), vget_high_u16(_alpha0)), vget_high_u16(_a1_1), vget_high_u16(_alpha1)), 5);
                uint16x4_t _a00_2h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_a0_2), vget_high_u16(_alpha0)), vget_high_u16(_a1_2), vget_high_u16(_alpha1)), 5);
                uint16x4_t _a00_3h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_a0_3), vget_high_u16(_alpha0)), vget_high_u16(_a1_3), vget_high_u16(_alpha1)), 5);
                uint16x4_t _b00_0l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_b0_0), vget_low_u16(_alpha0)), vget_low_u16(_b1_0), vget_low_u16(_alpha1)), 5);
                uint16x4_t _b00_1l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_b0_1), vget_low_u16(_alpha0)), vget_low_u16(_b1_1), vget_low_u16(_alpha1)), 5);
                uint16x4_t _b00_2l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_b0_2), vget_low_u16(_alpha0)), vget_low_u16(_b1_2), vget_low_u16(_alpha1)), 5);
                uint16x4_t _b00_3l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_b0_3), vget_low_u16(_alpha0)), vget_low_u16(_b1_3), vget_low_u16(_alpha1)), 5);
                uint16x4_t _b00_0h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_b0_0), vget_high_u16(_alpha0)), vget_high_u16(_b1_0), vget_high_u16(_alpha1)), 5);
                uint16x4_t _b00_1h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_b0_1), vget_high_u16(_alpha0)), vget_high_u16(_b1_1), vget_high_u16(_alpha1)), 5);
                uint16x4_t _b00_2h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_b0_2), vget_high_u16(_alpha0)), vget_high_u16(_b1_2), vget_high_u16(_alpha1)), 5);
                uint16x4_t _b00_3h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_b0_3), vget_high_u16(_alpha0)), vget_high_u16(_b1_3), vget_high_u16(_alpha1)), 5);

                uint16x4_t _dst_0l = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_0l, vget_low_u16(_beta0)), _b00_0l, vget_low_u16(_beta1)), 15);
                uint16x4_t _dst_1l = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_1l, vget_low_u16(_beta0)), _b00_1l, vget_low_u16(_beta1)), 15);
                uint16x4_t _dst_2l = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_2l, vget_low_u16(_beta0)), _b00_2l, vget_low_u16(_beta1)), 15);
                uint16x4_t _dst_3l = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_3l, vget_low_u16(_beta0)), _b00_3l, vget_low_u16(_beta1)), 15);
                uint16x4_t _dst_0h = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_0h, vget_high_u16(_beta0)), _b00_0h, vget_high_u16(_beta1)), 15);
                uint16x4_t _dst_1h = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_1h, vget_high_u16(_beta0)), _b00_1h, vget_high_u16(_beta1)), 15);
                uint16x4_t _dst_2h = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_2h, vget_high_u16(_beta0)), _b00_2h, vget_high_u16(_beta1)), 15);
                uint16x4_t _dst_3h = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_3h, vget_high_u16(_beta0)), _b00_3h, vget_high_u16(_beta1)), 15);

                uint8x8x4_t _dst;
                _dst.val[0] = vqmovn_u16(vcombine_u16(_dst_0l, _dst_0h));
                _dst.val[1] = vqmovn_u16(vcombine_u16(_dst_1l, _dst_1h));
                _dst.val[2] = vqmovn_u16(vcombine_u16(_dst_2l, _dst_2h));
                _dst.val[3] = vqmovn_u16(vcombine_u16(_dst_3l, _dst_3h));

                vst4_u8(dst0, _dst);

                dst0 += 4 * 8;
#else
                for (int xi = 0; xi < 8; xi++)
                {
                    int X = X0 + adelta[x + xi];
                    int Y = Y0 + bdelta[x + xi];

                    short sx = SATURATE_CAST_SHORT((X >> 10));
                    short sy = SATURATE_CAST_SHORT((Y >> 10));

                    short fx = X & ((1 << 10) - 1);
                    short fy = Y & ((1 << 10) - 1);

                    short alpha0 = (1 << 10) - fx;
                    short alpha1 = fx;

                    short beta0 = (1 << 10) - fy;
                    short beta1 = fy;

                    const unsigned char* a0 = src0 + srcstride * sy + sx * 4;
                    const unsigned char* a1 = src0 + srcstride * sy + sx * 4 + 4;
                    const unsigned char* b0 = src0 + srcstride * (sy + 1) + sx * 4;
                    const unsigned char* b1 = src0 + srcstride * (sy + 1) + sx * 4 + 4;

                    dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
                    dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);
                    dst0[2] = (unsigned char)(((((unsigned short)((a0[2] * alpha0 + a1[2] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[2] * alpha0 + b1[2] * alpha1) >> 5) * beta1))) >> 15);
                    dst0[3] = (unsigned char)(((((unsigned short)((a0[3] * alpha0 + a1[3] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[3] * alpha0 + b1[3] * alpha1) >> 5) * beta1))) >> 15);

                    dst0 += 4;
                }
#endif // __ARM_NEON
            }
            else if (sxy_inout == 2)
            {
                // all outside
                if (type != -233)
                {
#if __ARM_NEON
                    uint8x8x4_t _border_color;
                    _border_color.val[0] = vdup_n_u8(border_color[0]);
                    _border_color.val[1] = vdup_n_u8(border_color[1]);
                    _border_color.val[2] = vdup_n_u8(border_color[2]);
                    _border_color.val[3] = vdup_n_u8(border_color[3]);

                    vst4_u8(dst0, _border_color);
#else
                    for (int xi = 0; xi < 8; xi++)
                    {
                        dst0[xi * 4] = border_color[0];
                        dst0[xi * 4 + 1] = border_color[1];
                        dst0[xi * 4 + 2] = border_color[2];
                        dst0[xi * 4 + 3] = border_color[3];
                    }
#endif // __ARM_NEON
                }
                else
                {
                    // skip
                }

                dst0 += 32;
            }
            else // if (sxy_inout == 0)
            {
                for (int xi = 0; xi < 8; xi++)
                {
                    int X = X0 + adelta[x + xi];
                    int Y = Y0 + bdelta[x + xi];

                    short sx = SATURATE_CAST_SHORT((X >> 10));
                    short sy = SATURATE_CAST_SHORT((Y >> 10));

                    if (type != -233 && (sx < -1 || sx >= srcw || sy < -1 || sy >= srch))
                    {
                        dst0[0] = border_color[0];
                        dst0[1] = border_color[1];
                        dst0[2] = border_color[2];
                        dst0[3] = border_color[3];
                    }
                    else if (type == -233 && ((unsigned short)sx >= srcw - 1 || (unsigned short)sy >= srch - 1))
                    {
                        // skip
                    }
                    else
                    {
                        short fx = X & ((1 << 10) - 1);
                        short fy = Y & ((1 << 10) - 1);

                        short alpha0 = (1 << 10) - fx;
                        short alpha1 = fx;

                        short beta0 = (1 << 10) - fy;
                        short beta1 = fy;

                        short sx1 = sx + 1;
                        short sy1 = sy + 1;

                        const unsigned char* a0 = src0 + srcstride * sy + sx * 4;
                        const unsigned char* a1 = src0 + srcstride * sy + sx * 4 + 4;
                        const unsigned char* b0 = src0 + srcstride * (sy + 1) + sx * 4;
                        const unsigned char* b1 = src0 + srcstride * (sy + 1) + sx * 4 + 4;

                        if ((unsigned short)sx >= srcw || (unsigned short)sy >= srch)
                        {
                            a0 = type != -233 ? border_color : dst0;
                        }
                        if ((unsigned short)sx1 >= srcw || (unsigned short)sy >= srch)
                        {
                            a1 = type != -233 ? border_color : dst0;
                        }
                        if ((unsigned short)sx >= srcw || (unsigned short)sy1 >= srch)
                        {
                            b0 = type != -233 ? border_color : dst0;
                        }
                        if ((unsigned short)sx1 >= srcw || (unsigned short)sy1 >= srch)
                        {
                            b1 = type != -233 ? border_color : dst0;
                        }

                        dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
                        dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);
                        dst0[2] = (unsigned char)(((((unsigned short)((a0[2] * alpha0 + a1[2] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[2] * alpha0 + b1[2] * alpha1) >> 5) * beta1))) >> 15);
                        dst0[3] = (unsigned char)(((((unsigned short)((a0[3] * alpha0 + a1[3] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[3] * alpha0 + b1[3] * alpha1) >> 5) * beta1))) >> 15);
                    }

                    dst0 += 4;
                }
            }
        }
        for (; x < w; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));

            if (type != -233 && (sx < -1 || sx >= srcw || sy < -1 || sy >= srch))
            {
                dst0[0] = border_color[0];
                dst0[1] = border_color[1];
                dst0[2] = border_color[2];
                dst0[3] = border_color[3];
            }
            else if (type == -233 && ((unsigned short)sx >= srcw - 1 || (unsigned short)sy >= srch - 1))
            {
                // skip
            }
            else
            {
                short fx = X & ((1 << 10) - 1);
                short fy = Y & ((1 << 10) - 1);

                short alpha0 = (1 << 10) - fx;
                short alpha1 = fx;

                short beta0 = (1 << 10) - fy;
                short beta1 = fy;

                short sx1 = sx + 1;
                short sy1 = sy + 1;

                const unsigned char* a0 = src0 + srcstride * sy + sx * 4;
                const unsigned char* a1 = src0 + srcstride * sy + sx * 4 + 4;
                const unsigned char* b0 = src0 + srcstride * (sy + 1) + sx * 4;
                const unsigned char* b1 = src0 + srcstride * (sy + 1) + sx * 4 + 4;

                if ((unsigned short)sx >= srcw || (unsigned short)sy >= srch)
                {
                    a0 = type != -233 ? border_color : dst0;
                }
                if ((unsigned short)sx1 >= srcw || (unsigned short)sy >= srch)
                {
                    a1 = type != -233 ? border_color : dst0;
                }
                if ((unsigned short)sx >= srcw || (unsigned short)sy1 >= srch)
                {
                    b0 = type != -233 ? border_color : dst0;
                }
                if ((unsigned short)sx1 >= srcw || (unsigned short)sy1 >= srch)
                {
                    b1 = type != -233 ? border_color : dst0;
                }

                dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
                dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);
                dst0[2] = (unsigned char)(((((unsigned short)((a0[2] * alpha0 + a1[2] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[2] * alpha0 + b1[2] * alpha1) >> 5) * beta1))) >> 15);
                dst0[3] = (unsigned char)(((((unsigned short)((a0[3] * alpha0 + a1[3] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[3] * alpha0 + b1[3] * alpha1) >> 5) * beta1))) >> 15);
            }

            dst0 += 4;
        }

        dst0 += wgap;
    }

#undef SATURATE_CAST_SHORT
#undef SATURATE_CAST_INT
}

void warpaffine_bilinear_yuv420sp(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int type, unsigned int v)
{
    // assert srcw % 2 == 0
    // assert srch % 2 == 0
    // assert w % 2 == 0
    // assert h % 2 == 0

    const unsigned char* border_color = (const unsigned char*)&v;

    unsigned int v_y;
    unsigned int v_uv;
    unsigned char* border_color_y = (unsigned char*)&v_y;
    unsigned char* border_color_uv = (unsigned char*)&v_uv;
    border_color_y[0] = border_color[0];
    border_color_uv[0] = border_color[1];
    border_color_uv[1] = border_color[2];

    const unsigned char* srcY = src;
    unsigned char* dstY = dst;
    warpaffine_bilinear_c1(srcY, srcw, srch, dstY, w, h, tm, type, v_y);

    const float tm_uv[6] = {
        tm[0],
        tm[1],
        tm[2] / 2.0f,
        tm[3],
        tm[4],
        tm[5] / 2.0f,
    };

    const unsigned char* srcUV = src + srcw * srch;
    unsigned char* dstUV = dst + w * h;
    warpaffine_bilinear_c2(srcUV, srcw / 2, srch / 2, dstUV, w / 2, h / 2, tm_uv, type, v_uv);
}
#endif // NCNN_PIXEL_AFFINE

} // namespace ncnn
