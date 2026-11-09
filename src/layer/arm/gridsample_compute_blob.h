// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static float grid_sample_unormalize_arm(int w, float coordx, int align_corner)
{
    return align_corner ? (coordx + 1) / 2.f * (w - 1) : ((coordx + 1) * w - 1) / 2.f;
}

static float border_coord_arm(float x, float border)
{
    if (x < 0.f)
        return 0.f;
    if (x > border)
        return border;
    return x;
}

static float reflect_coord_arm(float x, int high)
{
    x = fabsf(x);
    x = high - fabsf(x - high);
    return x;
}

static float compute_coord_arm(float sx, int w, int padding_mode, int align_corner)
{
    if (padding_mode == GridSample::Padding_BORDER)
    {
        sx = border_coord_arm(sx, w - 1);
    }
    else if (padding_mode == GridSample::Padding_REFLECTION)
    {
        if (align_corner)
        {
            sx = reflect_coord_arm(sx, w - 1);
        }
        else
        {
            sx = reflect_coord_arm(sx + 0.5f, w) - 0.5f;
            sx = border_coord_arm(sx, w - 1);
        }
    }

    return sx;
}

static bool in_bounds_arm(int x, int y, int w, int h)
{
    return x >= 0 && y >= 0 && x < w && y < h;
}

static bool in_bounds_arm(int x, int y, int z, int w, int h, int d)
{
    return x >= 0 && y >= 0 && z >= 0 && x < w && y < h && z < d;
}

static void gridsample_store_2d_bilinear_arm(float* offset_value_ptr, float sample_x, float sample_y, int w, int h, int elempack)
{
    int x0 = (int)floorf(sample_x);
    int y0 = (int)floorf(sample_y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    int* offset_ptr = (int*)offset_value_ptr;
    offset_ptr[0] = in_bounds_arm(x0, y0, w, h) ? (x0 + y0 * w) * elempack : -1;
    offset_ptr[1] = in_bounds_arm(x1, y0, w, h) ? (x1 + y0 * w) * elempack : -1;
    offset_ptr[2] = in_bounds_arm(x0, y1, w, h) ? (x0 + y1 * w) * elempack : -1;
    offset_ptr[3] = in_bounds_arm(x1, y1, w, h) ? (x1 + y1 * w) * elempack : -1;

    offset_value_ptr[4] = sample_x - x0;
    offset_value_ptr[5] = sample_y - y0;
}

static void gridsample_store_2d_nearest_arm(int* offset_ptr, float sample_x, float sample_y, int w, int h, int elempack)
{
    int x0 = (int)floorf(sample_x + 0.5f);
    int y0 = (int)floorf(sample_y + 0.5f);

    offset_ptr[0] = in_bounds_arm(x0, y0, w, h) ? (x0 + y0 * w) * elempack : -1;
}

static void gridsample_store_3d_bilinear_arm(float* offset_value_ptr, float sample_x, float sample_y, float sample_z, int w, int h, int d, int elempack)
{
    int x0 = (int)floorf(sample_x);
    int y0 = (int)floorf(sample_y);
    int z0 = (int)floorf(sample_z);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    int* offset_ptr = (int*)offset_value_ptr;
    offset_ptr[0] = in_bounds_arm(x0, y0, z0, w, h, d) ? (x0 + y0 * w + z0 * w * h) * elempack : -1;
    offset_ptr[1] = in_bounds_arm(x1, y0, z0, w, h, d) ? (x1 + y0 * w + z0 * w * h) * elempack : -1;
    offset_ptr[2] = in_bounds_arm(x0, y1, z0, w, h, d) ? (x0 + y1 * w + z0 * w * h) * elempack : -1;
    offset_ptr[3] = in_bounds_arm(x1, y1, z0, w, h, d) ? (x1 + y1 * w + z0 * w * h) * elempack : -1;
    offset_ptr[4] = in_bounds_arm(x0, y0, z1, w, h, d) ? (x0 + y0 * w + z1 * w * h) * elempack : -1;
    offset_ptr[5] = in_bounds_arm(x1, y0, z1, w, h, d) ? (x1 + y0 * w + z1 * w * h) * elempack : -1;
    offset_ptr[6] = in_bounds_arm(x0, y1, z1, w, h, d) ? (x0 + y1 * w + z1 * w * h) * elempack : -1;
    offset_ptr[7] = in_bounds_arm(x1, y1, z1, w, h, d) ? (x1 + y1 * w + z1 * w * h) * elempack : -1;

    offset_value_ptr[8] = sample_x - x0;
    offset_value_ptr[9] = sample_y - y0;
    offset_value_ptr[10] = sample_z - z0;
}

static void gridsample_store_3d_nearest_arm(int* offset_ptr, float sample_x, float sample_y, float sample_z, int w, int h, int d, int elempack)
{
    int x0 = (int)floorf(sample_x + 0.5f);
    int y0 = (int)floorf(sample_y + 0.5f);
    int z0 = (int)floorf(sample_z + 0.5f);

    offset_ptr[0] = in_bounds_arm(x0, y0, z0, w, h, d) ? (x0 + y0 * w + z0 * w * h) * elempack : -1;
}

static void gridsample_store_2d_bicubic_arm(float* offset_value_ptr, float sample_x, float sample_y, int w, int h, int elempack, int padding_mode, int align_corner)
{
    int x1 = (int)floorf(sample_x);
    int y1 = (int)floorf(sample_y);
    int x0 = x1 - 1;
    int x2 = x1 + 1;
    int x3 = x1 + 2;

    offset_value_ptr[0] = sample_x - x1;
    offset_value_ptr[1] = sample_y - y1;

    x0 = (int)compute_coord_arm((float)x0, w, padding_mode, align_corner);
    x1 = (int)compute_coord_arm((float)x1, w, padding_mode, align_corner);
    x2 = (int)compute_coord_arm((float)x2, w, padding_mode, align_corner);
    x3 = (int)compute_coord_arm((float)x3, w, padding_mode, align_corner);

    bool x0_in_range = x0 >= 0 && x0 < w;
    bool x1_in_range = x1 >= 0 && x1 < w;
    bool x2_in_range = x2 >= 0 && x2 < w;
    bool x3_in_range = x3 >= 0 && x3 < w;

    int* offset_ptr = (int*)offset_value_ptr + 2;

    for (int i = 0; i < 4; i++)
    {
        int gy = (int)compute_coord_arm((float)(y1 + i - 1), h, padding_mode, align_corner);
        int offset_y = gy * w;
        bool y_in_range = gy >= 0 && gy < h;

        offset_ptr[0] = (x0_in_range && y_in_range) ? (offset_y + x0) * elempack : -1;
        offset_ptr[1] = (x1_in_range && y_in_range) ? (offset_y + x1) * elempack : -1;
        offset_ptr[2] = (x2_in_range && y_in_range) ? (offset_y + x2) * elempack : -1;
        offset_ptr[3] = (x3_in_range && y_in_range) ? (offset_y + x3) * elempack : -1;

        offset_ptr += 4;
    }
}

#if __ARM_NEON
static float32x4_t gridsample_floor_neon(float32x4_t _v)
{
#if __aarch64__
    return vrndmq_f32(_v);
#else
    int32x4_t _vi = vcvtq_s32_f32(_v);
    uint32x4_t _mask = vcgtq_f32(vcvtq_f32_s32(_vi), _v);
    return vcvtq_f32_s32(vaddq_s32(_vi, vreinterpretq_s32_u32(_mask)));
#endif
}

static float32x4_t grid_sample_unormalize_neon(int w, float32x4_t _coordx, int align_corner)
{
    float32x4_t _one = vdupq_n_f32(1.f);
    float32x4_t _half = vdupq_n_f32(0.5f);

    if (align_corner)
        return vmulq_f32(vmulq_f32(vaddq_f32(_coordx, _one), _half), vdupq_n_f32((float)(w - 1)));

    return vmulq_f32(vsubq_f32(vmulq_f32(vaddq_f32(_coordx, _one), vdupq_n_f32((float)w)), _one), _half);
}

static float32x4_t compute_coord_neon(float32x4_t _sx, int w, int padding_mode, int align_corner)
{
    if (padding_mode == GridSample::Padding_BORDER)
    {
        _sx = vminq_f32(vdupq_n_f32((float)(w - 1)), vmaxq_f32(_sx, vdupq_n_f32(0.f)));
    }
    else if (padding_mode == GridSample::Padding_REFLECTION)
    {
        if (align_corner)
        {
            float32x4_t _high = vdupq_n_f32((float)(w - 1));
            _sx = vabsq_f32(_sx);
            _sx = vsubq_f32(_high, vabsq_f32(vsubq_f32(_sx, _high)));
        }
        else
        {
            float32x4_t _half = vdupq_n_f32(0.5f);
            float32x4_t _length = vdupq_n_f32((float)w);
            _sx = vabsq_f32(vaddq_f32(_sx, _half));
            _sx = vsubq_f32(vsubq_f32(_length, vabsq_f32(vsubq_f32(_sx, _length))), _half);
            _sx = vminq_f32(vdupq_n_f32((float)(w - 1)), vmaxq_f32(_sx, vdupq_n_f32(0.f)));
        }
    }

    return _sx;
}

// _v holds integral values, so _v >= 0 is the same as _v > -1
static uint32x4_t gridsample_in_range_neon(float32x4_t _v, int size)
{
    return vandq_u32(vcgeq_f32(_v, vdupq_n_f32(0.f)), vcltq_f32(_v, vdupq_n_f32((float)size)));
}

static int32x4_t gridsample_select_offset_neon(int32x4_t _offset, uint32x4_t _mask)
{
    return vbslq_s32(_mask, _offset, vdupq_n_s32(-1));
}

static void gridsample_store_2d_bilinear_neon(float* offset_value_ptr, float32x4_t _sample_x, float32x4_t _sample_y, int w, int h, int elempack)
{
    float32x4_t _x0f = gridsample_floor_neon(_sample_x);
    float32x4_t _y0f = gridsample_floor_neon(_sample_y);
    float32x4_t _x1f = vaddq_f32(_x0f, vdupq_n_f32(1.f));
    float32x4_t _y1f = vaddq_f32(_y0f, vdupq_n_f32(1.f));

    uint32x4_t _x0_in_range = gridsample_in_range_neon(_x0f, w);
    uint32x4_t _x1_in_range = gridsample_in_range_neon(_x1f, w);
    uint32x4_t _y0_in_range = gridsample_in_range_neon(_y0f, h);
    uint32x4_t _y1_in_range = gridsample_in_range_neon(_y1f, h);

    int32x4_t _x0 = vcvtq_s32_f32(_x0f);
    int32x4_t _y0 = vcvtq_s32_f32(_y0f);

    int32x4_t _nw_offset = vmulq_n_s32(vmlaq_n_s32(_x0, _y0, w), elempack);
    int32x4_t _ne_offset = vaddq_s32(_nw_offset, vdupq_n_s32(elempack));
    int32x4_t _sw_offset = vaddq_s32(_nw_offset, vdupq_n_s32(w * elempack));
    int32x4_t _se_offset = vaddq_s32(_sw_offset, vdupq_n_s32(elempack));

    _nw_offset = gridsample_select_offset_neon(_nw_offset, vandq_u32(_x0_in_range, _y0_in_range));
    _ne_offset = gridsample_select_offset_neon(_ne_offset, vandq_u32(_x1_in_range, _y0_in_range));
    _sw_offset = gridsample_select_offset_neon(_sw_offset, vandq_u32(_x0_in_range, _y1_in_range));
    _se_offset = gridsample_select_offset_neon(_se_offset, vandq_u32(_x1_in_range, _y1_in_range));

    // transpose so that the four offsets of one output point are contiguous
    float32x4_t _offset0 = vreinterpretq_f32_s32(_nw_offset);
    float32x4_t _offset1 = vreinterpretq_f32_s32(_ne_offset);
    float32x4_t _offset2 = vreinterpretq_f32_s32(_sw_offset);
    float32x4_t _offset3 = vreinterpretq_f32_s32(_se_offset);
    transpose4x4_ps(_offset0, _offset1, _offset2, _offset3);

    float32x4_t _alpha = vsubq_f32(_sample_x, _x0f);
    float32x4_t _beta = vsubq_f32(_sample_y, _y0f);
    float32x4x2_t _alpha_beta = vzipq_f32(_alpha, _beta);

    vst1q_s32((int*)offset_value_ptr, vreinterpretq_s32_f32(_offset0));
    vst1_f32(offset_value_ptr + 4, vget_low_f32(_alpha_beta.val[0]));
    vst1q_s32((int*)offset_value_ptr + 6, vreinterpretq_s32_f32(_offset1));
    vst1_f32(offset_value_ptr + 10, vget_high_f32(_alpha_beta.val[0]));
    vst1q_s32((int*)offset_value_ptr + 12, vreinterpretq_s32_f32(_offset2));
    vst1_f32(offset_value_ptr + 16, vget_low_f32(_alpha_beta.val[1]));
    vst1q_s32((int*)offset_value_ptr + 18, vreinterpretq_s32_f32(_offset3));
    vst1_f32(offset_value_ptr + 22, vget_high_f32(_alpha_beta.val[1]));
}

static void gridsample_store_2d_nearest_neon(int* offset_ptr, float32x4_t _sample_x, float32x4_t _sample_y, int w, int h, int elempack)
{
    float32x4_t _x0f = gridsample_floor_neon(vaddq_f32(_sample_x, vdupq_n_f32(0.5f)));
    float32x4_t _y0f = gridsample_floor_neon(vaddq_f32(_sample_y, vdupq_n_f32(0.5f)));

    uint32x4_t _in_range = vandq_u32(gridsample_in_range_neon(_x0f, w), gridsample_in_range_neon(_y0f, h));

    int32x4_t _offset = vmulq_n_s32(vmlaq_n_s32(vcvtq_s32_f32(_x0f), vcvtq_s32_f32(_y0f), w), elempack);
    _offset = gridsample_select_offset_neon(_offset, _in_range);

    vst1q_s32(offset_ptr, _offset);
}

static void gridsample_store_3d_bilinear_neon(float* offset_value_ptr, float32x4_t _sample_x, float32x4_t _sample_y, float32x4_t _sample_z, int w, int h, int d, int elempack)
{
    float32x4_t _x0f = gridsample_floor_neon(_sample_x);
    float32x4_t _y0f = gridsample_floor_neon(_sample_y);
    float32x4_t _z0f = gridsample_floor_neon(_sample_z);
    float32x4_t _x1f = vaddq_f32(_x0f, vdupq_n_f32(1.f));
    float32x4_t _y1f = vaddq_f32(_y0f, vdupq_n_f32(1.f));
    float32x4_t _z1f = vaddq_f32(_z0f, vdupq_n_f32(1.f));

    uint32x4_t _x0_in_range = gridsample_in_range_neon(_x0f, w);
    uint32x4_t _x1_in_range = gridsample_in_range_neon(_x1f, w);
    uint32x4_t _y0_in_range = gridsample_in_range_neon(_y0f, h);
    uint32x4_t _y1_in_range = gridsample_in_range_neon(_y1f, h);
    uint32x4_t _z0_in_range = gridsample_in_range_neon(_z0f, d);
    uint32x4_t _z1_in_range = gridsample_in_range_neon(_z1f, d);

    int32x4_t _x0 = vcvtq_s32_f32(_x0f);
    int32x4_t _y0 = vcvtq_s32_f32(_y0f);
    int32x4_t _z0 = vcvtq_s32_f32(_z0f);

    int32x4_t _offset0 = vmulq_n_s32(vmlaq_n_s32(vmlaq_n_s32(_x0, _y0, w), _z0, w * h), elempack);
    int32x4_t _offset1 = vaddq_s32(_offset0, vdupq_n_s32(elempack));
    int32x4_t _offset2 = vaddq_s32(_offset0, vdupq_n_s32(w * elempack));
    int32x4_t _offset3 = vaddq_s32(_offset2, vdupq_n_s32(elempack));
    int32x4_t _offset4 = vaddq_s32(_offset0, vdupq_n_s32(w * h * elempack));
    int32x4_t _offset5 = vaddq_s32(_offset4, vdupq_n_s32(elempack));
    int32x4_t _offset6 = vaddq_s32(_offset4, vdupq_n_s32(w * elempack));
    int32x4_t _offset7 = vaddq_s32(_offset6, vdupq_n_s32(elempack));

    uint32x4_t _xy00_in_range = vandq_u32(_x0_in_range, _y0_in_range);
    uint32x4_t _xy01_in_range = vandq_u32(_x1_in_range, _y0_in_range);
    uint32x4_t _xy10_in_range = vandq_u32(_x0_in_range, _y1_in_range);
    uint32x4_t _xy11_in_range = vandq_u32(_x1_in_range, _y1_in_range);

    _offset0 = gridsample_select_offset_neon(_offset0, vandq_u32(_xy00_in_range, _z0_in_range));
    _offset1 = gridsample_select_offset_neon(_offset1, vandq_u32(_xy01_in_range, _z0_in_range));
    _offset2 = gridsample_select_offset_neon(_offset2, vandq_u32(_xy10_in_range, _z0_in_range));
    _offset3 = gridsample_select_offset_neon(_offset3, vandq_u32(_xy11_in_range, _z0_in_range));
    _offset4 = gridsample_select_offset_neon(_offset4, vandq_u32(_xy00_in_range, _z1_in_range));
    _offset5 = gridsample_select_offset_neon(_offset5, vandq_u32(_xy01_in_range, _z1_in_range));
    _offset6 = gridsample_select_offset_neon(_offset6, vandq_u32(_xy10_in_range, _z1_in_range));
    _offset7 = gridsample_select_offset_neon(_offset7, vandq_u32(_xy11_in_range, _z1_in_range));

    // transpose so that the eight offsets of one output point are contiguous
    float32x4_t _r0 = vreinterpretq_f32_s32(_offset0);
    float32x4_t _r1 = vreinterpretq_f32_s32(_offset1);
    float32x4_t _r2 = vreinterpretq_f32_s32(_offset2);
    float32x4_t _r3 = vreinterpretq_f32_s32(_offset3);
    float32x4_t _r4 = vreinterpretq_f32_s32(_offset4);
    float32x4_t _r5 = vreinterpretq_f32_s32(_offset5);
    float32x4_t _r6 = vreinterpretq_f32_s32(_offset6);
    float32x4_t _r7 = vreinterpretq_f32_s32(_offset7);
    transpose4x4_ps(_r0, _r1, _r2, _r3);
    transpose4x4_ps(_r4, _r5, _r6, _r7);

    vst1q_s32((int*)offset_value_ptr, vreinterpretq_s32_f32(_r0));
    vst1q_s32((int*)offset_value_ptr + 4, vreinterpretq_s32_f32(_r4));
    vst1q_s32((int*)offset_value_ptr + 11, vreinterpretq_s32_f32(_r1));
    vst1q_s32((int*)offset_value_ptr + 15, vreinterpretq_s32_f32(_r5));
    vst1q_s32((int*)offset_value_ptr + 22, vreinterpretq_s32_f32(_r2));
    vst1q_s32((int*)offset_value_ptr + 26, vreinterpretq_s32_f32(_r6));
    vst1q_s32((int*)offset_value_ptr + 33, vreinterpretq_s32_f32(_r3));
    vst1q_s32((int*)offset_value_ptr + 37, vreinterpretq_s32_f32(_r7));

    float alpha[4];
    float beta[4];
    float gamma[4];
    vst1q_f32(alpha, vsubq_f32(_sample_x, _x0f));
    vst1q_f32(beta, vsubq_f32(_sample_y, _y0f));
    vst1q_f32(gamma, vsubq_f32(_sample_z, _z0f));

    for (int i = 0; i < 4; i++)
    {
        offset_value_ptr[i * 11 + 8] = alpha[i];
        offset_value_ptr[i * 11 + 9] = beta[i];
        offset_value_ptr[i * 11 + 10] = gamma[i];
    }
}

static void gridsample_store_3d_nearest_neon(int* offset_ptr, float32x4_t _sample_x, float32x4_t _sample_y, float32x4_t _sample_z, int w, int h, int d, int elempack)
{
    float32x4_t _x0f = gridsample_floor_neon(vaddq_f32(_sample_x, vdupq_n_f32(0.5f)));
    float32x4_t _y0f = gridsample_floor_neon(vaddq_f32(_sample_y, vdupq_n_f32(0.5f)));
    float32x4_t _z0f = gridsample_floor_neon(vaddq_f32(_sample_z, vdupq_n_f32(0.5f)));

    uint32x4_t _in_range = vandq_u32(vandq_u32(gridsample_in_range_neon(_x0f, w), gridsample_in_range_neon(_y0f, h)), gridsample_in_range_neon(_z0f, d));

    int32x4_t _offset = vmlaq_n_s32(vmlaq_n_s32(vcvtq_s32_f32(_x0f), vcvtq_s32_f32(_y0f), w), vcvtq_s32_f32(_z0f), w * h);
    _offset = vmulq_n_s32(_offset, elempack);
    _offset = gridsample_select_offset_neon(_offset, _in_range);

    vst1q_s32(offset_ptr, _offset);
}

static void gridsample_store_2d_bicubic_neon(float* offset_value_ptr, float32x4_t _sample_x, float32x4_t _sample_y, int w, int h, int elempack, int padding_mode, int align_corner)
{
    float32x4_t _x1f = gridsample_floor_neon(_sample_x);
    float32x4_t _y1f = gridsample_floor_neon(_sample_y);

    float32x4x2_t _tx_ty = vzipq_f32(vsubq_f32(_sample_x, _x1f), vsubq_f32(_sample_y, _y1f));

    float32x4_t _gx0 = compute_coord_neon(vsubq_f32(_x1f, vdupq_n_f32(1.f)), w, padding_mode, align_corner);
    float32x4_t _gx1 = compute_coord_neon(_x1f, w, padding_mode, align_corner);
    float32x4_t _gx2 = compute_coord_neon(vaddq_f32(_x1f, vdupq_n_f32(1.f)), w, padding_mode, align_corner);
    float32x4_t _gx3 = compute_coord_neon(vaddq_f32(_x1f, vdupq_n_f32(2.f)), w, padding_mode, align_corner);

    uint32x4_t _x0_in_range = gridsample_in_range_neon(_gx0, w);
    uint32x4_t _x1_in_range = gridsample_in_range_neon(_gx1, w);
    uint32x4_t _x2_in_range = gridsample_in_range_neon(_gx2, w);
    uint32x4_t _x3_in_range = gridsample_in_range_neon(_gx3, w);

    int32x4_t _x0 = vcvtq_s32_f32(_gx0);
    int32x4_t _x1 = vcvtq_s32_f32(_gx1);
    int32x4_t _x2 = vcvtq_s32_f32(_gx2);
    int32x4_t _x3 = vcvtq_s32_f32(_gx3);

    for (int i = 0; i < 4; i++)
    {
        float32x4_t _gy = compute_coord_neon(vaddq_f32(_y1f, vdupq_n_f32((float)(i - 1))), h, padding_mode, align_corner);
        uint32x4_t _y_in_range = gridsample_in_range_neon(_gy, h);
        int32x4_t _offset_y = vmulq_n_s32(vcvtq_s32_f32(_gy), w);

        int32x4_t _offset0 = vmulq_n_s32(vaddq_s32(_offset_y, _x0), elempack);
        int32x4_t _offset1 = vmulq_n_s32(vaddq_s32(_offset_y, _x1), elempack);
        int32x4_t _offset2 = vmulq_n_s32(vaddq_s32(_offset_y, _x2), elempack);
        int32x4_t _offset3 = vmulq_n_s32(vaddq_s32(_offset_y, _x3), elempack);

        _offset0 = gridsample_select_offset_neon(_offset0, vandq_u32(_x0_in_range, _y_in_range));
        _offset1 = gridsample_select_offset_neon(_offset1, vandq_u32(_x1_in_range, _y_in_range));
        _offset2 = gridsample_select_offset_neon(_offset2, vandq_u32(_x2_in_range, _y_in_range));
        _offset3 = gridsample_select_offset_neon(_offset3, vandq_u32(_x3_in_range, _y_in_range));

        // transpose so that the four offsets of row i of one output point are contiguous
        float32x4_t _r0 = vreinterpretq_f32_s32(_offset0);
        float32x4_t _r1 = vreinterpretq_f32_s32(_offset1);
        float32x4_t _r2 = vreinterpretq_f32_s32(_offset2);
        float32x4_t _r3 = vreinterpretq_f32_s32(_offset3);
        transpose4x4_ps(_r0, _r1, _r2, _r3);

        vst1q_s32((int*)offset_value_ptr + 2 + i * 4, vreinterpretq_s32_f32(_r0));
        vst1q_s32((int*)offset_value_ptr + 20 + i * 4, vreinterpretq_s32_f32(_r1));
        vst1q_s32((int*)offset_value_ptr + 38 + i * 4, vreinterpretq_s32_f32(_r2));
        vst1q_s32((int*)offset_value_ptr + 56 + i * 4, vreinterpretq_s32_f32(_r3));
    }

    vst1_f32(offset_value_ptr, vget_low_f32(_tx_ty.val[0]));
    vst1_f32(offset_value_ptr + 18, vget_high_f32(_tx_ty.val[0]));
    vst1_f32(offset_value_ptr + 36, vget_low_f32(_tx_ty.val[1]));
    vst1_f32(offset_value_ptr + 54, vget_high_f32(_tx_ty.val[1]));
}
#endif // __ARM_NEON

#define GRIDSAMPLE_COMPUTE_BLOB_DISPATCH(func, src, grid, offset_value, padding_mode, align_corner, permute_fusion) \
    do                                                                                                              \
    {                                                                                                               \
        if (padding_mode == GridSample::Padding_ZEROS)                                                              \
        {                                                                                                           \
            if (align_corner == 0)                                                                                  \
                func<GridSample::Padding_ZEROS, false>(src, grid, offset_value, permute_fusion);                    \
            else                                                                                                    \
                func<GridSample::Padding_ZEROS, true>(src, grid, offset_value, permute_fusion);                     \
        }                                                                                                           \
        else if (padding_mode == GridSample::Padding_BORDER)                                                        \
        {                                                                                                           \
            if (align_corner == 0)                                                                                  \
                func<GridSample::Padding_BORDER, false>(src, grid, offset_value, permute_fusion);                   \
            else                                                                                                    \
                func<GridSample::Padding_BORDER, true>(src, grid, offset_value, permute_fusion);                    \
        }                                                                                                           \
        else if (padding_mode == GridSample::Padding_REFLECTION)                                                    \
        {                                                                                                           \
            if (align_corner == 0)                                                                                  \
                func<GridSample::Padding_REFLECTION, false>(src, grid, offset_value, permute_fusion);               \
            else                                                                                                    \
                func<GridSample::Padding_REFLECTION, true>(src, grid, offset_value, permute_fusion);                \
        }                                                                                                           \
    } while (0)

#include "gridsample_bilinear_compute_blob.h"
#include "gridsample_bicubic_compute_blob.h"
#include "gridsample_nearest_compute_blob.h"
#undef GRIDSAMPLE_COMPUTE_BLOB_DISPATCH
