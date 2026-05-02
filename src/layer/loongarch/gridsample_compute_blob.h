// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static float grid_sample_unormalize_loongarch(int w, float coordx, int align_corner)
{
    return align_corner ? (coordx + 1) / 2.f * (w - 1) : ((coordx + 1) * w - 1) / 2.f;
}

static float border_coord_loongarch(float x, float border)
{
    if (x < 0.f)
        return 0.f;
    if (x > border)
        return border;
    return x;
}

static float reflect_coord_loongarch(float x, int high)
{
    x = fabsf(x);
    x = high - fabsf(x - high);
    return x;
}

static float compute_coord_loongarch(float sx, int w, int padding_mode, int align_corner)
{
    if (padding_mode == GridSample::Padding_BORDER)
    {
        sx = border_coord_loongarch(sx, w - 1);
    }
    else if (padding_mode == GridSample::Padding_REFLECTION)
    {
        if (align_corner)
        {
            sx = reflect_coord_loongarch(sx, w - 1);
        }
        else
        {
            sx = reflect_coord_loongarch(sx + 0.5f, w) - 0.5f;
            sx = border_coord_loongarch(sx, w - 1);
        }
    }

    return sx;
}

static bool in_bounds_loongarch(int x, int y, int w, int h)
{
    return x >= 0 && y >= 0 && x < w && y < h;
}

static bool in_bounds_loongarch(int x, int y, int z, int w, int h, int d)
{
    return x >= 0 && y >= 0 && z >= 0 && x < w && y < h && z < d;
}

#if __loongarch_sx
static __m128 gridsample_set4_ps_lsx(float v0, float v1, float v2, float v3)
{
    FloatInt fi0;
    FloatInt fi1;
    FloatInt fi2;
    FloatInt fi3;
    fi0.f = v0;
    fi1.f = v1;
    fi2.f = v2;
    fi3.f = v3;
    __m128i _v = __lsx_vreplgr2vr_w(fi0.i);
    _v = __lsx_vinsgr2vr_w(_v, fi1.i, 1);
    _v = __lsx_vinsgr2vr_w(_v, fi2.i, 2);
    _v = __lsx_vinsgr2vr_w(_v, fi3.i, 3);
    return (__m128)_v;
}

static __m128 grid_sample_unormalize_lsx(int w, __m128 _coordx, int align_corner)
{
    __m128 _one = (__m128)__lsx_vreplgr2vr_w(0x3f800000);
    __m128 _half = (__m128)__lsx_vreplgr2vr_w(0x3f000000);

    if (align_corner)
        return __lsx_vfmul_s(__lsx_vfmul_s(__lsx_vfadd_s(_coordx, _one), _half), (__m128)__lsx_vreplfr2vr_s((float)(w - 1)));

    return __lsx_vfmul_s(__lsx_vfsub_s(__lsx_vfmul_s(__lsx_vfadd_s(_coordx, _one), (__m128)__lsx_vreplfr2vr_s((float)w)), _one), _half);
}

static __m128 compute_coord_lsx(__m128 _sx, int w, int padding_mode, int align_corner)
{
    if (padding_mode == GridSample::Padding_BORDER)
    {
        _sx = __lsx_vfmin_s((__m128)__lsx_vreplfr2vr_s((float)(w - 1)), __lsx_vfmax_s(_sx, (__m128)__lsx_vldi(0)));
    }
    else if (padding_mode == GridSample::Padding_REFLECTION)
    {
        if (align_corner)
        {
            __m128 _high = (__m128)__lsx_vreplfr2vr_s((float)(w - 1));
            _sx = (__m128)__lsx_vbitclri_w((__m128i)_sx, 31);
            _sx = __lsx_vfsub_s(_high, (__m128)__lsx_vbitclri_w((__m128i)__lsx_vfsub_s(_sx, _high), 31));
        }
        else
        {
            __m128 _half = (__m128)__lsx_vreplgr2vr_w(0x3f000000);
            __m128 _length = (__m128)__lsx_vreplfr2vr_s((float)w);
            _sx = (__m128)__lsx_vbitclri_w((__m128i)__lsx_vfadd_s(_sx, _half), 31);
            _sx = __lsx_vfsub_s(__lsx_vfsub_s(_length, (__m128)__lsx_vbitclri_w((__m128i)__lsx_vfsub_s(_sx, _length), 31)), _half);
            _sx = __lsx_vfmin_s((__m128)__lsx_vreplfr2vr_s((float)(w - 1)), __lsx_vfmax_s(_sx, (__m128)__lsx_vldi(0)));
        }
    }

    return _sx;
}

static __m128i gridsample_in_range_lsx(__m128 _v, int size)
{
    __m128i _ge0 = __lsx_vfcmp_clt_s((__m128)__lsx_vreplfr2vr_s(-1.f), _v);
    __m128i _lt_size = __lsx_vfcmp_clt_s(_v, (__m128)__lsx_vreplfr2vr_s((float)size));
    return __lsx_vand_v(_ge0, _lt_size);
}

static __m128i gridsample_select_offset_lsx(__m128i _offset, __m128i _mask)
{
    return __lsx_vbitsel_v(__lsx_vreplgr2vr_w(-1), _offset, _mask);
}

static void gridsample_store_2d_bilinear_lsx(float* offset_value_ptr, __m128 _sample_x, __m128 _sample_y, int w, int h, int elempack)
{
    __m128 _x0f = (__m128)__lsx_vfrintrm_s(_sample_x);
    __m128 _y0f = (__m128)__lsx_vfrintrm_s(_sample_y);
    __m128 _x1f = __lsx_vfadd_s(_x0f, (__m128)__lsx_vreplfr2vr_s(1.f));
    __m128 _y1f = __lsx_vfadd_s(_y0f, (__m128)__lsx_vreplfr2vr_s(1.f));

    __m128i _x0_in_range = gridsample_in_range_lsx(_x0f, w);
    __m128i _x1_in_range = gridsample_in_range_lsx(_x1f, w);
    __m128i _y0_in_range = gridsample_in_range_lsx(_y0f, h);
    __m128i _y1_in_range = gridsample_in_range_lsx(_y1f, h);

    __m128i _x0 = __lsx_vftintrz_w_s(_x0f);
    __m128i _y0 = __lsx_vftintrz_w_s(_y0f);

    __m128i _base = __lsx_vmul_w(_y0, __lsx_vreplgr2vr_w(w));
    _base = __lsx_vadd_w(_base, _x0);
    _base = __lsx_vmul_w(_base, __lsx_vreplgr2vr_w(elempack));

    __m128i _elempack = __lsx_vreplgr2vr_w(elempack);
    __m128i _wstep = __lsx_vreplgr2vr_w(w * elempack);
    __m128i _nw_offset = _base;
    __m128i _ne_offset = __lsx_vadd_w(_nw_offset, _elempack);
    __m128i _sw_offset = __lsx_vadd_w(_nw_offset, _wstep);
    __m128i _se_offset = __lsx_vadd_w(_sw_offset, _elempack);

    __m128i _v00_in_range = __lsx_vand_v(_x0_in_range, _y0_in_range);
    __m128i _v01_in_range = __lsx_vand_v(_x1_in_range, _y0_in_range);
    __m128i _v10_in_range = __lsx_vand_v(_x0_in_range, _y1_in_range);
    __m128i _v11_in_range = __lsx_vand_v(_x1_in_range, _y1_in_range);

    _nw_offset = gridsample_select_offset_lsx(_nw_offset, _v00_in_range);
    _ne_offset = gridsample_select_offset_lsx(_ne_offset, _v01_in_range);
    _sw_offset = gridsample_select_offset_lsx(_sw_offset, _v10_in_range);
    _se_offset = gridsample_select_offset_lsx(_se_offset, _v11_in_range);

    __m128 _nw = (__m128)_nw_offset;
    __m128 _ne = (__m128)_ne_offset;
    __m128 _sw = (__m128)_sw_offset;
    __m128 _se = (__m128)_se_offset;
    transpose4x4_ps(_nw, _ne, _sw, _se);

    __lsx_vst((__m128i)_nw, (int*)offset_value_ptr, 0);
    __lsx_vst((__m128i)_ne, (int*)(offset_value_ptr + 6), 0);
    __lsx_vst((__m128i)_sw, (int*)(offset_value_ptr + 12), 0);
    __lsx_vst((__m128i)_se, (int*)(offset_value_ptr + 18), 0);

    float alpha[4];
    float beta[4];
    __lsx_vst(__lsx_vfsub_s(_sample_x, _x0f), alpha, 0);
    __lsx_vst(__lsx_vfsub_s(_sample_y, _y0f), beta, 0);

    offset_value_ptr[4] = alpha[0];
    offset_value_ptr[5] = beta[0];
    offset_value_ptr[10] = alpha[1];
    offset_value_ptr[11] = beta[1];
    offset_value_ptr[16] = alpha[2];
    offset_value_ptr[17] = beta[2];
    offset_value_ptr[22] = alpha[3];
    offset_value_ptr[23] = beta[3];
}

static void gridsample_store_2d_nearest_lsx(int* offset_ptr, __m128 _sample_x, __m128 _sample_y, int w, int h, int elempack)
{
    __m128 _x0f = (__m128)__lsx_vfrintrm_s(__lsx_vfadd_s(_sample_x, (__m128)__lsx_vreplfr2vr_s(0.5f)));
    __m128 _y0f = (__m128)__lsx_vfrintrm_s(__lsx_vfadd_s(_sample_y, (__m128)__lsx_vreplfr2vr_s(0.5f)));

    __m128i _in_range = __lsx_vand_v(gridsample_in_range_lsx(_x0f, w), gridsample_in_range_lsx(_y0f, h));
    __m128i _offset = __lsx_vmul_w(__lsx_vftintrz_w_s(_y0f), __lsx_vreplgr2vr_w(w));
    _offset = __lsx_vadd_w(_offset, __lsx_vftintrz_w_s(_x0f));
    _offset = __lsx_vmul_w(_offset, __lsx_vreplgr2vr_w(elempack));
    _offset = gridsample_select_offset_lsx(_offset, _in_range);

    __lsx_vst(_offset, offset_ptr, 0);
}

static void gridsample_store_3d_bilinear_lsx(float* offset_value_ptr, __m128 _sample_x, __m128 _sample_y, __m128 _sample_z, int w, int h, int d, int elempack)
{
    __m128 _x0f = (__m128)__lsx_vfrintrm_s(_sample_x);
    __m128 _y0f = (__m128)__lsx_vfrintrm_s(_sample_y);
    __m128 _z0f = (__m128)__lsx_vfrintrm_s(_sample_z);
    __m128 _x1f = __lsx_vfadd_s(_x0f, (__m128)__lsx_vreplfr2vr_s(1.f));
    __m128 _y1f = __lsx_vfadd_s(_y0f, (__m128)__lsx_vreplfr2vr_s(1.f));
    __m128 _z1f = __lsx_vfadd_s(_z0f, (__m128)__lsx_vreplfr2vr_s(1.f));

    __m128i _x0_in_range = gridsample_in_range_lsx(_x0f, w);
    __m128i _x1_in_range = gridsample_in_range_lsx(_x1f, w);
    __m128i _y0_in_range = gridsample_in_range_lsx(_y0f, h);
    __m128i _y1_in_range = gridsample_in_range_lsx(_y1f, h);
    __m128i _z0_in_range = gridsample_in_range_lsx(_z0f, d);
    __m128i _z1_in_range = gridsample_in_range_lsx(_z1f, d);

    __m128i _base = __lsx_vmul_w(__lsx_vftintrz_w_s(_z0f), __lsx_vreplgr2vr_w(w * h));
    _base = __lsx_vadd_w(_base, __lsx_vmul_w(__lsx_vftintrz_w_s(_y0f), __lsx_vreplgr2vr_w(w)));
    _base = __lsx_vadd_w(_base, __lsx_vftintrz_w_s(_x0f));
    _base = __lsx_vmul_w(_base, __lsx_vreplgr2vr_w(elempack));

    __m128i _elempack = __lsx_vreplgr2vr_w(elempack);
    __m128i _wstep = __lsx_vreplgr2vr_w(w * elempack);
    __m128i _dstep = __lsx_vreplgr2vr_w(w * h * elempack);
    __m128i _offset0 = _base;
    __m128i _offset1 = __lsx_vadd_w(_offset0, _elempack);
    __m128i _offset2 = __lsx_vadd_w(_offset0, _wstep);
    __m128i _offset3 = __lsx_vadd_w(_offset2, _elempack);
    __m128i _offset4 = __lsx_vadd_w(_offset0, _dstep);
    __m128i _offset5 = __lsx_vadd_w(_offset4, _elempack);
    __m128i _offset6 = __lsx_vadd_w(_offset4, _wstep);
    __m128i _offset7 = __lsx_vadd_w(_offset6, _elempack);

    __m128i _xy00 = __lsx_vand_v(_x0_in_range, _y0_in_range);
    __m128i _xy01 = __lsx_vand_v(_x1_in_range, _y0_in_range);
    __m128i _xy10 = __lsx_vand_v(_x0_in_range, _y1_in_range);
    __m128i _xy11 = __lsx_vand_v(_x1_in_range, _y1_in_range);

    _offset0 = gridsample_select_offset_lsx(_offset0, __lsx_vand_v(_xy00, _z0_in_range));
    _offset1 = gridsample_select_offset_lsx(_offset1, __lsx_vand_v(_xy01, _z0_in_range));
    _offset2 = gridsample_select_offset_lsx(_offset2, __lsx_vand_v(_xy10, _z0_in_range));
    _offset3 = gridsample_select_offset_lsx(_offset3, __lsx_vand_v(_xy11, _z0_in_range));
    _offset4 = gridsample_select_offset_lsx(_offset4, __lsx_vand_v(_xy00, _z1_in_range));
    _offset5 = gridsample_select_offset_lsx(_offset5, __lsx_vand_v(_xy01, _z1_in_range));
    _offset6 = gridsample_select_offset_lsx(_offset6, __lsx_vand_v(_xy10, _z1_in_range));
    _offset7 = gridsample_select_offset_lsx(_offset7, __lsx_vand_v(_xy11, _z1_in_range));

    int offsets[8][4] = {};
    float alpha[4];
    float beta[4];
    float gamma[4];
    __lsx_vst(_offset0, offsets[0], 0);
    __lsx_vst(_offset1, offsets[1], 0);
    __lsx_vst(_offset2, offsets[2], 0);
    __lsx_vst(_offset3, offsets[3], 0);
    __lsx_vst(_offset4, offsets[4], 0);
    __lsx_vst(_offset5, offsets[5], 0);
    __lsx_vst(_offset6, offsets[6], 0);
    __lsx_vst(_offset7, offsets[7], 0);
    __lsx_vst(__lsx_vfsub_s(_sample_x, _x0f), alpha, 0);
    __lsx_vst(__lsx_vfsub_s(_sample_y, _y0f), beta, 0);
    __lsx_vst(__lsx_vfsub_s(_sample_z, _z0f), gamma, 0);

    for (int i = 0; i < 4; i++)
    {
        int* offset_ptr = (int*)(offset_value_ptr + i * 11);
        for (int j = 0; j < 8; j++)
        {
            offset_ptr[j] = offsets[j][i];
        }
        offset_value_ptr[i * 11 + 8] = alpha[i];
        offset_value_ptr[i * 11 + 9] = beta[i];
        offset_value_ptr[i * 11 + 10] = gamma[i];
    }
}

static void gridsample_store_3d_nearest_lsx(int* offset_ptr, __m128 _sample_x, __m128 _sample_y, __m128 _sample_z, int w, int h, int d, int elempack)
{
    __m128 _x0f = (__m128)__lsx_vfrintrm_s(__lsx_vfadd_s(_sample_x, (__m128)__lsx_vreplfr2vr_s(0.5f)));
    __m128 _y0f = (__m128)__lsx_vfrintrm_s(__lsx_vfadd_s(_sample_y, (__m128)__lsx_vreplfr2vr_s(0.5f)));
    __m128 _z0f = (__m128)__lsx_vfrintrm_s(__lsx_vfadd_s(_sample_z, (__m128)__lsx_vreplfr2vr_s(0.5f)));

    __m128i _in_range = __lsx_vand_v(__lsx_vand_v(gridsample_in_range_lsx(_x0f, w), gridsample_in_range_lsx(_y0f, h)), gridsample_in_range_lsx(_z0f, d));
    __m128i _offset = __lsx_vmul_w(__lsx_vftintrz_w_s(_z0f), __lsx_vreplgr2vr_w(w * h));
    _offset = __lsx_vadd_w(_offset, __lsx_vmul_w(__lsx_vftintrz_w_s(_y0f), __lsx_vreplgr2vr_w(w)));
    _offset = __lsx_vadd_w(_offset, __lsx_vftintrz_w_s(_x0f));
    _offset = __lsx_vmul_w(_offset, __lsx_vreplgr2vr_w(elempack));
    _offset = gridsample_select_offset_lsx(_offset, _in_range);

    __lsx_vst(_offset, offset_ptr, 0);
}

static void gridsample_store_2d_bicubic_lsx(float* offset_value_ptr, __m128 _sample_x, __m128 _sample_y, int w, int h, int elempack, int padding_mode, int align_corner)
{
    __m128 _x1f = (__m128)__lsx_vfrintrm_s(_sample_x);
    __m128 _y1f = (__m128)__lsx_vfrintrm_s(_sample_y);
    __m128 _tx = __lsx_vfsub_s(_sample_x, _x1f);
    __m128 _ty = __lsx_vfsub_s(_sample_y, _y1f);

    __m128 _gx0 = compute_coord_lsx(__lsx_vfsub_s(_x1f, (__m128)__lsx_vreplfr2vr_s(1.f)), w, padding_mode, align_corner);
    __m128 _gx1 = compute_coord_lsx(_x1f, w, padding_mode, align_corner);
    __m128 _gx2 = compute_coord_lsx(__lsx_vfadd_s(_x1f, (__m128)__lsx_vreplfr2vr_s(1.f)), w, padding_mode, align_corner);
    __m128 _gx3 = compute_coord_lsx(__lsx_vfadd_s(_x1f, (__m128)__lsx_vreplfr2vr_s(2.f)), w, padding_mode, align_corner);

    __m128i _x0_in_range = gridsample_in_range_lsx(_gx0, w);
    __m128i _x1_in_range = gridsample_in_range_lsx(_gx1, w);
    __m128i _x2_in_range = gridsample_in_range_lsx(_gx2, w);
    __m128i _x3_in_range = gridsample_in_range_lsx(_gx3, w);

    __m128i _x0 = __lsx_vftintrz_w_s(_gx0);
    __m128i _x1 = __lsx_vftintrz_w_s(_gx1);
    __m128i _x2 = __lsx_vftintrz_w_s(_gx2);
    __m128i _x3 = __lsx_vftintrz_w_s(_gx3);

    int offsets[16][4] = {};
    for (int i = 0; i < 4; i++)
    {
        __m128 _gy = compute_coord_lsx(__lsx_vfadd_s(_y1f, (__m128)__lsx_vreplfr2vr_s((float)(i - 1))), h, padding_mode, align_corner);
        __m128i _y_in_range = gridsample_in_range_lsx(_gy, h);
        __m128i _offset_y = __lsx_vmul_w(__lsx_vftintrz_w_s(_gy), __lsx_vreplgr2vr_w(w));

        __m128i _offset0 = __lsx_vmul_w(__lsx_vadd_w(_offset_y, _x0), __lsx_vreplgr2vr_w(elempack));
        __m128i _offset1 = __lsx_vmul_w(__lsx_vadd_w(_offset_y, _x1), __lsx_vreplgr2vr_w(elempack));
        __m128i _offset2 = __lsx_vmul_w(__lsx_vadd_w(_offset_y, _x2), __lsx_vreplgr2vr_w(elempack));
        __m128i _offset3 = __lsx_vmul_w(__lsx_vadd_w(_offset_y, _x3), __lsx_vreplgr2vr_w(elempack));

        _offset0 = gridsample_select_offset_lsx(_offset0, __lsx_vand_v(_x0_in_range, _y_in_range));
        _offset1 = gridsample_select_offset_lsx(_offset1, __lsx_vand_v(_x1_in_range, _y_in_range));
        _offset2 = gridsample_select_offset_lsx(_offset2, __lsx_vand_v(_x2_in_range, _y_in_range));
        _offset3 = gridsample_select_offset_lsx(_offset3, __lsx_vand_v(_x3_in_range, _y_in_range));

        __lsx_vst(_offset0, offsets[i * 4], 0);
        __lsx_vst(_offset1, offsets[i * 4 + 1], 0);
        __lsx_vst(_offset2, offsets[i * 4 + 2], 0);
        __lsx_vst(_offset3, offsets[i * 4 + 3], 0);
    }

    float tx[4];
    float ty[4];
    __lsx_vst(_tx, tx, 0);
    __lsx_vst(_ty, ty, 0);

    for (int i = 0; i < 4; i++)
    {
        float* value_ptr = offset_value_ptr + i * 18;
        int* offset_ptr = (int*)value_ptr + 2;
        value_ptr[0] = tx[i];
        value_ptr[1] = ty[i];
        for (int j = 0; j < 16; j++)
        {
            offset_ptr[j] = offsets[j][i];
        }
    }
}

#if __loongarch_asx
static __m256 gridsample_set8_ps_lasx(float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7)
{
    FloatInt fi0;
    FloatInt fi1;
    FloatInt fi2;
    FloatInt fi3;
    FloatInt fi4;
    FloatInt fi5;
    FloatInt fi6;
    FloatInt fi7;
    fi0.f = v0;
    fi1.f = v1;
    fi2.f = v2;
    fi3.f = v3;
    fi4.f = v4;
    fi5.f = v5;
    fi6.f = v6;
    fi7.f = v7;
    __m256i _v = __lasx_xvreplgr2vr_w(fi0.i);
    _v = __lasx_xvinsgr2vr_w(_v, fi1.i, 1);
    _v = __lasx_xvinsgr2vr_w(_v, fi2.i, 2);
    _v = __lasx_xvinsgr2vr_w(_v, fi3.i, 3);
    _v = __lasx_xvinsgr2vr_w(_v, fi4.i, 4);
    _v = __lasx_xvinsgr2vr_w(_v, fi5.i, 5);
    _v = __lasx_xvinsgr2vr_w(_v, fi6.i, 6);
    _v = __lasx_xvinsgr2vr_w(_v, fi7.i, 7);
    return (__m256)_v;
}

static __m256 grid_sample_unormalize_lasx(int w, __m256 _coordx, int align_corner)
{
    __m256 _one = (__m256)__lasx_xvreplgr2vr_w(0x3f800000);
    __m256 _half = (__m256)__lasx_xvreplgr2vr_w(0x3f000000);

    if (align_corner)
        return __lasx_xvfmul_s(__lasx_xvfmul_s(__lasx_xvfadd_s(_coordx, _one), _half), (__m256)__lasx_xvreplfr2vr_s((float)(w - 1)));

    return __lasx_xvfmul_s(__lasx_xvfsub_s(__lasx_xvfmul_s(__lasx_xvfadd_s(_coordx, _one), (__m256)__lasx_xvreplfr2vr_s((float)w)), _one), _half);
}

static __m256 compute_coord_lasx(__m256 _sx, int w, int padding_mode, int align_corner)
{
    if (padding_mode == GridSample::Padding_BORDER)
    {
        _sx = __lasx_xvfmin_s((__m256)__lasx_xvreplfr2vr_s((float)(w - 1)), __lasx_xvfmax_s(_sx, (__m256)__lasx_xvldi(0)));
    }
    else if (padding_mode == GridSample::Padding_REFLECTION)
    {
        if (align_corner)
        {
            __m256 _high = (__m256)__lasx_xvreplfr2vr_s((float)(w - 1));
            _sx = (__m256)__lasx_xvbitclri_w((__m256i)_sx, 31);
            _sx = __lasx_xvfsub_s(_high, (__m256)__lasx_xvbitclri_w((__m256i)__lasx_xvfsub_s(_sx, _high), 31));
        }
        else
        {
            __m256 _half = (__m256)__lasx_xvreplgr2vr_w(0x3f000000);
            __m256 _length = (__m256)__lasx_xvreplfr2vr_s((float)w);
            _sx = (__m256)__lasx_xvbitclri_w((__m256i)__lasx_xvfadd_s(_sx, _half), 31);
            _sx = __lasx_xvfsub_s(__lasx_xvfsub_s(_length, (__m256)__lasx_xvbitclri_w((__m256i)__lasx_xvfsub_s(_sx, _length), 31)), _half);
            _sx = __lasx_xvfmin_s((__m256)__lasx_xvreplfr2vr_s((float)(w - 1)), __lasx_xvfmax_s(_sx, (__m256)__lasx_xvldi(0)));
        }
    }

    return _sx;
}

static __m256i gridsample_in_range_lasx(__m256 _v, int size)
{
    __m256i _ge0 = __lasx_xvfcmp_clt_s((__m256)__lasx_xvreplfr2vr_s(-1.f), _v);
    __m256i _lt_size = __lasx_xvfcmp_clt_s(_v, (__m256)__lasx_xvreplfr2vr_s((float)size));
    return __lasx_xvand_v(_ge0, _lt_size);
}

static __m256i gridsample_select_offset_lasx(__m256i _offset, __m256i _mask)
{
    return __lasx_xvbitsel_v(__lasx_xvreplgr2vr_w(-1), _offset, _mask);
}

static void gridsample_store_2d_bilinear_lasx(float* offset_value_ptr, __m256 _sample_x, __m256 _sample_y, int w, int h, int elempack)
{
    __m256 _x0f = (__m256)__lasx_xvfrintrm_s(_sample_x);
    __m256 _y0f = (__m256)__lasx_xvfrintrm_s(_sample_y);
    __m256 _x1f = __lasx_xvfadd_s(_x0f, (__m256)__lasx_xvreplfr2vr_s(1.f));
    __m256 _y1f = __lasx_xvfadd_s(_y0f, (__m256)__lasx_xvreplfr2vr_s(1.f));

    __m256i _x0_in_range = gridsample_in_range_lasx(_x0f, w);
    __m256i _x1_in_range = gridsample_in_range_lasx(_x1f, w);
    __m256i _y0_in_range = gridsample_in_range_lasx(_y0f, h);
    __m256i _y1_in_range = gridsample_in_range_lasx(_y1f, h);

    __m256i _base = __lasx_xvmul_w(__lasx_xvftintrz_w_s(_y0f), __lasx_xvreplgr2vr_w(w));
    _base = __lasx_xvadd_w(_base, __lasx_xvftintrz_w_s(_x0f));
    _base = __lasx_xvmul_w(_base, __lasx_xvreplgr2vr_w(elempack));

    __m256i _elempack = __lasx_xvreplgr2vr_w(elempack);
    __m256i _wstep = __lasx_xvreplgr2vr_w(w * elempack);
    __m256i _nw_offset = _base;
    __m256i _ne_offset = __lasx_xvadd_w(_nw_offset, _elempack);
    __m256i _sw_offset = __lasx_xvadd_w(_nw_offset, _wstep);
    __m256i _se_offset = __lasx_xvadd_w(_sw_offset, _elempack);

    _nw_offset = gridsample_select_offset_lasx(_nw_offset, __lasx_xvand_v(_x0_in_range, _y0_in_range));
    _ne_offset = gridsample_select_offset_lasx(_ne_offset, __lasx_xvand_v(_x1_in_range, _y0_in_range));
    _sw_offset = gridsample_select_offset_lasx(_sw_offset, __lasx_xvand_v(_x0_in_range, _y1_in_range));
    _se_offset = gridsample_select_offset_lasx(_se_offset, __lasx_xvand_v(_x1_in_range, _y1_in_range));

    __m256 _nw_offset_f = (__m256)_nw_offset;
    __m256 _ne_offset_f = (__m256)_ne_offset;
    __m256 _sw_offset_f = (__m256)_sw_offset;
    __m256 _se_offset_f = (__m256)_se_offset;
    transpose8x4_ps(_nw_offset_f, _ne_offset_f, _sw_offset_f, _se_offset_f);

    float alpha[8];
    float beta[8];
    __lasx_xvst(__lasx_xvfsub_s(_sample_x, _x0f), alpha, 0);
    __lasx_xvst(__lasx_xvfsub_s(_sample_y, _y0f), beta, 0);

    __lsx_vst(__lasx_extract_128_lo((__m256i)_nw_offset_f), (int*)offset_value_ptr, 0);
    __lsx_vst(__lasx_extract_128_hi((__m256i)_nw_offset_f), (int*)(offset_value_ptr + 6), 0);
    __lsx_vst(__lasx_extract_128_lo((__m256i)_ne_offset_f), (int*)(offset_value_ptr + 12), 0);
    __lsx_vst(__lasx_extract_128_hi((__m256i)_ne_offset_f), (int*)(offset_value_ptr + 18), 0);
    __lsx_vst(__lasx_extract_128_lo((__m256i)_sw_offset_f), (int*)(offset_value_ptr + 24), 0);
    __lsx_vst(__lasx_extract_128_hi((__m256i)_sw_offset_f), (int*)(offset_value_ptr + 30), 0);
    __lsx_vst(__lasx_extract_128_lo((__m256i)_se_offset_f), (int*)(offset_value_ptr + 36), 0);
    __lsx_vst(__lasx_extract_128_hi((__m256i)_se_offset_f), (int*)(offset_value_ptr + 42), 0);

    for (int i = 0; i < 8; i++)
    {
        offset_value_ptr[i * 6 + 4] = alpha[i];
        offset_value_ptr[i * 6 + 5] = beta[i];
    }
}

static void gridsample_store_2d_nearest_lasx(int* offset_ptr, __m256 _sample_x, __m256 _sample_y, int w, int h, int elempack)
{
    __m256 _x0f = (__m256)__lasx_xvfrintrm_s(__lasx_xvfadd_s(_sample_x, (__m256)__lasx_xvreplfr2vr_s(0.5f)));
    __m256 _y0f = (__m256)__lasx_xvfrintrm_s(__lasx_xvfadd_s(_sample_y, (__m256)__lasx_xvreplfr2vr_s(0.5f)));

    __m256i _in_range = __lasx_xvand_v(gridsample_in_range_lasx(_x0f, w), gridsample_in_range_lasx(_y0f, h));
    __m256i _offset = __lasx_xvmul_w(__lasx_xvftintrz_w_s(_y0f), __lasx_xvreplgr2vr_w(w));
    _offset = __lasx_xvadd_w(_offset, __lasx_xvftintrz_w_s(_x0f));
    _offset = __lasx_xvmul_w(_offset, __lasx_xvreplgr2vr_w(elempack));
    _offset = gridsample_select_offset_lasx(_offset, _in_range);

    __lasx_xvst(_offset, offset_ptr, 0);
}

static void gridsample_store_3d_bilinear_lasx(float* offset_value_ptr, __m256 _sample_x, __m256 _sample_y, __m256 _sample_z, int w, int h, int d, int elempack)
{
    __m256 _x0f = (__m256)__lasx_xvfrintrm_s(_sample_x);
    __m256 _y0f = (__m256)__lasx_xvfrintrm_s(_sample_y);
    __m256 _z0f = (__m256)__lasx_xvfrintrm_s(_sample_z);
    __m256 _x1f = __lasx_xvfadd_s(_x0f, (__m256)__lasx_xvreplfr2vr_s(1.f));
    __m256 _y1f = __lasx_xvfadd_s(_y0f, (__m256)__lasx_xvreplfr2vr_s(1.f));
    __m256 _z1f = __lasx_xvfadd_s(_z0f, (__m256)__lasx_xvreplfr2vr_s(1.f));

    __m256i _x0_in_range = gridsample_in_range_lasx(_x0f, w);
    __m256i _x1_in_range = gridsample_in_range_lasx(_x1f, w);
    __m256i _y0_in_range = gridsample_in_range_lasx(_y0f, h);
    __m256i _y1_in_range = gridsample_in_range_lasx(_y1f, h);
    __m256i _z0_in_range = gridsample_in_range_lasx(_z0f, d);
    __m256i _z1_in_range = gridsample_in_range_lasx(_z1f, d);

    __m256i _base = __lasx_xvmul_w(__lasx_xvftintrz_w_s(_z0f), __lasx_xvreplgr2vr_w(w * h));
    _base = __lasx_xvadd_w(_base, __lasx_xvmul_w(__lasx_xvftintrz_w_s(_y0f), __lasx_xvreplgr2vr_w(w)));
    _base = __lasx_xvadd_w(_base, __lasx_xvftintrz_w_s(_x0f));
    _base = __lasx_xvmul_w(_base, __lasx_xvreplgr2vr_w(elempack));

    __m256i _elempack = __lasx_xvreplgr2vr_w(elempack);
    __m256i _wstep = __lasx_xvreplgr2vr_w(w * elempack);
    __m256i _dstep = __lasx_xvreplgr2vr_w(w * h * elempack);
    __m256i _offset0 = _base;
    __m256i _offset1 = __lasx_xvadd_w(_offset0, _elempack);
    __m256i _offset2 = __lasx_xvadd_w(_offset0, _wstep);
    __m256i _offset3 = __lasx_xvadd_w(_offset2, _elempack);
    __m256i _offset4 = __lasx_xvadd_w(_offset0, _dstep);
    __m256i _offset5 = __lasx_xvadd_w(_offset4, _elempack);
    __m256i _offset6 = __lasx_xvadd_w(_offset4, _wstep);
    __m256i _offset7 = __lasx_xvadd_w(_offset6, _elempack);

    __m256i _xy00 = __lasx_xvand_v(_x0_in_range, _y0_in_range);
    __m256i _xy01 = __lasx_xvand_v(_x1_in_range, _y0_in_range);
    __m256i _xy10 = __lasx_xvand_v(_x0_in_range, _y1_in_range);
    __m256i _xy11 = __lasx_xvand_v(_x1_in_range, _y1_in_range);

    _offset0 = gridsample_select_offset_lasx(_offset0, __lasx_xvand_v(_xy00, _z0_in_range));
    _offset1 = gridsample_select_offset_lasx(_offset1, __lasx_xvand_v(_xy01, _z0_in_range));
    _offset2 = gridsample_select_offset_lasx(_offset2, __lasx_xvand_v(_xy10, _z0_in_range));
    _offset3 = gridsample_select_offset_lasx(_offset3, __lasx_xvand_v(_xy11, _z0_in_range));
    _offset4 = gridsample_select_offset_lasx(_offset4, __lasx_xvand_v(_xy00, _z1_in_range));
    _offset5 = gridsample_select_offset_lasx(_offset5, __lasx_xvand_v(_xy01, _z1_in_range));
    _offset6 = gridsample_select_offset_lasx(_offset6, __lasx_xvand_v(_xy10, _z1_in_range));
    _offset7 = gridsample_select_offset_lasx(_offset7, __lasx_xvand_v(_xy11, _z1_in_range));

    int offsets[8][8] = {};
    float alpha[8];
    float beta[8];
    float gamma[8];
    __lasx_xvst(_offset0, offsets[0], 0);
    __lasx_xvst(_offset1, offsets[1], 0);
    __lasx_xvst(_offset2, offsets[2], 0);
    __lasx_xvst(_offset3, offsets[3], 0);
    __lasx_xvst(_offset4, offsets[4], 0);
    __lasx_xvst(_offset5, offsets[5], 0);
    __lasx_xvst(_offset6, offsets[6], 0);
    __lasx_xvst(_offset7, offsets[7], 0);
    __lasx_xvst(__lasx_xvfsub_s(_sample_x, _x0f), alpha, 0);
    __lasx_xvst(__lasx_xvfsub_s(_sample_y, _y0f), beta, 0);
    __lasx_xvst(__lasx_xvfsub_s(_sample_z, _z0f), gamma, 0);

    for (int i = 0; i < 8; i++)
    {
        int* offset_ptr = (int*)(offset_value_ptr + i * 11);
        for (int j = 0; j < 8; j++)
        {
            offset_ptr[j] = offsets[j][i];
        }
        offset_value_ptr[i * 11 + 8] = alpha[i];
        offset_value_ptr[i * 11 + 9] = beta[i];
        offset_value_ptr[i * 11 + 10] = gamma[i];
    }
}

static void gridsample_store_3d_nearest_lasx(int* offset_ptr, __m256 _sample_x, __m256 _sample_y, __m256 _sample_z, int w, int h, int d, int elempack)
{
    __m256 _x0f = (__m256)__lasx_xvfrintrm_s(__lasx_xvfadd_s(_sample_x, (__m256)__lasx_xvreplfr2vr_s(0.5f)));
    __m256 _y0f = (__m256)__lasx_xvfrintrm_s(__lasx_xvfadd_s(_sample_y, (__m256)__lasx_xvreplfr2vr_s(0.5f)));
    __m256 _z0f = (__m256)__lasx_xvfrintrm_s(__lasx_xvfadd_s(_sample_z, (__m256)__lasx_xvreplfr2vr_s(0.5f)));

    __m256i _in_range = __lasx_xvand_v(__lasx_xvand_v(gridsample_in_range_lasx(_x0f, w), gridsample_in_range_lasx(_y0f, h)), gridsample_in_range_lasx(_z0f, d));
    __m256i _offset = __lasx_xvmul_w(__lasx_xvftintrz_w_s(_z0f), __lasx_xvreplgr2vr_w(w * h));
    _offset = __lasx_xvadd_w(_offset, __lasx_xvmul_w(__lasx_xvftintrz_w_s(_y0f), __lasx_xvreplgr2vr_w(w)));
    _offset = __lasx_xvadd_w(_offset, __lasx_xvftintrz_w_s(_x0f));
    _offset = __lasx_xvmul_w(_offset, __lasx_xvreplgr2vr_w(elempack));
    _offset = gridsample_select_offset_lasx(_offset, _in_range);

    __lasx_xvst(_offset, offset_ptr, 0);
}

static void gridsample_store_2d_bicubic_lasx(float* offset_value_ptr, __m256 _sample_x, __m256 _sample_y, int w, int h, int elempack, int padding_mode, int align_corner)
{
    __m256 _x1f = (__m256)__lasx_xvfrintrm_s(_sample_x);
    __m256 _y1f = (__m256)__lasx_xvfrintrm_s(_sample_y);
    __m256 _tx = __lasx_xvfsub_s(_sample_x, _x1f);
    __m256 _ty = __lasx_xvfsub_s(_sample_y, _y1f);

    __m256 _gx0 = compute_coord_lasx(__lasx_xvfsub_s(_x1f, (__m256)__lasx_xvreplfr2vr_s(1.f)), w, padding_mode, align_corner);
    __m256 _gx1 = compute_coord_lasx(_x1f, w, padding_mode, align_corner);
    __m256 _gx2 = compute_coord_lasx(__lasx_xvfadd_s(_x1f, (__m256)__lasx_xvreplfr2vr_s(1.f)), w, padding_mode, align_corner);
    __m256 _gx3 = compute_coord_lasx(__lasx_xvfadd_s(_x1f, (__m256)__lasx_xvreplfr2vr_s(2.f)), w, padding_mode, align_corner);

    __m256i _x0_in_range = gridsample_in_range_lasx(_gx0, w);
    __m256i _x1_in_range = gridsample_in_range_lasx(_gx1, w);
    __m256i _x2_in_range = gridsample_in_range_lasx(_gx2, w);
    __m256i _x3_in_range = gridsample_in_range_lasx(_gx3, w);

    __m256i _x0 = __lasx_xvftintrz_w_s(_gx0);
    __m256i _x1 = __lasx_xvftintrz_w_s(_gx1);
    __m256i _x2 = __lasx_xvftintrz_w_s(_gx2);
    __m256i _x3 = __lasx_xvftintrz_w_s(_gx3);

    int offsets[16][8] = {};
    for (int i = 0; i < 4; i++)
    {
        __m256 _gy = compute_coord_lasx(__lasx_xvfadd_s(_y1f, (__m256)__lasx_xvreplfr2vr_s((float)(i - 1))), h, padding_mode, align_corner);
        __m256i _y_in_range = gridsample_in_range_lasx(_gy, h);
        __m256i _offset_y = __lasx_xvmul_w(__lasx_xvftintrz_w_s(_gy), __lasx_xvreplgr2vr_w(w));

        __m256i _offset0 = __lasx_xvmul_w(__lasx_xvadd_w(_offset_y, _x0), __lasx_xvreplgr2vr_w(elempack));
        __m256i _offset1 = __lasx_xvmul_w(__lasx_xvadd_w(_offset_y, _x1), __lasx_xvreplgr2vr_w(elempack));
        __m256i _offset2 = __lasx_xvmul_w(__lasx_xvadd_w(_offset_y, _x2), __lasx_xvreplgr2vr_w(elempack));
        __m256i _offset3 = __lasx_xvmul_w(__lasx_xvadd_w(_offset_y, _x3), __lasx_xvreplgr2vr_w(elempack));

        _offset0 = gridsample_select_offset_lasx(_offset0, __lasx_xvand_v(_x0_in_range, _y_in_range));
        _offset1 = gridsample_select_offset_lasx(_offset1, __lasx_xvand_v(_x1_in_range, _y_in_range));
        _offset2 = gridsample_select_offset_lasx(_offset2, __lasx_xvand_v(_x2_in_range, _y_in_range));
        _offset3 = gridsample_select_offset_lasx(_offset3, __lasx_xvand_v(_x3_in_range, _y_in_range));

        __lasx_xvst(_offset0, offsets[i * 4], 0);
        __lasx_xvst(_offset1, offsets[i * 4 + 1], 0);
        __lasx_xvst(_offset2, offsets[i * 4 + 2], 0);
        __lasx_xvst(_offset3, offsets[i * 4 + 3], 0);
    }

    float tx[8];
    float ty[8];
    __lasx_xvst(_tx, tx, 0);
    __lasx_xvst(_ty, ty, 0);

    for (int i = 0; i < 8; i++)
    {
        float* value_ptr = offset_value_ptr + i * 18;
        int* offset_ptr = (int*)value_ptr + 2;
        value_ptr[0] = tx[i];
        value_ptr[1] = ty[i];
        for (int j = 0; j < 16; j++)
        {
            offset_ptr[j] = offsets[j][i];
        }
    }
}
#endif // __loongarch_asx
#endif // __loongarch_sx

#define GRIDSAMPLE_COMPUTE_BLOB_DISPATCH(func, src, grid, offset_value, padding_mode, align_corner, permute_fusion) \
    do \
    { \
        if (padding_mode == GridSample::Padding_ZEROS) \
        { \
            if (align_corner == 0) \
                func<GridSample::Padding_ZEROS, false>(src, grid, offset_value, permute_fusion); \
            else \
                func<GridSample::Padding_ZEROS, true>(src, grid, offset_value, permute_fusion); \
        } \
        else if (padding_mode == GridSample::Padding_BORDER) \
        { \
            if (align_corner == 0) \
                func<GridSample::Padding_BORDER, false>(src, grid, offset_value, permute_fusion); \
            else \
                func<GridSample::Padding_BORDER, true>(src, grid, offset_value, permute_fusion); \
        } \
        else if (padding_mode == GridSample::Padding_REFLECTION) \
        { \
            if (align_corner == 0) \
                func<GridSample::Padding_REFLECTION, false>(src, grid, offset_value, permute_fusion); \
            else \
                func<GridSample::Padding_REFLECTION, true>(src, grid, offset_value, permute_fusion); \
        } \
    } while (0)

#include "gridsample_bilinear_compute_blob.h"
#include "gridsample_bicubic_compute_blob.h"
#include "gridsample_nearest_compute_blob.h"
#undef GRIDSAMPLE_COMPUTE_BLOB_DISPATCH
