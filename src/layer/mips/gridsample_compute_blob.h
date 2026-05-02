// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static float grid_sample_unormalize_mips(int w, float coordx, int align_corner)
{
    return align_corner ? (coordx + 1) / 2.f * (w - 1) : ((coordx + 1) * w - 1) / 2.f;
}

static float border_coord_mips(float x, float border)
{
    if (x < 0.f)
        return 0.f;
    if (x > border)
        return border;
    return x;
}

static float reflect_coord_mips(float x, int high)
{
    x = fabsf(x);
    x = high - fabsf(x - high);
    return x;
}

static float compute_coord_mips(float sx, int w, int padding_mode, int align_corner)
{
    if (padding_mode == GridSample::Padding_BORDER)
    {
        sx = border_coord_mips(sx, w - 1);
    }
    else if (padding_mode == GridSample::Padding_REFLECTION)
    {
        if (align_corner)
        {
            sx = reflect_coord_mips(sx, w - 1);
        }
        else
        {
            sx = reflect_coord_mips(sx + 0.5f, w) - 0.5f;
            sx = border_coord_mips(sx, w - 1);
        }
    }

    return sx;
}

static bool in_bounds_mips(int x, int y, int w, int h)
{
    return x >= 0 && y >= 0 && x < w && y < h;
}

static bool in_bounds_mips(int x, int y, int z, int w, int h, int d)
{
    return x >= 0 && y >= 0 && z >= 0 && x < w && y < h && z < d;
}

#if __mips_msa
static v4f32 gridsample_set4_ps_msa(float v0, float v1, float v2, float v3)
{
    FloatInt fi0;
    FloatInt fi1;
    FloatInt fi2;
    FloatInt fi3;
    fi0.f = v0;
    fi1.f = v1;
    fi2.f = v2;
    fi3.f = v3;
    return (v4f32)__msa_set_w(fi0.i, fi1.i, fi2.i, fi3.i);
}

static v4f32 gridsample_floor_msa(v4f32 _v)
{
    v4i32 _vi = __msa_ftrunc_s_w(_v);
    v4f32 _vf = __msa_ffint_s_w(_vi);
    v4i32 _mask = __msa_fclt_w(_v, _vf);
    return __msa_ffint_s_w(__msa_addv_w(_vi, _mask));
}

static v4f32 grid_sample_unormalize_msa(int w, v4f32 _coordx, int align_corner)
{
    v4f32 _one = __msa_fill_w_f32(1.f);
    v4f32 _half = __msa_fill_w_f32(0.5f);

    if (align_corner)
        return __msa_fmul_w(__msa_fmul_w(__msa_fadd_w(_coordx, _one), _half), __msa_fill_w_f32((float)(w - 1)));

    return __msa_fmul_w(__msa_fsub_w(__msa_fmul_w(__msa_fadd_w(_coordx, _one), __msa_fill_w_f32((float)w)), _one), _half);
}

static v4f32 compute_coord_msa(v4f32 _sx, int w, int padding_mode, int align_corner)
{
    if (padding_mode == GridSample::Padding_BORDER)
    {
        _sx = __msa_fmin_w(__msa_fill_w_f32((float)(w - 1)), __msa_fmax_w(_sx, __msa_fill_w_f32(0.f)));
    }
    else if (padding_mode == GridSample::Padding_REFLECTION)
    {
        if (align_corner)
        {
            v4f32 _high = __msa_fill_w_f32((float)(w - 1));
            _sx = (v4f32)__msa_bclri_w((v4u32)_sx, 31);
            _sx = __msa_fsub_w(_high, (v4f32)__msa_bclri_w((v4u32)__msa_fsub_w(_sx, _high), 31));
        }
        else
        {
            v4f32 _half = __msa_fill_w_f32(0.5f);
            v4f32 _length = __msa_fill_w_f32((float)w);
            _sx = (v4f32)__msa_bclri_w((v4u32)__msa_fadd_w(_sx, _half), 31);
            _sx = __msa_fsub_w(__msa_fsub_w(_length, (v4f32)__msa_bclri_w((v4u32)__msa_fsub_w(_sx, _length), 31)), _half);
            _sx = __msa_fmin_w(__msa_fill_w_f32((float)(w - 1)), __msa_fmax_w(_sx, __msa_fill_w_f32(0.f)));
        }
    }

    return _sx;
}

static v4i32 gridsample_in_range_msa(v4f32 _v, int size)
{
    v4i32 _ge0 = __msa_fclt_w(__msa_fill_w_f32(-1.f), _v);
    v4i32 _lt_size = __msa_fclt_w(_v, __msa_fill_w_f32((float)size));
    return (v4i32)__msa_and_v((v16u8)_ge0, (v16u8)_lt_size);
}

static v4i32 gridsample_select_offset_msa(v4i32 _offset, v4i32 _mask)
{
    return (v4i32)__msa_bsel_v((v16u8)_mask, (v16u8)__msa_fill_w(-1), (v16u8)_offset);
}

static void gridsample_store_2d_bilinear_msa(float* offset_value_ptr, v4f32 _sample_x, v4f32 _sample_y, int w, int h, int elempack)
{
    v4f32 _x0f = gridsample_floor_msa(_sample_x);
    v4f32 _y0f = gridsample_floor_msa(_sample_y);
    v4f32 _x1f = __msa_fadd_w(_x0f, __msa_fill_w_f32(1.f));
    v4f32 _y1f = __msa_fadd_w(_y0f, __msa_fill_w_f32(1.f));

    v4i32 _x0_in_range = gridsample_in_range_msa(_x0f, w);
    v4i32 _x1_in_range = gridsample_in_range_msa(_x1f, w);
    v4i32 _y0_in_range = gridsample_in_range_msa(_y0f, h);
    v4i32 _y1_in_range = gridsample_in_range_msa(_y1f, h);

    v4i32 _x0 = __msa_ftrunc_s_w(_x0f);
    v4i32 _y0 = __msa_ftrunc_s_w(_y0f);

    v4i32 _base = __msa_mulv_w(_y0, __msa_fill_w(w));
    _base = __msa_addv_w(_base, _x0);
    _base = __msa_mulv_w(_base, __msa_fill_w(elempack));

    v4i32 _elempack = __msa_fill_w(elempack);
    v4i32 _wstep = __msa_fill_w(w * elempack);
    v4i32 _nw_offset = _base;
    v4i32 _ne_offset = __msa_addv_w(_nw_offset, _elempack);
    v4i32 _sw_offset = __msa_addv_w(_nw_offset, _wstep);
    v4i32 _se_offset = __msa_addv_w(_sw_offset, _elempack);

    v4i32 _v00_in_range = (v4i32)__msa_and_v((v16u8)_x0_in_range, (v16u8)_y0_in_range);
    v4i32 _v01_in_range = (v4i32)__msa_and_v((v16u8)_x1_in_range, (v16u8)_y0_in_range);
    v4i32 _v10_in_range = (v4i32)__msa_and_v((v16u8)_x0_in_range, (v16u8)_y1_in_range);
    v4i32 _v11_in_range = (v4i32)__msa_and_v((v16u8)_x1_in_range, (v16u8)_y1_in_range);

    _nw_offset = gridsample_select_offset_msa(_nw_offset, _v00_in_range);
    _ne_offset = gridsample_select_offset_msa(_ne_offset, _v01_in_range);
    _sw_offset = gridsample_select_offset_msa(_sw_offset, _v10_in_range);
    _se_offset = gridsample_select_offset_msa(_se_offset, _v11_in_range);

    v4f32 _nw = (v4f32)_nw_offset;
    v4f32 _ne = (v4f32)_ne_offset;
    v4f32 _sw = (v4f32)_sw_offset;
    v4f32 _se = (v4f32)_se_offset;
    transpose4x4_ps(_nw, _ne, _sw, _se);

    __msa_st_w((v4i32)_nw, (int*)offset_value_ptr, 0);
    __msa_st_w((v4i32)_ne, (int*)(offset_value_ptr + 6), 0);
    __msa_st_w((v4i32)_sw, (int*)(offset_value_ptr + 12), 0);
    __msa_st_w((v4i32)_se, (int*)(offset_value_ptr + 18), 0);

    v4f32 _alpha = __msa_fsub_w(_sample_x, _x0f);
    v4f32 _beta = __msa_fsub_w(_sample_y, _y0f);
    float alpha[4];
    float beta[4];
    __msa_st_w((v4i32)_alpha, alpha, 0);
    __msa_st_w((v4i32)_beta, beta, 0);

    offset_value_ptr[4] = alpha[0];
    offset_value_ptr[5] = beta[0];
    offset_value_ptr[10] = alpha[1];
    offset_value_ptr[11] = beta[1];
    offset_value_ptr[16] = alpha[2];
    offset_value_ptr[17] = beta[2];
    offset_value_ptr[22] = alpha[3];
    offset_value_ptr[23] = beta[3];
}

static void gridsample_store_2d_nearest_msa(int* offset_ptr, v4f32 _sample_x, v4f32 _sample_y, int w, int h, int elempack)
{
    v4f32 _x0f = gridsample_floor_msa(__msa_fadd_w(_sample_x, __msa_fill_w_f32(0.5f)));
    v4f32 _y0f = gridsample_floor_msa(__msa_fadd_w(_sample_y, __msa_fill_w_f32(0.5f)));

    v4i32 _x0_in_range = gridsample_in_range_msa(_x0f, w);
    v4i32 _y0_in_range = gridsample_in_range_msa(_y0f, h);
    v4i32 _in_range = (v4i32)__msa_and_v((v16u8)_x0_in_range, (v16u8)_y0_in_range);

    v4i32 _x0 = __msa_ftrunc_s_w(_x0f);
    v4i32 _y0 = __msa_ftrunc_s_w(_y0f);
    v4i32 _offset = __msa_mulv_w(_y0, __msa_fill_w(w));
    _offset = __msa_addv_w(_offset, _x0);
    _offset = __msa_mulv_w(_offset, __msa_fill_w(elempack));
    _offset = gridsample_select_offset_msa(_offset, _in_range);

    __msa_st_w(_offset, offset_ptr, 0);
}

static void gridsample_store_3d_bilinear_msa(float* offset_value_ptr, v4f32 _sample_x, v4f32 _sample_y, v4f32 _sample_z, int w, int h, int d, int elempack)
{
    v4f32 _x0f = gridsample_floor_msa(_sample_x);
    v4f32 _y0f = gridsample_floor_msa(_sample_y);
    v4f32 _z0f = gridsample_floor_msa(_sample_z);
    v4f32 _x1f = __msa_fadd_w(_x0f, __msa_fill_w_f32(1.f));
    v4f32 _y1f = __msa_fadd_w(_y0f, __msa_fill_w_f32(1.f));
    v4f32 _z1f = __msa_fadd_w(_z0f, __msa_fill_w_f32(1.f));

    v4i32 _x0_in_range = gridsample_in_range_msa(_x0f, w);
    v4i32 _x1_in_range = gridsample_in_range_msa(_x1f, w);
    v4i32 _y0_in_range = gridsample_in_range_msa(_y0f, h);
    v4i32 _y1_in_range = gridsample_in_range_msa(_y1f, h);
    v4i32 _z0_in_range = gridsample_in_range_msa(_z0f, d);
    v4i32 _z1_in_range = gridsample_in_range_msa(_z1f, d);

    v4i32 _x0 = __msa_ftrunc_s_w(_x0f);
    v4i32 _y0 = __msa_ftrunc_s_w(_y0f);
    v4i32 _z0 = __msa_ftrunc_s_w(_z0f);

    v4i32 _base = __msa_mulv_w(_z0, __msa_fill_w(w * h));
    _base = __msa_addv_w(_base, __msa_mulv_w(_y0, __msa_fill_w(w)));
    _base = __msa_addv_w(_base, _x0);
    _base = __msa_mulv_w(_base, __msa_fill_w(elempack));

    v4i32 _elempack = __msa_fill_w(elempack);
    v4i32 _wstep = __msa_fill_w(w * elempack);
    v4i32 _dstep = __msa_fill_w(w * h * elempack);

    v4i32 _offset0 = _base;
    v4i32 _offset1 = __msa_addv_w(_offset0, _elempack);
    v4i32 _offset2 = __msa_addv_w(_offset0, _wstep);
    v4i32 _offset3 = __msa_addv_w(_offset2, _elempack);
    v4i32 _offset4 = __msa_addv_w(_offset0, _dstep);
    v4i32 _offset5 = __msa_addv_w(_offset4, _elempack);
    v4i32 _offset6 = __msa_addv_w(_offset4, _wstep);
    v4i32 _offset7 = __msa_addv_w(_offset6, _elempack);

    v4i32 _xy00 = (v4i32)__msa_and_v((v16u8)_x0_in_range, (v16u8)_y0_in_range);
    v4i32 _xy01 = (v4i32)__msa_and_v((v16u8)_x1_in_range, (v16u8)_y0_in_range);
    v4i32 _xy10 = (v4i32)__msa_and_v((v16u8)_x0_in_range, (v16u8)_y1_in_range);
    v4i32 _xy11 = (v4i32)__msa_and_v((v16u8)_x1_in_range, (v16u8)_y1_in_range);

    _offset0 = gridsample_select_offset_msa(_offset0, (v4i32)__msa_and_v((v16u8)_xy00, (v16u8)_z0_in_range));
    _offset1 = gridsample_select_offset_msa(_offset1, (v4i32)__msa_and_v((v16u8)_xy01, (v16u8)_z0_in_range));
    _offset2 = gridsample_select_offset_msa(_offset2, (v4i32)__msa_and_v((v16u8)_xy10, (v16u8)_z0_in_range));
    _offset3 = gridsample_select_offset_msa(_offset3, (v4i32)__msa_and_v((v16u8)_xy11, (v16u8)_z0_in_range));
    _offset4 = gridsample_select_offset_msa(_offset4, (v4i32)__msa_and_v((v16u8)_xy00, (v16u8)_z1_in_range));
    _offset5 = gridsample_select_offset_msa(_offset5, (v4i32)__msa_and_v((v16u8)_xy01, (v16u8)_z1_in_range));
    _offset6 = gridsample_select_offset_msa(_offset6, (v4i32)__msa_and_v((v16u8)_xy10, (v16u8)_z1_in_range));
    _offset7 = gridsample_select_offset_msa(_offset7, (v4i32)__msa_and_v((v16u8)_xy11, (v16u8)_z1_in_range));

    int offsets[8][4] = {};
    float alpha[4];
    float beta[4];
    float gamma[4];
    __msa_st_w(_offset0, offsets[0], 0);
    __msa_st_w(_offset1, offsets[1], 0);
    __msa_st_w(_offset2, offsets[2], 0);
    __msa_st_w(_offset3, offsets[3], 0);
    __msa_st_w(_offset4, offsets[4], 0);
    __msa_st_w(_offset5, offsets[5], 0);
    __msa_st_w(_offset6, offsets[6], 0);
    __msa_st_w(_offset7, offsets[7], 0);
    __msa_st_w((v4i32)__msa_fsub_w(_sample_x, _x0f), alpha, 0);
    __msa_st_w((v4i32)__msa_fsub_w(_sample_y, _y0f), beta, 0);
    __msa_st_w((v4i32)__msa_fsub_w(_sample_z, _z0f), gamma, 0);

    for (int i = 0; i < 4; i++)
    {
        int* offset_ptr = (int*)(offset_value_ptr + i * 11);
        offset_ptr[0] = offsets[0][i];
        offset_ptr[1] = offsets[1][i];
        offset_ptr[2] = offsets[2][i];
        offset_ptr[3] = offsets[3][i];
        offset_ptr[4] = offsets[4][i];
        offset_ptr[5] = offsets[5][i];
        offset_ptr[6] = offsets[6][i];
        offset_ptr[7] = offsets[7][i];
        offset_value_ptr[i * 11 + 8] = alpha[i];
        offset_value_ptr[i * 11 + 9] = beta[i];
        offset_value_ptr[i * 11 + 10] = gamma[i];
    }
}

static void gridsample_store_3d_nearest_msa(int* offset_ptr, v4f32 _sample_x, v4f32 _sample_y, v4f32 _sample_z, int w, int h, int d, int elempack)
{
    v4f32 _x0f = gridsample_floor_msa(__msa_fadd_w(_sample_x, __msa_fill_w_f32(0.5f)));
    v4f32 _y0f = gridsample_floor_msa(__msa_fadd_w(_sample_y, __msa_fill_w_f32(0.5f)));
    v4f32 _z0f = gridsample_floor_msa(__msa_fadd_w(_sample_z, __msa_fill_w_f32(0.5f)));

    v4i32 _x0_in_range = gridsample_in_range_msa(_x0f, w);
    v4i32 _y0_in_range = gridsample_in_range_msa(_y0f, h);
    v4i32 _z0_in_range = gridsample_in_range_msa(_z0f, d);
    v4i32 _in_range = (v4i32)__msa_and_v(__msa_and_v((v16u8)_x0_in_range, (v16u8)_y0_in_range), (v16u8)_z0_in_range);

    v4i32 _x0 = __msa_ftrunc_s_w(_x0f);
    v4i32 _y0 = __msa_ftrunc_s_w(_y0f);
    v4i32 _z0 = __msa_ftrunc_s_w(_z0f);
    v4i32 _offset = __msa_mulv_w(_z0, __msa_fill_w(w * h));
    _offset = __msa_addv_w(_offset, __msa_mulv_w(_y0, __msa_fill_w(w)));
    _offset = __msa_addv_w(_offset, _x0);
    _offset = __msa_mulv_w(_offset, __msa_fill_w(elempack));
    _offset = gridsample_select_offset_msa(_offset, _in_range);

    __msa_st_w(_offset, offset_ptr, 0);
}

static void gridsample_store_2d_bicubic_msa(float* offset_value_ptr, v4f32 _sample_x, v4f32 _sample_y, int w, int h, int elempack, int padding_mode, int align_corner)
{
    v4f32 _x1f = gridsample_floor_msa(_sample_x);
    v4f32 _y1f = gridsample_floor_msa(_sample_y);

    v4f32 _tx = __msa_fsub_w(_sample_x, _x1f);
    v4f32 _ty = __msa_fsub_w(_sample_y, _y1f);

    v4f32 _gx0 = compute_coord_msa(__msa_fsub_w(_x1f, __msa_fill_w_f32(1.f)), w, padding_mode, align_corner);
    v4f32 _gx1 = compute_coord_msa(_x1f, w, padding_mode, align_corner);
    v4f32 _gx2 = compute_coord_msa(__msa_fadd_w(_x1f, __msa_fill_w_f32(1.f)), w, padding_mode, align_corner);
    v4f32 _gx3 = compute_coord_msa(__msa_fadd_w(_x1f, __msa_fill_w_f32(2.f)), w, padding_mode, align_corner);

    v4i32 _x0_in_range = gridsample_in_range_msa(_gx0, w);
    v4i32 _x1_in_range = gridsample_in_range_msa(_gx1, w);
    v4i32 _x2_in_range = gridsample_in_range_msa(_gx2, w);
    v4i32 _x3_in_range = gridsample_in_range_msa(_gx3, w);

    v4i32 _x0 = __msa_ftrunc_s_w(_gx0);
    v4i32 _x1 = __msa_ftrunc_s_w(_gx1);
    v4i32 _x2 = __msa_ftrunc_s_w(_gx2);
    v4i32 _x3 = __msa_ftrunc_s_w(_gx3);

    int offsets[16][4] = {};
    for (int i = 0; i < 4; i++)
    {
        v4f32 _gy = compute_coord_msa(__msa_fadd_w(_y1f, __msa_fill_w_f32((float)(i - 1))), h, padding_mode, align_corner);
        v4i32 _y_in_range = gridsample_in_range_msa(_gy, h);
        v4i32 _offset_y = __msa_mulv_w(__msa_ftrunc_s_w(_gy), __msa_fill_w(w));

        v4i32 _offset0 = __msa_mulv_w(__msa_addv_w(_offset_y, _x0), __msa_fill_w(elempack));
        v4i32 _offset1 = __msa_mulv_w(__msa_addv_w(_offset_y, _x1), __msa_fill_w(elempack));
        v4i32 _offset2 = __msa_mulv_w(__msa_addv_w(_offset_y, _x2), __msa_fill_w(elempack));
        v4i32 _offset3 = __msa_mulv_w(__msa_addv_w(_offset_y, _x3), __msa_fill_w(elempack));

        _offset0 = gridsample_select_offset_msa(_offset0, (v4i32)__msa_and_v((v16u8)_x0_in_range, (v16u8)_y_in_range));
        _offset1 = gridsample_select_offset_msa(_offset1, (v4i32)__msa_and_v((v16u8)_x1_in_range, (v16u8)_y_in_range));
        _offset2 = gridsample_select_offset_msa(_offset2, (v4i32)__msa_and_v((v16u8)_x2_in_range, (v16u8)_y_in_range));
        _offset3 = gridsample_select_offset_msa(_offset3, (v4i32)__msa_and_v((v16u8)_x3_in_range, (v16u8)_y_in_range));

        __msa_st_w(_offset0, offsets[i * 4], 0);
        __msa_st_w(_offset1, offsets[i * 4 + 1], 0);
        __msa_st_w(_offset2, offsets[i * 4 + 2], 0);
        __msa_st_w(_offset3, offsets[i * 4 + 3], 0);
    }

    float tx[4];
    float ty[4];
    __msa_st_w((v4i32)_tx, tx, 0);
    __msa_st_w((v4i32)_ty, ty, 0);

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
#endif // __mips_msa

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
