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

static void linear_coeffs(int w, int outw, int* xofs, float* alpha, int align_corner)
{
    double scale = (double)w / outw;
    if (align_corner)
    {
        scale = (double)(w - 1) / (outw - 1);
    }

    for (int dx = 0; dx < outw; dx++)
    {
        float fx = (float)((dx + 0.5) * scale - 0.5);
        if (align_corner)
        {
            fx = (float)(dx * scale);
        }

        int sx = floor(fx);
        fx -= sx;

        if (sx < 0)
        {
            sx = 0;
            fx = 0.f;
        }
        if (sx >= w - 1)
        {
            sx = w - 2;
            fx = 1.f;
        }

        xofs[dx] = sx;

        alpha[dx * 2] = 1.f - fx;
        alpha[dx * 2 + 1] = fx;
    }
}

static void resize_bilinear_image(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    int w = dst.w;
    int h = dst.h;

    // loop body
    Mat rowsbuf0(w);
    Mat rowsbuf1(w);
    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++)
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            float* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const float* S1 = src.row(sy + 1);

            const float* alphap = alpha;
            float* rows1p = rows1;

#if __riscv_vector
            const unsigned int* pxofs = (const unsigned int*)xofs;
            int n = w;
            while (n > 0)
            {
                size_t vl = vsetvl_e32m4(n);

                vuint32m4_t _sx = vmul_vx_u32m4(vle32_v_u32m4(pxofs, vl), sizeof(float), vl);

                vfloat32m4_t _S1p0;
                vfloat32m4_t _S1p1;
                vloxseg2ei32_v_f32m4(&_S1p0, &_S1p1, S1, _sx, vl);

                vfloat32m4_t _a0;
                vfloat32m4_t _a1;
                vlseg2e32_v_f32m4(&_a0, &_a1, alphap, vl);

                vfloat32m4_t _rows1 = vfmacc_vv_f32m4(vfmul_vv_f32m4(_S1p0, _a0, vl), _S1p1, _a1, vl);

                vse32_v_f32m4(rows1p, _rows1, vl);

                pxofs += vl;
                alphap += vl * 2;
                rows1p += vl;
                n -= vl;
            }
#else  // __riscv_vector
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows1p[dx] = S1p[0] * a0 + S1p[1] * a1;

                alphap += 2;
            }
#endif // __riscv_vector
        }
        else
        {
            // hresize two rows
            const float* S0 = src.row(sy);
            const float* S1 = src.row(sy + 1);

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;

#if __riscv_vector
            const unsigned int* pxofs = (const unsigned int*)xofs;
            int n = w;
            while (n > 0)
            {
                size_t vl = vsetvl_e32m4(n);

                vuint32m4_t _sx = vmul_vx_u32m4(vle32_v_u32m4(pxofs, vl), sizeof(float), vl);

                vfloat32m4_t _S0p0;
                vfloat32m4_t _S0p1;
                vfloat32m4_t _S1p0;
                vfloat32m4_t _S1p1;

                vloxseg2ei32_v_f32m4(&_S0p0, &_S0p1, S0, _sx, vl);
                vloxseg2ei32_v_f32m4(&_S1p0, &_S1p1, S1, _sx, vl);

                vfloat32m4_t _a0;
                vfloat32m4_t _a1;
                vlseg2e32_v_f32m4(&_a0, &_a1, alphap, vl);

                vfloat32m4_t _rows0 = vfmacc_vv_f32m4(vfmul_vv_f32m4(_S0p0, _a0, vl), _S0p1, _a1, vl);
                vfloat32m4_t _rows1 = vfmacc_vv_f32m4(vfmul_vv_f32m4(_S1p0, _a0, vl), _S1p1, _a1, vl);

                vse32_v_f32m4(rows0p, _rows0, vl);
                vse32_v_f32m4(rows1p, _rows1, vl);

                pxofs += vl;
                alphap += vl * 2;
                rows0p += vl;
                rows1p += vl;
                n -= vl;
            }
#else  // __riscv_vector
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows0p[dx] = S0p[0] * a0 + S0p[1] * a1;
                rows1p[dx] = S1p[0] * a0 + S1p[1] * a1;

                alphap += 2;
            }
#endif // __riscv_vector
        }

        prev_sy1 = sy;

        // vresize
        float b0 = beta[0];
        float b1 = beta[1];

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* Dp = dst.row(dy);

#if __riscv_vector
        int n = w;
        while (n > 0)
        {
            size_t vl = vsetvl_e32m8(n);

            vfloat32m8_t _rows0 = vle32_v_f32m8(rows0p, vl);
            vfloat32m8_t _rows1 = vle32_v_f32m8(rows1p, vl);

            vfloat32m8_t _Dp = vfmacc_vf_f32m8(vfmul_vf_f32m8(_rows0, b0, vl), b1, _rows1, vl);

            vse32_v_f32m8(Dp, _Dp, vl);

            Dp += vl;
            rows0p += vl;
            rows1p += vl;
            n -= vl;
        }
#else  // __riscv_vector
        for (int i = 0; i < w; i++)
        {
            //             D[x] = rows0[x]*b0 + rows1[x]*b1;
            *Dp++ = *rows0p++ * b0 + *rows1p++ * b1;
        }
#endif // __riscv_vector

        beta += 2;
    }
}
