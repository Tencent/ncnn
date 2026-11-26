// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "reshape_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

#include <string.h>

namespace ncnn {

Reshape_powerpc::Reshape_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
#endif // __ALTIVEC__
}

// base implementation is not packing aware; re-pack the output afterwards
static int reshape_base(const Reshape_powerpc& self, const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt)
{
    int ret = self.Reshape::forward(bottom_blobs, top_blobs, opt);
    if (ret != 0)
        return ret;

    Mat& top_blob = top_blobs[0];

    int repack_count = 0;
    if (top_blob.dims == 1)
        repack_count = top_blob.w;
    if (top_blob.dims == 2)
        repack_count = top_blob.h;
    if (top_blob.dims == 3 || top_blob.dims == 4)
        repack_count = top_blob.c;

    if (opt.use_packing_layout && repack_count % 4 == 0)
    {
        Mat top_blob_packed;
        convert_packing(top_blob, top_blob_packed, 4, opt);
        if (!top_blob_packed.empty())
        {
            top_blob = top_blob_packed;
            return 0;
        }
    }

    return 0;
}

int Reshape_powerpc::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];

#if NCNN_BATCH
    if (support_batch)
    {
        // batch path -> base implementation, unpacked
        if (bottom_blob.elempack != 1)
        {
            Option opt_pack1 = opt;
            opt_pack1.blob_allocator = opt.workspace_allocator;

            Mat bottom_blob_unpacked;
            convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack1);
            if (bottom_blob_unpacked.empty())
                return -100;

            std::vector<Mat> bottom_blobs_unpacked;
            bottom_blobs_unpacked.push_back(bottom_blob_unpacked);
            for (size_t i = 1; i < bottom_blobs.size(); i++)
                bottom_blobs_unpacked.push_back(bottom_blobs[i]);

            return reshape_base(*this, bottom_blobs_unpacked, top_blobs, opt);
        }

        return reshape_base(*this, bottom_blobs, top_blobs, opt);
    }
#endif

    int elembits = bottom_blob.elembits();
    if (elembits != 32)
    {
        // int8 / bf16 -> base implementation, unpacked
        if (bottom_blob.elempack != 1)
        {
            Option opt_pack1 = opt;
            opt_pack1.blob_allocator = opt.workspace_allocator;

            Mat bottom_blob_unpacked;
            convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack1);
            if (bottom_blob_unpacked.empty())
                return -100;

            std::vector<Mat> bottom_blobs_unpacked;
            bottom_blobs_unpacked.push_back(bottom_blob_unpacked);
            for (size_t i = 1; i < bottom_blobs.size(); i++)
                bottom_blobs_unpacked.push_back(bottom_blobs[i]);

            return reshape_base(*this, bottom_blobs_unpacked, top_blobs, opt);
        }

        return reshape_base(*this, bottom_blobs, top_blobs, opt);
    }

#if __ALTIVEC__
    Mat& top_blob = top_blobs[0];

    int outw = w;
    int outh = h;
    int outd = d;
    int outc = c;

    if (!shape_expr.empty())
    {
        int er = eval_shape_expr(bottom_blobs, outw, outh, outd, outc);
        if (er != 0)
            return -1;
    }

    if (ndim == 1)
    {
        flatten(bottom_blob, top_blob, opt);
        if (top_blob.empty())
            return -100;

        return 0;
    }

    const int dims = bottom_blob.dims;
    const int elempack = bottom_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;
    const int total = bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.c * elempack;

    if (ndim == 2)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (outh == 0)
            outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;

        if (outw == -1)
            outw = total / outh;
        if (outh == -1)
            outh = total / outw;

        int out_elempack = 1;
        if (opt.use_packing_layout)
            out_elempack = outh % 4 == 0 ? 4 : 1;
        const size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 2 && bottom_blob.h * elempack == outh && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        if (out_elempack == 1)
        {
            flatten(bottom_blob, top_blob, opt);
            if (top_blob.empty())
                return -100;

            top_blob.dims = 2;
            top_blob.w = outw;
            top_blob.h = outh;
            top_blob.cstep = top_blob.cstep * top_blob.elempack;
            top_blob.elemsize = out_elemsize;
            top_blob.elempack = out_elempack;

            return 0;
        }

        Mat bottom_blob_flattened = bottom_blob;
        {
            Option opt_flatten = opt;
            opt_flatten.blob_allocator = opt.workspace_allocator;

            flatten(bottom_blob, bottom_blob_flattened, opt_flatten);
            if (bottom_blob_flattened.empty())
                return -100;
        }

        top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (out_elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < top_blob.h; i++)
            {
                const float* ptr0 = (const float*)bottom_blob_flattened + outw * i * 4;
                const float* ptr1 = (const float*)bottom_blob_flattened + outw * (i * 4 + 1);
                const float* ptr2 = (const float*)bottom_blob_flattened + outw * (i * 4 + 2);
                const float* ptr3 = (const float*)bottom_blob_flattened + outw * (i * 4 + 3);
                float* outptr = top_blob.row(i);

                int j = 0;
                for (; j + 3 < outw; j += 4)
                {
                    v4f32 _row0 = __ncnn_altivec_load(ptr0);
                    v4f32 _row1 = __ncnn_altivec_load(ptr1);
                    v4f32 _row2 = __ncnn_altivec_load(ptr2);
                    v4f32 _row3 = __ncnn_altivec_load(ptr3);

                    transpose4x4_ps(_row0, _row1, _row2, _row3);

                    __ncnn_altivec_store(outptr, _row0);
                    __ncnn_altivec_store(outptr + 4, _row1);
                    __ncnn_altivec_store(outptr + 8, _row2);
                    __ncnn_altivec_store(outptr + 12, _row3);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
                for (; j < outw; j++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }

            return 0;
        }

        return 0;
    }

    if (ndim == 3)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (outh == 0)
            outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
        if (outc == 0)
            outc = dims == 3 ? bottom_blob.c * elempack : bottom_blob.c;

        if (outw == -1)
            outw = total / outc / outh;
        if (outh == -1)
            outh = total / outc / outw;
        if (outc == -1)
            outc = total / outh / outw;

        outd = 1;
    }

    if (ndim == 4)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (outh == 0)
            outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
        if (outd == 0)
            outd = bottom_blob.d;
        if (outc == 0)
            outc = (dims == 3 || dims == 4) ? bottom_blob.c * elempack : bottom_blob.c;

        if (outw == -1)
            outw = total / outc / outd / outh;
        if (outh == -1)
            outh = total / outc / outd / outw;
        if (outd == -1)
            outd = total / outc / outh / outw;
        if (outc == -1)
            outc = total / outd / outh / outw;
    }

    int out_elempack = 1;
    if (opt.use_packing_layout)
        out_elempack = outc % 4 == 0 ? 4 : 1;
    const size_t out_elemsize = elemsize / elempack * out_elempack;

    if ((dims == 3 || dims == 4) && bottom_blob.c * elempack == outc && elempack == out_elempack)
    {
        top_blob = bottom_blob;
        top_blob.dims = ndim;
        top_blob.w = outw;
        top_blob.h = outh;
        top_blob.d = outd;
        return 0;
    }

    Mat bottom_blob_flattened = bottom_blob;
    {
        Option opt_flatten = opt;
        opt_flatten.blob_allocator = opt.workspace_allocator;

        flatten(bottom_blob, bottom_blob_flattened, opt_flatten);
        if (bottom_blob_flattened.empty())
            return -100;
    }

    if (ndim == 3)
        top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    else // if (ndim == 4)
        top_blob.create(outw, outh, outd, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int size = top_blob.w * top_blob.h * top_blob.d;

    if (out_elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < top_blob.c; q++)
        {
            const float* ptr0 = (const float*)bottom_blob_flattened + size * q * 4;
            const float* ptr1 = (const float*)bottom_blob_flattened + size * (q * 4 + 1);
            const float* ptr2 = (const float*)bottom_blob_flattened + size * (q * 4 + 2);
            const float* ptr3 = (const float*)bottom_blob_flattened + size * (q * 4 + 3);
            float* outptr = top_blob.channel(q);

            int i = 0;
            for (; i + 3 < size; i += 4)
            {
                v4f32 _row0 = __ncnn_altivec_load(ptr0);
                v4f32 _row1 = __ncnn_altivec_load(ptr1);
                v4f32 _row2 = __ncnn_altivec_load(ptr2);
                v4f32 _row3 = __ncnn_altivec_load(ptr3);

                transpose4x4_ps(_row0, _row1, _row2, _row3);

                __ncnn_altivec_store(outptr, _row0);
                __ncnn_altivec_store(outptr + 4, _row1);
                __ncnn_altivec_store(outptr + 8, _row2);
                __ncnn_altivec_store(outptr + 12, _row3);

                ptr0 += 4;
                ptr1 += 4;
                ptr2 += 4;
                ptr3 += 4;
                outptr += 16;
            }
            for (; i < size; i++)
            {
                outptr[0] = *ptr0++;
                outptr[1] = *ptr1++;
                outptr[2] = *ptr2++;
                outptr[3] = *ptr3++;

                outptr += 4;
            }
        }

        return 0;
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < top_blob.c; q++)
    {
        const float* ptr = (const float*)bottom_blob_flattened + size * q;
        float* outptr = top_blob.channel(q);

        int i = 0;
        for (; i + 3 < size; i += 4)
        {
            __ncnn_altivec_store(outptr, __ncnn_altivec_load(ptr));
            ptr += 4;
            outptr += 4;
        }
        for (; i < size; i++)
        {
            *outptr++ = *ptr++;
        }
    }

    return 0;
#else
    // no altivec -> base implementation
    return reshape_base(*this, bottom_blobs, top_blobs, opt);
#endif // __ALTIVEC__
}

} // namespace ncnn
