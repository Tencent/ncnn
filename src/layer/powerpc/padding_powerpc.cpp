// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "padding_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

Padding_powerpc::Padding_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
    support_any_packing = true;
#endif // __ALTIVEC__
}

#if __ALTIVEC__
// vectorized contiguous fill of n floats
static void fill_vec(float* dst, float v, int n)
{
    v4f32 _v = vec_splats(v);

    int i = 0;
    for (; i + 4 <= n; i += 4)
    {
        __ncnn_altivec_store(dst + i, _v);
    }
    for (; i < n; i++)
    {
        dst[i] = v;
    }
}

// vectorized contiguous copy of n floats
static void copy_vec(float* dst, const float* src, int n)
{
    int i = 0;
    for (; i + 4 <= n; i += 4)
    {
        v4f32 _v = __ncnn_altivec_load(src + i);
        __ncnn_altivec_store(dst + i, _v);
    }
    for (; i < n; i++)
    {
        dst[i] = src[i];
    }
}

// constant border padding, one element = one 4-float pack
static void padding_constant_pack4_altivec(const Mat& src, Mat& dst, int top, int bottom, int left, int right, v4f32 v)
{
    const float* ptr = src;
    float* outptr = dst;
    int top_size = top * dst.w;
    int bottom_size = bottom * dst.w;

    // fill top
    for (int y = 0; y < top_size; y++)
    {
        __ncnn_altivec_store(outptr, v);
        outptr += 4;
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            __ncnn_altivec_store(outptr, v);
            outptr += 4;
        }
        for (int x = 0; x < src.w; x++)
        {
            v4f32 _p = __ncnn_altivec_load(ptr);
            __ncnn_altivec_store(outptr, _p);
            ptr += 4;
            outptr += 4;
        }
        for (int x = 0; x < right; x++)
        {
            __ncnn_altivec_store(outptr, v);
            outptr += 4;
        }
    }
    // fill bottom
    for (int y = 0; y < bottom_size; y++)
    {
        __ncnn_altivec_store(outptr, v);
        outptr += 4;
    }
}

// constant border padding, pack1, vectorized fill and interior copy
static void padding_constant_pack1_altivec(const Mat& src, Mat& dst, int top, int bottom, int left, int right, float v)
{
    const float* ptr = src;
    float* outptr = dst;

    // fill top
    fill_vec(outptr, v, top * dst.w);
    outptr += top * dst.w;
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        fill_vec(outptr, v, left);
        outptr += left;

        copy_vec(outptr, ptr, src.w);
        ptr += src.w;
        outptr += src.w;

        fill_vec(outptr, v, right);
        outptr += right;
    }
    // fill bottom
    fill_vec(outptr, v, bottom * dst.w);
}
#endif // __ALTIVEC__

int Padding_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (top == 0 && bottom == 0 && left == 0 && right == 0 && front == 0 && behind == 0)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int elembits = bottom_blob.elembits();

    if (elembits == 8)
    {
        // int8 -> base implementation, unpacked
        Mat bottom_blob_unpacked = bottom_blob;
        if (bottom_blob.elempack != 1)
        {
            Option opt_pack1 = opt;
            opt_pack1.blob_allocator = opt.workspace_allocator;

            convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack1);
            if (bottom_blob_unpacked.empty())
                return -100;
        }

        return Padding::forward(bottom_blob_unpacked, top_blob, opt);
    }

    if (elembits == 16)
    {
        // fp16 / bf16 -> base implementation, unpacked
        Mat bottom_blob_unpacked = bottom_blob;
        if (bottom_blob.elempack != 1)
        {
            Option opt_pack1 = opt;
            opt_pack1.blob_allocator = opt.workspace_allocator;

            convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack1);
            if (bottom_blob_unpacked.empty())
                return -100;
        }

        return Padding::forward(bottom_blob_unpacked, top_blob, opt);
    }

#if __ALTIVEC__
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    if (elempack == 4)
    {
        if (dims == 1)
        {
            int outw = w * elempack + left + right;

            int out_elempack = outw % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (left % 4 == 0 && out_elempack == 4 && type == 0)
            {
                top_blob.create(outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                v4f32 pad_value = vec_splats(value);
                padding_constant_pack4_altivec(bottom_blob, top_blob, 0, 0, left / 4, right / 4, pad_value);

                return 0;
            }
        }

        if (dims == 2)
        {
            int outw = w + left + right;
            int outh = h * elempack + top + bottom;

            int out_elempack = outh % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (top % 4 == 0 && out_elempack == 4 && type == 0)
            {
                top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                v4f32 pad_value = vec_splats(value);
                padding_constant_pack4_altivec(bottom_blob, top_blob, top / 4, bottom / 4, left, right, pad_value);

                return 0;
            }
        }

        if (dims == 3)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outc = channels * elempack + front + behind;

            int out_elempack = outc % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (front % 4 == 0 && out_elempack == 4 && type == 0 && !(outc != channels * elempack && type != 0))
            {
                top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                int front_ = front / elempack;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < outc / out_elempack; q++)
                {
                    Mat borderm = top_blob.channel(q);

                    v4f32 pad_value = per_channel_pad_data_size ? __ncnn_altivec_load((const float*)per_channel_pad_data + q * 4) : vec_splats(value);
                    // Channel padding
                    if ((q - front_) < 0 || (q - front_) >= channels)
                    {
                        borderm.fill(pad_value);
                    }
                    else
                    {
                        const Mat m = bottom_blob.channel(q - front_);
                        if (type == 0)
                            padding_constant_pack4_altivec(m, borderm, top, bottom, left, right, pad_value);
                    }
                }

                return 0;
            }
        }

        if (dims == 4)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outd = d + front + behind;

            if (type == 0)
            {
                top_blob.create(outw, outh, outd, channels, elemsize, elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    v4f32 pad_value = per_channel_pad_data_size ? __ncnn_altivec_load((const float*)per_channel_pad_data + q * 4) : vec_splats(value);

                    for (int z = 0; z < outd; z++)
                    {
                        Mat borderm = top_blob.channel(q).depth(z);

                        // depth padding
                        if ((z - front) < 0 || (z - front) >= d)
                        {
                            borderm.fill(pad_value);
                        }
                        else
                        {
                            const Mat m = bottom_blob.channel(q).depth(z - front);
                            padding_constant_pack4_altivec(m, borderm, top, bottom, left, right, pad_value);
                        }
                    }
                }

                return 0;
            }
        }
    }

    if (elempack == 1 && type == 0 && per_channel_pad_data_size == 0)
    {
        if (dims == 1)
        {
            int outw = w + left + right;

            top_blob.create(outw, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            padding_constant_pack1_altivec(bottom_blob, top_blob, 0, 0, left, right, value);

            return 0;
        }

        if (dims == 2)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;

            top_blob.create(outw, outh, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            padding_constant_pack1_altivec(bottom_blob, top_blob, top, bottom, left, right, value);

            return 0;
        }

        if (dims == 3)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outc = channels + front + behind;

            top_blob.create(outw, outh, outc, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                Mat borderm = top_blob.channel(q);

                // Channel padding
                if ((q - front) < 0 || (q - front) >= channels)
                {
                    borderm.fill(value);
                }
                else
                {
                    const Mat m = bottom_blob.channel(q - front);
                    padding_constant_pack1_altivec(m, borderm, top, bottom, left, right, value);
                }
            }

            return 0;
        }

        if (dims == 4)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outd = d + front + behind;

            top_blob.create(outw, outh, outd, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int z = 0; z < outd; z++)
                {
                    Mat borderm = top_blob.channel(q).depth(z);

                    // depth padding
                    if ((z - front) < 0 || (z - front) >= d)
                    {
                        borderm.fill(value);
                    }
                    else
                    {
                        const Mat m = bottom_blob.channel(q).depth(z - front);
                        padding_constant_pack1_altivec(m, borderm, top, bottom, left, right, value);
                    }
                }
            }

            return 0;
        }
    }
#endif // __ALTIVEC__

    // unsupported combination -> base implementation, unpacked
    Mat bottom_blob_unpacked = bottom_blob;
    if (bottom_blob.elempack != 1)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_allocator = opt.workspace_allocator;

        Mat bottom_blob_unpacked_;
        convert_packing(bottom_blob, bottom_blob_unpacked_, 1, opt_pack1);
        if (bottom_blob_unpacked_.empty())
            return -100;
        bottom_blob_unpacked = bottom_blob_unpacked_;
    }

    Mat top_blob_unpacked;
    int ret = Padding::forward(bottom_blob_unpacked, top_blob_unpacked, opt);
    if (ret != 0)
    {
        top_blob.release();
        return ret;
    }

    int out_elempack = 1;
#if __ALTIVEC__
    if (opt.use_packing_layout)
    {
        out_elempack = top_blob_unpacked.c % 4 == 0 ? 4 : 1;
    }
#endif

    convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);
    if (top_blob.empty())
        return -100;

    return 0;
}

} // namespace ncnn
