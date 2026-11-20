// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "innerproduct_powerpc.h"

#include "layer_type.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

#include "fused_activation.h"

namespace ncnn {

InnerProduct_powerpc::InnerProduct_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
#endif // __ALTIVEC__

    flatten = 0;
}

int InnerProduct_powerpc::create_pipeline(const Option& opt)
{
    {
        flatten = ncnn::create_layer_cpu(ncnn::LayerType::Flatten);

        ncnn::ParamDict pd;

        flatten->load_param(pd);

        flatten->create_pipeline(opt);
    }

#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        // no int8 kernel on powerpc yet, the generic InnerProduct::forward
        // consumes the raw int8 weight_data, so keep it untouched
        return 0;
    }
#endif

    const int num_input = weight_data_size / num_output;

    int out_elempack = 1;

#if __ALTIVEC__
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif // __ALTIVEC__

    if (out_elempack == 4)
    {
        // src = inch-outch
        // dst = 4-inch-outch/4
        {
            Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

            weight_data_tm.create(num_input, num_output / 4, (size_t)4u * 4, 4);

            for (int q = 0; q + 3 < num_output; q += 4)
            {
                float* g0 = weight_data_tm.row(q / 4);

                for (int p = 0; p < num_input; p++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        *g0++ = weight_data_r2.row(q + j)[p];
                    }
                }
            }
        }
    }
    else
    {
        weight_data_tm = weight_data;
    }

    if (opt.lightmode)
        weight_data.release();

    return 0;
}

int InnerProduct_powerpc::destroy_pipeline(const Option& opt)
{
    if (flatten)
    {
        flatten->destroy_pipeline(opt);
        delete flatten;
        flatten = 0;
    }

    return 0;
}

#if __ALTIVEC__
// scalar fused activation applied element-wise, mirroring activation_ss
static NCNN_FORCEINLINE v4f32 activation_ss_ps(v4f32 _v, int activation_type, const ncnn::Mat& activation_params)
{
    float v[4];
    v[0] = vec_extract(_v, 0);
    v[1] = vec_extract(_v, 1);
    v[2] = vec_extract(_v, 2);
    v[3] = vec_extract(_v, 3);

    v[0] = activation_ss(v[0], activation_type, activation_params);
    v[1] = activation_ss(v[1], activation_type, activation_params);
    v[2] = activation_ss(v[2], activation_type, activation_params);
    v[3] = activation_ss(v[3], activation_type, activation_params);

    return __ncnn_altivec_load(v);
}
#endif // __ALTIVEC__

int InnerProduct_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if NCNN_INT8
    if (opt.use_int8_inference && int8_scale_term)
    {
        return forward_int8_scalar(bottom_blob, top_blob, opt);
    }
#endif

    if (bottom_blob.elembits() == 16)
    {
        return InnerProduct::forward(bottom_blob, top_blob, opt);
    }

    const int num_input = weight_data_size / num_output;

    if (bottom_blob.dims == 2 && bottom_blob.w == num_input)
    {
        // gemm
        int h = bottom_blob.h;
        size_t elemsize = bottom_blob.elemsize;
        int elempack = bottom_blob.elempack;

        top_blob.create(num_output, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int num_output_elempack = 1;
#if __ALTIVEC__
        if (opt.use_packing_layout)
        {
            num_output_elempack = num_output % 4 == 0 ? 4 : 1;
        }
#endif

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int j = 0; j < h; j++)
        {
#if __ALTIVEC__
            if (elempack == 4 && num_output_elempack == 4)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const float* kptr = weight_data_tm.row(p);
                    const float* m = bottom_blob.row(j);

                    v4f32 _sum0 = (v4f32)vec_splats(0);
                    v4f32 _sum1 = (v4f32)vec_splats(0);
                    v4f32 _sum2 = (v4f32)vec_splats(0);
                    v4f32 _sum3 = (v4f32)vec_splats(0);

                    if (bias_term)
                    {
                        _sum0 = __ncnn_altivec_fill_f32(bias_data[p * 4 + 0]);
                        _sum1 = __ncnn_altivec_fill_f32(bias_data[p * 4 + 1]);
                        _sum2 = __ncnn_altivec_fill_f32(bias_data[p * 4 + 2]);
                        _sum3 = __ncnn_altivec_fill_f32(bias_data[p * 4 + 3]);
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        __builtin_prefetch(m + 16);
                        __builtin_prefetch(kptr + 16);
                        v4f32 _val = __ncnn_altivec_load(m);
                        v4f32 _w = __ncnn_altivec_load(kptr);
                        _sum0 = __ncnn_altivec_fmadd(_sum0, _val, vec_splat(_w, 0));
                        _sum1 = __ncnn_altivec_fmadd(_sum1, _val, vec_splat(_w, 1));
                        _sum2 = __ncnn_altivec_fmadd(_sum2, _val, vec_splat(_w, 2));
                        _sum3 = __ncnn_altivec_fmadd(_sum3, _val, vec_splat(_w, 3));

                        m += 4;
                        kptr += 4;
                    }

                    _sum0 = activation_ss_ps(_sum0, activation_type, activation_params);
                    _sum1 = activation_ss_ps(_sum1, activation_type, activation_params);
                    _sum2 = activation_ss_ps(_sum2, activation_type, activation_params);
                    _sum3 = activation_ss_ps(_sum3, activation_type, activation_params);

                    __ncnn_altivec_store(outptr, _sum0);
                    __ncnn_altivec_store(outptr + 4, _sum1);
                    __ncnn_altivec_store(outptr + 8, _sum2);
                    __ncnn_altivec_store(outptr + 12, _sum3);
                    outptr += 16;
                }
            }

            if (elempack == 1 && num_output_elempack == 4)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const float* kptr = weight_data_tm.row(p);
                    const float* m = bottom_blob.row(j);

                    v4f32 _sum0 = (v4f32)vec_splats(0);
                    v4f32 _sum1 = (v4f32)vec_splats(0);
                    v4f32 _sum2 = (v4f32)vec_splats(0);
                    v4f32 _sum3 = (v4f32)vec_splats(0);

                    if (bias_term)
                    {
                        _sum0 = __ncnn_altivec_load((const float*)bias_data + p * 4);
                    }

                    int i = 0;
                    for (; i + 3 < num_input; i += 4)
                    {
                        __builtin_prefetch(m + 16);
                        __builtin_prefetch(kptr + 64);
                        v4f32 _val = __ncnn_altivec_load(m);
                        v4f32 _w0 = __ncnn_altivec_load(kptr);
                        v4f32 _w1 = __ncnn_altivec_load(kptr + 4);
                        v4f32 _w2 = __ncnn_altivec_load(kptr + 8);
                        v4f32 _w3 = __ncnn_altivec_load(kptr + 12);
                        _sum0 = __ncnn_altivec_fmadd(_sum0, vec_splat(_val, 0), _w0);
                        _sum1 = __ncnn_altivec_fmadd(_sum1, vec_splat(_val, 1), _w1);
                        _sum2 = __ncnn_altivec_fmadd(_sum2, vec_splat(_val, 2), _w2);
                        _sum3 = __ncnn_altivec_fmadd(_sum3, vec_splat(_val, 3), _w3);

                        m += 4;
                        kptr += 16;
                    }
                    for (; i < num_input; i++)
                    {
                        v4f32 _val = __ncnn_altivec_fill_f32(m[0]);
                        v4f32 _w = __ncnn_altivec_load(kptr);
                        _sum0 = __ncnn_altivec_fmadd(_sum0, _val, _w);

                        m += 1;
                        kptr += 4;
                    }

                    _sum0 = vec_add(_sum0, _sum1);
                    _sum2 = vec_add(_sum2, _sum3);
                    _sum0 = vec_add(_sum0, _sum2);

                    _sum0 = activation_ss_ps(_sum0, activation_type, activation_params);

                    __ncnn_altivec_store(outptr, _sum0);
                    outptr += 4;
                }
            }

            if (elempack == 4 && num_output_elempack == 1)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output; p++)
                {
                    const float* kptr = (const float*)weight_data_tm + num_input * p;
                    const float* m = bottom_blob.row(j);

                    v4f32 _sum = (v4f32)vec_splats(0);

                    if (bias_term)
                    {
                        _sum = __ncnn_altivec_fill_f32(bias_data[p]);
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        __builtin_prefetch(m + 16);
                        __builtin_prefetch(kptr + 4);
                        v4f32 _val = __ncnn_altivec_load(m);
                        v4f32 _k = __ncnn_altivec_fill_f32(kptr[0]);
                        _sum = __ncnn_altivec_fmadd(_sum, _val, _k);

                        m += 4;
                        kptr += 1;
                    }

                    _sum = activation_ss_ps(_sum, activation_type, activation_params);

                    __ncnn_altivec_store(outptr, _sum);
                    outptr += 4;
                }
            }
#endif // __ALTIVEC__

            if (elempack == 1 && num_output_elempack == 1)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output; p++)
                {
                    const float* kptr = (const float*)weight_data_tm + num_input * p;
                    const float* m = bottom_blob.row(j);

                    float sum = 0.f;

                    if (bias_term)
                    {
                        sum = bias_data[p];
                    }

                    int i = 0;
#if __ALTIVEC__
                    v4f32 _sum = (v4f32)vec_splats(0);
                    for (; i + 3 < num_input; i += 4)
                    {
                        __builtin_prefetch(m + 16);
                        __builtin_prefetch(kptr + 16);
                        v4f32 _m = __ncnn_altivec_load(m);
                        v4f32 _w = __ncnn_altivec_load(kptr);
                        _sum = __ncnn_altivec_fmadd(_sum, _m, _w);

                        m += 4;
                        kptr += 4;
                    }
                    sum += __ncnn_altivec_reduce_add(_sum);
#endif // __ALTIVEC__
                    for (; i < num_input; i++)
                    {
                        sum += *m * *kptr;

                        m += 1;
                        kptr += 1;
                    }

                    sum = activation_ss(sum, activation_type, activation_params);

                    outptr[0] = sum;
                    outptr += 1;
                }
            }
        }

        return 0;
    }

    // flatten
    Mat bottom_blob_flattened = bottom_blob;
    if (bottom_blob.dims != 1)
    {
        Option opt_flatten = opt;
        opt_flatten.blob_allocator = opt.workspace_allocator;

        flatten->forward(bottom_blob, bottom_blob_flattened, opt_flatten);
        if (bottom_blob_flattened.empty())
            return -100;
    }

    size_t elemsize = bottom_blob_flattened.elemsize;
    int elempack = bottom_blob_flattened.elempack;

    int out_elempack = 1;
#if __ALTIVEC__
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif // __ALTIVEC__
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#if __ALTIVEC__
    if (out_elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            v4f32 _sum0 = (v4f32)vec_splats(0);
            v4f32 _sum1 = (v4f32)vec_splats(0);
            v4f32 _sum2 = (v4f32)vec_splats(0);
            v4f32 _sum3 = (v4f32)vec_splats(0);

            if (bias_term)
            {
                _sum0 = __ncnn_altivec_load((const float*)bias_data + p * 4);
            }

            const float* kptr = weight_data_tm.row(p);

            const float* sptr = bottom_blob_flattened;

            int i = 0;
            for (; i + 3 < num_input; i += 4)
            {
                __builtin_prefetch(sptr + 16);
                __builtin_prefetch(kptr + 64);
                v4f32 _val = __ncnn_altivec_load(sptr);
                v4f32 _w0 = __ncnn_altivec_load(kptr);
                v4f32 _w1 = __ncnn_altivec_load(kptr + 4);
                v4f32 _w2 = __ncnn_altivec_load(kptr + 8);
                v4f32 _w3 = __ncnn_altivec_load(kptr + 12);
                _sum0 = __ncnn_altivec_fmadd(_sum0, vec_splat(_val, 0), _w0);
                _sum1 = __ncnn_altivec_fmadd(_sum1, vec_splat(_val, 1), _w1);
                _sum2 = __ncnn_altivec_fmadd(_sum2, vec_splat(_val, 2), _w2);
                _sum3 = __ncnn_altivec_fmadd(_sum3, vec_splat(_val, 3), _w3);

                sptr += 4;
                kptr += 16;
            }
            for (; i < num_input; i++)
            {
                v4f32 _val = __ncnn_altivec_fill_f32(sptr[0]);
                v4f32 _w = __ncnn_altivec_load(kptr);
                _sum0 = __ncnn_altivec_fmadd(_sum0, _val, _w);

                sptr += 1;
                kptr += 4;
            }

            _sum0 = vec_add(_sum0, _sum1);
            _sum2 = vec_add(_sum2, _sum3);
            _sum0 = vec_add(_sum0, _sum2);

            _sum0 = activation_ss_ps(_sum0, activation_type, activation_params);

            float* outptr = top_blob;
            __ncnn_altivec_store(outptr + p * 4, _sum0);
        }
    }
#endif // __ALTIVEC__

    if (out_elempack == 1)
    {
        int nn_num_output = num_output / 4;
        int remain_num_output_start = nn_num_output * 4;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_num_output; pp++)
        {
            int p = pp * 4;

            float sum0 = 0.f;
            float sum1 = 0.f;
            float sum2 = 0.f;
            float sum3 = 0.f;

            if (bias_term)
            {
                sum0 = bias_data[p];
                sum1 = bias_data[p + 1];
                sum2 = bias_data[p + 2];
                sum3 = bias_data[p + 3];
            }

            const float* w0 = (const float*)weight_data_tm + num_input * p;
            const float* w1 = (const float*)weight_data_tm + num_input * (p + 1);
            const float* w2 = (const float*)weight_data_tm + num_input * (p + 2);
            const float* w3 = (const float*)weight_data_tm + num_input * (p + 3);

            const float* m = bottom_blob_flattened;

            int i = 0;
#if __ALTIVEC__
            v4f32 _sum0 = (v4f32)vec_splats(0);
            v4f32 _sum1 = (v4f32)vec_splats(0);
            v4f32 _sum2 = (v4f32)vec_splats(0);
            v4f32 _sum3 = (v4f32)vec_splats(0);
            for (; i + 3 < num_input; i += 4)
            {
                __builtin_prefetch(m + 16);
                __builtin_prefetch(w0 + 16);
                __builtin_prefetch(w1 + 16);
                __builtin_prefetch(w2 + 16);
                __builtin_prefetch(w3 + 16);
                v4f32 _m = __ncnn_altivec_load(m);
                v4f32 _w0 = __ncnn_altivec_load(w0);
                v4f32 _w1 = __ncnn_altivec_load(w1);
                v4f32 _w2 = __ncnn_altivec_load(w2);
                v4f32 _w3 = __ncnn_altivec_load(w3);
                _sum0 = __ncnn_altivec_fmadd(_sum0, _m, _w0);
                _sum1 = __ncnn_altivec_fmadd(_sum1, _m, _w1);
                _sum2 = __ncnn_altivec_fmadd(_sum2, _m, _w2);
                _sum3 = __ncnn_altivec_fmadd(_sum3, _m, _w3);

                m += 4;
                w0 += 4;
                w1 += 4;
                w2 += 4;
                w3 += 4;
            }
#endif // __ALTIVEC__
            for (; i < num_input; i++)
            {
                sum0 += *m * *w0;
                sum1 += *m * *w1;
                sum2 += *m * *w2;
                sum3 += *m * *w3;

                m++;
                w0++;
                w1++;
                w2++;
                w3++;
            }

#if __ALTIVEC__
            sum0 += __ncnn_altivec_reduce_add(_sum0);
            sum1 += __ncnn_altivec_reduce_add(_sum1);
            sum2 += __ncnn_altivec_reduce_add(_sum2);
            sum3 += __ncnn_altivec_reduce_add(_sum3);
#endif // __ALTIVEC__

            sum0 = activation_ss(sum0, activation_type, activation_params);
            sum1 = activation_ss(sum1, activation_type, activation_params);
            sum2 = activation_ss(sum2, activation_type, activation_params);
            sum3 = activation_ss(sum3, activation_type, activation_params);

            top_blob[p] = sum0;
            top_blob[p + 1] = sum1;
            top_blob[p + 2] = sum2;
            top_blob[p + 3] = sum3;
        }

        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = remain_num_output_start; p < num_output; p++)
        {
            float sum = 0.f;

            if (bias_term)
                sum = bias_data[p];

            const float* w = (const float*)weight_data_tm + num_input * p;

            const float* m = bottom_blob_flattened;

            int i = 0;
#if __ALTIVEC__
            v4f32 _sum0 = (v4f32)vec_splats(0);
            for (; i + 3 < num_input; i += 4)
            {
                __builtin_prefetch(m + 16);
                __builtin_prefetch(w + 16);
                v4f32 _m = __ncnn_altivec_load(m);
                v4f32 _w = __ncnn_altivec_load(w);
                _sum0 = __ncnn_altivec_fmadd(_sum0, _m, _w);

                m += 4;
                w += 4;
            }
            sum += __ncnn_altivec_reduce_add(_sum0);
#endif // __ALTIVEC__
            for (; i < num_input; i++)
            {
                sum += *m * *w;

                m++;
                w++;
            }

            sum = activation_ss(sum, activation_type, activation_params);

            top_blob[p] = sum;
        }
    }

    return 0;
}

#if NCNN_INT8
// powerpc has no int8 kernel yet; run the generic scalar int8 compute on
// unpacked blobs and apply the powerpc fp32 output packing afterwards
int InnerProduct_powerpc::forward_int8_scalar(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int num_input = weight_data_size / num_output;

    int elembits = bottom_blob.elembits();

    Mat bottom_blob_int8 = bottom_blob;
    if (elembits != 8)
    {
        Option opt_q = opt;
        opt_q.blob_allocator = opt.workspace_allocator;
        quantize_to_int8(bottom_blob, bottom_blob_int8, bottom_blob_int8_scales, opt_q);
        if (bottom_blob_int8.empty())
            return -100;
    }

    Mat top_blob_unpacked;
    int ret = 0;

    if (bottom_blob_int8.dims == 2 && bottom_blob_int8.w == num_input)
    {
        // gemm; dims2 packing interleaves rows, unpack before the scalar kernel
        Mat bottom_blob_int8_unpacked;
        Option opt_unpack = opt;
        opt_unpack.blob_allocator = opt.workspace_allocator;
        convert_packing(bottom_blob_int8, bottom_blob_int8_unpacked, 1, opt_unpack);
        if (bottom_blob_int8_unpacked.empty())
            return -100;

        int h = bottom_blob_int8_unpacked.h;

        ret = InnerProduct::forward_int8(bottom_blob_int8_unpacked, top_blob_unpacked, opt);
        if (ret != 0)
            return ret;

        int out_elempack = 1;
#if __ALTIVEC__
        if (opt.use_packing_layout)
        {
            out_elempack = h % 4 == 0 ? 4 : 1;
        }
#endif

        if (out_elempack == 1)
        {
            top_blob = top_blob_unpacked;
            return 0;
        }

        convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);
        return 0;
    }

    // flatten; unpack first since the generic kernel is pack-agnostic
    Mat bottom_blob_int8_unpacked;
    if (bottom_blob_int8.elempack != 1)
    {
        Option opt_unpack = opt;
        opt_unpack.blob_allocator = opt.workspace_allocator;
        convert_packing(bottom_blob_int8, bottom_blob_int8_unpacked, 1, opt_unpack);
        if (bottom_blob_int8_unpacked.empty())
            return -100;
    }
    else
    {
        bottom_blob_int8_unpacked = bottom_blob_int8;
    }

    Mat bottom_blob_int8_flattened = bottom_blob_int8_unpacked;
    if (bottom_blob_int8_unpacked.dims != 1)
    {
        Option opt_flatten = opt;
        opt_flatten.blob_allocator = opt.workspace_allocator;

        flatten->forward(bottom_blob_int8_unpacked, bottom_blob_int8_flattened, opt_flatten);
        if (bottom_blob_int8_flattened.empty())
            return -100;
    }

    ret = InnerProduct::forward_int8(bottom_blob_int8_flattened, top_blob_unpacked, opt);
    if (ret != 0)
        return ret;

    int out_elempack = 1;
#if __ALTIVEC__
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif

    if (out_elempack == 1)
    {
        top_blob = top_blob_unpacked;
        return 0;
    }

    convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);

    return 0;
}
#endif // NCNN_INT8

} // namespace ncnn
