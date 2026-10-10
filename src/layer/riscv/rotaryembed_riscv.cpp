// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "rotaryembed_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

namespace ncnn {

RotaryEmbed_riscv::RotaryEmbed_riscv()
{
}

int RotaryEmbed_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& cos_cache = bottom_blobs[1];
    const Mat& sin_cache = bottom_blobs[2];

    const int embed_dim = bottom_blob.w;
    /* input_hc_swapped=1：输入是 [w, c, h]（吸收前面的 Permute(order_type=2)） */
    const int seqlen = input_hc_swapped ? bottom_blob.c : bottom_blob.h;
    const int num_heads = input_hc_swapped ? bottom_blob.h : bottom_blob.c;

#if !__riscv_vector
    return RotaryEmbed::forward(bottom_blobs, top_blobs, opt);
#else
    // 只接管 fp32/pack1 常态；其余交给通用实现
    if (bottom_blob.elembits() != 32 || bottom_blob.elempack != 1 || embed_dim % 2 != 0)
        return RotaryEmbed::forward(bottom_blobs, top_blobs, opt);

    Mat& top_blob = top_blobs[0];
    if (input_hc_swapped)
    {
        /* 融合模式下输入是 [w, c, h]，输出必须回到标准布局 [w, h, c]
         * （原来的 Permute(order_type=2) 干的活由本层一并完成）。 */
        top_blob.create(embed_dim, seqlen, num_heads, bottom_blob.elemsize, opt.blob_allocator);
    }
    else
    {
        top_blob.create_like(bottom_blob, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    const int half = embed_dim / 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_heads; q++)
    {
        const Mat head = bottom_blob.channel(q);
        Mat out_head = top_blob.channel(q);

        for (int i = 0; i < seqlen; i++)
        {
            const float* ptr = input_hc_swapped ? bottom_blob.channel(i).row(q) : head.row(i);
            const float* cos_ptr = cos_cache.row(i);
            const float* sin_ptr = sin_cache.row(i);
            float* outptr = out_head.row(i);

            int j = 0;
            if (interleaved)
            {
                /* 交错布局：x0/x1 各取偶数位/奇数位（跨步 8 字节） */
                while (j < half)
                {
                    const size_t vl = __riscv_vsetvl_e32m2(half - j);
                    vfloat32m2_t x0 = __riscv_vlse32_v_f32m2(ptr + (size_t)j * 2, 8, vl);
                    vfloat32m2_t x1 = __riscv_vlse32_v_f32m2(ptr + (size_t)j * 2 + 1, 8, vl);
                    vfloat32m2_t c = __riscv_vle32_v_f32m2(cos_ptr + j, vl);
                    vfloat32m2_t s = __riscv_vle32_v_f32m2(sin_ptr + j, vl);
                    vfloat32m2_t o0 = __riscv_vfsub_vv_f32m2(__riscv_vfmul_vv_f32m2(x0, c, vl), __riscv_vfmul_vv_f32m2(x1, s, vl), vl);
                    vfloat32m2_t o1 = __riscv_vfadd_vv_f32m2(__riscv_vfmul_vv_f32m2(x0, s, vl), __riscv_vfmul_vv_f32m2(x1, c, vl), vl);
                    __riscv_vsse32_v_f32m2(outptr + (size_t)j * 2, 8, o0, vl);
                    __riscv_vsse32_v_f32m2(outptr + (size_t)j * 2 + 1, 8, o1, vl);
                    j += (int)vl;
                }
            }
            else
            {
                /* 半分解（本条模型走这条）：前一半与后一半各连续，纯连续访存，最好向量化 */
                const float* ptr0 = ptr;
                const float* ptr1 = ptr + half;
                float* outptr0 = outptr;
                float* outptr1 = outptr + half;
                while (j < half)
                {
                    const size_t vl = __riscv_vsetvl_e32m4(half - j);
                    vfloat32m4_t x0 = __riscv_vle32_v_f32m4(ptr0 + j, vl);
                    vfloat32m4_t x1 = __riscv_vle32_v_f32m4(ptr1 + j, vl);
                    vfloat32m4_t c = __riscv_vle32_v_f32m4(cos_ptr + j, vl);
                    vfloat32m4_t s = __riscv_vle32_v_f32m4(sin_ptr + j, vl);
                    vfloat32m4_t o0 = __riscv_vfsub_vv_f32m4(__riscv_vfmul_vv_f32m4(x0, c, vl), __riscv_vfmul_vv_f32m4(x1, s, vl), vl);
                    vfloat32m4_t o1 = __riscv_vfadd_vv_f32m4(__riscv_vfmul_vv_f32m4(x0, s, vl), __riscv_vfmul_vv_f32m4(x1, c, vl), vl);
                    __riscv_vse32_v_f32m4(outptr0 + j, o0, vl);
                    __riscv_vse32_v_f32m4(outptr1 + j, o1, vl);
                    j += (int)vl;
                }
            }
        }
    }
    return 0;
#endif // __riscv_vector
}

} // namespace ncnn
