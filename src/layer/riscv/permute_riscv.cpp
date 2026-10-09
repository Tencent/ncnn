// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "permute_riscv.h"

#include <string.h>

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

Permute_riscv::Permute_riscv()
{
    /* 不声明 fp16 storage：本层只对 order_type==2 做快速行拷贝，其余 order 一律委托父类
     * 实现（父类按 float* 处理）。让 ncnn 统一在层外做 fp16<->fp32 转换，
     * 可以保证所有模型/所有 order 都正确（曾因只处理 order 2 而让 int8/2B 模型输出为空）。 */
}

int Permute_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const size_t elemsize = bottom_blob.elemsize;
    const int elempack = bottom_blob.elempack;
    const int dims = bottom_blob.dims;

    // order_type == 2（w c h）：输出 [w, c, h]。注意这**不是**纯维度重标签——
    // 通用实现把"通道 i"与"行 q"做了真正的转置：out(q, i, :) = in(i, q, :)。
    // （最初误当成重标签直接 memcpy，被三腿闸门当场抓住。）
    // 这里按行做向量化拷贝：每行 w 个元素连续，源行距 = w*h，目标行距 = w*channels。
    if (dims == 3 && order_type == 2)
    {
        top_blob.create(w, channels, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const size_t row_bytes = (size_t)w * elemsize;
        const unsigned char* src = (const unsigned char*)bottom_blob.data;
        unsigned char* dst = (unsigned char*)top_blob.data;

        /* 每行 w 个元素连续（prefill 常见 128~256B），但行数很多（h*channels 达上千），
                                         * 逐行 memcpy 的调用开销就成了瓶颈（实测 115 token 时仅 1.75 GB/s，而流式墙是 15 GB/s）。
                                         * 改用显式 RVV 拷贝（vl 按实际 VLMAX 走，与 VLEN 无关），去掉每次调用的 memcpy 开销。 */
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < h; q++)
        {
            unsigned char* drow = dst + (size_t)q * channels * row_bytes;
            for (int i = 0; i < channels; i++)
            {
                const unsigned char* srow = src + ((size_t)i * h + q) * row_bytes;
#if __riscv_vector
                size_t off = 0;
                while (off < row_bytes)
                {
                    const size_t vl = __riscv_vsetvl_e8m8(row_bytes - off);
                    __riscv_vse8_v_u8m8(drow + (size_t)i * row_bytes + off,
                                        __riscv_vle8_v_u8m8(srow + off, vl), vl);
                    off += vl;
                }
#else
                memcpy(drow + (size_t)i * row_bytes, srow, row_bytes);
#endif
            }
        }
        return 0;
    }

    if (dims == 2 && order_type == 0)
    {
        top_blob = bottom_blob;
        return 0;
    }

    // 其余 order：全部交给父类（未声明 fp16 storage，父类拿到的必然是 fp32/pack1）
    return Permute::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
