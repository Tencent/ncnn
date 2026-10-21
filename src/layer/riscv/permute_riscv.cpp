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
    /* do not advertise fp16 storage: this layer only does a fast row copy for
     * order_type==2 and delegates every other order to the parent
     * implementation (which treats the data as float*). Letting ncnn do the
     * fp16<->fp32 conversion uniformly outside the layer keeps every
     * model/every order correct (handling only order 2 once left int8/2-byte
     * models with empty output). */
}

int Permute_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const size_t elemsize = bottom_blob.elemsize;
    const int elempack = bottom_blob.elempack;
    const int dims = bottom_blob.dims;

    // order_type == 2 (w c h): the output is [w, c, h]. Note this is **not** a
    // pure dimension relabeling -- the generic implementation does a real
    // transpose of "channel i" and "row q": out(q, i, :) = in(i, q, :).
    // (It was first mistaken for a relabeling and memcpy'd directly; the
    // three-legged gate caught it on the spot.)
    // Here the copy is vectorized per row: each row has w contiguous elements,
    // source row stride = w*h, destination row stride = w*channels.
    if (dims == 3 && order_type == 2)
    {
        top_blob.create(w, channels, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const size_t row_bytes = (size_t)w * elemsize;
        const unsigned char* src = (const unsigned char*)bottom_blob.data;
        unsigned char* dst = (unsigned char*)top_blob.data;
        /* Mat pads every channel plane up to 16 bytes, so a plane stride is
         * cstep, not w * h. With w=1, h=2, c=7 in fp32 the input plane is only
         * 8 bytes; assuming tight packing then starts reading the second input
         * channel 8 bytes early and writing the second output channel 4 bytes
         * early, which corrupts the result for every misaligned shape. */
        const size_t src_cstep = bottom_blob.cstep;
        const size_t dst_cstep = top_blob.cstep;

        /* each row has w contiguous elements (commonly 128~256B at prefill), but
                         * there are many rows (h*channels reaches into the thousands), so the
                         * per-row memcpy call overhead becomes the bottleneck (measured only
                         * 1.75 GB/s at 115 tokens, against a 15 GB/s streaming wall). Switch to
                         * an explicit RVV copy (vl follows the actual VLMAX, VLEN-independent),
                         * removing the per-call memcpy overhead. */
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < h; q++)
        {
            unsigned char* drow = dst + (size_t)q * dst_cstep * elemsize;
            for (int i = 0; i < channels; i++)
            {
                const unsigned char* srow = src + (size_t)i * src_cstep * elemsize
                                            + (size_t)q * row_bytes;
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

    // all other orders: hand everything to the parent (fp16 storage is not advertised, so the parent always receives fp32/pack1)
    return Permute::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
