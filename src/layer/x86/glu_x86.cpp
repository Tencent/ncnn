// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "glu_x86.h"

#include "x86_activation.h"

#include "cpu.h"

namespace ncnn {

#include "glu_fp32.h"

GLU_x86::GLU_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

int GLU_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int dims = bottom_blob.dims;
    const int positive_axis = dims == 1 ? 0 : axis < 0 ? dims + axis : axis;
    if (positive_axis < 0 || positive_axis >= dims)
        return -100;

    const int elempack = bottom_blob.elempack;

    int outw = bottom_blob.w;
    int outh = bottom_blob.h;
    int outd = bottom_blob.d;
    int outc = bottom_blob.c;
    int out_elempack = elempack;

    if (positive_axis == 0)
    {
        // keep out_elempack no larger than elempack, so that values and gates never straddle input packs
        const int out_size = (dims == 1 ? outw : dims == 2 ? outh : outc) * elempack / 2;
        while (out_size % out_elempack != 0)
            out_elempack = out_elempack == 4 ? 1 : out_elempack / 2;

        if (dims == 1)
            outw = out_size / out_elempack;
        else if (dims == 2)
            outh = out_size / out_elempack;
        else
            outc = out_size / out_elempack;
    }
    else if (positive_axis == dims - 1)
    {
        outw /= 2;
    }
    else if (positive_axis == dims - 2)
    {
        outh /= 2;
    }
    else
    {
        // dims == 4 && positive_axis == 1
        outd /= 2;
    }

    const size_t out_elemsize = bottom_blob.elemsize / elempack * out_elempack;

    if (dims == 1)
        top_blob.create(outw, out_elemsize, out_elempack, opt.blob_allocator);
    else if (dims == 2)
        top_blob.create(outw, outh, out_elemsize, out_elempack, opt.blob_allocator);
    else if (dims == 3)
        top_blob.create(outw, outh, outc, out_elemsize, out_elempack, opt.blob_allocator);
    else
        top_blob.create(outw, outh, outd, outc, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    glu_fp32(bottom_blob, top_blob, positive_axis, opt);

    return 0;
}

} // namespace ncnn
