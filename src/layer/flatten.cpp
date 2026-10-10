// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "flatten.h"

#include <string.h>

namespace ncnn {

Flatten::Flatten()
{
    one_blob_only = true;
    support_inplace = false;
}

int Flatten::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int size = w * h * d;

    top_blob.create(size * channels, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const unsigned char* ptr = bottom_blob.channel(q);
        unsigned char* outptr = (unsigned char*)top_blob + size * elemsize * q;

        memcpy(outptr, ptr, size * elemsize);
    }

    return 0;
}

} // namespace ncnn
