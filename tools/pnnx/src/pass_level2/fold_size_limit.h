// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_PASS_LEVEL2_FOLD_SIZE_LIMIT_H
#define PNNX_PASS_LEVEL2_FOLD_SIZE_LIMIT_H

#include <stddef.h>
#include <vector>

namespace pnnx {

// pnnx tensor type -> element size in bytes. `type` is a pnnx dtype (see
// ir.h): 1=f32 2=f64 3=f16 4=i32 5=i64 6=i16 7=i8 8=u8 9=bool 10=c64 11=c128
// 12=c32 13=bf16; an unknown type materializes as f32.
inline size_t fold_elemsize(int type)
{
    size_t es = 4;
    if (type == 2 || type == 5) es = 8;               // f64/i64
    if (type == 3 || type == 6 || type == 13) es = 2; // f16/i16/bf16
    if (type == 7 || type == 8 || type == 9) es = 1;  // i8/u8/bool
    if (type == 10) es = 8;                           // complex64 (2 x f32)
    if (type == 11) es = 16;                          // complex128 (2 x f64)
    return es;
}

// A constant fold materializes its payload inside the converter, so a valid but
// huge static shape must be declined while matching rather than truncated into
// an empty payload with a positive declared shape (which only moves the failure
// into the generated model, where mapping the missing bytes raises) or allowed
// to allocate gigabytes and OOM the process. The limit is the same 1 GiB byte
// cap the ncnn new_empty fold applies, and the element size is resolved by
// fold_elemsize() so the cap and the allocation cannot disagree.
inline bool fold_size_within_limit(const std::vector<int>& shape, int type)
{
    const size_t es = fold_elemsize(type);

    size_t count = 1;
    for (size_t i = 0; i < shape.size(); i++)
    {
        const int s = shape[i];
        if (s < 0)
            return false;
        if (s == 0)
            return true; // an empty constant allocates nothing
        if (count > (size_t)-1 / (size_t)s)
            return false;
        count *= (size_t)s;
    }

    return count <= (size_t)0x40000000 / es;
}

} // namespace pnnx

#endif // PNNX_PASS_LEVEL2_FOLD_SIZE_LIMIT_H
