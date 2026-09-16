// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// serde dtype / memory-format / target normalization helpers used across the
// pt2 loader (split out of load_exportedprogram.cpp).

#include "load_pt2_serde.h"

#include <stdio.h>
#include <string.h>

#include "pnnx_json.h"

namespace pnnx {

// torch serde ScalarType -> pnnx type
//   1=uint8 2=int8 28=uint16 3=int16 4=int32 5=int64 6=float16 7=float32 8=float64
//   9=complex32 10=complex64 11=complex128 12=bool 13=bfloat16
int serde_dtype_to_pnnx_type(int64_t dtype)
{
    switch (dtype)
    {
    case 1:
        return 8; // uint8 -> u8
    case 2:
        return 7; // int8 -> i8
    case 3:
        return 6; // int16 -> i16
    case 4:
        return 4; // int32 -> i32
    case 5:
        return 5; // int64 -> i64
    case 6:
        return 3; // float16 -> f16
    case 7:
        return 1; // float32 -> f32
    case 8:
        return 2; // float64 -> f64
    case 9:
        return 12; // complex32 -> c32
    case 10:
        return 10; // complex64 -> c64
    case 11:
        return 11; // complex128 -> c128
    case 12:
        return 9; // bool
    case 13:
        return 13; // bfloat16
    default:
        fprintf(stderr, "unsupported serde dtype %lld\n", (long long)dtype);
        return 0;
    }
}

// torch serde ScalarType -> pnnx dtype input enum (prim::Constant value)
//   pnnx dtype enum (see pass_level2/Tensor_to.cpp): 0=uint8 1=int8 2=short 3=int
//   4=long 5=half 6=float 7=double 8=complex32 9=complex64 10=complex128 11=bool 15=bfloat16
int serde_dtype_to_pnnx_dtype_value(int64_t dtype)
{
    switch (dtype)
    {
    case 1:
        return 0; // uint8
    case 2:
        return 1; // int8
    case 3:
        return 2; // int16 -> short
    case 4:
        return 3; // int32 -> int
    case 5:
        return 4; // int64 -> long
    case 6:
        return 5; // float16 -> half
    case 7:
        return 6; // float32 -> float
    case 8:
        return 7; // float64 -> double
    case 9:
        return 8; // complex32
    case 10:
        return 9; // complex64
    case 11:
        return 10; // complex128
    case 12:
        return 11; // bool
    case 13:
        return 15; // bfloat16
    default:
        fprintf(stderr, "unsupported serde dtype %lld\n", (long long)dtype);
        return -1;
    }
}

// scalar-dtype handling helpers: reject a serde ScalarType with no pnnx
// representation instead of silently emitting a bogus 0/-1 constant (mirrors
// the tensor/weight dtype rejection used when materializing attributes)
bool scalar_dtype_to_pnnx_type(int64_t serde_dtype, int& pnnx_type)
{
    pnnx_type = serde_dtype_to_pnnx_type(serde_dtype);
    if (pnnx_type <= 0)
    {
        fprintf(stderr, "unsupported scalar dtype %lld\n", (long long)serde_dtype);
        return false;
    }
    return true;
}

bool scalar_dtype_to_pnnx_dtype_value(int64_t serde_dtype, int& dtype_value)
{
    dtype_value = serde_dtype_to_pnnx_dtype_value(serde_dtype);
    if (dtype_value < 0)
    {
        fprintf(stderr, "unsupported scalar dtype %lld\n", (long long)serde_dtype);
        return false;
    }
    return true;
}

// torch serde MemoryFormat -> pnnx memory_format enum
//   serde: 0=none 1=contiguous 2=channels_last 3=channels_last_3d 4=preserve
//   pnnx enum (see pass_level2/Tensor_to.cpp): 0=contiguous 1=preserve 2=channels_last
int serde_memory_format_to_pnnx(int64_t mf)
{
    switch (mf)
    {
    case 1:
        return 0; // contiguous
    case 2:
        return 2; // channels_last
    case 4:
        return 1; // preserve
    case 0:       // none -> preserve
    case 3:       // channels_last_3d has no counterpart, fall back to preserve
    default:
        return 1;
    }
}

size_t type_to_elemsize(int type)
{
    if (type == 1) return 4;
    if (type == 2) return 8;
    if (type == 3) return 2;
    if (type == 4) return 4;
    if (type == 5) return 8;
    if (type == 6) return 2;
    if (type == 7) return 1;
    if (type == 8) return 1;
    if (type == 9) return 1;
    if (type == 10) return 8;
    if (type == 11) return 16;
    if (type == 12) return 4;
    if (type == 13) return 2;
    return 0;
}

// torch.ops.aten.conv2d.default -> aten::conv2d
std::string normalize_target(const std::string& target)
{
    std::string t = target;

    const char* prefix = "torch.ops.";
    if (t.compare(0, strlen(prefix), prefix) == 0)
        t = t.substr(strlen(prefix));

    size_t dot = t.rfind('.');
    if (dot != std::string::npos)
    {
        // keep the non-default overloads only for the arange family (pt2
        // arange.start_step / arange.start variants have different arguments
        // than the overload-less aten::arange and cannot be mixed), return
        // them as "aten::arange.start_step" (overload separated by a dot);
        // arange.default / arange.end are normalized to the suffix-less
        // aten::arange (matches the pass_level2 torch_arange patterns)
        if (t.compare(0, 12, "aten.arange.") == 0)
        {
            std::string body = t.substr(0, dot); // "aten.arange"
            std::string overload = t.substr(dot + 1);
            if (overload == "default" || overload == "end")
            {
                t = body;
            }
            else
            {
                std::string rb;
                for (size_t i = 0; i < body.size(); i++)
                {
                    if (body[i] == '.')
                        rb += "::";
                    else
                        rb += body[i];
                }
                return rb + "." + overload;
            }
        }
        else
        {
            t = t.substr(0, dot);
        }
    }

    std::string r;
    for (size_t i = 0; i < t.size(); i++)
    {
        if (t[i] == '.')
            r += "::";
        else
            r += t[i];
    }

    return r;
}

// read the sizes array (each element is {"as_int": n} or {"as_sym_int": ...})
void read_sizes(const JsonValue& meta, std::vector<int>& shape)
{
    shape.clear();

    if (!meta.is_object())
        return;

    const JsonValue& sizes = meta["sizes"];
    for (size_t i = 0; i < sizes.size(); i++)
    {
        const JsonValue& s = sizes[i];
        if (s.has("as_int"))
            shape.push_back((int)s["as_int"].as_int());
        else
            shape.push_back(-1); // symbolic dimension, unresolved
    }
}

int read_dtype(const JsonValue& meta)
{
    if (!meta.is_object() || !meta.has("dtype"))
        return 0;

    return serde_dtype_to_pnnx_type(meta["dtype"].as_int());
}

} // namespace pnnx
