// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// serde / tensor_meta helpers shared by the pt2 loader translation units.
// split out of load_exportedprogram.cpp so the dtype/type mapping layer can be
// reused without dragging in the whole loader.

#ifndef PNNX_LOAD_PT2_SERDE_H
#define PNNX_LOAD_PT2_SERDE_H

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

namespace pnnx {

class JsonValue;

// torch serde ScalarType -> pnnx tensor type / dtype enum; returns 0 / -1 and
// prints on unsupported dtypes
int serde_dtype_to_pnnx_type(int64_t dtype);
int serde_dtype_to_pnnx_dtype_value(int64_t dtype);

// scalar-dtype handling: reject a serde ScalarType with no pnnx representation
bool scalar_dtype_to_pnnx_type(int64_t serde_dtype, int& pnnx_type);
bool scalar_dtype_to_pnnx_dtype_value(int64_t serde_dtype, int& dtype_value);

// torch serde MemoryFormat -> pnnx memory_format enum
int serde_memory_format_to_pnnx(int64_t mf);

// pnnx tensor type -> element size in bytes (0 when unknown)
size_t type_to_elemsize(int type);

// torch.ops.aten.conv2d.default -> aten::conv2d (keeps arange overloads)
std::string normalize_target(const std::string& target);

// read the sizes array of a tensor_meta (symbolic dims become -1)
void read_sizes(const JsonValue& meta, std::vector<int>& shape);

// tensor_meta dtype -> pnnx type (0 when absent/unsupported)
int read_dtype(const JsonValue& meta);

} // namespace pnnx

#endif // PNNX_LOAD_PT2_SERDE_H
