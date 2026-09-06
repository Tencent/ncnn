// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// tensor_meta materialization shared by the pt2 loader translation units.
// split out of load_exportedprogram.cpp: decodes a serialized tensor_meta
// (sizes / strides / storage_offset) out of a raw storage byte buffer into an
// Attribute. used by both the 2.8+ archive path (raw byte records) and the
// legacy (<2.8) path (raw storage shards from a pickled state dict).

#ifndef PNNX_LOAD_PT2_TENSOR_H
#define PNNX_LOAD_PT2_TENSOR_H

#include <string>
#include <vector>

namespace pnnx {

class Attribute;
class JsonValue;
class StoreZipReader;

// materialize the logical row-major tensor described by tensor_meta out of raw
// storage bytes into a (transposed/sliced/shared-storage views handled here)
void load_tensor_from_raw(std::vector<char> raw, const JsonValue& meta, Attribute& a);

// read one weight/constant record (raw storage bytes) from the zip into an
// Attribute. returns 0 on success, -1 when the referenced payload record is
// missing (caller can reject an incomplete archive instead of installing an
// empty attribute).
int load_tensor_data(StoreZipReader& zip, const std::vector<std::string>& names,
                     const std::string& dir, const std::string& path_name,
                     const JsonValue& meta, Attribute& a);

} // namespace pnnx

#endif // PNNX_LOAD_PT2_TENSOR_H
