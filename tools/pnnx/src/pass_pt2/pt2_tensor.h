// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_PT2_TENSOR_H
#define PNNX_PT2_TENSOR_H

#include "ir.h"
#include "json.h"
#include "storezip.h"

namespace pnnx {

struct Pt2TensorMeta
{
    int type;
    std::vector<int> shape;
    std::vector<std::string> shape_symbols;
    std::vector<int64_t> strides;
    int64_t storage_offset;
    std::string device;
    int layout;
};

struct Pt2TensorStorage
{
    Pt2TensorStorage()
        : remaining(0)
    {
    }

    size_t remaining;
    std::vector<char> data;
};

int pt2_tensor_type(int serde_type);
int pt2_scalar_type(int type);

int parse_pt2_tensor_meta(const JsonValue& value, Pt2TensorMeta& meta, bool allow_symbolic);

int load_pt2_tensor(StoreZipReader& zip, const std::string& directory, const JsonValue& payload, Attribute& attr, Pt2TensorMeta& meta,
                    std::map<std::string, Pt2TensorStorage>& storage_cache);

} // namespace pnnx

#endif // PNNX_PT2_TENSOR_H
