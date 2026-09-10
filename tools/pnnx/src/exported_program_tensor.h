// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_EXPORTED_PROGRAM_TENSOR_H
#define PNNX_EXPORTED_PROGRAM_TENSOR_H

#include "exported_program_schema.h"
#include "pt2_archive.h"

#include <string>
#include <vector>

namespace pnnx {

struct MaterializedExportedTensor
{
    MaterializedExportedTensor();

    int pnnx_type;
    std::vector<int> shape;
    std::vector<char> data;
};

int exported_tensor_dtype_to_pnnx_type(int64_t dtype);

int materialize_exported_tensor(const ExportedTensorMeta& meta,
                                const std::vector<char>& storage,
                                Pt2ByteOrder byte_order,
                                MaterializedExportedTensor& tensor,
                                std::string& error);

} // namespace pnnx

#endif // PNNX_EXPORTED_PROGRAM_TENSOR_H
