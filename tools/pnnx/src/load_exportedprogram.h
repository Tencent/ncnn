// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_LOAD_EXPORTEDPROGRAM_H
#define PNNX_LOAD_EXPORTEDPROGRAM_H

#include "ir.h"

namespace pnnx {

// load torch dynamo exported program (.pt2, PT2 Archive Spec, torch >= 2.7)
// and convert it to the pnnx graph. No third-party library is involved:
// the archive is read with StoreZipReader and the serialized graph is parsed
// with the built-in JsonParser.
int load_exportedprogram(const std::string& pt2path, Graph& g,
                         const std::vector<std::vector<int64_t> >& input_shapes,
                         const std::vector<std::string>& input_types);

} // namespace pnnx

#endif // PNNX_LOAD_EXPORTEDPROGRAM_H
