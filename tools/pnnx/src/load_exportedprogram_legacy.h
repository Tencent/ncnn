// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// legacy (<2.8) torch.export container support.
//
// torch.export.save switched to the pt2 archive layout (models/model.json +
// *config.json weight records + raw byte shards) in torch 2.8. older releases
// (2.5..2.7 here) save a flat archive instead:
//
//   serialized_exported_program.json   graph JSON (same graph_module schema,
//                                      the loader's existing json path parses
//                                      it unchanged)
//   serialized_state_dict.pt           weights: a nested torch.save zip whose
//   serialized_constants.pt            archive/data.pkl pickle holds the state
//                                      dict and whose storage shards live in
//                                      archive/data/<n>
//   version
//
// this file implements the pickled side only: it decodes the two .pt records
// into the same fqn -> (payload, tensor_meta) maps the archive path builds from
// its *config.json records, plus the raw storage bytes per fqn. the caller
// (load_exportedprogram.cpp) then materializes attributes through the shared
// load_tensor_from_raw() path, so tensor_meta views/offsets are handled exactly
// like the 2.8+ format.

#ifndef PNNX_LOAD_EXPORTEDPROGRAM_LEGACY_H
#define PNNX_LOAD_EXPORTEDPROGRAM_LEGACY_H

#include <map>
#include <string>
#include <vector>

#include "pnnx_json.h"

namespace pnnx {

class StoreZipReader;

// decode the legacy pickled weights/constants into the same
// fqn -> (payload_name, tensor_meta) maps the archive path builds from its
// config json, and fill legacy_raw[fqn] with the raw storage bytes for each
// parameter/buffer/tensor_constant in the signature. tensor_meta is taken from
// graph.tensor_values (keyed by the placeholder arg name), matching the archive
// path's config records.
//
// payload_name in the maps is only a placeholder: the caller reads the bytes
// from legacy_raw instead of the outer zip.
//
// returns 0 on success, -1 with a message on stderr on any inconsistency
// (missing payload, unknown pickle construct, unsupported byte order, ...).
int pnnx_load_legacy_payloads(StoreZipReader& zip, const std::vector<std::string>& names,
                              const JsonValue& root,
                              std::map<std::string, std::pair<std::string, JsonValue> >& weights,
                              std::map<std::string, std::pair<std::string, JsonValue> >& constants,
                              std::map<std::string, std::vector<char> >& legacy_raw);

} // namespace pnnx

#endif // PNNX_LOAD_EXPORTEDPROGRAM_LEGACY_H
