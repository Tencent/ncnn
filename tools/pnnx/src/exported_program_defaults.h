// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_EXPORTED_PROGRAM_DEFAULTS_H
#define PNNX_EXPORTED_PROGRAM_DEFAULTS_H

#include <string>

#include "exported_program.h"

namespace pnnx {
namespace pt2 {

// Whole-program normalization proves single-use, unaliased local whitelisted
// pointwise/fill_ targets from known allocators and uses functional counterparts.
// Arithmetic additionally requires static metadata proving no dtype promotion:
// floating self and concrete real scalars or same-dtype broadcastable tensors.
// detach_ is metadata-only solely for requires_grad=false lifted TensorConstants
// and their non-view detach aliases; its functional result still aliases self.
// Other writes remain unsupported; there is no general alias-rewiring engine.
bool append_default_arguments(ExportedProgram& program, std::string& error);
// Checks effects and common schema types, then orders arguments and fills
// dispatcher defaults. Optional graph metadata permits concrete metadata guards,
// NOT writes. The whole-program path/importer must also resolve guard references.
// Discharging input metadata guards assumes the exported input contract: callers
// must enforce dtype, static sizes, CPU/Strided and any asserted strides at the
// runtime boundary. Direct guarded signature UserInputs are static float32;
// lifted constants and state tensors are not subject to that runtime dtype cap.
bool normalize_exported_program_node(Node& node, std::string& error, const Graph* metadata_context = 0);

} // namespace pt2
} // namespace pnnx

#endif // PNNX_EXPORTED_PROGRAM_DEFAULTS_H