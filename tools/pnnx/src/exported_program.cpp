// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "exported_program.h"

namespace pnnx {
namespace pt2 {

SymInt::SymInt()
    : type(Integer), integer(0), has_hint(false), hint(0)
{
}

Device::Device()
    : has_index(false), index(0)
{
}

TensorMeta::TensorMeta()
    : scalar_type(0), requires_grad(false), layout(0)
{
}

Argument::Argument()
    : type(Unknown), boolean(false), integer(0), floating_point(0.0), complex_real(0.0), complex_imag(0.0)
{
}

NamedArgument::NamedArgument()
    : kind(KindUnknown)
{
}

Graph::Graph()
    : is_single_tensor_return(false)
{
}

InputSpec::InputSpec()
    : type(UserInput), persistent(false)
{
}

OutputSpec::OutputSpec()
    : type(UserOutput)
{
}

RangeConstraint::RangeConstraint()
    : has_min(false), min(0), has_max(false), max(0)
{
}

SchemaVersion::SchemaVersion()
    : major(0), minor(0)
{
}

PayloadMeta::PayloadMeta()
    : is_parameter(false), use_pickle(false), has_tensor_meta(false)
{
}

ExportedProgramArchive::ExportedProgramArchive()
    : archive_version(0)
{
}

} // namespace pt2
} // namespace pnnx