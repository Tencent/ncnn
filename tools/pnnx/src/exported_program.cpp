// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "exported_program.h"

#include <limits>

#include <torch/csrc/jit/operator_upgraders/utils.h>

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

bool validate_exported_program_version(const ExportedProgram& program, std::string& error)
{
    // 8.20 is covered by the serialized fixtures and the inspected torch serde
    // schema. Do not infer support for other minor versions from the major alone.
    if (program.schema_version.major != 8 || program.schema_version.minor != 20)
    {
        error = "schema_version: unsupported exported program schema " + std::to_string(program.schema_version.major) + "." + std::to_string(program.schema_version.minor) + "; supported schema is 8.20";
        return false;
    }

    // torch.export's serde uses this operator version, not the torch release or
    // an ONNX opset. We do not run JIT upgraders, so require an exact match.
    const uint64_t expected = torch::jit::getMaxOperatorVersion();
    std::map<std::string, int>::const_iterator aten = program.opset_version.find("aten");
    if (aten == program.opset_version.end() || aten->second < 0 || (uint64_t)aten->second != expected)
    {
        error = "opset_version.aten: expected linked Torch operator version " + std::to_string(expected) + ", got " + (aten == program.opset_version.end() ? "missing" : std::to_string(aten->second)) + "; operator upgrading/downgrading is not supported";
        return false;
    }
    for (std::map<std::string, int>::const_iterator it = program.opset_version.begin(); it != program.opset_version.end(); ++it)
    {
        if (it->first != "aten")
        {
            error = "opset_version." + it->first + ": unsupported operator namespace contract";
            return false;
        }
    }
    return true;
}

static bool checked_add(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs)
        return false;
    result = lhs + rhs;
    return true;
}

static bool checked_multiply(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
        return false;
    result = lhs * rhs;
    return true;
}

bool validate_tensor_storage(const PayloadMeta& payload, uint64_t storage_size, std::string& error)
{
    if (payload.use_pickle || !payload.has_tensor_meta)
    {
        error = payload.use_pickle ? "pickled tensor payload is not supported" : "tensor metadata is missing";
        return false;
    }
    const TensorMeta& meta = payload.tensor_meta;
    if (meta.device.type != "cpu" || (meta.device.has_index && meta.device.index != 0))
    {
        error = "raw tensor payload requires CPU device";
        return false;
    }
    if (meta.layout != 7)
    {
        error = "raw tensor payload requires serde Strided layout (7), got " + std::to_string(meta.layout);
        return false;
    }
    if (meta.sizes.size() != meta.strides.size() || meta.sizes.size() > 64)
    {
        error = "tensor size and stride rank mismatch or rank exceeds limit (64)";
        return false;
    }
    // Only serde scalar types representable by pnnx::Attribute. In particular,
    // newer float8/uint16/uint32/uint64 enums are not implicitly byte blobs.
    const unsigned int element_sizes[] = {0, 1, 1, 2, 4, 8, 2, 4, 8, 4, 8, 16, 1, 2};
    if (meta.scalar_type <= 0 || meta.scalar_type >= (int)(sizeof(element_sizes) / sizeof(element_sizes[0])))
    {
        error = "unsupported tensor scalar type " + std::to_string(meta.scalar_type);
        return false;
    }
    const uint64_t element_size = element_sizes[meta.scalar_type];
    if (meta.storage_offset.type != SymInt::Integer || meta.storage_offset.integer < 0)
    {
        error = "storage offset must be a nonnegative integer";
        return false;
    }
    bool empty = false;
    uint64_t maximum_element = (uint64_t)meta.storage_offset.integer;
    for (size_t i = 0; i < meta.sizes.size(); i++)
    {
        if (meta.sizes[i].type != SymInt::Integer || meta.strides[i].type != SymInt::Integer || meta.sizes[i].integer < 0 || meta.strides[i].integer < 0)
        {
            error = "tensor sizes and strides must be nonnegative integers";
            return false;
        }
        const uint64_t size = (uint64_t)meta.sizes[i].integer;
        empty = empty || size == 0;
        uint64_t extent = 0;
        if (size && (!checked_multiply(size - 1, (uint64_t)meta.strides[i].integer, extent) || !checked_add(maximum_element, extent, maximum_element)))
        {
            error = "tensor storage range overflows uint64";
            return false;
        }
    }
    uint64_t required_elements = (uint64_t)meta.storage_offset.integer;
    uint64_t required_bytes = 0;
    if ((!empty && !checked_add(maximum_element, 1, required_elements)) || !checked_multiply(required_elements, element_size, required_bytes))
    {
        error = "tensor storage size overflows uint64";
        return false;
    }
    if (required_bytes > storage_size)
    {
        error = "tensor view exceeds storage payload";
        return false;
    }

    uint64_t elements = empty ? 0 : 1;
    for (size_t i = 0; i < meta.sizes.size(); i++)
    {
        if (meta.sizes[i].integer > std::numeric_limits<int>::max())
        {
            error = "attribute dimension exceeds pnnx integer range";
            return false;
        }
        if (!checked_multiply(elements, (uint64_t)meta.sizes[i].integer, elements))
        {
            error = "attribute element count overflows uint64";
            return false;
        }
    }
    uint64_t bytes = 0;
    if (!checked_multiply(elements, element_size, bytes))
    {
        error = "attribute byte size overflows uint64";
        return false;
    }
    // A small broadcast/zero-stride storage can describe an enormous dense
    // tensor. Bound each materialized attribute to 512 MiB (the reader's record
    // budget), independently of source bounds; this is not a total-model budget.
    if (bytes > 512ull * 1024 * 1024 || bytes > std::numeric_limits<size_t>::max())
    {
        error = "attribute exceeds 512 MiB materialization budget";
        return false;
    }
    return true;
}

} // namespace pt2
} // namespace pnnx