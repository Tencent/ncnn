// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "load_exported_program.h"

#include <stdio.h>
#include <string.h>

#include <limits>

namespace pnnx {

static int to_pnnx_type(int scalar_type)
{
    if (scalar_type == 1) return 8;
    if (scalar_type == 2) return 7;
    if (scalar_type == 3) return 6;
    if (scalar_type == 4) return 4;
    if (scalar_type == 5) return 5;
    if (scalar_type == 6) return 3;
    if (scalar_type == 7) return 1;
    if (scalar_type == 8) return 2;
    if (scalar_type == 9) return 12;
    if (scalar_type == 10) return 10;
    if (scalar_type == 11) return 11;
    if (scalar_type == 12) return 9;
    if (scalar_type == 13) return 13;
    return 0;
}

static bool to_pnnx_shape(const std::vector<pt2::SymInt>& dimensions, std::vector<int>& shape, std::string& error)
{
    for (size_t i = 0; i < dimensions.size(); i++)
    {
        const pt2::SymInt& dimension = dimensions[i];
        int64_t value = -1;
        if (dimension.type == pt2::SymInt::Integer)
            value = dimension.integer;
        else if (dimension.has_hint)
            value = dimension.hint;

        if (value < -1 || value > INT_MAX)
        {
            error = "tensor dimension " + std::to_string(i) + " is out of range";
            return false;
        }
        shape.push_back((int)value);
    }
    return true;
}

static bool checked_multiply_size(size_t lhs, size_t rhs, size_t& result)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
        return false;
    result = lhs * rhs;
    return true;
}

static bool materialize_attribute(const pt2::PayloadMeta& payload, const std::vector<char>& storage, Attribute& attribute, std::string& error)
{
    attribute.type = to_pnnx_type(payload.tensor_meta.scalar_type);
    if (attribute.type == 0)
    {
        error = "unsupported tensor scalar type " + std::to_string(payload.tensor_meta.scalar_type);
        return false;
    }
    if (!to_pnnx_shape(payload.tensor_meta.sizes, attribute.shape, error))
        return false;

    size_t element_count = 1;
    for (size_t i = 0; i < attribute.shape.size(); i++)
    {
        if (attribute.shape[i] < 0)
        {
            error = "attribute shape must be static";
            return false;
        }
        if (!checked_multiply_size(element_count, (size_t)attribute.shape[i], element_count))
        {
            error = "attribute element count overflows size_t";
            return false;
        }
    }

    size_t byte_count = 0;
    if (!checked_multiply_size(element_count, attribute.elemsize(), byte_count))
    {
        error = "attribute byte size overflows size_t";
        return false;
    }
    attribute.data.resize(byte_count);
    if (element_count == 0)
        return true;

    size_t source_element = (size_t)payload.tensor_meta.storage_offset.integer;
    for (size_t output_element = 0; output_element < element_count; output_element++)
    {
        size_t remaining = output_element;
        source_element = (size_t)payload.tensor_meta.storage_offset.integer;
        for (size_t axis = attribute.shape.size(); axis > 0; axis--)
        {
            const size_t dimension = (size_t)attribute.shape[axis - 1];
            const size_t index = remaining % dimension;
            remaining /= dimension;
            source_element += index * (size_t)payload.tensor_meta.strides[axis - 1].integer;
        }
        memcpy(attribute.data.data() + output_element * attribute.elemsize(), storage.data() + source_element * attribute.elemsize(), attribute.elemsize());
    }
    return true;
}

static const pt2::PayloadMeta* find_payload(const pt2::ExportedProgramArchive& archive, const pt2::InputSpec& spec, const std::map<std::string, std::vector<char> >*& storages)
{
    if (spec.type == pt2::InputSpec::Parameter)
    {
        storages = &archive.state_dict_storages;
        std::map<std::string, pt2::PayloadMeta>::const_iterator it = archive.state_dict.find(spec.target);
        return it == archive.state_dict.end() ? 0 : &it->second;
    }

    std::map<std::string, pt2::PayloadMeta>::const_iterator state = archive.state_dict.find(spec.target);
    if (state != archive.state_dict.end())
    {
        storages = &archive.state_dict_storages;
        return &state->second;
    }

    storages = &archive.constant_storages;
    std::map<std::string, pt2::PayloadMeta>::const_iterator constant = archive.constants.find(spec.target);
    return constant == archive.constants.end() ? 0 : &constant->second;
}

int import_exported_program_inputs(const pt2::ExportedProgramArchive& archive, Graph& graph, std::string& error)
{
    error.clear();
    if (archive.program.graph.inputs.size() != archive.program.signature.inputs.size())
    {
        error = "graph input count does not match graph signature";
        return -1;
    }

    int user_input_index = 0;
    for (size_t i = 0; i < archive.program.signature.inputs.size(); i++)
    {
        const pt2::InputSpec& spec = archive.program.signature.inputs[i];
        if (spec.type == pt2::InputSpec::UserInput && spec.argument.type == pt2::Argument::Tensor)
        {
            std::map<std::string, pt2::TensorMeta>::const_iterator meta = archive.program.graph.tensor_values.find(spec.argument.name);
            if (meta == archive.program.graph.tensor_values.end())
            {
                error = spec.argument.name + ": tensor metadata is missing";
                return -1;
            }

            Operator* op = graph.new_operator("pnnx.Input", "pnnx_input_" + std::to_string(user_input_index++));
            Operand* operand = graph.new_operand(spec.argument.name);
            operand->producer = op;
            operand->type = to_pnnx_type(meta->second.scalar_type);
            if (operand->type == 0 || !to_pnnx_shape(meta->second.sizes, operand->shape, error))
            {
                if (error.empty()) error = spec.argument.name + ": unsupported input tensor type";
                return -1;
            }
            op->outputs.push_back(operand);
            continue;
        }

        if (spec.type == pt2::InputSpec::Parameter || spec.type == pt2::InputSpec::Buffer || spec.type == pt2::InputSpec::TensorConstant)
        {
            const std::map<std::string, std::vector<char> >* storages = 0;
            const pt2::PayloadMeta* payload = find_payload(archive, spec, storages);
            if (!payload)
            {
                error = spec.target + ": tensor payload is missing";
                return -1;
            }

            const std::string storage_path = (storages == &archive.state_dict_storages ? "data/weights/" : "data/constants/") + payload->path;
            std::map<std::string, std::vector<char> >::const_iterator storage = storages->find(storage_path);
            if (storage == storages->end())
            {
                error = storage_path + ": tensor storage is missing";
                return -1;
            }

            Operator* op = graph.new_operator("pnnx.Attribute", spec.target);
            if (!materialize_attribute(*payload, storage->second, op->attrs["data"], error))
            {
                error = spec.target + ": " + error;
                return -1;
            }
            Operand* operand = graph.new_operand(spec.argument.name);
            operand->producer = op;
            operand->type = op->attrs["data"].type;
            operand->shape = op->attrs["data"].shape;
            op->outputs.push_back(operand);
            continue;
        }

        error = "unsupported graph input at index " + std::to_string(i);
        return -1;
    }
    return 0;
}

int load_exported_program(const std::string& path, Graph& graph)
{
    pt2::ExportedProgramArchive archive;
    std::string error;
    if (!pt2::load_exported_program_archive(path, archive, error))
    {
        fprintf(stderr, "load exported program failed: %s\n", error.c_str());
        return -1;
    }

    if (import_exported_program_inputs(archive, graph, error) != 0)
    {
        fprintf(stderr, "load exported program failed: %s\n", error.c_str());
        return -1;
    }

    fprintf(stderr, "load exported program failed: graph node import is not supported yet\n");
    return -1;
}

} // namespace pnnx