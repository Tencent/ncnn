// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "exported_program_tensor.h"

#include <limits.h>
#include <stdint.h>
#include <string.h>

#include <limits>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace pnnx {

MaterializedExportedTensor::MaterializedExportedTensor()
{
    pnnx_type = 0;
}

struct ExportedDtypeInfo
{
    int pnnx_type;
    size_t element_size;
    size_t endian_unit_size;
};

static int exported_dtype_info(int64_t dtype, ExportedDtypeInfo& info)
{
    // PyTorch serialized ScalarType -> pnnx Attribute type
    if (dtype == 1)
        info = ExportedDtypeInfo{8, 1, 1}; // Byte -> u8
    else if (dtype == 2)
        info = ExportedDtypeInfo{7, 1, 1}; // Char -> i8
    else if (dtype == 3)
        info = ExportedDtypeInfo{6, 2, 2}; // Short -> i16
    else if (dtype == 4)
        info = ExportedDtypeInfo{4, 4, 4}; // Int -> i32
    else if (dtype == 5)
        info = ExportedDtypeInfo{5, 8, 8}; // Long -> i64
    else if (dtype == 6)
        info = ExportedDtypeInfo{3, 2, 2}; // Half -> f16
    else if (dtype == 7)
        info = ExportedDtypeInfo{1, 4, 4}; // Float -> f32
    else if (dtype == 8)
        info = ExportedDtypeInfo{2, 8, 8}; // Double -> f64
    else if (dtype == 9)
        info = ExportedDtypeInfo{12, 4, 2}; // ComplexHalf -> c32
    else if (dtype == 10)
        info = ExportedDtypeInfo{10, 8, 4}; // ComplexFloat -> c64
    else if (dtype == 11)
        info = ExportedDtypeInfo{11, 16, 8}; // ComplexDouble -> c128
    else if (dtype == 12)
        info = ExportedDtypeInfo{9, 1, 1}; // Bool -> bool
    else if (dtype == 13)
        info = ExportedDtypeInfo{13, 2, 2}; // BFloat16 -> bf16
    else
        return -1;

    return 0;
}

int exported_tensor_dtype_to_pnnx_type(int64_t dtype)
{
    ExportedDtypeInfo info;
    if (exported_dtype_info(dtype, info) != 0)
        return 0;

    return info.pnnx_type;
}

static bool checked_multiply(uint64_t a, uint64_t b, uint64_t& result)
{
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a)
        return false;

    result = a * b;
    return true;
}

static bool checked_add(uint64_t a, uint64_t b, uint64_t& result)
{
    if (b > std::numeric_limits<uint64_t>::max() - a)
        return false;

    result = a + b;
    return true;
}

static bool host_is_little_endian()
{
    const uint16_t marker = 1;
    unsigned char first_byte = 0;
    memcpy(&first_byte, &marker, sizeof(first_byte));
    return first_byte == 1;
}

static void copy_element(char* destination, const char* source, const ExportedDtypeInfo& info, bool swap_byte_order)
{
    if (!swap_byte_order || info.endian_unit_size == 1)
    {
        memcpy(destination, source, info.element_size);
        return;
    }

    for (size_t component = 0; component < info.element_size; component += info.endian_unit_size)
    {
        for (size_t i = 0; i < info.endian_unit_size; i++)
            destination[component + i] = source[component + info.endian_unit_size - 1 - i];
    }
}

int materialize_exported_tensor(const ExportedTensorMeta& meta,
                                const std::vector<char>& storage,
                                Pt2ByteOrder byte_order,
                                MaterializedExportedTensor& tensor,
                                std::string& error)
{
    tensor = MaterializedExportedTensor();
    error.clear();

    if (meta.layout != 7)
    {
        std::ostringstream message;
        message << "unsupported tensor layout " << meta.layout;
        error = message.str();
        return -1;
    }

    ExportedDtypeInfo dtype_info;
    if (exported_dtype_info(meta.dtype, dtype_info) != 0)
    {
        std::ostringstream message;
        message << "unsupported exported tensor dtype " << meta.dtype;
        error = message.str();
        return -1;
    }

    if (meta.strides.size() != meta.sizes.size())
    {
        error = "tensor stride rank does not match shape";
        return -1;
    }

    if (meta.storage_offset < 0)
    {
        error = "negative storage offset";
        return -1;
    }

    MaterializedExportedTensor parsed_tensor;
    parsed_tensor.pnnx_type = dtype_info.pnnx_type;
    parsed_tensor.shape.reserve(meta.sizes.size());

    bool has_zero_dimension = false;
    for (size_t i = 0; i < meta.sizes.size(); i++)
    {
        if (meta.sizes[i] < 0)
        {
            std::ostringstream message;
            message << "negative tensor size at dimension " << i;
            error = message.str();
            return -1;
        }
        if (meta.sizes[i] > INT_MAX)
        {
            std::ostringstream message;
            message << "tensor size at dimension " << i << " does not fit pnnx shape";
            error = message.str();
            return -1;
        }
        if (meta.strides[i] < 0)
        {
            std::ostringstream message;
            message << "negative tensor stride at dimension " << i;
            error = message.str();
            return -1;
        }

        parsed_tensor.shape.push_back((int)meta.sizes[i]);
        has_zero_dimension = has_zero_dimension || meta.sizes[i] == 0;
    }

    if (storage.size() % dtype_info.element_size != 0)
    {
        error = "tensor storage size is not aligned to element size";
        return -1;
    }

    const uint64_t storage_element_count = (uint64_t)(storage.size() / dtype_info.element_size);
    const uint64_t storage_offset = (uint64_t)meta.storage_offset;
    if (has_zero_dimension)
    {
        if (storage_offset > storage_element_count)
        {
            error = "tensor view exceeds storage";
            return -1;
        }

        tensor.pnnx_type = parsed_tensor.pnnx_type;
        tensor.shape.swap(parsed_tensor.shape);
        return 0;
    }

    uint64_t element_count = 1;
    for (size_t i = 0; i < meta.sizes.size(); i++)
    {
        uint64_t next_count = 0;
        if (!checked_multiply(element_count, (uint64_t)meta.sizes[i], next_count))
        {
            error = "tensor element count overflow";
            return -1;
        }
        element_count = next_count;
    }

    if (element_count > INT_MAX)
    {
        error = "tensor element count does not fit pnnx attribute";
        return -1;
    }

    uint64_t maximum_source_offset = storage_offset;
    for (size_t i = 0; i < meta.sizes.size(); i++)
    {
        uint64_t dimension_offset = 0;
        if (!checked_multiply((uint64_t)(meta.sizes[i] - 1), (uint64_t)meta.strides[i], dimension_offset)
                || !checked_add(maximum_source_offset, dimension_offset, maximum_source_offset))
        {
            error = "tensor view offset overflow";
            return -1;
        }
    }

    if (maximum_source_offset >= storage_element_count)
    {
        error = "tensor view exceeds storage";
        return -1;
    }

    uint64_t output_size = 0;
    if (!checked_multiply(element_count, (uint64_t)dtype_info.element_size, output_size)
            || output_size > (uint64_t)std::numeric_limits<size_t>::max()
            || output_size > (uint64_t)parsed_tensor.data.max_size())
    {
        error = "tensor output size overflow";
        return -1;
    }

    try
    {
        parsed_tensor.data.resize((size_t)output_size);
    }
    catch (const std::length_error&)
    {
        error = "tensor output is too large for this platform";
        return -1;
    }
    catch (const std::bad_alloc&)
    {
        error = "cannot allocate materialized tensor";
        return -1;
    }

    const bool source_is_little_endian = byte_order == PT2_BYTE_ORDER_LITTLE;
    const bool swap_byte_order = source_is_little_endian != host_is_little_endian();

    bool is_contiguous = true;
    uint64_t expected_stride = 1;
    for (size_t reverse_i = meta.sizes.size(); reverse_i > 0; reverse_i--)
    {
        const size_t i = reverse_i - 1;
        if (meta.sizes[i] > 1 && (uint64_t)meta.strides[i] != expected_stride)
            is_contiguous = false;

        uint64_t next_stride = 0;
        if (!checked_multiply(expected_stride, (uint64_t)meta.sizes[i], next_stride))
        {
            error = "tensor contiguous stride overflow";
            return -1;
        }
        expected_stride = next_stride;
    }

    if (is_contiguous && !swap_byte_order)
    {
        const size_t source_offset = (size_t)(storage_offset * dtype_info.element_size);
        memcpy(&parsed_tensor.data[0], &storage[source_offset], (size_t)output_size);
    }
    else
    {
        std::vector<uint64_t> coordinate(meta.sizes.size(), 0);
        for (uint64_t output_index = 0; output_index < element_count; output_index++)
        {
            uint64_t source_index = storage_offset;
            for (size_t i = 0; i < coordinate.size(); i++)
                source_index += coordinate[i] * (uint64_t)meta.strides[i];

            const size_t source_byte = (size_t)(source_index * dtype_info.element_size);
            const size_t destination_byte = (size_t)(output_index * dtype_info.element_size);
            copy_element(&parsed_tensor.data[destination_byte], &storage[source_byte], dtype_info, swap_byte_order);

            for (size_t reverse_i = coordinate.size(); reverse_i > 0; reverse_i--)
            {
                const size_t i = reverse_i - 1;
                coordinate[i]++;
                if (coordinate[i] < (uint64_t)meta.sizes[i])
                    break;
                coordinate[i] = 0;
            }
        }
    }

    tensor.pnnx_type = parsed_tensor.pnnx_type;
    tensor.shape.swap(parsed_tensor.shape);
    tensor.data.swap(parsed_tensor.data);
    return 0;
}

} // namespace pnnx
