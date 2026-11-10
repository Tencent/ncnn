// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pt2_tensor.h"

#include <limits.h>
#include <stdint.h>
#include <string.h>
#include <utility>

#include <stdio.h>

namespace pnnx {

int pt2_scalar_type(int type)
{
    // pnnx null/f32/f64/f16/i32/i64/i16/i8/u8/bool/c64/c128/c32/bf16 -> c10 ScalarType
    static const int types[] = {-1, 6, 7, 5, 3, 4, 2, 1, 0, 11, 9, 10, 8, 15};
    return type >= 1 && type <= 13 ? types[type] : -1;
}

int pt2_tensor_type(int serde_type)
{
    switch (serde_type)
    {
    case 1:
        return 8; // uint8
    case 2:
        return 7; // int8
    case 3:
        return 6; // int16
    case 4:
        return 4; // int32
    case 5:
        return 5; // int64
    case 6:
        return 3; // float16
    case 7:
        return 1; // float32
    case 8:
        return 2; // float64
    case 9:
        return 12; // complex32
    case 10:
        return 10; // complex64
    case 11:
        return 11; // complex128
    case 12:
        return 9; // bool
    case 13:
        return 13; // bfloat16
    default:
        return 0;
    }
}

static int parse_symint(const JsonValue& value, int64_t& n, std::string& symbol, bool allow_symbolic)
{
    if (value.type != JsonValue::Object || value.o.size() != 1)
    {
        fprintf(stderr, "pnnx: pt2 tensor: expected a SymInt union\n");
        return -1;
    }

    const JsonValue* as_int = value.get("as_int");
    if (as_int)
    {
        if (as_int->type != JsonValue::Int || as_int->i < 0)
        {
            fprintf(stderr, "pnnx: pt2 tensor: expected a nonnegative integer\n");
            return -1;
        }

        n = as_int->i;
        symbol.clear();
        return 0;
    }

    const JsonValue* as_expr = value.get("as_expr");
    if (!as_expr || as_expr->type != JsonValue::Object)
    {
        fprintf(stderr, "pnnx: pt2 tensor: unsupported SymInt union\n");
        return -1;
    }
    if (!allow_symbolic)
    {
        fprintf(stderr, "pnnx: pt2 tensor: symbolic tensor storage is unsupported\n");
        return -1;
    }

    const JsonValue* expr_str = as_expr->get("expr_str");
    if (!expr_str || expr_str->type != JsonValue::String || expr_str->s.empty())
    {
        fprintf(stderr, "pnnx: pt2 tensor: missing symbolic expression\n");
        return -1;
    }

    const JsonValue* hint = as_expr->get("hint");
    if (hint && hint->type != JsonValue::Null)
    {
        const JsonValue* hint_int = hint->get("as_int");
        if (hint->type != JsonValue::Object || hint->o.size() != 1 || !hint_int || hint_int->type != JsonValue::Int || hint_int->i < 0)
        {
            fprintf(stderr, "pnnx: pt2 tensor: invalid SymInt hint\n");
            return -1;
        }
    }

    // hints describe an example, not the runtime dimension
    n = -1;
    symbol = expr_str->s;
    return 0;
}

int parse_pt2_tensor_meta(const JsonValue& value, Pt2TensorMeta& meta, bool allow_symbolic)
{
    const JsonValue* dtype = value.get("dtype");
    const JsonValue* sizes = value.get("sizes");
    const JsonValue* strides = value.get("strides");
    const JsonValue* storage_offset = value.get("storage_offset");
    const JsonValue* device = value.get("device");
    const JsonValue* layout = value.get("layout");
    const JsonValue* requires_grad = value.get("requires_grad");
    if (!dtype || !sizes || !strides || !storage_offset || !device || !layout || !requires_grad)
    {
        fprintf(stderr, "pnnx: pt2 tensor: incomplete tensor metadata\n");
        return -1;
    }

    Pt2TensorMeta m;
    if (dtype->type != JsonValue::Int || dtype->i < 0 || dtype->i > INT_MAX || (m.type = pt2_tensor_type((int)dtype->i)) == 0)
    {
        fprintf(stderr, "pnnx: pt2 tensor: unsupported tensor dtype\n");
        return -1;
    }
    if (layout->type != JsonValue::Int || layout->i != 7)
    {
        fprintf(stderr, "pnnx: pt2 tensor: only strided tensor layout is supported\n");
        return -1;
    }
    m.layout = 7;

    const JsonValue* device_type = device->get("type");
    const JsonValue* device_index = device->get("index");
    if (!device_type || device_type->type != JsonValue::String || device_type->s != "cpu"
            || (device_index && device_index->type != JsonValue::Null && (device_index->type != JsonValue::Int || device_index->i < 0 || device_index->i > INT_MAX)))
    {
        fprintf(stderr, "pnnx: pt2 tensor: only CPU tensor metadata is supported\n");
        return -1;
    }
    m.device = "cpu";
    if (requires_grad->type != JsonValue::Bool)
    {
        fprintf(stderr, "pnnx: pt2 tensor: invalid requires_grad\n");
        return -1;
    }

    if (sizes->type != JsonValue::Array || strides->type != JsonValue::Array || sizes->a.size() != strides->a.size())
    {
        fprintf(stderr, "pnnx: pt2 tensor: tensor sizes and strides must have the same rank\n");
        return -1;
    }

    for (size_t i = 0; i < sizes->a.size(); i++)
    {
        int64_t n;
        std::string symbol;
        if (parse_symint(sizes->a[i], n, symbol, allow_symbolic) != 0)
        {
            return -1;
        }
        if (n > INT_MAX)
        {
            fprintf(stderr, "pnnx: pt2 tensor: tensor dimension exceeds pnnx int32 shape range\n");
            return -1;
        }
        m.shape.push_back((int)n);
        m.shape_symbols.push_back(symbol);

        if (parse_symint(strides->a[i], n, symbol, allow_symbolic) != 0)
        {
            return -1;
        }
        m.strides.push_back(n);
    }

    std::string symbol;
    if (parse_symint(*storage_offset, m.storage_offset, symbol, allow_symbolic) != 0)
    {
        return -1;
    }

    std::swap(meta, m);
    return 0;
}

static bool valid_storage_name(const std::string& name)
{
    if (name.empty() || name.find_first_of("\\:") != std::string::npos || name.find('\0') != std::string::npos)
        return false;

    size_t pos = 0;
    while (pos < name.size())
    {
        const size_t end = name.find('/', pos);
        const std::string part = name.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        if (part.empty() || part == "." || part == "..")
            return false;
        if (end == std::string::npos)
            return true;
        pos = end + 1;
    }
    return false;
}

static size_t tensor_elemsize(int type)
{
    static const size_t elemsizes[] = {0, 4, 8, 2, 4, 8, 2, 1, 1, 1, 8, 16, 4, 2};
    return elemsizes[type];
}

int load_pt2_tensor(StoreZipReader& zip, const std::string& directory, const JsonValue& payload, Attribute& attr, Pt2TensorMeta& meta,
                    std::map<std::string, Pt2TensorStorage>& storage_cache)
{
    const JsonValue* path_name = payload.get("path_name");
    const JsonValue* is_param = payload.get("is_param");
    const JsonValue* use_pickle = payload.get("use_pickle");
    const JsonValue* tensor_meta = payload.get("tensor_meta");
    if (!path_name || path_name->type != JsonValue::String || !valid_storage_name(path_name->s)
            || !is_param || is_param->type != JsonValue::Bool || !use_pickle || use_pickle->type != JsonValue::Bool || !tensor_meta)
    {
        fprintf(stderr, "pnnx: pt2 tensor: invalid tensor payload metadata or storage path\n");
        return -1;
    }
    if (use_pickle->b)
    {
        fprintf(stderr, "pnnx: pt2 tensor: pickle tensor payload is unsupported\n");
        return -1;
    }

    const uint16_t endian = 1;
    if (*(const unsigned char*)&endian != 1)
    {
        fprintf(stderr, "pnnx: pt2 tensor: only little-endian tensor storage is supported\n");
        return -1;
    }

    Pt2TensorMeta m;
    if (parse_pt2_tensor_meta(*tensor_meta, m, false) != 0)
        return -1;

    size_t numel = 1;
    bool empty = false;
    for (size_t i = 0; i < m.shape.size(); i++)
    {
        if (m.shape[i] == 0)
            empty = true;
    }
    if (empty)
    {
        numel = 0;
    }
    else
    {
        for (size_t i = 0; i < m.shape.size(); i++)
        {
            if (numel > (size_t)INT_MAX / (size_t)m.shape[i])
            {
                fprintf(stderr, "pnnx: pt2 tensor: tensor element count exceeds pnnx int32 range\n");
                return -1;
            }
            numel *= (size_t)m.shape[i];
        }
    }
    const size_t elemsize = tensor_elemsize(m.type);
    if (numel > SIZE_MAX / elemsize)
    {
        fprintf(stderr, "pnnx: pt2 tensor: tensor byte size overflow\n");
        return -1;
    }

    const std::string path = directory + (directory.empty() || directory[directory.size() - 1] == '/' ? "" : "/") + path_name->s;
    if (!zip.has_file(path))
    {
        fprintf(stderr, "pnnx: pt2 tensor: missing tensor storage %s\n", path.c_str());
        return -1;
    }

    const uint64_t size = zip.get_file_size(path);
    if (size > SIZE_MAX || size > std::vector<char>().max_size() || size % elemsize != 0)
    {
        fprintf(stderr, "pnnx: pt2 tensor: invalid tensor storage byte size %s\n", path.c_str());
        return -1;
    }

    if (numel != 0)
    {
        uint64_t last = (uint64_t)m.storage_offset;
        for (size_t i = 0; i < m.shape.size(); i++)
        {
            const uint64_t extent = (uint64_t)m.shape[i] - 1;
            const uint64_t stride = (uint64_t)m.strides[i];
            if (extent != 0 && stride > (UINT64_MAX - last) / extent)
            {
                fprintf(stderr, "pnnx: pt2 tensor: tensor storage address overflow %s\n", path.c_str());
                return -1;
            }
            last += extent * stride;
        }
        if (last >= size / elemsize)
        {
            fprintf(stderr, "pnnx: pt2 tensor: tensor view exceeds storage %s\n", path.c_str());
            return -1;
        }
    }

    Pt2TensorStorage& storage = storage_cache[path];
    if (storage.data.empty())
    {
        std::vector<char> data((size_t)size);
        char empty_data;
        if (zip.read_file(path, data.empty() ? &empty_data : data.data()) != 0)
        {
            fprintf(stderr, "pnnx: pt2 tensor: failed to read tensor storage %s\n", path.c_str());
            return -1;
        }
        storage.data.swap(data);
    }
    if (storage.data.size() != size)
    {
        fprintf(stderr, "pnnx: pt2 tensor: inconsistent cached tensor storage %s\n", path.c_str());
        return -1;
    }

    Attribute r;
    r.type = m.type;
    r.shape = m.shape;
    const bool last_use = storage.remaining <= 1;
    if (numel != 0)
    {
        // coalesce contiguous and size-one suffix axes before walking outer axes
        size_t block = 1;
        size_t outer_rank = m.shape.size();
        while (outer_rank > 0)
        {
            const size_t axis = outer_rank - 1;
            if (m.shape[axis] > 1 && (uint64_t)m.strides[axis] != block)
                break;
            block *= (size_t)m.shape[axis];
            outer_rank--;
        }
        if (outer_rank == 0 && m.storage_offset == 0 && numel * elemsize == storage.data.size() && last_use)
        {
            std::swap(r.data, storage.data);
        }
        else
        {
            r.data.resize(numel * elemsize);
            std::vector<int> indices(outer_rank, 0);
            uint64_t offset = (uint64_t)m.storage_offset;
            for (size_t i = 0; i < numel; i += block)
            {
                memcpy(r.data.data() + i * elemsize, storage.data.data() + (size_t)offset * elemsize, block * elemsize);
                if (i + block == numel)
                    break;
                for (size_t j = outer_rank; j > 0; j--)
                {
                    const size_t axis = j - 1;
                    if (++indices[axis] < m.shape[axis])
                    {
                        offset += (uint64_t)m.strides[axis];
                        break;
                    }
                    indices[axis] = 0;
                    offset -= (uint64_t)(m.shape[axis] - 1) * (uint64_t)m.strides[axis];
                }
            }
        }
    }
    if (last_use)
        storage_cache.erase(path);
    else
        storage.remaining--;
    std::swap(attr, r);
    std::swap(meta, m);
    return 0;
}

} // namespace pnnx
