// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pt2_weights.h"

#include <limits.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "pt2_archive.h"
#include "pt2_json.h"
#include "storezip.h"

namespace pnnx {

struct Pt2LoadedTensor
{
    Pt2Tensor meta;
    Attribute attribute;
    bool is_parameter;
};

static uint32_t read_le32(const unsigned char* p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static bool checked_add_u64(uint64_t a, uint64_t b, uint64_t& result)
{
    if (a > UINT64_MAX - b)
        return false;
    result = a + b;
    return true;
}

static bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t& result)
{
    if (a != 0 && b > UINT64_MAX / a)
        return false;
    result = a * b;
    return true;
}

static int scalar_type_info(int dtype, int& attribute_type, size_t& element_size, size_t& component_size)
{
    static const int attribute_types[] = {0, 8, 7, 6, 4, 5, 3, 1, 2, 12, 10, 11, 9, 13};
    static const unsigned char element_sizes[] = {0, 1, 1, 2, 4, 8, 2, 4, 8, 4, 8, 16, 1, 2};
    static const unsigned char component_sizes[] = {0, 1, 1, 2, 4, 8, 2, 4, 8, 2, 4, 8, 1, 2};
    if (dtype < 1 || dtype > 13)
        return -1;
    attribute_type = attribute_types[dtype];
    element_size = element_sizes[dtype];
    component_size = component_sizes[dtype];
    return 0;
}

static int storage_scalar_type(const std::string& name)
{
    static const char* names[] = {
        "", "torch ByteStorage", "torch CharStorage", "torch ShortStorage", "torch IntStorage", "torch LongStorage",
        "torch HalfStorage", "torch FloatStorage", "torch DoubleStorage", "torch ComplexHalfStorage",
        "torch ComplexFloatStorage", "torch ComplexDoubleStorage", "torch BoolStorage", "torch BFloat16Storage"};
    for (int i = 1; i <= 13; i++)
    {
        if (name == names[i])
            return i;
    }
    return 0;
}

static std::string trim_ascii_whitespace(const std::vector<unsigned char>& data)
{
    size_t begin = 0;
    while (begin < data.size() && (data[begin] == ' ' || data[begin] == '\t' || data[begin] == '\r' || data[begin] == '\n'))
        begin++;
    size_t end = data.size();
    while (end > begin && (data[end - 1] == ' ' || data[end - 1] == '\t' || data[end - 1] == '\r' || data[end - 1] == '\n'))
        end--;
    return std::string(data.begin() + begin, data.begin() + end);
}

static bool host_is_little_endian()
{
    const uint16_t value = 1;
    return *(const unsigned char*)&value == 1;
}

static void byte_swap_components(char* data, uint64_t count, size_t element_size, size_t component_size)
{
    if (component_size == 1)
        return;
    const size_t components = element_size / component_size;
    for (uint64_t i = 0; i < count; i++)
    {
        for (size_t j = 0; j < components; j++)
            std::reverse(data + i * element_size + j * component_size, data + i * element_size + (j + 1) * component_size);
    }
}

static int make_attribute(const Pt2Tensor& tensor, const std::vector<unsigned char>& storage, uint64_t storage_elements,
                          const std::string& byteorder, Attribute& attribute, std::string& error)
{
    int attribute_type;
    size_t element_size;
    size_t component_size;
    if (scalar_type_info(tensor.dtype, attribute_type, element_size, component_size) != 0)
    {
        error = "unsupported scalar type " + std::to_string(tensor.dtype);
        return -1;
    }
    if (tensor.layout != 7 || tensor.device != "cpu" || tensor.device_index != -1)
    {
        error = "only dense strided CPU tensors are supported";
        return -1;
    }
    if (tensor.sizes.size() != tensor.strides.size() || tensor.storage_offset.symbolic || tensor.storage_offset.value < 0)
    {
        error = "invalid static tensor view metadata";
        return -1;
    }

    uint64_t expected_storage_bytes;
    if (!checked_mul_u64(storage_elements, element_size, expected_storage_bytes) || expected_storage_bytes != storage.size())
    {
        error = "storage byte size does not match dtype and element count";
        return -1;
    }

    uint64_t numel = 1;
    uint64_t max_index = (uint64_t)tensor.storage_offset.value;
    attribute.shape.clear();
    attribute.shape.reserve(tensor.sizes.size());
    for (size_t i = 0; i < tensor.sizes.size(); i++)
    {
        if (tensor.sizes[i].symbolic || tensor.strides[i].symbolic || tensor.sizes[i].value < 0 || tensor.strides[i].value < 0 || tensor.sizes[i].value > INT_MAX)
        {
            error = "symbolic, negative, or oversized tensor view metadata";
            return -1;
        }
        attribute.shape.push_back((int)tensor.sizes[i].value);
        if (!checked_mul_u64(numel, (uint64_t)tensor.sizes[i].value, numel))
        {
            error = "tensor element count overflow";
            return -1;
        }
        if (tensor.sizes[i].value != 0)
        {
            uint64_t extent;
            if (!checked_mul_u64((uint64_t)(tensor.sizes[i].value - 1), (uint64_t)tensor.strides[i].value, extent) ||
                !checked_add_u64(max_index, extent, max_index))
            {
                error = "tensor view offset overflow";
                return -1;
            }
        }
    }
    if ((numel == 0 && (uint64_t)tensor.storage_offset.value > storage_elements) || (numel != 0 && max_index >= storage_elements))
    {
        error = "tensor view is outside its storage";
        return -1;
    }

    uint64_t output_size;
    if (!checked_mul_u64(numel, element_size, output_size) || output_size > SIZE_MAX)
    {
        error = "tensor byte size overflow";
        return -1;
    }
    attribute.type = attribute_type;
    attribute.data.resize((size_t)output_size);

    bool contiguous = true;
    uint64_t expected_stride = 1;
    for (size_t i = tensor.sizes.size(); i-- > 0;)
    {
        if (tensor.sizes[i].value > 1 && (uint64_t)tensor.strides[i].value != expected_stride)
            contiguous = false;
        if (!checked_mul_u64(expected_stride, (uint64_t)tensor.sizes[i].value, expected_stride))
            contiguous = false;
    }

    if (numel != 0 && contiguous)
    {
        memcpy(&attribute.data[0], &storage[(size_t)tensor.storage_offset.value * element_size], (size_t)output_size);
    }
    else
    {
        for (uint64_t i = 0; i < numel; i++)
        {
            uint64_t remaining = i;
            uint64_t source_index = (uint64_t)tensor.storage_offset.value;
            for (size_t j = tensor.sizes.size(); j-- > 0;)
            {
                const uint64_t size = (uint64_t)tensor.sizes[j].value;
                const uint64_t index = remaining % size;
                remaining /= size;
                source_index += index * (uint64_t)tensor.strides[j].value;
            }
            memcpy(&attribute.data[(size_t)i * element_size], &storage[(size_t)source_index * element_size], element_size);
        }
    }

    if (byteorder != "little" && byteorder != "big")
    {
        error = "unsupported byte order " + byteorder;
        return -1;
    }
    if ((byteorder == "little") != host_is_little_endian() && !attribute.data.empty())
        byte_swap_components(&attribute.data[0], numel, element_size, component_size);
    return 0;
}

static bool same_sym_int(const Pt2SymInt& a, const Pt2SymInt& b)
{
    return a.symbolic == b.symbolic && (a.symbolic ? a.has_hint == b.has_hint && a.hint == b.hint && a.expression == b.expression : a.value == b.value);
}

static bool same_tensor_meta(const Pt2Tensor& a, const Pt2Tensor& b)
{
    if (a.dtype != b.dtype || a.requires_grad != b.requires_grad || a.device != b.device || a.device_index != b.device_index ||
        a.layout != b.layout || a.sizes.size() != b.sizes.size() || a.strides.size() != b.strides.size())
        return false;
    for (size_t i = 0; i < a.sizes.size(); i++)
    {
        if (!same_sym_int(a.sizes[i], b.sizes[i]) || !same_sym_int(a.strides[i], b.strides[i]))
            return false;
    }
    return true;
}

class Pt2PickleReader
{
public:
    struct Value
    {
        enum Type
        {
            None,
            Bool,
            Int,
            String,
            Global,
            Tuple,
            List,
            Dict,
            Storage,
            Tensor
        };

        Value()
        {
            type = None;
            b = false;
            i = 0;
            storage_elements = 0;
            is_parameter = false;
        }

        Type type;
        bool b;
        int64_t i;
        std::string s;
        std::vector<Value> list;
        std::map<std::string, Value> dict;
        Pt2Tensor tensor;
        uint64_t storage_elements;
        bool is_parameter;
    };

    Pt2PickleReader(const std::vector<unsigned char>& _data, const std::string& _storage_prefix)
        : data(_data), storage_prefix(_storage_prefix)
    {
        pos = 0;
        operations = 0;
    }

    int parse(std::map<std::string, Value>& tensors, std::string& output_error)
    {
        if (data.size() > 64 * 1024 * 1024)
            return finish_error("pickle is too large", output_error);
        if (!read_opcode(0x80) || !read_opcode(2))
            return finish_error("only pickle protocol 2 is supported", output_error);

        while (pos < data.size())
        {
            if (++operations > 1000000 || stack.size() > 1000000)
                return finish_error("pickle operation or stack limit exceeded", output_error);
            const unsigned char opcode = data[pos++];
            if (opcode == '.')
            {
                if (pos != data.size() || !marks.empty() || stack.size() != 1 || stack[0].type != Value::Dict)
                    return finish_error("invalid pickle STOP state", output_error);
                for (std::map<std::string, Value>::const_iterator it = stack[0].dict.begin(); it != stack[0].dict.end(); ++it)
                {
                    if (it->second.type != Value::Tensor)
                        return finish_error("root dictionary contains a non-tensor value at " + it->first, output_error);
                    tensors[it->first] = it->second;
                }
                return 0;
            }
            if (!execute(opcode))
                return finish_error(error, output_error);
        }
        return finish_error("pickle has no STOP opcode", output_error);
    }

private:
    int finish_error(const std::string& message, std::string& output_error)
    {
        output_error = "pickle offset " + std::to_string(pos) + ": " + message;
        return -1;
    }

    bool fail(const std::string& message)
    {
        if (error.empty())
            error = message;
        return false;
    }

    bool read_opcode(unsigned char expected)
    {
        return pos < data.size() && data[pos++] == expected;
    }

    bool read_u32(uint32_t& value)
    {
        if (data.size() - pos < 4)
            return fail("truncated 32-bit value");
        value = read_le32(&data[pos]);
        pos += 4;
        return true;
    }

    bool read_string(size_t size, std::string& value)
    {
        if (size > 16 * 1024 * 1024 || size > data.size() - pos)
            return fail("truncated or oversized string");
        if (size == 0)
        {
            value.clear();
            return true;
        }
        value.assign((const char*)&data[pos], size);
        pos += size;
        return true;
    }

    bool read_line(std::string& value)
    {
        const size_t begin = pos;
        while (pos < data.size() && data[pos] != '\n')
            pos++;
        if (pos == data.size() || pos - begin > 1024)
            return fail("truncated or oversized GLOBAL name");
        value.assign((const char*)&data[begin], pos - begin);
        pos++;
        return true;
    }

    bool push_int(int64_t value)
    {
        Value item;
        item.type = Value::Int;
        item.i = value;
        stack.push_back(item);
        return true;
    }

    bool memo_put(uint32_t index)
    {
        if (stack.empty() || index >= 1000000)
            return fail("invalid memo write");
        if (memo.size() <= index)
        {
            memo.resize(index + 1);
            memo_set.resize(index + 1);
        }
        memo[index] = stack.back();
        memo_set[index] = 1;
        return true;
    }

    bool memo_get(uint32_t index)
    {
        if (index >= memo.size() || !memo_set[index])
            return fail("invalid memo read");
        stack.push_back(memo[index]);
        return true;
    }

    bool make_tuple(size_t count)
    {
        if (stack.size() < count)
            return fail("tuple stack underflow");
        Value tuple;
        tuple.type = Value::Tuple;
        tuple.list.assign(stack.end() - count, stack.end());
        stack.erase(stack.end() - count, stack.end());
        stack.push_back(std::move(tuple));
        return true;
    }

    bool tuple_from_mark()
    {
        if (marks.empty() || marks.back() > stack.size())
            return fail("tuple has no MARK");
        const size_t mark = marks.back();
        marks.pop_back();
        Value tuple;
        tuple.type = Value::Tuple;
        tuple.list.assign(stack.begin() + mark, stack.end());
        stack.erase(stack.begin() + mark, stack.end());
        stack.push_back(std::move(tuple));
        return true;
    }

    bool set_item(bool multiple)
    {
        size_t begin;
        if (multiple)
        {
            if (marks.empty())
                return fail("SETITEMS has no MARK");
            begin = marks.back();
            marks.pop_back();
        }
        else
        {
            if (stack.size() < 3)
                return fail("SETITEM stack underflow");
            begin = stack.size() - 2;
        }
        if (begin == 0 || begin > stack.size() || (stack.size() - begin) % 2 != 0 || stack[begin - 1].type != Value::Dict)
            return fail("invalid dictionary item sequence");
        Value& dict = stack[begin - 1];
        for (size_t i = begin; i < stack.size(); i += 2)
        {
            if (stack[i].type != Value::String)
                return fail("dictionary key is not a string");
            if (!dict.dict.insert(std::make_pair(stack[i].s, stack[i + 1])).second)
                return fail("duplicate dictionary key " + stack[i].s);
        }
        stack.erase(stack.begin() + begin, stack.end());
        return true;
    }

    bool append(bool multiple)
    {
        size_t begin;
        if (multiple)
        {
            if (marks.empty())
                return fail("APPENDS has no MARK");
            begin = marks.back();
            marks.pop_back();
        }
        else
        {
            if (stack.size() < 2)
                return fail("APPEND stack underflow");
            begin = stack.size() - 1;
        }
        if (begin == 0 || stack[begin - 1].type != Value::List)
            return fail("APPEND target is not a list");
        stack[begin - 1].list.insert(stack[begin - 1].list.end(), stack.begin() + begin, stack.end());
        stack.erase(stack.begin() + begin, stack.end());
        return true;
    }

    bool persistent_storage()
    {
        if (stack.empty() || stack.back().type != Value::Tuple)
            return fail("BINPERSID requires a tuple");
        Value id = std::move(stack.back());
        stack.pop_back();
        if (id.list.size() != 5 || id.list[0].type != Value::String || id.list[0].s != "storage" ||
            id.list[1].type != Value::Global || id.list[2].type != Value::String ||
            id.list[3].type != Value::String || id.list[3].s != "cpu" || id.list[4].type != Value::Int || id.list[4].i < 0)
            return fail("unsupported persistent storage id");
        const int dtype = storage_scalar_type(id.list[1].s);
        if (!dtype)
            return fail("unsupported storage type " + id.list[1].s);

        Value storage;
        storage.type = Value::Storage;
        storage.tensor.dtype = dtype;
        storage.tensor.device = "cpu";
        storage.tensor.device_index = -1;
        storage.tensor.layout = 7;
        storage.s = storage_prefix + "data/" + id.list[2].s;
        storage.storage_elements = (uint64_t)id.list[4].i;
        stack.push_back(std::move(storage));
        return true;
    }

    bool reduce()
    {
        if (stack.size() < 2 || stack.back().type != Value::Tuple || stack[stack.size() - 2].type != Value::Global)
            return fail("invalid REDUCE operands");
        Value args = std::move(stack.back());
        stack.pop_back();
        const std::string callable = stack.back().s;
        stack.pop_back();

        if (callable == "collections OrderedDict")
        {
            if (!args.list.empty())
                return fail("OrderedDict constructor arguments are unsupported");
            Value dict;
            dict.type = Value::Dict;
            stack.push_back(std::move(dict));
            return true;
        }
        if (callable == "torch._utils _rebuild_tensor_v2")
        {
            if (args.list.size() != 6 || args.list[0].type != Value::Storage || args.list[1].type != Value::Int ||
                args.list[2].type != Value::Tuple || args.list[3].type != Value::Tuple || args.list[4].type != Value::Bool ||
                args.list[2].list.size() != args.list[3].list.size() || args.list[1].i < 0)
                return fail("invalid _rebuild_tensor_v2 arguments");

            Value tensor = args.list[0];
            tensor.type = Value::Tensor;
            tensor.tensor.requires_grad = args.list[4].b;
            tensor.tensor.storage_offset.value = args.list[1].i;
            for (size_t i = 0; i < args.list[2].list.size(); i++)
            {
                if (args.list[2].list[i].type != Value::Int || args.list[3].list[i].type != Value::Int)
                    return fail("tensor size or stride is not an integer");
                Pt2SymInt size;
                Pt2SymInt stride;
                size.value = args.list[2].list[i].i;
                stride.value = args.list[3].list[i].i;
                tensor.tensor.sizes.push_back(size);
                tensor.tensor.strides.push_back(stride);
            }
            stack.push_back(std::move(tensor));
            return true;
        }
        if (callable == "torch._utils _rebuild_parameter_with_state")
        {
            if (args.list.size() != 4 || args.list[0].type != Value::Tensor || args.list[1].type != Value::Bool)
                return fail("invalid _rebuild_parameter_with_state arguments");
            Value tensor = args.list[0];
            tensor.is_parameter = true;
            tensor.tensor.requires_grad = args.list[1].b;
            stack.push_back(std::move(tensor));
            return true;
        }
        return fail("unsupported REDUCE callable " + callable);
    }

    bool execute(unsigned char opcode)
    {
        if (opcode == '}')
        {
            Value value;
            value.type = Value::Dict;
            stack.push_back(value);
            return true;
        }
        if (opcode == ']')
        {
            Value value;
            value.type = Value::List;
            stack.push_back(value);
            return true;
        }
        if (opcode == ')')
            return make_tuple(0);
        if (opcode == '(')
        {
            marks.push_back(stack.size());
            return true;
        }
        if (opcode == 'N')
        {
            stack.push_back(Value());
            return true;
        }
        if (opcode == 0x88 || opcode == 0x89)
        {
            Value value;
            value.type = Value::Bool;
            value.b = opcode == 0x88;
            stack.push_back(value);
            return true;
        }
        if (opcode == 'K')
        {
            if (pos == data.size())
                return fail("truncated BININT1");
            return push_int(data[pos++]);
        }
        if (opcode == 'M')
        {
            if (data.size() - pos < 2)
                return fail("truncated BININT2");
            const uint16_t value = (uint16_t)data[pos] | ((uint16_t)data[pos + 1] << 8);
            pos += 2;
            return push_int(value);
        }
        if (opcode == 'J')
        {
            uint32_t value;
            if (!read_u32(value))
                return false;
            return push_int((int32_t)value);
        }
        if (opcode == 0x8a || opcode == 0x8b)
        {
            uint32_t size = 0;
            if (opcode == 0x8a)
            {
                if (pos == data.size())
                    return fail("truncated LONG1");
                size = data[pos++];
            }
            else if (!read_u32(size))
            {
                return false;
            }
            if (size == 0)
                return push_int(0);
            if (size > 8 || size > data.size() - pos)
                return fail("oversized or truncated pickle integer");
            uint64_t value = 0;
            for (uint32_t i = 0; i < size; i++)
                value |= (uint64_t)data[pos + i] << (i * 8);
            if (size < 8 && (data[pos + size - 1] & 0x80))
                value |= UINT64_MAX << (size * 8);
            pos += size;
            return push_int((int64_t)value);
        }
        if (opcode == 'X')
        {
            uint32_t size;
            Value value;
            value.type = Value::String;
            if (!read_u32(size) || !read_string(size, value.s))
                return false;
            stack.push_back(std::move(value));
            return true;
        }
        if (opcode == 'c')
        {
            std::string module;
            std::string name;
            if (!read_line(module) || !read_line(name))
                return false;
            Value value;
            value.type = Value::Global;
            value.s = module + " " + name;
            if (value.s != "collections OrderedDict" && value.s != "torch._utils _rebuild_tensor_v2" &&
                value.s != "torch._utils _rebuild_parameter_with_state" && !storage_scalar_type(value.s))
                return fail("unsupported GLOBAL " + value.s);
            stack.push_back(std::move(value));
            return true;
        }
        if (opcode == 'q')
        {
            if (pos == data.size())
                return fail("truncated BINPUT");
            return memo_put(data[pos++]);
        }
        if (opcode == 'r')
        {
            uint32_t index;
            return read_u32(index) && memo_put(index);
        }
        if (opcode == 'h')
        {
            if (pos == data.size())
                return fail("truncated BINGET");
            return memo_get(data[pos++]);
        }
        if (opcode == 'j')
        {
            uint32_t index;
            return read_u32(index) && memo_get(index);
        }
        if (opcode == 't')
            return tuple_from_mark();
        if (opcode == 0x85)
            return make_tuple(1);
        if (opcode == 0x86)
            return make_tuple(2);
        if (opcode == 0x87)
            return make_tuple(3);
        if (opcode == 'Q')
            return persistent_storage();
        if (opcode == 'R')
            return reduce();
        if (opcode == 's')
            return set_item(false);
        if (opcode == 'u')
            return set_item(true);
        if (opcode == 'a')
            return append(false);
        if (opcode == 'e')
            return append(true);
        return fail("unsupported pickle opcode " + std::to_string(opcode));
    }

    const std::vector<unsigned char>& data;
    std::string storage_prefix;
    size_t pos;
    size_t operations;
    std::vector<Value> stack;
    std::vector<size_t> marks;
    std::vector<Value> memo;
    std::vector<unsigned char> memo_set;
    std::string error;
};

class Pt2WeightsLoader
{
public:
    Pt2WeightsLoader(Pt2ArchiveReader& _archive, const Pt2Program& _program, Pt2Weights& _weights)
        : archive(_archive), program(_program), weights(_weights)
    {
    }

    int load()
    {
        std::map<std::string, Pt2LoadedTensor> tensors;
        if (archive.records.find("data/weights/model_weights_config.json") != archive.records.end())
        {
            if (load_raw_config("data/weights/model_weights_config.json", "data/weights/", tensors) != 0 ||
                load_raw_config("data/constants/model_constants_config.json", "data/constants/", tensors) != 0)
                return -1;
        }
        else
        {
            const char* weights_record = archive.container_kind == Pt2ContainerLegacyExportedProgram ? "serialized_state_dict.pt" : "data/weights/model.pt";
            const char* constants_record = archive.container_kind == Pt2ContainerLegacyExportedProgram ? "serialized_constants.pt" : "data/constants/model.pt";
            if (load_pickle_archive(weights_record, tensors) != 0 || load_pickle_archive(constants_record, tensors) != 0)
                return -1;
        }
        return bind(tensors);
    }

private:
    int fail(const std::string& message)
    {
        weights.error = message;
        return -1;
    }

    const Pt2JsonValue* field(const Pt2JsonValue& object, const char* name, const std::string& path)
    {
        if (object.type != Pt2JsonValue::Object)
        {
            fail(path + " at json offset " + std::to_string(object.offset) + ": expected object");
            return 0;
        }
        std::map<std::string, Pt2JsonValue>::const_iterator it = object.object.find(name);
        if (it == object.object.end())
        {
            fail(path + "." + name + " at json offset " + std::to_string(object.offset) + ": missing required field");
            return 0;
        }
        return &it->second;
    }

    bool string_value(const Pt2JsonValue& value, const std::string& path, std::string& result)
    {
        if (value.type != Pt2JsonValue::String)
        {
            fail(path + " at json offset " + std::to_string(value.offset) + ": expected string");
            return false;
        }
        result = value.value;
        return true;
    }

    bool bool_value(const Pt2JsonValue& value, const std::string& path, bool& result)
    {
        if (value.type != Pt2JsonValue::Bool)
        {
            fail(path + " at json offset " + std::to_string(value.offset) + ": expected boolean");
            return false;
        }
        result = value.boolean;
        return true;
    }

    int read_byteorder(const std::string& record, std::string& byteorder)
    {
        std::vector<unsigned char> data;
        if (archive.read_file(record, data) != 0)
            return fail(archive.error);
        byteorder = trim_ascii_whitespace(data);
        if (byteorder != "little" && byteorder != "big")
            return fail("unsupported byte order in " + record);
        return 0;
    }

    int load_raw_config(const std::string& config_record, const std::string& data_prefix,
                        std::map<std::string, Pt2LoadedTensor>& tensors)
    {
        std::vector<unsigned char> data;
        if (archive.read_file(config_record, data) != 0)
            return fail(archive.error);
        Pt2JsonValue root;
        std::string json_error;
        if (parse_pt2_json(data.empty() ? 0 : &data[0], data.size(), root, json_error) != 0)
            return fail(config_record + ": " + json_error);
        const Pt2JsonValue* config = field(root, "config", "$");
        if (!config || config->type != Pt2JsonValue::Object)
            return config ? fail("$.config at json offset " + std::to_string(config->offset) + ": expected object") : -1;
        if (root.object.size() != 1)
        {
            for (std::map<std::string, Pt2JsonValue>::const_iterator it = root.object.begin(); it != root.object.end(); ++it)
            {
                if (it->first != "config")
                    return fail("$." + it->first + " at json offset " + std::to_string(it->second.offset) + ": unknown field");
            }
        }

        std::string byteorder;
        if (read_byteorder("byteorder", byteorder) != 0)
            return -1;
        std::map<std::string, std::vector<unsigned char> > storage_cache;
        for (std::map<std::string, Pt2JsonValue>::const_iterator it = config->object.begin(); it != config->object.end(); ++it)
        {
            const std::string path = "$.config." + it->first;
            const Pt2JsonValue* path_name = field(it->second, "path_name", path);
            const Pt2JsonValue* is_param = field(it->second, "is_param", path);
            const Pt2JsonValue* use_pickle = field(it->second, "use_pickle", path);
            const Pt2JsonValue* tensor_meta = field(it->second, "tensor_meta", path);
            std::string name;
            bool parameter;
            bool pickle;
            Pt2Tensor meta;
            if (!path_name || !is_param || !use_pickle || !tensor_meta || !string_value(*path_name, path + ".path_name", name) ||
                !bool_value(*is_param, path + ".is_param", parameter) || !bool_value(*use_pickle, path + ".use_pickle", pickle))
                return -1;
            if (name.empty() || name.find('/') != std::string::npos || name.find('\\') != std::string::npos)
                return fail(path + ".path_name: unsafe payload record name");
            if (pickle)
                return fail(path + ": pickled raw payload is unsupported");
            if (tensor_meta->type == Pt2JsonValue::Null)
                return fail(path + ": custom object payload is unsupported");
            if (decode_pt2_tensor_meta(*tensor_meta, path + ".tensor_meta", meta, json_error) != 0)
                return fail(config_record + ": " + json_error);
            for (std::map<std::string, Pt2JsonValue>::const_iterator field_it = it->second.object.begin(); field_it != it->second.object.end(); ++field_it)
            {
                if (field_it->first != "path_name" && field_it->first != "is_param" && field_it->first != "use_pickle" && field_it->first != "tensor_meta")
                    return fail(path + "." + field_it->first + " at json offset " + std::to_string(field_it->second.offset) + ": unknown field");
            }

            const std::string record = data_prefix + name;
            std::map<std::string, std::vector<unsigned char> >::iterator storage_it = storage_cache.find(record);
            if (storage_it == storage_cache.end())
            {
                std::vector<unsigned char> storage;
                if (archive.read_file(record, storage) != 0)
                    return fail(archive.error);
                storage_it = storage_cache.insert(std::make_pair(record, std::move(storage))).first;
            }
            int attribute_type;
            size_t element_size;
            size_t component_size;
            if (scalar_type_info(meta.dtype, attribute_type, element_size, component_size) != 0 || storage_it->second.size() % element_size != 0)
                return fail(path + ": invalid storage dtype or byte size");

            Pt2LoadedTensor loaded;
            loaded.meta = meta;
            loaded.is_parameter = parameter;
            if (make_attribute(meta, storage_it->second, storage_it->second.size() / element_size, byteorder, loaded.attribute, json_error) != 0)
                return fail(path + ": " + json_error);
            if (!tensors.insert(std::make_pair(it->first, std::move(loaded))).second)
                return fail("duplicate payload FQN " + it->first);
        }
        return 0;
    }

    int load_pickle_archive(const std::string& record, std::map<std::string, Pt2LoadedTensor>& tensors)
    {
        std::vector<unsigned char> nested_data;
        if (archive.read_file(record, nested_data) != 0)
            return fail(archive.error);
        StoreZipReader nested;
        if (nested.open(nested_data.empty() ? 0 : &nested_data[0], nested_data.size()) != 0)
            return fail(record + ": " + nested.error);

        const std::vector<std::string> names = nested.get_names();
        std::string pickle_record;
        for (size_t i = 0; i < names.size(); i++)
        {
            if (names[i].size() >= 8 && names[i].compare(names[i].size() - 8, 8, "data.pkl") == 0)
            {
                if (!pickle_record.empty())
                    return fail(record + ": multiple data.pkl records");
                pickle_record = names[i];
            }
        }
        if (pickle_record.empty())
            return fail(record + ": missing data.pkl");
        const std::string prefix = pickle_record.substr(0, pickle_record.size() - 8);

        std::vector<unsigned char> pickle_data;
        std::vector<unsigned char> byteorder_data;
        if (nested.read_file(pickle_record, pickle_data) != 0 || nested.read_file(prefix + "byteorder", byteorder_data) != 0)
            return fail(record + ": " + nested.error);
        const std::string byteorder = trim_ascii_whitespace(byteorder_data);
        if (byteorder != "little" && byteorder != "big")
            return fail(record + ": unsupported byte order");

        std::map<std::string, Pt2PickleReader::Value> pickle_tensors;
        Pt2PickleReader parser(pickle_data, prefix);
        std::string pickle_error;
        if (parser.parse(pickle_tensors, pickle_error) != 0)
            return fail(record + ": " + pickle_error);

        std::map<std::string, std::vector<unsigned char> > storage_cache;
        for (std::map<std::string, Pt2PickleReader::Value>::const_iterator it = pickle_tensors.begin(); it != pickle_tensors.end(); ++it)
        {
            std::map<std::string, std::vector<unsigned char> >::iterator storage_it = storage_cache.find(it->second.s);
            if (storage_it == storage_cache.end())
            {
                std::vector<unsigned char> storage;
                if (nested.read_file(it->second.s, storage) != 0)
                    return fail(record + ": " + nested.error);
                storage_it = storage_cache.insert(std::make_pair(it->second.s, std::move(storage))).first;
            }
            Pt2LoadedTensor loaded;
            loaded.meta = it->second.tensor;
            loaded.is_parameter = it->second.is_parameter;
            if (make_attribute(loaded.meta, storage_it->second, it->second.storage_elements, byteorder, loaded.attribute, pickle_error) != 0)
                return fail(record + ": tensor " + it->first + ": " + pickle_error);
            if (!tensors.insert(std::make_pair(it->first, std::move(loaded))).second)
                return fail("duplicate payload FQN " + it->first);
        }
        return 0;
    }

    int bind(std::map<std::string, Pt2LoadedTensor>& tensors)
    {
        std::set<std::string> used;
        for (size_t i = 0; i < program.input_specs.size(); i++)
        {
            const Pt2InputSpec& spec = program.input_specs[i];
            if (spec.kind != Pt2InputSpec::Parameter && spec.kind != Pt2InputSpec::Buffer && spec.kind != Pt2InputSpec::TensorConstant)
                continue;
            std::map<std::string, Pt2LoadedTensor>::iterator tensor_it = tensors.find(spec.target);
            if (tensor_it == tensors.end())
                return fail("missing payload for " + spec.target);
            if (!used.insert(spec.target).second)
                return fail("payload is bound more than once: " + spec.target);
            if ((spec.kind == Pt2InputSpec::Parameter) != tensor_it->second.is_parameter)
                return fail("parameter classification mismatch for " + spec.target);
            std::map<std::string, Pt2Tensor>::const_iterator graph_meta = program.tensors.find(spec.arg.s);
            if (graph_meta == program.tensors.end() || !same_tensor_meta(graph_meta->second, tensor_it->second.meta))
                return fail("payload tensor metadata does not match graph input " + spec.arg.s);

            Pt2Weight weight;
            weight.kind = spec.kind;
            weight.attribute = std::move(tensor_it->second.attribute);
            weights.values[spec.target] = std::move(weight);
        }
        if (used.size() != tensors.size())
            return fail("payload contains a tensor that is not present in the graph signature");
        return 0;
    }

    Pt2ArchiveReader& archive;
    const Pt2Program& program;
    Pt2Weights& weights;
};

int load_pt2_weights(Pt2ArchiveReader& archive, const Pt2Program& program, Pt2Weights& weights)
{
    weights.values.clear();
    weights.error.clear();
    Pt2WeightsLoader loader(archive, program, weights);
    return loader.load();
}

} // namespace pnnx
