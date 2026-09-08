// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "load_exportedprogram_legacy.h"

#include "storezip.h"

#include <stdio.h>
#include <string.h>

#include <stdint.h>

#include <algorithm>
#include <map>
#include <string>
#include <vector>

namespace pnnx {

// ---------------------------------------------------------------------------
// minimal in-memory zip reader for the nested torch.save container.
//
// torch.save writes a standard (uncompressed) zip: a data.pkl pickle plus one
// storage shard per externalized storage (archive/data/<n>), a byteorder
// marker and version records. only stored (compression 0) entries are
// supported, which is all torch.save produces for these containers.
// ---------------------------------------------------------------------------

static uint16_t read_le16(const char* p)
{
    return (uint16_t)(unsigned char)p[0] | ((uint16_t)(unsigned char)p[1] << 8);
}

static uint32_t read_le32(const char* p)
{
    return (uint32_t)(unsigned char)p[0] | ((uint32_t)(unsigned char)p[1] << 8) | ((uint32_t)(unsigned char)p[2] << 16) | ((uint32_t)(unsigned char)p[3] << 24);
}

static uint64_t read_le64(const char* p)
{
    uint64_t v = 0;
    for (int i = 0; i < 8; i++)
        v |= (uint64_t)(unsigned char)p[i] << (i * 8);
    return v;
}

// read one stored entry (by exact name) out of a zip held in memory
static int memzip_read_entry(const std::vector<char>& z, const std::string& want, std::vector<char>& out)
{
    const char* data = z.data();
    const size_t size = z.size();

    // a zip is at least a 22-byte EOCD record; anything shorter (or a payload
    // truncated to a few bytes) would walk past the vector below
    if (size < 22)
    {
        fprintf(stderr, "legacy weights: nested zip payload too short (%zu bytes)\n", size);
        return -1;
    }
    // locate the end of central directory record by scanning backwards (the
    // eocd is at most 65557 bytes from the end: 22 fixed + 65535 comment)
    size_t eocd_pos = (size_t)-1;
    {
        const size_t minpos = size >= 65557 ? size - 65557 : 0;
        for (size_t pos = size >= 22 ? size - 22 : 0; pos >= minpos; pos--)
        {
            if (read_le32(data + pos) != 0x06054b50)
            {
                if (pos == 0)
                    break;
                continue;
            }
            const uint16_t comment_length = read_le16(data + pos + 20);
            if (pos + 22 + comment_length != size)
            {
                if (pos == 0)
                    break;
                continue;
            }
            eocd_pos = pos;
            break;
        }
    }
    if (eocd_pos == (size_t)-1)
    {
        fprintf(stderr, "legacy weights: end of central directory not found\n");
        return -1;
    }

    uint32_t cd_records = read_le16(data + eocd_pos + 10);
    uint32_t cd_offset = read_le32(data + eocd_pos + 16);
    uint32_t cd_size = read_le32(data + eocd_pos + 12);
    if (cd_records == 0xffff || cd_offset == 0xffffffff || cd_size == 0xffffffff)
    {
        fprintf(stderr, "legacy weights: zip64 nested container is not supported\n");
        return -1;
    }
    if (cd_offset + cd_size > eocd_pos)
    {
        fprintf(stderr, "legacy weights: invalid central directory\n");
        return -1;
    }

    size_t pos = cd_offset;
    const size_t cd_end = cd_offset + cd_size;
    while (pos + 46 <= cd_end)
    {
        if (read_le32(data + pos) != 0x02014b50)
        {
            fprintf(stderr, "legacy weights: bad central directory signature\n");
            return -1;
        }
        const uint16_t method = read_le16(data + pos + 10);
        const uint32_t compressed_size = read_le32(data + pos + 20);
        const uint32_t name_length = read_le16(data + pos + 28);
        const uint32_t extra_length = read_le16(data + pos + 30);
        const uint32_t comment_length = read_le16(data + pos + 32);
        const uint32_t lfh_offset = read_le32(data + pos + 42);
        if (name_length > cd_end - (pos + 46))
        {
            fprintf(stderr, "legacy weights: truncated file name\n");
            return -1;
        }
        const std::string name(data + pos + 46, name_length);

        if (name == want)
        {
            if (method != 0)
            {
                fprintf(stderr, "legacy weights: compressed entry %s is not supported\n", name.c_str());
                return -1;
            }
            if (compressed_size == 0xffffffff || lfh_offset == 0xffffffff)
            {
                fprintf(stderr, "legacy weights: zip64 entry %s is not supported\n", name.c_str());
                return -1;
            }
            // compute the data offset from the local file header (the central
            // directory sizes are reliable for stored entries)
            if (lfh_offset + 30 > size || read_le32(data + lfh_offset) != 0x04034b50)
            {
                fprintf(stderr, "legacy weights: bad local header for %s\n", name.c_str());
                return -1;
            }
            const uint32_t lname_length = read_le16(data + lfh_offset + 26);
            const uint32_t lextra_length = read_le16(data + lfh_offset + 28);
            const uint64_t data_offset = (uint64_t)lfh_offset + 30 + lname_length + lextra_length;
            if (data_offset + compressed_size > size)
            {
                fprintf(stderr, "legacy weights: truncated data for %s\n", name.c_str());
                return -1;
            }
            out.assign(data + data_offset, data + data_offset + compressed_size);
            return 0;
        }

        pos += 46 + name_length + extra_length + comment_length;
    }

    fprintf(stderr, "legacy weights: entry %s not found\n", want.c_str());
    return -1;
}

// ---------------------------------------------------------------------------
// restricted pickle (protocol 2) reader for a torch.save state dict.
//
// torch.save pickles the state dict as a dict { fqn : tensor }, where each
// tensor is
//   _rebuild_parameter_with_state(_rebuild_tensor_v2(storage, offset,
//       sizes, strides, requires_grad, OrderedDict()), True, ...)
// or the bare _rebuild_tensor_v2(...) for non-parameter tensors, and the
// storage argument is a BINPERSID reference to a shard
//   ('storage', 'torch FloatStorage', '<id>', '<device>', <numel>)
// that this reader maps to archive/data/<id>. only the constructs torch.save
// emits are supported; anything else fails with a clear message instead of
// silently mis-decoding.
// ---------------------------------------------------------------------------

namespace pickle {

struct Value
{
    enum Type
    {
        None,
        Bool,
        Int,
        Str,
        Global,
        Tuple,
        Dict,
        Storage,
        Tensor
    };

    Value()
        : type(None), b(false), i(0), is_parameter(false)
    {
    }

    Type type;
    bool b;
    int64_t i;
    std::string s;
    std::vector<Value> list; // tuple contents
    std::map<std::string, Value> dict;
    std::string shard; // storage shard id
    bool is_parameter;
};

// torch storage class name -> serde ScalarType
static int storage_scalar_type(const std::string& cls)
{
    if (cls == "torch ByteStorage") return 1;           // uint8
    if (cls == "torch CharStorage") return 2;           // int8
    if (cls == "torch ShortStorage") return 3;          // int16
    if (cls == "torch IntStorage") return 4;            // int32
    if (cls == "torch LongStorage") return 5;           // int64
    if (cls == "torch HalfStorage") return 6;           // float16
    if (cls == "torch FloatStorage") return 7;          // float32
    if (cls == "torch DoubleStorage") return 8;         // float64
    if (cls == "torch ComplexFloatStorage") return 10;  // complex64
    if (cls == "torch ComplexDoubleStorage") return 11; // complex128
    if (cls == "torch BoolStorage") return 12;          // bool
    if (cls == "torch BFloat16Storage") return 13;      // bfloat16
    return 0;
}

class Reader
{
public:
    Reader(const std::vector<char>& _data, const std::string& _storage_prefix)
        : data(_data), storage_prefix(_storage_prefix), pos(0)
    {
    }

    // decode into tensors[fqn] = Value (Storage-backed Tensor)
    int parse(std::map<std::string, Value>& tensors)
    {
        if (data.size() < 2 || (unsigned char)data[0] != 0x80 || (unsigned char)data[1] != 2)
        {
            fprintf(stderr, "legacy weights: only pickle protocol 2 is supported\n");
            return -1;
        }
        pos = 2;

        while (pos < data.size())
        {
            const unsigned char opcode = (unsigned char)data[pos++];
            if (opcode == '.')
            {
                if (pos != data.size() || !marks.empty() || stack.size() != 1 || stack[0].type != Value::Dict)
                {
                    fprintf(stderr, "legacy weights: invalid pickle end state\n");
                    return -1;
                }
                for (std::map<std::string, Value>::const_iterator it = stack[0].dict.begin(); it != stack[0].dict.end(); ++it)
                {
                    if (it->second.type != Value::Tensor)
                    {
                        fprintf(stderr, "legacy weights: state dict value for '%s' is not a tensor\n", it->first.c_str());
                        return -1;
                    }
                    tensors[it->first] = it->second;
                }
                return 0;
            }
            if (!execute(opcode))
                return -1;
        }
        fprintf(stderr, "legacy weights: pickle has no end marker\n");
        return -1;
    }

private:
    bool fail(const std::string& message)
    {
        if (error.empty())
        {
            fprintf(stderr, "legacy weights: pickle at offset %d: %s\n", (int)pos, message.c_str());
            error = message;
        }
        return false;
    }

    void push_value(const Value& v)
    {
        stack.push_back(v);
    }

    bool read_u8(uint8_t& v)
    {
        if (pos >= data.size())
            return fail("truncated 8-bit value");
        v = (unsigned char)data[pos++];
        return true;
    }

    bool read_u16(uint16_t& v)
    {
        if (data.size() - pos < 2)
            return fail("truncated 16-bit value");
        v = read_le16(&data[pos]);
        pos += 2;
        return true;
    }

    bool read_u32(uint32_t& v)
    {
        if (data.size() - pos < 4)
            return fail("truncated 32-bit value");
        v = read_le32(&data[pos]);
        pos += 4;
        return true;
    }

    bool read_bytes(size_t n, std::string& out)
    {
        if (n > data.size() - pos)
            return fail("truncated byte string");
        out.assign(data.data() + pos, n);
        pos += n;
        return true;
    }

    bool read_line(std::string& out)
    {
        const size_t begin = pos;
        while (pos < data.size() && data[pos] != '\n')
            pos++;
        if (pos == data.size())
            return fail("truncated line");
        out.assign(data.data() + begin, pos - begin);
        pos++;
        return true;
    }

    bool push_int(int64_t v)
    {
        Value item;
        item.type = Value::Int;
        item.i = v;
        push_value(item);
        return true;
    }

    bool memo_put(uint32_t index)
    {
        // memo indexes come straight from the pickle stream; cap them so a
        // hostile 'r' (u32 index) cannot drive memo.resize() to gigabytes (OOM)
        if (stack.empty() || index >= 65536)
            return fail("invalid memo write");
        if (memo.size() <= index)
            memo.resize((size_t)index + 1);
        memo[index] = stack.back();
        memo_set.resize(memo.size(), 0);
        memo_set[index] = 1;
        return true;
    }

    bool memo_get(uint32_t index)
    {
        if (index >= memo.size() || !memo_set[index])
            return fail("invalid memo read");
        push_value(memo[index]);
        return true;
    }

    bool make_tuple(size_t count)
    {
        if (stack.size() < count)
            return fail("tuple underflow");
        Value tuple;
        tuple.type = Value::Tuple;
        tuple.list.assign(stack.end() - count, stack.end());
        stack.erase(stack.end() - count, stack.end());
        push_value(tuple);
        return true;
    }

    bool tuple_from_mark()
    {
        if (marks.empty() || marks.back() > stack.size())
            return fail("tuple without MARK");
        const size_t count = stack.size() - marks.back();
        marks.pop_back();
        return make_tuple(count);
    }

    bool set_items(bool multiple)
    {
        size_t begin;
        if (multiple)
        {
            if (marks.empty())
                return fail("SETITEMS without MARK");
            begin = marks.back();
            marks.pop_back();
        }
        else
        {
            if (stack.size() < 3)
                return fail("SETITEM underflow");
            begin = stack.size() - 2;
        }
        if (begin == 0 || stack.size() < begin || (stack.size() - begin) % 2 != 0 || stack[begin - 1].type != Value::Dict)
            return fail("invalid dict item sequence");
        Value& dict = stack[begin - 1];
        for (size_t i = begin; i < stack.size(); i += 2)
        {
            if (stack[i].type != Value::Str)
                return fail("dict key is not a string");
            if (!dict.dict.insert(std::make_pair(stack[i].s, stack[i + 1])).second)
                return fail("duplicate dict key " + stack[i].s);
        }
        stack.erase(stack.begin() + begin, stack.end());
        return true;
    }

    bool persistent_storage()
    {
        if (stack.empty() || stack.back().type != Value::Tuple)
            return fail("BINPERSID requires a tuple");
        Value id = stack.back();
        stack.pop_back();
        // ('storage', 'torch FloatStorage', '<id>', '<device>', <numel>)
        if (id.list.size() != 5 || id.list[0].type != Value::Str || id.list[0].s != "storage" || id.list[1].type != Value::Global || id.list[2].type != Value::Str || id.list[3].type != Value::Str || id.list[4].type != Value::Int || id.list[4].i < 0)
            return fail("unsupported persistent storage id");
        if (!storage_scalar_type(id.list[1].s))
            return fail("unsupported storage type " + id.list[1].s);

        Value storage;
        storage.type = Value::Storage;
        storage.s = storage_prefix + "data/" + id.list[2].s;
        push_value(storage);
        return true;
    }

    bool reduce()
    {
        if (stack.size() < 2 || stack.back().type != Value::Tuple || stack[stack.size() - 2].type != Value::Global)
            return fail("invalid REDUCE");
        Value args = stack.back();
        stack.pop_back();
        const std::string callable = stack.back().s;
        stack.pop_back();

        if (callable == "collections OrderedDict")
        {
            if (!args.list.empty())
                return fail("OrderedDict with arguments is not supported");
            Value dict;
            dict.type = Value::Dict;
            push_value(dict);
            return true;
        }
        if (callable == "torch._utils _rebuild_tensor_v2")
        {
            if (args.list.size() != 6 || args.list[0].type != Value::Storage || args.list[1].type != Value::Int || args.list[2].type != Value::Tuple || args.list[3].type != Value::Tuple || args.list[2].list.size() != args.list[3].list.size() || args.list[1].i < 0)
                return fail("invalid _rebuild_tensor_v2 arguments");
            Value tensor = args.list[0];
            tensor.type = Value::Tensor;
            tensor.is_parameter = false;
            push_value(tensor);
            return true;
        }
        if (callable == "torch._utils _rebuild_parameter_with_state")
        {
            if (args.list.size() != 4 || args.list[0].type != Value::Tensor || args.list[1].type != Value::Bool)
                return fail("invalid _rebuild_parameter_with_state arguments");
            Value tensor = args.list[0];
            tensor.is_parameter = true;
            push_value(tensor);
            return true;
        }
        if (callable == "torch._utils _rebuild_parameter")
        {
            // older pickle variant used by some torch releases: takes the
            // tensor plus a requires_grad flag, optionally followed by state;
            // both forms are accepted (the state is not needed here)
            if (args.list.size() < 2 || args.list.size() > 3 || args.list[0].type != Value::Tensor || args.list[1].type != Value::Bool)
                return fail("invalid _rebuild_parameter arguments");
            Value tensor = args.list[0];
            tensor.is_parameter = true;
            push_value(tensor);
            return true;
        }
        return fail("unsupported REDUCE callable " + callable);
    }

    bool execute(unsigned char opcode)
    {
        if (opcode == '}')
        {
            Value v;
            v.type = Value::Dict;
            push_value(v);
            return true;
        }
        if (opcode == ']')
            return fail("unsupported pickle list construct");
        if (opcode == ')')
            return make_tuple(0);
        if (opcode == '(')
        {
            marks.push_back(stack.size());
            return true;
        }
        if (opcode == 'N')
        {
            push_value(Value());
            return true;
        }
        if (opcode == 0x88 || opcode == 0x89)
        {
            Value v;
            v.type = Value::Bool;
            v.b = opcode == 0x88;
            push_value(v);
            return true;
        }
        if (opcode == 'K')
        {
            uint8_t v;
            if (!read_u8(v))
                return false;
            return push_int(v);
        }
        if (opcode == 'M')
        {
            uint16_t v;
            if (!read_u16(v))
                return false;
            return push_int(v);
        }
        if (opcode == 'J')
        {
            uint32_t v;
            if (!read_u32(v))
                return false;
            return push_int((int32_t)v);
        }
        if (opcode == 0x8a || opcode == 0x8b)
        {
            uint32_t size = 0;
            if (opcode == 0x8a)
            {
                uint8_t s8;
                if (!read_u8(s8))
                    return false;
                size = s8;
            }
            else if (!read_u32(size))
            {
                return false;
            }
            if (size == 0)
                return push_int(0);
            if (size > 8 || size > data.size() - pos)
                return fail("oversized or truncated long integer");
            uint64_t v = 0;
            for (uint32_t i = 0; i < size; i++)
                v |= (uint64_t)(unsigned char)data[pos + i] << (i * 8);
            if (size < 8 && (data[pos + size - 1] & 0x80))
                v |= UINT64_MAX << (size * 8);
            pos += size;
            return push_int((int64_t)v);
        }
        if (opcode == 'X')
        {
            uint32_t size;
            if (!read_u32(size))
                return false;
            Value v;
            v.type = Value::Str;
            if (!read_bytes(size, v.s))
                return false;
            push_value(v);
            return true;
        }
        if (opcode == 'c')
        {
            std::string module;
            std::string name;
            if (!read_line(module) || !read_line(name))
                return false;
            Value v;
            v.type = Value::Global;
            v.s = module + " " + name;
            if (v.s != "collections OrderedDict" && v.s != "torch._utils _rebuild_tensor_v2" && v.s != "torch._utils _rebuild_parameter_with_state" && v.s != "torch._utils _rebuild_parameter" && !storage_scalar_type(v.s))
                return fail("unsupported GLOBAL " + v.s);
            push_value(v);
            return true;
        }
        if (opcode == 'q')
        {
            uint8_t index;
            if (!read_u8(index))
                return false;
            return memo_put(index);
        }
        if (opcode == 'r')
        {
            uint32_t index;
            if (!read_u32(index))
                return false;
            return memo_put(index);
        }
        if (opcode == 'h')
        {
            uint8_t index;
            if (!read_u8(index))
                return false;
            return memo_get(index);
        }
        if (opcode == 'j')
        {
            uint32_t index;
            if (!read_u32(index))
                return false;
            return memo_get(index);
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
            return set_items(false);
        if (opcode == 'u')
            return set_items(true);
        return fail("unsupported pickle opcode " + std::to_string(opcode));
    }

    const std::vector<char>& data;
    std::string storage_prefix;
    size_t pos;
    std::vector<Value> stack;
    std::vector<size_t> marks;
    std::vector<Value> memo;
    std::vector<unsigned char> memo_set;
    std::string error;
};

} // namespace pickle

// find the directory prefix that holds data.pkl inside a nested torch.save zip
static int memzip_find_data_pkl_prefix(const std::vector<char>& bytes, std::string& prefix)
{
    const char* p = bytes.data();
    const size_t sz = bytes.size();

    // a zip is at least a 22-byte EOCD record; anything shorter (or a payload
    // truncated to a few bytes) would walk past the vector below
    if (sz < 22)
    {
        fprintf(stderr, "legacy weights: nested zip payload too short (%zu bytes)\n", sz);
        return -1;
    }
    size_t eocd_pos = (size_t)-1;
    const size_t minpos = sz >= 65557 ? sz - 65557 : 0;
    for (size_t q = sz >= 22 ? sz - 22 : 0; q >= minpos; q--)
    {
        if (read_le32(p + q) != 0x06054b50)
        {
            if (q == 0)
                break;
            continue;
        }
        if (q + 22 + read_le16(p + q + 20) == sz)
        {
            eocd_pos = q;
            break;
        }
        if (q == 0)
            break;
    }
    if (eocd_pos == (size_t)-1)
    {
        fprintf(stderr, "legacy weights: no zip end record\n");
        return -1;
    }
    const uint32_t cd_offset = read_le32(p + eocd_pos + 16);
    const uint32_t cd_size = read_le32(p + eocd_pos + 12);
    if (cd_offset == 0xffffffff || cd_size == 0xffffffff)
    {
        fprintf(stderr, "legacy weights: zip64 nested container is not supported\n");
        return -1;
    }
    size_t pos = cd_offset;
    const size_t cd_end = cd_offset + cd_size;
    while (pos + 46 <= cd_end)
    {
        if (read_le32(p + pos) != 0x02014b50)
            break;
        const uint32_t name_length = read_le16(p + pos + 28);
        const uint32_t extra_length = read_le16(p + pos + 30);
        const uint32_t comment_length = read_le16(p + pos + 32);
        if (name_length > cd_end - (pos + 46))
            break;
        const std::string nm(p + pos + 46, name_length);
        const char* suffix = "data.pkl";
        const size_t suffix_len = 8;
        if (nm.size() >= suffix_len && nm.compare(nm.size() - suffix_len, suffix_len, suffix) == 0)
        {
            prefix = nm.substr(0, nm.size() - suffix_len);
            return 0;
        }
        pos += 46 + name_length + extra_length + comment_length;
    }
    fprintf(stderr, "legacy weights: no data.pkl record\n");
    return -1;
}

// a decoded legacy .pt payload: the container bytes stay alive so storage
// shards can be read on demand, plus the decoded state dict and its shard
// directory prefix
struct LegacyStateDict
{
    std::vector<char> bytes;
    std::string prefix;
    std::string tag; // outer record name, disambiguates shards across payloads
    std::map<std::string, pickle::Value> tensors;
};

// decode one legacy .pt record (a nested torch.save zip) into out.tensors.
// a missing record is not an error (serialized_constants.pt is absent when the
// model has no non-persistent buffers / tensor constants); missing payloads
// are caught later when binding against the signature.
static int load_legacy_state_dict(StoreZipReader& zip, const std::vector<std::string>& names,
                                  const std::string& record, LegacyStateDict& out)
{
    if (std::find(names.begin(), names.end(), record) == names.end())
    {
        out.bytes.clear();
        out.prefix.clear();
        out.tensors.clear();
        out.tag = record;
        return 0;
    }

    out.tag = record;
    uint64_t size = zip.get_file_size(record);
    out.bytes.resize((size_t)size);
    if (zip.read_file(record, out.bytes.data()) != 0)
        return -1;

    if (memzip_find_data_pkl_prefix(out.bytes, out.prefix) != 0)
        return -1;

    // byte order marker must agree with the reader's native little-endian
    {
        std::vector<char> bo;
        if (memzip_read_entry(out.bytes, out.prefix + "byteorder", bo) != 0)
            return -1;
        const std::string s(bo.begin(), bo.end());
        const std::string trimmed = s.find('\n') != std::string::npos ? s.substr(0, s.find('\n')) : s;
        if (trimmed != "little")
        {
            fprintf(stderr, "legacy weights: unsupported byte order '%s'\n", trimmed.c_str());
            return -1;
        }
    }

    std::vector<char> pkl;
    if (memzip_read_entry(out.bytes, out.prefix + "data.pkl", pkl) != 0)
        return -1;

    pickle::Reader reader(pkl, out.prefix);
    return reader.parse(out.tensors);
}

int pnnx_load_legacy_payloads(StoreZipReader& zip, const std::vector<std::string>& names,
                              const JsonValue& root,
                              std::map<std::string, std::pair<std::string, JsonValue> >& weights,
                              std::map<std::string, std::pair<std::string, JsonValue> >& constants,
                              std::map<std::string, std::vector<char> >& legacy_raw)
{
    // decode both pickled state dicts (weights + non-persistent constants)
    LegacyStateDict w;
    LegacyStateDict c;
    if (load_legacy_state_dict(zip, names, "serialized_state_dict.pt", w) != 0)
        return -1;
    if (load_legacy_state_dict(zip, names, "serialized_constants.pt", c) != 0)
        return -1;

    // shard bytes are read once per storage record (views may share a storage,
    // but the pickle already resolved each fqn to exactly one shard record)
    std::map<std::string, std::vector<char> > shard_cache;

    const JsonValue& gm = root["graph_module"];
    const JsonValue& graph = gm["graph"];
    const JsonValue& signature = gm["signature"];
    const JsonValue& tensor_values = graph["tensor_values"];
    const JsonValue& input_specs = signature["input_specs"];

    for (size_t i = 0; i < input_specs.size(); i++)
    {
        const JsonValue& spec = input_specs[i];

        bool is_parameter = false;
        bool is_buffer = false;
        std::string fqn;
        std::string arg_name;
        if (spec.has("parameter"))
        {
            is_parameter = true;
            fqn = spec["parameter"]["parameter_name"].as_string();
            arg_name = spec["parameter"]["arg"]["name"].as_string();
        }
        else if (spec.has("buffer"))
        {
            is_buffer = true;
            fqn = spec["buffer"]["buffer_name"].as_string();
            arg_name = spec["buffer"]["arg"]["name"].as_string();
        }
        else if (spec.has("tensor_constant"))
        {
            fqn = spec["tensor_constant"]["tensor_constant_name"].as_string();
            arg_name = spec["tensor_constant"]["arg"]["name"].as_string();
        }
        else
        {
            continue; // user_input etc.
        }

        // find the fqn in weights first, then constants
        const LegacyStateDict* src = 0;
        const pickle::Value* rec = 0;
        {
            std::map<std::string, pickle::Value>::const_iterator it = w.tensors.find(fqn);
            if (it != w.tensors.end())
            {
                src = &w;
                rec = &it->second;
            }
            else
            {
                std::map<std::string, pickle::Value>::const_iterator jt = c.tensors.find(fqn);
                if (jt != c.tensors.end())
                {
                    src = &c;
                    rec = &jt->second;
                }
            }
        }
        if (!rec)
        {
            fprintf(stderr, "legacy weights: missing pickled payload for '%s'\n", fqn.c_str());
            return -1;
        }

        if ((is_parameter && !rec->is_parameter) || (!is_parameter && !is_buffer && rec->is_parameter))
        {
            fprintf(stderr, "legacy weights: parameter classification mismatch for '%s'\n", fqn.c_str());
            return -1;
        }

        // tensor metadata comes from the graph (same schema as the archive path)
        if (!tensor_values.has(arg_name))
        {
            fprintf(stderr, "legacy weights: graph has no tensor_value for '%s'\n", arg_name.c_str());
            return -1;
        }
        const JsonValue& meta = tensor_values[arg_name];

        // read the storage shard bytes (once per unique shard; the shard id is
        // only unique within one .pt payload, so key by record + shard)
        const std::string cache_key = src->tag + "/" + rec->s;
        std::map<std::string, std::vector<char> >::iterator sit = shard_cache.find(cache_key);
        if (sit == shard_cache.end())
        {
            std::vector<char> shard;
            if (memzip_read_entry(src->bytes, rec->s, shard) != 0)
                return -1;
            sit = shard_cache.insert(std::make_pair(cache_key, std::move(shard))).first;
        }
        legacy_raw[fqn] = sit->second;

        // persistent buffer data lives with the weights; non-persistent buffer
        // and tensor_constant data with the constants (mirrors the archive path)
        bool persistent = is_buffer && (!spec["buffer"].has("persistent") || spec["buffer"]["persistent"].as_bool());
        std::map<std::string, std::pair<std::string, JsonValue> >& target = (is_parameter || persistent) ? weights : constants;

        target[fqn] = std::make_pair(rec->s, meta);
    }

    return 0;
}

} // namespace pnnx
