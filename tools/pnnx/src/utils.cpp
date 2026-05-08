// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "utils.h"

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <limits>
#include <math.h>
#include <string>

namespace pnnx {

static std::string normalize_exponent(std::string s)
{
    // keep scientific notation compact and parser-friendly, e.g. 1.0e+06 -> 1.0e6.
    size_t pos = s.find_first_of("eE");
    if (pos == std::string::npos || pos + 1 == s.size())
        return s;

    size_t exponent_pos = pos + 1;
    if (s[exponent_pos] == '+')
    {
        s.erase(exponent_pos, 1);
    }
    else if (s[exponent_pos] == '-')
    {
        exponent_pos++;
    }

    while (exponent_pos + 1 < s.size() && s[exponent_pos] == '0')
        s.erase(exponent_pos, 1);

    return s;
}

static bool fractional_digits_exceed(const char* s, int max_digits)
{
    // ncnn ParamDict parses decimals with an unsigned int pow10 accumulator.
    // prefer scientific notation before long fractional parts can overflow it.
    const char* dot = strchr(s, '.');
    if (!dot)
        return false;

    int digits = 0;
    const char* p = dot + 1;
    while (*p != '\0' && *p != 'e' && *p != 'E')
    {
        digits++;
        if (digits > max_digits)
            return true;
        p++;
    }

    return false;
}

unsigned short float32_to_float16(float value)
{
    // 1 : 8 : 23
    union
    {
        unsigned int u;
        float f;
    } tmp;

    tmp.f = value;

    // 1 : 8 : 23
    unsigned short sign = (tmp.u & 0x80000000) >> 31;
    unsigned short exponent = (tmp.u & 0x7F800000) >> 23;
    unsigned int significand = tmp.u & 0x7FFFFF;

    //     NCNN_LOGE("%d %d %d", sign, exponent, significand);

    // 1 : 5 : 10
    unsigned short fp16;
    if (exponent == 0)
    {
        // zero or denormal, always underflow
        fp16 = (sign << 15) | (0x00 << 10) | 0x00;
    }
    else if (exponent == 0xFF)
    {
        // infinity or NaN
        fp16 = (sign << 15) | (0x1F << 10) | (significand ? 0x200 : 0x00);
    }
    else
    {
        // normalized
        short newexp = exponent + (-127 + 15);
        if (newexp >= 31)
        {
            // overflow, return infinity
            fp16 = (sign << 15) | (0x1F << 10) | 0x00;
        }
        else if (newexp <= 0)
        {
            // Some normal fp32 cannot be expressed as normal fp16
            fp16 = (sign << 15) | (0x00 << 10) | 0x00;
        }
        else
        {
            // normal fp16
            fp16 = (sign << 15) | (newexp << 10) | (significand >> 13);
        }
    }

    return fp16;
}

float float16_to_float32(unsigned short value)
{
    // 1 : 5 : 10
    unsigned short sign = (value & 0x8000) >> 15;
    unsigned short exponent = (value & 0x7c00) >> 10;
    unsigned short significand = value & 0x03FF;

    //     NCNN_LOGE("%d %d %d", sign, exponent, significand);

    // 1 : 8 : 23
    union
    {
        unsigned int u;
        float f;
    } tmp;
    if (exponent == 0)
    {
        if (significand == 0)
        {
            // zero
            tmp.u = (sign << 31);
        }
        else
        {
            // denormal
            exponent = 0;
            // find non-zero bit
            while ((significand & 0x200) == 0)
            {
                significand <<= 1;
                exponent++;
            }
            significand <<= 1;
            significand &= 0x3FF;
            tmp.u = (sign << 31) | ((-exponent + (-15 + 127)) << 23) | (significand << 13);
        }
    }
    else if (exponent == 0x1F)
    {
        // infinity or NaN
        tmp.u = (sign << 31) | (0xFF << 23) | (significand << 13);
    }
    else
    {
        // normalized
        tmp.u = (sign << 31) | ((exponent + (-15 + 127)) << 23) | (significand << 13);
    }

    return tmp.f;
}

std::string float_to_string(float f)
{
    if (f == 0.f)
        return "0.0";

    const float abs_f = std::abs(f);
    char buffer[64];

    if (abs_f < 0.0001f || abs_f >= 1000000.0f)
    {
        snprintf(buffer, sizeof(buffer), "%e", f);
        // keep short output when it round-trips; otherwise emit enough digits
        // for exact float32 recovery from text.
        if (strtof(buffer, 0) != f)
            snprintf(buffer, sizeof(buffer), "%.*e", std::numeric_limits<float>::max_digits10 - 1, f);
        return normalize_exponent(buffer);
    }

    int len = snprintf(buffer, sizeof(buffer), "%g", f);
    if (strtof(buffer, 0) != f)
    {
        // scalarized attributes, such as PReLU weight -> LeakyReLU slope,
        // must survive text serialization without changing the float32 value.
        len = snprintf(buffer, sizeof(buffer), "%.*g", std::numeric_limits<float>::max_digits10, f);
        if (fractional_digits_exceed(buffer, 9))
        {
            snprintf(buffer, sizeof(buffer), "%.*e", std::numeric_limits<float>::max_digits10 - 1, f);
            return normalize_exponent(buffer);
        }
    }

    bool is_integer = true;
    for (int i = 0; i < len; i++)
    {
        if (buffer[i] == '.' || buffer[i] == 'e' || buffer[i] == 'E')
        {
            is_integer = false;
            break;
        }
    }

    // maintain point-zero
    if (is_integer)
    {
        buffer[len] = '.';
        buffer[len + 1] = '0';
        buffer[len + 2] = '\0';
    }

    return std::string(buffer);
}

std::string double_to_string(double d)
{
    if (d == 0.0)
        return "0.0";

    const double abs_d = std::abs(d);
    char buffer[128];

    if (abs_d < 0.0001 || abs_d >= 1000000.0)
    {
        snprintf(buffer, sizeof(buffer), "%e", d);
        // mirror float_to_string for f64 constants used in pnnx expressions.
        if (strtod(buffer, 0) != d)
            snprintf(buffer, sizeof(buffer), "%.*e", std::numeric_limits<double>::max_digits10 - 1, d);
        return normalize_exponent(buffer);
    }

    int len = snprintf(buffer, sizeof(buffer), "%g", d);
    if (strtod(buffer, 0) != d)
    {
        // preserve f64 constants while avoiding extremely long plain decimals.
        len = snprintf(buffer, sizeof(buffer), "%.*g", std::numeric_limits<double>::max_digits10, d);
        if (fractional_digits_exceed(buffer, 17))
        {
            snprintf(buffer, sizeof(buffer), "%.*e", std::numeric_limits<double>::max_digits10 - 1, d);
            return normalize_exponent(buffer);
        }
    }

    bool is_integer = true;
    for (int i = 0; i < len; i++)
    {
        if (buffer[i] == '.' || buffer[i] == 'e' || buffer[i] == 'E')
        {
            is_integer = false;
            break;
        }
    }

    // maintain point-zero
    if (is_integer)
    {
        buffer[len] = '.';
        buffer[len + 1] = '0';
        buffer[len + 2] = '\0';
    }

    return std::string(buffer);
}

void apply_weight_norm(std::vector<float>& weight, const std::vector<float>& weight_g, int dim0, int size)
{
    for (int i = 0; i < dim0; i++)
    {
        float* pw = weight.data() + i * size;

        double norm = 0.f;
        for (int j = 0; j < size; j++)
        {
            float w = pw[j];
            norm += w * w;
        }
        norm = sqrt(norm);

        for (int j = 0; j < size; j++)
        {
            pw[j] = pw[j] * (weight_g[i] / norm);
        }
    }
}

static char system_endian()
{
    uint16_t x = 1;
    return (*(const uint8_t*)&x) ? '<' : '>';
}

static uint16_t read_le16(const unsigned char* p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t read_le32(const unsigned char* p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static bool fread_exact(FILE* fp, void* data, size_t size, const char* path, const char* what)
{
    if (size == 0)
        return true;

    if (fread(data, 1, size, fp) != size)
    {
        fprintf(stderr, "npy load failed %s: failed to read %s\n", path, what);
        return false;
    }

    return true;
}

static bool get_remaining_file_size(FILE* fp, const char* path, size_t& remaining_size)
{
    const long pos = ftell(fp);
    if (pos < 0)
    {
        fprintf(stderr, "npy load failed %s: ftell failed\n", path);
        return false;
    }

    if (fseek(fp, 0, SEEK_END) != 0)
    {
        fprintf(stderr, "npy load failed %s: fseek failed\n", path);
        return false;
    }

    const long end = ftell(fp);
    if (end < 0)
    {
        fprintf(stderr, "npy load failed %s: ftell failed\n", path);
        return false;
    }

    if (fseek(fp, pos, SEEK_SET) != 0)
    {
        fprintf(stderr, "npy load failed %s: fseek failed\n", path);
        return false;
    }

    if (end < pos)
    {
        fprintf(stderr, "npy load failed %s: invalid file position\n", path);
        return false;
    }

    remaining_size = (size_t)(end - pos);
    return true;
}

static std::string trim_string(const std::string& s)
{
    size_t begin = 0;
    while (begin < s.size() && isspace((unsigned char)s[begin]))
        begin++;

    size_t end = s.size();
    while (end > begin && isspace((unsigned char)s[end - 1]))
        end--;

    return s.substr(begin, end - begin);
}

static size_t find_header_key(const std::string& header, const char* key)
{
    std::string key1 = std::string("'") + key + "'";
    size_t pos = header.find(key1);
    if (pos != std::string::npos)
        return pos;

    std::string key2 = std::string("\"") + key + "\"";
    return header.find(key2);
}

static bool parse_header_string(const std::string& header, const char* key, std::string& value, const char* path)
{
    const size_t keypos = find_header_key(header, key);
    if (keypos == std::string::npos)
    {
        fprintf(stderr, "npy load failed %s: missing header key %s\n", path, key);
        return false;
    }

    const size_t colon = header.find(':', keypos);
    if (colon == std::string::npos)
    {
        fprintf(stderr, "npy load failed %s: malformed header key %s\n", path, key);
        return false;
    }

    const size_t quote = header.find_first_of("'\"", colon + 1);
    if (quote == std::string::npos)
    {
        fprintf(stderr, "npy load failed %s: malformed string header key %s\n", path, key);
        return false;
    }

    const char quote_char = header[quote];
    const size_t quote2 = header.find(quote_char, quote + 1);
    if (quote2 == std::string::npos)
    {
        fprintf(stderr, "npy load failed %s: unterminated string header key %s\n", path, key);
        return false;
    }

    value = header.substr(quote + 1, quote2 - quote - 1);
    return true;
}

static bool parse_header_bool(const std::string& header, const char* key, bool& value, const char* path)
{
    const size_t keypos = find_header_key(header, key);
    if (keypos == std::string::npos)
    {
        fprintf(stderr, "npy load failed %s: missing header key %s\n", path, key);
        return false;
    }

    const size_t colon = header.find(':', keypos);
    if (colon == std::string::npos)
    {
        fprintf(stderr, "npy load failed %s: malformed header key %s\n", path, key);
        return false;
    }

    size_t pos = colon + 1;
    while (pos < header.size() && isspace((unsigned char)header[pos]))
        pos++;

    if (header.compare(pos, 4, "True") == 0)
    {
        value = true;
        return true;
    }

    if (header.compare(pos, 5, "False") == 0)
    {
        value = false;
        return true;
    }

    fprintf(stderr, "npy load failed %s: malformed bool header key %s\n", path, key);
    return false;
}

static bool parse_header_shape(const std::string& header, std::vector<int64_t>& shape, const char* path)
{
    const size_t keypos = find_header_key(header, "shape");
    if (keypos == std::string::npos)
    {
        fprintf(stderr, "npy load failed %s: missing header key shape\n", path);
        return false;
    }

    const size_t colon = header.find(':', keypos);
    const size_t lparen = header.find('(', colon == std::string::npos ? keypos : colon);
    const size_t rparen = header.find(')', lparen == std::string::npos ? keypos : lparen);
    if (colon == std::string::npos || lparen == std::string::npos || rparen == std::string::npos || rparen < lparen)
    {
        fprintf(stderr, "npy load failed %s: malformed shape header\n", path);
        return false;
    }

    shape.clear();

    const std::string body = header.substr(lparen + 1, rparen - lparen - 1);
    size_t pos = 0;
    while (pos <= body.size())
    {
        const size_t comma = body.find(',', pos);
        const size_t end = comma == std::string::npos ? body.size() : comma;
        const std::string token = trim_string(body.substr(pos, end - pos));

        if (!token.empty())
        {
            char* endptr = 0;
            const long long dim = strtoll(token.c_str(), &endptr, 10);
            if (!endptr || *endptr != '\0' || dim < 0)
            {
                fprintf(stderr, "npy load failed %s: invalid shape dimension %s\n", path, token.c_str());
                return false;
            }

            shape.push_back((int64_t)dim);
        }

        if (comma == std::string::npos)
            break;

        pos = comma + 1;
    }

    return true;
}

struct NumpyDType
{
    char byte_order;
    char kind;
    size_t item_size;
    std::string type;
};

static bool parse_numpy_dtype(const std::string& descr, NumpyDType& dtype, const char* path)
{
    if (descr.empty())
    {
        fprintf(stderr, "npy load failed %s: empty dtype descriptor\n", path);
        return false;
    }

    size_t pos = 0;
    dtype.byte_order = '|';

    if (descr[pos] == '<' || descr[pos] == '>' || descr[pos] == '|' || descr[pos] == '=')
        dtype.byte_order = descr[pos++];

    if (pos >= descr.size())
    {
        fprintf(stderr, "npy load failed %s: malformed dtype descriptor %s\n", path, descr.c_str());
        return false;
    }

    dtype.kind = descr[pos++];

    if (dtype.kind == '?' && pos == descr.size())
    {
        dtype.item_size = 1;
    }
    else
    {
        if (pos >= descr.size())
        {
            fprintf(stderr, "npy load failed %s: dtype descriptor %s misses item size\n", path, descr.c_str());
            return false;
        }

        for (size_t i = pos; i < descr.size(); i++)
        {
            if (!isdigit((unsigned char)descr[i]))
            {
                fprintf(stderr, "npy load failed %s: unsupported dtype descriptor %s\n", path, descr.c_str());
                return false;
            }
        }

        dtype.item_size = (size_t)strtoull(descr.c_str() + pos, 0, 10);
    }

    if (dtype.item_size == 0)
    {
        fprintf(stderr, "npy load failed %s: invalid zero-sized dtype %s\n", path, descr.c_str());
        return false;
    }

    if (dtype.kind == 'f')
    {
        if (dtype.item_size == 2) dtype.type = "f16";
        if (dtype.item_size == 4) dtype.type = "f32";
        if (dtype.item_size == 8) dtype.type = "f64";
    }
    else if (dtype.kind == 'i')
    {
        if (dtype.item_size == 1) dtype.type = "i8";
        if (dtype.item_size == 2) dtype.type = "i16";
        if (dtype.item_size == 4) dtype.type = "i32";
        if (dtype.item_size == 8) dtype.type = "i64";
    }
    else if (dtype.kind == 'u')
    {
        if (dtype.item_size == 1) dtype.type = "u8";
    }
    else if (dtype.kind == 'b' || dtype.kind == '?')
    {
        if (dtype.item_size == 1) dtype.type = "bool";
    }
    else if (dtype.kind == 'c')
    {
        if (dtype.item_size == 4) dtype.type = "c32";
        if (dtype.item_size == 8) dtype.type = "c64";
        if (dtype.item_size == 16) dtype.type = "c128";
    }

    if (dtype.type.empty())
    {
        fprintf(stderr, "npy load failed %s: unsupported dtype descriptor %s\n", path, descr.c_str());
        return false;
    }

    return true;
}

static bool compute_element_count(const std::vector<int64_t>& shape, size_t& elem_count, const char* path)
{
    elem_count = 1;

    for (size_t i = 0; i < shape.size(); i++)
    {
        if (shape[i] < 0)
        {
            fprintf(stderr, "npy load failed %s: negative shape dimension\n", path);
            return false;
        }

        const size_t dim = (size_t)shape[i];
        if (dim != 0 && elem_count > std::numeric_limits<size_t>::max() / dim)
        {
            fprintf(stderr, "npy load failed %s: shape element count overflow\n", path);
            return false;
        }

        elem_count *= dim;
    }

    return true;
}

static void swap_bytes(char* p, size_t size)
{
    for (size_t i = 0; i < size / 2; i++)
        std::swap(p[i], p[size - 1 - i]);
}

static void swap_numpy_data_endian(std::vector<char>& data, const NumpyDType& dtype)
{
    if (dtype.item_size == 1)
        return;

    size_t swap_size = dtype.item_size;
    if (dtype.kind == 'c')
        swap_size = dtype.item_size / 2;

    if (swap_size <= 1)
        return;

    const size_t elem_count = data.size() / dtype.item_size;
    for (size_t i = 0; i < elem_count; i++)
    {
        char* elem = data.data() + i * dtype.item_size;
        for (size_t j = 0; j < dtype.item_size; j += swap_size)
            swap_bytes(elem + j, swap_size);
    }
}

static bool convert_fortran_to_c_order(std::vector<char>& data, const std::vector<int64_t>& shape, size_t item_size, const char* path)
{
    const size_t dims = shape.size();
    if (dims <= 1 || data.empty())
        return true;

    std::vector<size_t> shape_size(dims);
    for (size_t i = 0; i < dims; i++)
    {
        if (shape[i] < 0)
        {
            fprintf(stderr, "npy load failed %s: negative shape dimension\n", path);
            return false;
        }
        shape_size[i] = (size_t)shape[i];
    }

    std::vector<size_t> c_strides(dims);
    std::vector<size_t> f_strides(dims);

    c_strides[dims - 1] = 1;
    for (int i = (int)dims - 2; i >= 0; i--)
        c_strides[i] = c_strides[i + 1] * shape_size[i + 1];

    f_strides[0] = 1;
    for (size_t i = 1; i < dims; i++)
        f_strides[i] = f_strides[i - 1] * shape_size[i - 1];

    const size_t elem_count = data.size() / item_size;
    std::vector<char> reordered(data.size());

    for (size_t c_index = 0; c_index < elem_count; c_index++)
    {
        size_t remain = c_index;
        size_t f_index = 0;

        for (size_t i = 0; i < dims; i++)
        {
            const size_t idx = c_strides[i] == 0 ? 0 : remain / c_strides[i];
            remain = c_strides[i] == 0 ? 0 : remain % c_strides[i];
            f_index += idx * f_strides[i];
        }

        memcpy(reordered.data() + c_index * item_size, data.data() + f_index * item_size, item_size);
    }

    data.swap(reordered);
    return true;
}

bool load_numpy_file(const char* path, NumpyArray& array, bool load_data)
{
    array.shape.clear();
    array.type.clear();
    array.data.clear();

    FILE* fp = fopen(path, "rb");
    if (!fp)
    {
        fprintf(stderr, "npy load failed %s: fopen failed\n", path);
        return false;
    }

    bool ok = false;

    do
    {
        unsigned char magic[6];
        if (!fread_exact(fp, magic, 6, path, "magic"))
            break;

        static const unsigned char numpy_magic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
        if (memcmp(magic, numpy_magic, 6) != 0)
        {
            fprintf(stderr, "npy load failed %s: invalid magic\n", path);
            break;
        }

        unsigned char version[2];
        if (!fread_exact(fp, version, 2, path, "version"))
            break;

        size_t header_len = 0;
        if (version[0] == 1)
        {
            unsigned char header_len_bytes[2];
            if (!fread_exact(fp, header_len_bytes, 2, path, "header length"))
                break;
            header_len = read_le16(header_len_bytes);
        }
        else if (version[0] == 2 || version[0] == 3)
        {
            unsigned char header_len_bytes[4];
            if (!fread_exact(fp, header_len_bytes, 4, path, "header length"))
                break;
            header_len = read_le32(header_len_bytes);
        }
        else
        {
            fprintf(stderr, "npy load failed %s: unsupported npy version %u.%u\n", path, version[0], version[1]);
            break;
        }

        static const size_t numpy_max_header_len = 1024 * 1024;
        if (header_len > numpy_max_header_len)
        {
            fprintf(stderr, "npy load failed %s: header length %zu exceeds limit %zu\n", path, header_len, numpy_max_header_len);
            break;
        }

        std::vector<char> header_buf(header_len);
        if (!fread_exact(fp, header_buf.data(), header_len, path, "header"))
            break;

        const std::string header(header_buf.begin(), header_buf.end());

        std::string descr;
        bool fortran_order = false;
        std::vector<int64_t> shape;
        if (!parse_header_string(header, "descr", descr, path))
            break;
        if (!parse_header_bool(header, "fortran_order", fortran_order, path))
            break;
        if (!parse_header_shape(header, shape, path))
            break;

        NumpyDType dtype;
        if (!parse_numpy_dtype(descr, dtype, path))
            break;

        size_t elem_count = 0;
        if (!compute_element_count(shape, elem_count, path))
            break;

        if (elem_count != 0 && dtype.item_size > std::numeric_limits<size_t>::max() / elem_count)
        {
            fprintf(stderr, "npy load failed %s: data size overflow\n", path);
            break;
        }

        const size_t data_size = elem_count * dtype.item_size;

        size_t remaining_size = 0;
        if (!get_remaining_file_size(fp, path, remaining_size))
            break;

        if (data_size > remaining_size)
        {
            fprintf(stderr, "npy load failed %s: data size %zu exceeds remaining file size %zu\n", path, data_size, remaining_size);
            break;
        }

        array.shape = shape;
        array.type = dtype.type;

        if (!load_data)
        {
            ok = true;
            break;
        }

        array.data.resize(data_size);
        if (!fread_exact(fp, array.data.data(), data_size, path, "data"))
            break;

        const char endian = system_endian();
        const bool need_swap = (dtype.byte_order == '<' && endian == '>') || (dtype.byte_order == '>' && endian == '<');
        if (need_swap)
            swap_numpy_data_endian(array.data, dtype);

        if (fortran_order && !convert_fortran_to_c_order(array.data, shape, dtype.item_size, path))
            break;

        ok = true;
    } while (false);

    fclose(fp);
    return ok;
}

} // namespace pnnx
