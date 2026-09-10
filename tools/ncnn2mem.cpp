// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "allocator.h"
#include "layer.h"
#include "layer_type.h"

#include <cstddef>
#include <ctype.h>
#include <float.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

static std::vector<std::string> blob_names;

static int find_blob_index_by_name(const char* name)
{
    for (std::size_t i = 0; i < blob_names.size(); i++)
    {
        if (blob_names[i] == name)
        {
            return static_cast<int>(i);
        }
    }

    fprintf(stderr, "find_blob_index_by_name %s failed\n", name);
    return -1;
}

static void sanitize_name(char* name)
{
    for (char* p = name; *p; p++)
    {
        if (!isalnum((unsigned char)*p))
        {
            *p = '_';
        }
    }
}

static std::string path_to_varname(const char* path)
{
    const char* lastslash = strrchr(path, '/');
    const char* name = lastslash == NULL ? path : lastslash + 1;

    std::string varname = name;
    for (std::size_t i = 0; i < varname.size(); i++)
    {
        if (!isalnum((unsigned char)varname[i]))
        {
            varname[i] = '_';
        }
    }

    return varname;
}

// keep array length checks in sync with src/paramdict.cpp
static size_t max_array_length()
{
    // leave room for Mat alignment, the reference count and fastMalloc overhead
    const size_t max_len = ((size_t)-1 - 15 - sizeof(int) - sizeof(void*) - NCNN_MALLOC_ALIGN - NCNN_MALLOC_OVERREAD) / sizeof(float);
    return std::min((size_t)INT_MAX, max_len);
}

static bool valid_array_length(size_t len)
{
    return len <= max_array_length();
}

// keep numeric parsing in sync with src/paramdict.cpp
static bool vstr_is_float(const char* vstr)
{
    return strchr(vstr, '.') || strchr(vstr, 'e') || strchr(vstr, 'E');
}

static bool vstr_to_int(const char* p, int& v)
{
    const bool negative = *p == '-';
    if (*p == '+' || *p == '-')
        p++;

    if (*p < '0' || *p > '9')
        return false;

    const unsigned int limit = negative ? (unsigned int)INT_MAX + 1u : (unsigned int)INT_MAX;
    unsigned int magnitude = 0;
    while (*p >= '0' && *p <= '9')
    {
        const unsigned int digit = *p++ - '0';
        if (magnitude > (limit - digit) / 10)
            return false;
        magnitude = magnitude * 10 + digit;
    }
    if (*p != '\0')
        return false;

    v = negative ? (magnitude == (unsigned int)INT_MAX + 1u ? INT_MIN : -(int)magnitude) : (int)magnitude;
    return true;
}

// the input is a validated unsigned decimal token of at most 127 characters
static bool vstr_fits_float(const char* p)
{
    char digits[128];
    int len = 0;
    int point = -1;
    while (*p && *p != 'e' && *p != 'E')
    {
        if (*p == '.')
            point = len;
        else
            digits[len++] = *p;
        p++;
    }
    if (point < 0)
        point = len;

    int exponent = 0;
    if (*p)
    {
        p++;
        const bool negative = *p == '-';
        if (*p == '+' || *p == '-')
            p++;
        while (*p)
        {
            if (exponent < 1024)
                exponent = exponent * 10 + (*p - '0');
            p++;
        }
        if (negative)
            exponent = -exponent;
    }

    int first = 0;
    while (first < len && digits[first] == '0')
        first++;
    if (first == len)
        return true;

    const int decimal_digits = point - first + exponent;
    if (decimal_digits != 39)
        return decimal_digits < 39;

    // 2^128 - 2^103 is the exact midpoint between FLT_MAX and float overflow
    const char midpoint[] = "340282356779733661637539395458142568448";
    for (int i = 0; i < 39; i++)
    {
        const char digit = first + i < len ? digits[first + i] : '0';
        if (digit != midpoint[i])
            return digit < midpoint[i];
    }
    return false;
}

static bool vstr_to_float(const char* p, float& value)
{
    const bool negative = *p == '-';
    if (*p == '+' || *p == '-')
        p++;

    const char* digits = p;
    double v = 0.0;
    bool has_digit = false;
    while (*p >= '0' && *p <= '9')
    {
        has_digit = true;
        v = v * 10.0 + (*p++ - '0');
    }
    if (*p == '.')
    {
        p++;
        // accumulate the fraction before adding it to the integer part
        double fraction = 0.0;
        double scale = 1.0;
        while (*p >= '0' && *p <= '9')
        {
            has_digit = true;
            fraction = fraction * 10.0 + (*p++ - '0');
            scale *= 10.0;
        }
        v += fraction / scale;
    }
    if (!has_digit)
        return false;

    if (*p == 'e' || *p == 'E')
    {
        p++;
        const bool negative_exponent = *p == '-';
        if (*p == '+' || *p == '-')
            p++;
        if (*p < '0' || *p > '9')
            return false;

        // saturate the exponent instead of overflowing or looping over its value
        unsigned int exponent = 0;
        while (*p >= '0' && *p <= '9')
        {
            if (exponent < 1024)
            {
                exponent = exponent * 10 + (*p - '0');
                if (exponent > 1024)
                    exponent = 1024;
            }
            p++;
        }
        if (v != 0.0)
        {
            // tokens are limited to 127 characters, so an infinite scale implies float overflow or underflow
            double scale = 1.0;
            double base = 10.0;
            while (exponent)
            {
                if (exponent & 1)
                    scale *= base;
                exponent >>= 1;
                if (exponent)
                    base *= base;
            }
            v = negative_exponent ? v / scale : v * scale;
        }
    }
    if (*p != '\0')
        return false;

    // compare the original decimal near overflow, where double rounding can cross the midpoint
    if (v >= (double)FLT_MAX)
    {
        if (!vstr_fits_float(digits))
            return false;
        v = (double)FLT_MAX;
    }
    value = negative ? (float)-v : (float)v;
    return true;
}

static bool param_space(char c)
{
    return c == ' ' || c == '\t' || c == '\v' || c == '\f';
}

static int scan_numeric_value(const char*& p, char vstr[128], bool comma = false)
{
    vstr[0] = '\0';
    if (comma)
    {
        if (*p != ',')
            return 0;
        p++;
    }

    int len = 0;
    while (*p && *p != ',' && !param_space(*p))
    {
        if (len == 127)
        {
            vstr[len] = '\0';
            return -1;
        }
        vstr[len++] = *p++;
    }
    vstr[len] = '\0';
    return len > 0 ? 1 : 0;
}

static bool parse_numeric_value(const char* vstr, bool is_float, int& value)
{
    if (is_float)
    {
        float f;
        if (!vstr_to_float(vstr, f))
            return false;
        memcpy(&value, &f, sizeof(float));
        return true;
    }
    return vstr_to_int(vstr, value);
}

static bool write_param(FILE* fp, const void* data, size_t size)
{
    if (fwrite(data, 1, size, fp) != size)
    {
        fprintf(stderr, "write param failed\n");
        return false;
    }
    return true;
}

static int close_file(FILE* fp, const char* path)
{
    // fclose may report a buffered write failure even when earlier writes succeeded
    const int io_error = ferror(fp);
    const int close_error = fclose(fp);
    if (io_error || close_error != 0)
    {
        fprintf(stderr, "file io failed %s\n", path);
        return -1;
    }
    return 0;
}

static int dump_param_values(FILE* fp, FILE* mp)
{
    // each layer occupies one line
    // leave the newline for the next layer header scan
    char line[1024] = {0};
    std::vector<char> long_line;
    while (fscanf(fp, "%1023[^\r\n]", line) == 1)
    {
        const size_t len = strlen(line);
        if (long_line.empty() && len < sizeof(line) - 1)
            break;
        long_line.insert(long_line.end(), line, line + len);
        if (len < sizeof(line) - 1)
            break;
    }
    if (ferror(fp))
    {
        fprintf(stderr, "read param failed\n");
        return -1;
    }
    if (!long_line.empty())
        long_line.push_back('\0');
    const char* p = long_line.empty() ? line : &long_line[0];

    while (1)
    {
        while (param_space(*p))
            p++;
        if (!*p)
            break;

        char idstr[16];
        int idlen = 0;
        while ((*p == '-' || *p == '+' || (*p >= '0' && *p <= '9')) && idlen < 15)
            idstr[idlen++] = *p++;
        idstr[idlen] = '\0';
        int id;
        if (!vstr_to_int(idstr, id) || *p != '=')
        {
            fprintf(stderr, "invalid parameter id or missing equals sign\n");
            return -1;
        }
        p++;

        const bool old_array = id <= -23300;
        const int param_id = old_array ? -(id + 23300) : id;
        if (param_id < 0 || param_id >= NCNN_MAX_PARAM_COUNT)
        {
            fprintf(stderr, "invalid parameter id %d\n", id);
            return -1;
        }

        if (old_array)
        {
            char vstr[128];
            int len;
            if (scan_numeric_value(p, vstr) != 1 || !vstr_to_int(vstr, len) || !valid_array_length((size_t)len))
            {
                fprintf(stderr, "invalid array length (id=%d)\n", id);
                return -1;
            }
            if (!write_param(mp, &id, sizeof(int)) || !write_param(mp, &len, sizeof(int)))
                return -1;
            for (int j = 0; j < len; j++)
            {
                int value;
                if (scan_numeric_value(p, vstr, true) != 1
                        || !parse_numeric_value(vstr, vstr_is_float(vstr), value))
                {
                    fprintf(stderr, "invalid array element (id=%d, index=%d)\n", id, j);
                    return -1;
                }
                if (!write_param(mp, &value, sizeof(int)))
                    return -1;
            }
            if (*p == ',')
            {
                fprintf(stderr, "array length mismatch (id=%d)\n", id);
                return -1;
            }
            continue;
        }

        if (*p == '\"' || (*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z'))
        {
            char text[256] = {0};
            const bool quoted = *p == '\"';
            if (quoted)
                p++;
            int len = 0;
            while (*p && (quoted ? *p != '\"' : !param_space(*p)) && len < 255)
                text[len++] = *p++;
            if (quoted)
            {
                if (*p != '\"')
                {
                    fprintf(stderr, "unterminated or too long string (id=%d)\n", id);
                    return -1;
                }
                p++;
            }
            if (*p && !param_space(*p))
            {
                fprintf(stderr, "invalid string suffix or string too long (id=%d)\n", id);
                return -1;
            }

            id = -id - 23400;
            if (!write_param(mp, &id, sizeof(int)) || !write_param(mp, &len, sizeof(int)))
                return -1;
            if (!write_param(mp, text, (len + 3) / 4 * 4))
                return -1;
            continue;
        }

        char vstr[128];
        if (scan_numeric_value(p, vstr) != 1)
        {
            fprintf(stderr, "read value failed (id=%d)\n", id);
            return -1;
        }
        const bool is_float = vstr_is_float(vstr);
        int value;
        if (!parse_numeric_value(vstr, is_float, value))
        {
            fprintf(stderr, "invalid numeric value (id=%d)\n", id);
            return -1;
        }

        if (*p == ',')
        {
            p++;
            std::vector<int> values;
            values.push_back(value);
            while (1)
            {
                const int nscan = scan_numeric_value(p, vstr);
                if (nscan == 0)
                {
                    if (*p == ',')
                    {
                        fprintf(stderr, "missing array element (id=%d)\n", id);
                        return -1;
                    }
                    break;
                }
                if (nscan < 0 || !valid_array_length(values.size() + 1) || !parse_numeric_value(vstr, is_float, value))
                {
                    fprintf(stderr, "invalid array element (id=%d)\n", id);
                    return -1;
                }
                values.push_back(value);
                if (*p != ',')
                    break;
                p++;
            }
            id = -id - 23300;
            int len = (int)values.size();
            if (!write_param(mp, &id, sizeof(int)) || !write_param(mp, &len, sizeof(int)))
                return -1;
            if (!write_param(mp, &values[0], (size_t)len * sizeof(int)))
                return -1;
        }
        else
        {
            if (!write_param(mp, &id, sizeof(int)) || !write_param(mp, &value, sizeof(int)))
                return -1;
        }
    }

    int EOP = -233;
    return write_param(mp, &EOP, sizeof(int)) ? 0 : -1;
}

static int dump_param_impl(FILE* fp, FILE* mp, FILE* ip, const char* parampath, const char* idcpppath)
{
    std::string param_var = path_to_varname(parampath);

    std::string include_guard_var = path_to_varname(idcpppath);

    fprintf(ip, "#ifndef NCNN_INCLUDE_GUARD_%s\n", include_guard_var.c_str());
    fprintf(ip, "#define NCNN_INCLUDE_GUARD_%s\n", include_guard_var.c_str());
    fprintf(ip, "namespace %s_id {\n", param_var.c_str());

    int nscan = 0;
    int magic = 0;
    nscan = fscanf(fp, "%d", &magic);
    if (nscan != 1)
    {
        fprintf(stderr, "read magic failed %d\n", nscan);
        return -1;
    }
    if (!write_param(mp, &magic, sizeof(int)))
        return -1;

    int layer_count = 0;
    int blob_count = 0;
    nscan = fscanf(fp, "%d %d", &layer_count, &blob_count);
    if (nscan != 2)
    {
        fprintf(stderr, "read layer_count and blob_count failed %d\n", nscan);
        return -1;
    }
    if (!write_param(mp, &layer_count, sizeof(int)) || !write_param(mp, &blob_count, sizeof(int)))
        return -1;

    blob_names.resize(blob_count);

    std::vector<std::string> custom_layer_index;

    int blob_index = 0;
    for (int i = 0; i < layer_count; i++)
    {
        char layer_type[33];
        char layer_name[257];
        int bottom_count = 0;
        int top_count = 0;
        nscan = fscanf(fp, "%32s %256s %d %d", layer_type, layer_name, &bottom_count, &top_count);
        if (nscan != 4)
        {
            fprintf(stderr, "read layer params failed %d\n", nscan);
            return -1;
        }

        sanitize_name(layer_name);

        int typeindex = ncnn::layer_to_index(layer_type);
        if (typeindex == -1)
        {
            // lookup custom_layer_index
            for (size_t j = 0; j < custom_layer_index.size(); j++)
            {
                if (custom_layer_index[j] == layer_type)
                {
                    typeindex = ncnn::LayerType::CustomBit | j;
                    break;
                }
            }

            if (typeindex == -1)
            {
                // new custom layer type
                size_t j = custom_layer_index.size();
                custom_layer_index.push_back(layer_type);
                typeindex = ncnn::LayerType::CustomBit | j;
            }
        }
        if (!write_param(mp, &typeindex, sizeof(int))
                || !write_param(mp, &bottom_count, sizeof(int)) || !write_param(mp, &top_count, sizeof(int)))
            return -1;

        fprintf(ip, "const int LAYER_%s = %d;\n", layer_name, i);

        //         layer->bottoms.resize(bottom_count);
        for (int j = 0; j < bottom_count; j++)
        {
            char bottom_name[257];
            nscan = fscanf(fp, "%256s", bottom_name);
            if (nscan != 1)
            {
                fprintf(stderr, "read bottom_name failed %d\n", nscan);
                return -1;
            }

            sanitize_name(bottom_name);

            int bottom_blob_index = find_blob_index_by_name(bottom_name);

            if (!write_param(mp, &bottom_blob_index, sizeof(int)))
                return -1;
        }

        //         layer->tops.resize(top_count);
        for (int j = 0; j < top_count; j++)
        {
            char blob_name[257];
            nscan = fscanf(fp, "%256s", blob_name);
            if (nscan != 1)
            {
                fprintf(stderr, "read blob_name failed %d\n", nscan);
                return -1;
            }

            sanitize_name(blob_name);

            blob_names[blob_index] = std::string(blob_name);

            fprintf(ip, "const int BLOB_%s = %d;\n", blob_name, blob_index);

            if (!write_param(mp, &blob_index, sizeof(int)))
                return -1;

            blob_index++;
        }

        if (dump_param_values(fp, mp) != 0)
            return -1;
        if (ferror(ip))
        {
            fprintf(stderr, "write %s failed\n", idcpppath);
            return -1;
        }
    }

    // dump custom layer index
    for (size_t j = 0; j < custom_layer_index.size(); j++)
    {
        const std::string& layer_type = custom_layer_index[j];
        int typeindex = ncnn::LayerType::CustomBit | j;

        fprintf(ip, "const int TYPEINDEX_%s = %d;\n", layer_type.c_str(), typeindex);

        fprintf(stderr, "net.register_custom_layer(%s_id::TYPEINDEX_%s, %s_layer_creator);\n", param_var.c_str(), layer_type.c_str(), layer_type.c_str());
    }

    fprintf(ip, "} // namespace %s_id\n", param_var.c_str());
    fprintf(ip, "#endif // NCNN_INCLUDE_GUARD_%s\n", include_guard_var.c_str());

    return 0;
}

static int dump_param(const char* parampath, const char* parambinpath, const char* idcpppath)
{
    FILE* fp = fopen(parampath, "rb");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", parampath);
        return -1;
    }

    FILE* mp = fopen(parambinpath, "wb");
    if (!mp)
    {
        fprintf(stderr, "fopen %s failed\n", parambinpath);
        close_file(fp, parampath);
        return -1;
    }

    FILE* ip = fopen(idcpppath, "wb");
    if (!ip)
    {
        fprintf(stderr, "fopen %s failed\n", idcpppath);
        close_file(fp, parampath);
        close_file(mp, parambinpath);
        return -1;
    }

    int ret = dump_param_impl(fp, mp, ip, parampath, idcpppath);
    if (close_file(fp, parampath) != 0)
        ret = -1;
    if (close_file(mp, parambinpath) != 0)
        ret = -1;
    if (close_file(ip, idcpppath) != 0)
        ret = -1;
    return ret;
}

static int write_memcpp_impl(FILE* mp, FILE* bp, FILE* cppfp, const char* parambinpath, const char* modelpath, const char* memcpppath)
{
    // dump param
    std::string param_var = path_to_varname(parambinpath);

    std::string include_guard_var = path_to_varname(memcpppath);

    fprintf(cppfp, "#ifndef NCNN_INCLUDE_GUARD_%s\n", include_guard_var.c_str());
    fprintf(cppfp, "#define NCNN_INCLUDE_GUARD_%s\n", include_guard_var.c_str());

    fprintf(cppfp, "\n#ifdef _MSC_VER\n__declspec(align(4))\n#else\n__attribute__((aligned(4)))\n#endif\n");
    fprintf(cppfp, "static const unsigned char %s[] = {\n", param_var.c_str());

    int i = 0;
    while (!feof(mp))
    {
        int c = fgetc(mp);
        if (c == EOF)
            break;
        if (fprintf(cppfp, "0x%02x,", c) < 0)
        {
            fprintf(stderr, "write %s failed\n", memcpppath);
            return -1;
        }

        i++;
        if (i % 16 == 0)
        {
            fprintf(cppfp, "\n");
        }
    }

    fprintf(cppfp, "};\n");

    if (ferror(mp))
    {
        fprintf(stderr, "read %s failed\n", parambinpath);
        return -1;
    }

    // dump model
    std::string model_var = path_to_varname(modelpath);

    fprintf(cppfp, "\n#ifdef _MSC_VER\n__declspec(align(4))\n#else\n__attribute__((aligned(4)))\n#endif\n");
    fprintf(cppfp, "static const unsigned char %s[] = {\n", model_var.c_str());

    i = 0;
    while (!feof(bp))
    {
        int c = fgetc(bp);
        if (c == EOF)
            break;
        if (fprintf(cppfp, "0x%02x,", c) < 0)
        {
            fprintf(stderr, "write %s failed\n", memcpppath);
            return -1;
        }

        i++;
        if (i % 16 == 0)
        {
            fprintf(cppfp, "\n");
        }
    }

    fprintf(cppfp, "};\n");

    fprintf(cppfp, "#endif // NCNN_INCLUDE_GUARD_%s\n", include_guard_var.c_str());

    return 0;
}

static int write_memcpp(const char* parambinpath, const char* modelpath, const char* memcpppath)
{
    FILE* mp = fopen(parambinpath, "rb");
    if (!mp)
    {
        fprintf(stderr, "fopen %s failed\n", parambinpath);
        return -1;
    }

    FILE* bp = fopen(modelpath, "rb");
    if (!bp)
    {
        fprintf(stderr, "fopen %s failed\n", modelpath);
        close_file(mp, parambinpath);
        return -1;
    }

    FILE* cppfp = fopen(memcpppath, "wb");
    if (!cppfp)
    {
        fprintf(stderr, "fopen %s failed\n", memcpppath);
        close_file(mp, parambinpath);
        close_file(bp, modelpath);
        return -1;
    }

    int ret = write_memcpp_impl(mp, bp, cppfp, parambinpath, modelpath, memcpppath);
    if (close_file(mp, parambinpath) != 0)
        ret = -1;
    if (close_file(bp, modelpath) != 0)
        ret = -1;
    if (close_file(cppfp, memcpppath) != 0)
        ret = -1;
    return ret;
}

int main(int argc, char** argv)
{
    if (argc != 5)
    {
        fprintf(stderr, "Usage: %s [ncnnproto] [ncnnbin] [idcpppath] [memcpppath]\n", argv[0]);
        return -1;
    }

    const char* parampath = argv[1];
    const char* modelpath = argv[2];
    const char* idcpppath = argv[3];
    const char* memcpppath = argv[4];

    std::string parambinpath = std::string(parampath) + ".bin";

    if (dump_param(parampath, parambinpath.c_str(), idcpppath) != 0)
        return -1;

    return write_memcpp(parambinpath.c_str(), modelpath, memcpppath);
}
