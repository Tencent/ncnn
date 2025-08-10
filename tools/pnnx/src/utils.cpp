// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "utils.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include <string>
#include <vector>
namespace pnnx {

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

void prase_dtype(char* dtype, std::vector<std::string>& types, char& endian)
{
    if (dtype[0] != '<' && dtype[0] != '>' && dtype[0] != '|')
    {
        endian = '|';
    }
    else
    {
        endian = dtype[0];
        ++dtype;
    }
    std::string s;
    s += dtype[0];
    char p[10];
    sprintf(p, "%d", 8 * atoi(dtype + 1));
    s += p;
    types.push_back(s);
}

void prase_numpy_header(char* header_str,
                        std::vector<std::vector<int64_t> >& shapes,
                        std::vector<std::string>& types,
                        bool& fortran_order,
                        char& endian)
{
    char* start;
    char* end;
    char* ptr = strstr(header_str, "\'fortran_order\'");
    fortran_order = strstr(ptr, "True") != NULL;

    ptr = strstr(header_str, "\'descr\'");
    start = strchr(ptr + strlen("\'descr\'"), '\'');
    ++start;
    end = strstr(start, "\'");
    *end = '\0';
    prase_dtype(start, types, endian);
    *end = '\'';

    ptr = strstr(header_str, "\'shape\'");
    start = strchr(ptr, '(');
    ++start;
    end = strchr(ptr, ')');
    *end = '\0';

    std::vector<int64_t> v;
    char* token = strtok(start, ",");
    while (token != NULL)
    {
        v.push_back(atoi(token));
        token = strtok(NULL, ",");
    }
    shapes.push_back(v);
}

char get_system_endian()
{
    uint16_t i = 1;
    return (*(char*)&i) ? '<' : '>';
}

void swap_bytes(void* buffer, size_t type_size, size_t content_len)
{
    fprintf(stderr, "data endian is different from system endian, swapping bytes\n");

    char* p = (char*)buffer;
    for (size_t i = 0; i < content_len; ++i)
    {
        char* bytes = p + i * type_size;
        for (size_t j = 0; j < type_size / 2; ++j)
        {
            std::swap(bytes[j], bytes[type_size - j - 1]);
        }
    }
}

size_t get_type_size_from_input_type(const char* str)
{
    return atoi(str + 1) / 8;
}

void convert_to_c_order(void* src, const std::vector<int64_t>& shape, size_t type_size, size_t content_len)
{
    fprintf(stderr, "array is fortran order, converting to c order\n");
    void* dst = malloc(content_len * type_size);
    size_t dims = shape.size();
    int64_t* c_strides = (int64_t*)malloc(dims * sizeof(int64_t));
    int64_t* f_strides = (int64_t*)malloc(dims * sizeof(int64_t));
    int64_t* index = (int64_t*)malloc(dims * sizeof(int64_t));

    memset(index, 0, dims * sizeof(int64_t));

    c_strides[dims - 1] = 1;
    for (int i = dims - 2; i >= 0; --i)
    {
        c_strides[i] = c_strides[i + 1] * shape[i + 1];
    }

    f_strides[0] = 1;
    for (int i = 1; i <= dims - 1; ++i)
    {
        f_strides[i] = f_strides[i - 1] * shape[i - 1];
    }

    // todo: optimize this?
    for (int i = 1; i <= content_len; ++i)
    {
        int64_t c_index = 0;
        int64_t f_index = 0;
        for (int j = 0; j <= dims - 1; ++j)
        {
            c_index += c_strides[j] * index[j];
            f_index += f_strides[j] * index[j];
        }

        memcpy((char*)dst + c_index * type_size, (char*)src + f_index * type_size, type_size);

        ++index[dims - 1];
        for (int j = dims - 1; j >= 0; --j)
        {
            index[j]++;
            if (index[j] < shape[j])
            {
                break;
            }
            index[j] = 0;
        }
    }

    memcpy(src, dst, content_len * type_size);

    free(dst);
    free(c_strides);
    free(f_strides);
    free(index);
}

void prase_numpy_file(const char* path,
                      std::vector<std::vector<int64_t> >& shapes,
                      std::vector<std::string>& types,
                      std::vector<std::vector<char> >& contents)
{
    fprintf(stderr, "prasing numpy file: %s\n", path);
    FILE* fp = fopen(path, "rb");
    if (!fp)
    {
        fprintf(stderr, "open failed %s\n", path);
        fclose(fp);
        return;
    }

    char magic[6];
    fread(magic, sizeof(char), 6, fp);

    uint8_t major_version, minor_version;
    fread(&major_version, sizeof(uint8_t), 1, fp);
    fread(&minor_version, sizeof(uint8_t), 1, fp);

    uint16_t header_len_v1;
    uint32_t header_len_v2;
    size_t header_len;

    if (major_version == 1)
    {
        fread(&header_len_v1, sizeof(uint16_t), 1, fp);
        header_len = header_len_v1;
    }
    else if (major_version == 2)
    {
        fread(&header_len_v2, sizeof(uint32_t), 1, fp);
        header_len = header_len_v2;
    }

    char* header_str = (char*)malloc(header_len + 1);
    if (header_str == NULL)
    {
        fprintf(stderr, "malloc filed");
        fclose(fp);
        return;
    }
    fread(header_str, sizeof(char), header_len, fp);
    header_str[header_len] = '\0';

    bool fortran_order;
    char endian;
    prase_numpy_header(header_str, shapes, types, fortran_order, endian);
    free(header_str);

    size_t content_len = 1;
    for (auto& i : shapes[shapes.size() - 1])
    {
        content_len *= i;
    }

    size_t type_size = get_type_size_from_input_type(types[types.size() - 1].c_str());
    void* buffer = malloc(content_len * type_size);
    fread(buffer, type_size, content_len, fp);

    if (endian != '|' && endian != get_system_endian())
    {
        swap_bytes(buffer, type_size, content_len);
    }

    if (fortran_order)
    {
        convert_to_c_order(buffer, shapes[shapes.size() - 1], type_size, content_len);
    }

    std::vector<char> v;
    v.resize(type_size * content_len);
    memcpy(v.data(), buffer, type_size * content_len);
    contents.push_back(v);
    free(buffer);
}

} // namespace pnnx
