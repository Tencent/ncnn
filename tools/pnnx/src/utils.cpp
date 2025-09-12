// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "utils.h"

#include <math.h>

#include <cstdio>   
#include <cstdlib>  
#include <cstring>  
#include <regex>    
#include <fstream>
#include <vector>
#include <string>

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

static bool parse_npy_shape(const std::string& header_str, std::vector<int64_t>& out_shape)
{
    out_shape.clear();

    // 1. 先找到 "shape" 字段的位置（兼容双引号/单引号，如 "shape" 或 'shape'）
    size_t shape_pos = header_str.find("shape");
    if (shape_pos == std::string::npos)
    {
        fprintf(stderr, "[ERROR] utils: 'shape' key not found in npy header\n");
        return false;
    }

    // 2. 找到 shape 后的冒号 ':'（跳过可能的空格，如 "shape: " 或 "shape = "）
    size_t colon_pos = header_str.find(':', shape_pos);
    if (colon_pos == std::string::npos)
    {
        fprintf(stderr, "[ERROR] utils: Colon ':' after 'shape' not found\n");
        return false;
    }

    // 3. 找到 shape 开始的括号（只找 '[' 或 '('，跳过冒号后的空格）
    size_t start_bracket_pos = header_str.find_first_of("[(", colon_pos + 1);
    if (start_bracket_pos == std::string::npos)
    {
        fprintf(stderr, "[ERROR] utils: Shape start bracket ([ or () not found after 'shape'\n");
        return false;
    }
    // 确定对应的结束括号（'[' 对应 ']'，'(' 对应 ')'）
    char end_bracket = (header_str[start_bracket_pos] == '[') ? ']' : ')';

    // 4. 找到结束括号的位置（从开始括号后开始找，避免匹配到内部无关括号）
    size_t end_bracket_pos = header_str.find(end_bracket, start_bracket_pos + 1);
    if (end_bracket_pos == std::string::npos)
    {
        fprintf(stderr, "[ERROR] utils: Shape end bracket (%c) not found\n", end_bracket);
        return false;
    }

    // 5. 提取括号内的内容（如 "1,3,224,224"，去掉空格）
    std::string shape_content = header_str.substr(start_bracket_pos + 1, end_bracket_pos - start_bracket_pos - 1);
    // 移除内容中的所有空格（兼容 "1, 3, 224, 224" 或 "1,3,224,224" 格式）
    shape_content.erase(std::remove(shape_content.begin(), shape_content.end(), ' '), shape_content.end());

    // 6. 按逗号分割字符串，提取每个维度的数字
    std::stringstream ss(shape_content);
    std::string dim_str;
    while (std::getline(ss, dim_str, ','))
    {
        if (dim_str.empty()) continue; // 跳过空字符串（如最后一个逗号后的空值）
        // 转换为int64_t，处理可能的转换失败
        try
        {
            int64_t dim = std::stoll(dim_str);
            out_shape.push_back(dim);
        }
        catch (const std::exception& e)
        {
            fprintf(stderr, "[ERROR] utils: Invalid number in shape: %s (error: %s)\n", dim_str.c_str(), e.what());
            out_shape.clear();
            return false;
        }
    }

    // 7. 验证shape不为空
    if (out_shape.empty())
    {
        fprintf(stderr, "[ERROR] utils: npy shape is empty\n");
        return false;
    }

    // 打印解析结果（调试用）
    // fprintf(stderr, "[DEBUG] utils: Parsed shape from %c...%c: [", header_str[start_bracket_pos], end_bracket);
    // for (size_t i = 0; i < out_shape.size(); i++)
    // {
    //     if (i > 0) fprintf(stderr, ", ");
    //     fprintf(stderr, "%ld", out_shape[i]);
    // }
    // fprintf(stderr, "]\n");

    return true;
}

static bool parse_npy_dtype(const std::string& header_str, std::string& out_dtype)
{
    // 转为小写后再判断，兼容"Float32"、"FLOAT32"等写法
    std::string lower_header = header_str;
    std::transform(lower_header.begin(), lower_header.end(), lower_header.begin(), ::tolower);
    
    if (lower_header.find("float32") != std::string::npos || 
        lower_header.find("f4") != std::string::npos)
    {
        out_dtype = "f32";
        return true;
    }
    else
    {
        fprintf(stderr, "[ERROR] utils: Only support float32 npy file (current dtype not supported)\n");
        return false;
    }
}

static int get_npy_header_length(FILE* fp)
{
    // npy文件格式：前6字节魔法数(0x934E554D5059) + 4字节头长度（little-endian）
    fseek(fp, 6, SEEK_SET); // 跳过前6字节魔法数
    uint32_t header_len = 0;
    if (fread(&header_len, sizeof(uint32_t), 1, fp) != 1)
    {
        fprintf(stderr, "[ERROR] utils: Failed to read npy header length\n");
        return -1;
    }
    fseek(fp, 0, SEEK_SET); // 恢复文件指针到开头
    return header_len;
}


// load_npy_tensor 主函数实现

int load_npy_tensor(const std::string& npy_path,
                   std::vector<int64_t>& out_shape,
                   std::string& out_dtype,
                   std::vector<float>& out_data)
{
    // 1. 打开文件（二进制模式）
    FILE* fp = fopen(npy_path.c_str(), "rb");
    if (!fp)
    {
        fprintf(stderr, "[ERROR] 无法打开文件: %s\n", npy_path.c_str());
        return -1;
    }

    // 2. 定义npy格式固定偏移量
    const int MAGIC_OFFSET = 0;       // 魔法数起始位置
    const int VERSION_OFFSET = 6;     // 版本号起始位置
    const int HEADER_LEN_OFFSET = 8;  // 头长度起始位置

    // 3. 读取并验证魔法数（绝对定位）
    unsigned char magic[6];
    fseek(fp, MAGIC_OFFSET, SEEK_SET);  // 强制定位到0字节
    if (fread(magic, 1, 6, fp) != 6)
    {
        fprintf(stderr, "[ERROR] 读取魔法数失败\n");
        fclose(fp);
        return -1;
    }
    if (magic[0] != 0x93 || magic[1] != 'N' || magic[2] != 'U' ||
        magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y')
    {
        fprintf(stderr, "[ERROR] 不是有效的npy文件\n");
        fclose(fp);
        return -1;
    }

    // 4. 读取并验证版本号（绝对定位）
    unsigned char version[2];
    fseek(fp, VERSION_OFFSET, SEEK_SET);  // 强制定位到6字节
    if (fread(version, 1, 2, fp) != 2)
    {
        fprintf(stderr, "[ERROR] 读取版本号失败\n");
        fclose(fp);
        return -1;
    }
    if (version[0] != 1 || version[1] != 0)
    {
        fprintf(stderr, "[ERROR] 只支持npy 1.0版本\n");
        fclose(fp);
        return -1;
    }

    // 5. 读取并解析头长度（绝对定位，关键修复）
    uint8_t header_len_bytes[4];
    fseek(fp, HEADER_LEN_OFFSET, SEEK_SET);  // 强制定位到8字节
    if (fread(header_len_bytes, 1, 4, fp) != 4)
    {
        fprintf(stderr, "[ERROR] 读取头长度失败\n");
        fclose(fp);
        return -1;
    }
    // 打印实际读取的字节（用于调试）
    // fprintf(stderr, "[DEBUG] 头长度原始字节: 0x%02X 0x%02X 0x%02X 0x%02X\n",
    //         header_len_bytes[0], header_len_bytes[1], 
    //         header_len_bytes[2], header_len_bytes[3]);
    
    // 计算头长度（小端模式）
    uint32_t header_len = (uint32_t)header_len_bytes[0] | 
                         (uint32_t)header_len_bytes[1] << 8 | 
                         (uint32_t)header_len_bytes[2] << 16 | 
                         (uint32_t)header_len_bytes[3] << 24;
    
    // 验证头长度合理性
    if (header_len == 0 || header_len > 1024 * 1024)
    {
        fprintf(stderr, "[ERROR] 无效的头长度: %u 字节\n", header_len);
        fclose(fp);
        return -1;
    }

    // 调试用代码
    // fprintf(stderr, "[DEBUG] 有效头长度: %u 字节\n", header_len);

    // 6. 读取头内容（绝对定位）
    const int HEADER_CONTENT_OFFSET = 12;  // 头内容起始位置（8+4）
    char* header_buf = (char*)malloc(header_len + 1);
    if (!header_buf)
    {
        fprintf(stderr, "[ERROR] 内存分配失败\n");
        fclose(fp);
        return -1;
    }
    fseek(fp, HEADER_CONTENT_OFFSET, SEEK_SET);  // 强制定位到12字节
    if (fread(header_buf, 1, header_len, fp) != header_len)
    {
        fprintf(stderr, "[ERROR] 读取头内容失败\n");
        free(header_buf);
        fclose(fp);
        return -1;
    }
    header_buf[header_len] = '\0';
    std::string header_str(header_buf);
    free(header_buf);

    // 7. 解析shape和dtype
    if (!parse_npy_shape(header_str, out_shape) || !parse_npy_dtype(header_str, out_dtype))
    {
        fclose(fp);
        return -1;
    }

    // 8. 读取数据（绝对定位）
    int64_t total_elem = 1;
    for (int64_t dim : out_shape) total_elem *= dim;
    if (total_elem <= 0)
    {
        fprintf(stderr, "[ERROR] 无效的shape\n");
        fclose(fp);
        return -1;
    }

    const int DATA_OFFSET = HEADER_CONTENT_OFFSET + header_len;  // 数据起始位置
    out_data.resize(total_elem);
    fseek(fp, DATA_OFFSET, SEEK_SET);  // 强制定位到数据区
    if (fread(out_data.data(), sizeof(float), total_elem, fp) != (size_t)total_elem)
    {
        fprintf(stderr, "[ERROR] 读取数据失败\n");
        fclose(fp);
        return -1;
    }

    // 9. 成功完成
    fprintf(stderr, "[INFO] 成功加载npy文件: %s\n", npy_path.c_str());
    fclose(fp);
    return 0;
}



} // namespace pnnx

