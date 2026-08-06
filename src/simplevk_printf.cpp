// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "simplevk_printf.h"

#if NCNN_VULKAN

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

namespace ncnn {

struct SimplevkPrintfArg
{
    int lanes;
    char type;
};

struct SimplevkPrintfFormat
{
    uint32_t id;
    std::string format;
    std::vector<SimplevkPrintfArg> args;
};

static Mutex g_simplevk_printf_lock;
static std::vector<SimplevkPrintfFormat> g_simplevk_printf_formats;
static uint32_t g_simplevk_printf_next_id = 1;
static int g_simplevk_printf_warned_unsupported = 0;
static int g_simplevk_printf_warned_overflow = 0;

static int is_ident_char(char ch)
{
    return isalnum((unsigned char)ch) || ch == '_';
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

static int parse_string_literal(const char* data, size_t size, size_t pos, std::string& value, size_t& endpos)
{
    if (pos >= size || data[pos] != '"')
        return -1;

    value.clear();

    size_t i = pos + 1;
    while (i < size)
    {
        char ch = data[i++];
        if (ch == '"')
        {
            endpos = i;
            return 0;
        }

        if (ch == '\\' && i < size)
        {
            char esc = data[i++];
            if (esc == 'n')
                value += '\n';
            else if (esc == 'r')
                value += '\r';
            else if (esc == 't')
                value += '\t';
            else
                value += esc;
            continue;
        }

        value += ch;
    }

    return -1;
}

static int parse_format_args(const std::string& fmt, std::vector<SimplevkPrintfArg>& args, int& payload_words)
{
    args.clear();
    payload_words = 0;

    for (size_t i = 0; i < fmt.size(); i++)
    {
        if (fmt[i] != '%')
            continue;

        if (i + 1 < fmt.size() && fmt[i + 1] == '%')
        {
            i++;
            continue;
        }

        SimplevkPrintfArg arg;
        arg.lanes = 1;
        arg.type = 0;

        if (i + 3 < fmt.size() && fmt[i + 1] == 'v' && fmt[i + 2] >= '2' && fmt[i + 2] <= '4')
        {
            arg.lanes = fmt[i + 2] - '0';
            arg.type = fmt[i + 3];
            i += 3;
        }
        else if (i + 1 < fmt.size())
        {
            arg.type = fmt[i + 1];
            i += 1;
        }
        else
        {
            return -1;
        }

        if (arg.type == 'd' || arg.type == 'i' || arg.type == 'u' || arg.type == 'x' || arg.type == 'X' || arg.type == 'f' || arg.type == 'e' || arg.type == 'g')
        {
            args.push_back(arg);
            payload_words += arg.lanes;
            continue;
        }

        return -1;
    }

    return 0;
}

static uint32_t register_format(const std::string& fmt, const std::vector<SimplevkPrintfArg>& args)
{
    MutexLockGuard lock(g_simplevk_printf_lock);

    SimplevkPrintfFormat f;
    f.id = g_simplevk_printf_next_id++;
    f.format = fmt;
    f.args = args;

    g_simplevk_printf_formats.push_back(f);

    return f.id;
}

static const SimplevkPrintfFormat* find_format(uint32_t id)
{
    for (size_t i = 0; i < g_simplevk_printf_formats.size(); i++)
    {
        if (g_simplevk_printf_formats[i].id == id)
            return &g_simplevk_printf_formats[i];
    }

    return 0;
}

static size_t skip_space(const std::string& s, size_t pos)
{
    while (pos < s.size() && isspace((unsigned char)s[pos]))
        pos++;

    return pos;
}

static int split_call_args(const std::string& call, std::vector<std::string>& args)
{
    args.clear();

    size_t begin = 0;
    int paren_depth = 0;
    int bracket_depth = 0;
    int brace_depth = 0;
    for (size_t i = 0; i < call.size(); i++)
    {
        char ch = call[i];

        if (ch == '"')
        {
            i++;
            while (i < call.size())
            {
                if (call[i] == '\\')
                {
                    i += 2;
                    continue;
                }
                if (call[i] == '"')
                    break;
                i++;
            }
            continue;
        }
        if (ch == '\'')
        {
            i++;
            while (i < call.size())
            {
                if (call[i] == '\\')
                {
                    i += 2;
                    continue;
                }
                if (call[i] == '\'')
                    break;
                i++;
            }
            continue;
        }
        if (ch == '(') paren_depth++;
        if (ch == ')') paren_depth--;
        if (ch == '[') bracket_depth++;
        if (ch == ']') bracket_depth--;
        if (ch == '{') brace_depth++;
        if (ch == '}') brace_depth--;

        if (ch == ',' && paren_depth == 0 && bracket_depth == 0 && brace_depth == 0)
        {
            args.push_back(trim_string(call.substr(begin, i - begin)));
            begin = i + 1;
        }
    }

    args.push_back(trim_string(call.substr(begin)));

    return 0;
}

static std::string make_noop_call()
{
    return "{}";
}

static std::string make_store_expr(const SimplevkPrintfArg& spec, const std::string& expr, int lane)
{
    std::string v = expr;
    if (spec.lanes > 1)
    {
        static const char swizzle[4] = {'x', 'y', 'z', 'w'};
        v = "(" + expr + ").";
        v += swizzle[lane];
    }

    if (spec.type == 'f' || spec.type == 'e' || spec.type == 'g')
        return "floatBitsToUint(float(" + v + "))";

    if (spec.type == 'd' || spec.type == 'i')
        return "uint(int(" + v + "))";

    return "uint(" + v + ")";
}

static std::string rewrite_one_call(const std::string& call, int enable)
{
    std::vector<std::string> args;
    split_call_args(call, args);

    if (args.empty())
        return make_noop_call();

    size_t pos = skip_space(args[0], 0);
    std::string fmt;
    size_t fmt_end = 0;
    if (parse_string_literal(args[0].c_str(), args[0].size(), pos, fmt, fmt_end) != 0)
        return make_noop_call();

    fmt_end = skip_space(args[0], fmt_end);
    if (fmt_end != args[0].size())
        return make_noop_call();

    std::vector<SimplevkPrintfArg> specs;
    int payload_words = 0;
    if (parse_format_args(fmt, specs, payload_words) != 0)
    {
        if (!g_simplevk_printf_warned_unsupported)
        {
            NCNN_LOGE("simplevk shader printf unsupported format %s", fmt.c_str());
            g_simplevk_printf_warned_unsupported = 1;
        }
        return make_noop_call();
    }

    if (args.size() != specs.size() + 1)
    {
        if (!g_simplevk_printf_warned_unsupported)
        {
            NCNN_LOGE("simplevk shader printf argument count mismatch %s", fmt.c_str());
            g_simplevk_printf_warned_unsupported = 1;
        }
        return make_noop_call();
    }

    if (!enable)
        return make_noop_call();

    uint32_t fmt_id = register_format(fmt, specs);

    char tmp[128];
    std::string code;
    code += "{\n";
    sprintf(tmp, "    uint ncnn_printf_base = ncnn_simplevk_printf_reserve(%uu, %uu);\n", fmt_id, (uint32_t)payload_words);
    code += tmp;
    code += "    if (ncnn_printf_base != 0xffffffffu)\n";
    code += "    {\n";

    int word_index = 0;
    for (size_t i = 0; i < specs.size(); i++)
    {
        for (int j = 0; j < specs[i].lanes; j++)
        {
            sprintf(tmp, "        ncnn_simplevk_printf.data[ncnn_printf_base + %uu] = ", (uint32_t)word_index);
            code += tmp;
            code += make_store_expr(specs[i], args[i + 1], j);
            code += ";\n";
            word_index++;
        }
    }

    code += "    }\n";
    code += "}";

    return code;
}

static size_t find_matching_paren(const char* data, size_t size, size_t open_pos)
{
    int depth = 0;
    for (size_t i = open_pos; i < size; i++)
    {
        char ch = data[i];

        if (ch == '/' && i + 1 < size && data[i + 1] == '/')
        {
            i += 2;
            while (i < size && data[i] != '\n')
                i++;
            continue;
        }
        if (ch == '/' && i + 1 < size && data[i + 1] == '*')
        {
            i += 2;
            while (i + 1 < size && !(data[i] == '*' && data[i + 1] == '/'))
                i++;
            i++;
            continue;
        }
        if (ch == '"')
        {
            i++;
            while (i < size)
            {
                if (data[i] == '\\')
                {
                    i++;
                }
                else if (data[i] == '"')
                {
                    break;
                }
                i++;
            }
            continue;
        }
        if (ch == '\'')
        {
            i++;
            while (i < size)
            {
                if (data[i] == '\\')
                {
                    i++;
                }
                else if (data[i] == '\'')
                {
                    break;
                }
                i++;
            }
            continue;
        }

        if (ch == '(')
            depth++;
        else if (ch == ')')
        {
            depth--;
            if (depth == 0)
                return i;
        }
    }

    return (size_t)-1;
}

static int source_contains_identifier(const char* data, int size, const char* ident)
{
    const size_t ident_len = strlen(ident);

    for (int i = 0; i < size; i++)
    {
        char ch = data[i];

        if (ch == '/' && i + 1 < size && data[i + 1] == '/')
        {
            i += 2;
            while (i < size && data[i] != '\n')
                i++;
            continue;
        }
        if (ch == '/' && i + 1 < size && data[i + 1] == '*')
        {
            i += 2;
            while (i + 1 < size && !(data[i] == '*' && data[i + 1] == '/'))
                i++;
            i++;
            continue;
        }
        if (ch == '"')
        {
            i++;
            while (i < size)
            {
                if (data[i] == '\\')
                    i++;
                else if (data[i] == '"')
                    break;
                i++;
            }
            continue;
        }
        if (ch == '\'')
        {
            i++;
            while (i < size)
            {
                if (data[i] == '\\')
                    i++;
                else if (data[i] == '\'')
                    break;
                i++;
            }
            continue;
        }

        if ((i == 0 || !is_ident_char(data[i - 1])) && (size_t)(size - i) >= ident_len && strncmp(data + i, ident, ident_len) == 0 && (i + (int)ident_len == size || !is_ident_char(data[i + ident_len])))
            return 1;
    }

    return 0;
}

int simplevk_printf_source_contains_log(const char* data, int size)
{
    return source_contains_identifier(data, size, "NCNN_LOGE");
}

int simplevk_printf_source_conflicts(const char* data, int size)
{
    return source_contains_identifier(data, size, "ncnn_simplevk_printf");
}

int simplevk_printf_rewrite_shader(const char* data, int size, int enable, std::string& output, int* log_count)
{
    output.clear();
    if (log_count)
        *log_count = 0;

    for (int i = 0; i < size; i++)
    {
        char ch = data[i];

        if (ch == '/' && i + 1 < size && data[i + 1] == '/')
        {
            int begin = i;
            i += 2;
            while (i < size && data[i] != '\n')
                i++;
            if (i < size)
                i++;
            output.append(data + begin, i - begin);
            i--;
            continue;
        }
        if (ch == '/' && i + 1 < size && data[i + 1] == '*')
        {
            int begin = i;
            i += 2;
            while (i + 1 < size && !(data[i] == '*' && data[i + 1] == '/'))
                i++;
            if (i + 1 < size)
                i += 2;
            output.append(data + begin, i - begin);
            i--;
            continue;
        }
        if (ch == '"')
        {
            int begin = i;
            i++;
            while (i < size)
            {
                if (data[i] == '\\')
                    i++;
                else if (data[i] == '"')
                    break;
                i++;
            }
            if (i < size)
                i++;
            output.append(data + begin, i - begin);
            i--;
            continue;
        }
        if (ch == '\'')
        {
            int begin = i;
            i++;
            while (i < size)
            {
                if (data[i] == '\\')
                    i++;
                else if (data[i] == '\'')
                    break;
                i++;
            }
            if (i < size)
                i++;
            output.append(data + begin, i - begin);
            i--;
            continue;
        }

        const char* token = "NCNN_LOGE";
        const size_t token_len = strlen(token);
        if ((i == 0 || !is_ident_char(data[i - 1])) && (size_t)(size - i) >= token_len && strncmp(data + i, token, token_len) == 0 && (i + (int)token_len == size || !is_ident_char(data[i + token_len])))
        {
            int p = i + (int)token_len;
            while (p < size && isspace((unsigned char)data[p]))
                p++;

            if (p < size && data[p] == '(')
            {
                size_t end = find_matching_paren(data, size, p);
                if (end != (size_t)-1)
                {
                    std::string call(data + p + 1, end - p - 1);
                    output += rewrite_one_call(call, enable);
                    if (log_count)
                        (*log_count)++;
                    i = (int)end;
                    continue;
                }
            }
        }

        output += ch;
    }

    return 0;
}

std::string simplevk_printf_glsl_header()
{
    std::string s;
    s += "layout(set = 3, binding = 0, std430) buffer ncnn_simplevk_printf_t\n";
    s += "{\n";
    s += "    uint cursor;\n";
    s += "    uint cap;\n";
    s += "    uint dropped;\n";
    s += "    uint reserved;\n";
    s += "    uint data[];\n";
    s += "} ncnn_simplevk_printf;\n";
    s += "\n";
    s += "uint ncnn_simplevk_printf_reserve(uint fmt_id, uint words)\n";
    s += "{\n";
    s += "    uint pos = atomicAdd(ncnn_simplevk_printf.cursor, words + 2u);\n";
    s += "    if (pos + words + 2u > ncnn_simplevk_printf.cap)\n";
    s += "    {\n";
    s += "        atomicAdd(ncnn_simplevk_printf.dropped, 1u);\n";
    s += "        return 0xffffffffu;\n";
    s += "    }\n";
    s += "\n";
    s += "    ncnn_simplevk_printf.data[pos + 0u] = fmt_id;\n";
    s += "    ncnn_simplevk_printf.data[pos + 1u] = words;\n";
    s += "    return pos + 2u;\n";
    s += "}\n";
    return s;
}

int simplevk_printf_device_supported(const GpuInfo& info)
{
    const VkPhysicalDeviceLimits& limits = info.physicalDeviceProperties().limits;

    if (limits.maxBoundDescriptorSets < 4)
        return 0;
    if (limits.maxPerStageDescriptorStorageBuffers < 1)
        return 0;
    if (limits.maxDescriptorSetStorageBuffers < 1)
        return 0;
    if (limits.maxPerStageResources < 1)
        return 0;

    return 1;
}

int simplevk_printf_shader_info_supported(const GpuInfo& info, const ShaderInfo& shader_info)
{
    if (!simplevk_printf_device_supported(info))
        return 0;

    int storage_buffer_count = 0;
    int resource_count = 0;
    for (int i = 0; i < shader_info.binding_count; i++)
    {
        if (shader_info.binding_types[i] == 0)
            continue;

        resource_count++;
        if (shader_info.binding_types[i] == 1)
            storage_buffer_count++;
    }

    const VkPhysicalDeviceLimits& limits = info.physicalDeviceProperties().limits;

    if (storage_buffer_count + 1 > (int)limits.maxPerStageDescriptorStorageBuffers)
        return 0;
    if (storage_buffer_count + 1 > (int)limits.maxDescriptorSetStorageBuffers)
        return 0;
    if (resource_count + 1 > (int)limits.maxPerStageResources)
        return 0;

    return 1;
}

int simplevk_printf_shader_info_has_printf(const ShaderInfo& shader_info)
{
    return shader_info.reserved_0 == NCNN_SIMPLEVK_PRINTF_SHADER_INFO_FLAG;
}

void simplevk_printf_shader_info_set_has_printf(ShaderInfo& shader_info)
{
    shader_info.reserved_0 = NCNN_SIMPLEVK_PRINTF_SHADER_INFO_FLAG;
}

struct SimplevkPrintfCommand
{
    const VulkanDevice* vkdev;

    VkBuffer buffer;
    VkDeviceMemory memory;
    void* mapped_ptr;
    size_t buffer_size;
    uint32_t data_words;

    VkDescriptorPool descriptor_pool;
    VkDescriptorSet descriptor_set;

    int used;
    int reset_recorded;
    int readback_recorded;
};

SimplevkPrintfCommand* simplevk_printf_command_create(const VulkanDevice* vkdev)
{
    SimplevkPrintfCommand* state = new SimplevkPrintfCommand;
    state->vkdev = vkdev;
    state->buffer = 0;
    state->memory = 0;
    state->mapped_ptr = 0;
    state->buffer_size = 0;
    state->data_words = 0;
    state->descriptor_pool = 0;
    state->descriptor_set = 0;
    state->used = 0;
    state->reset_recorded = 0;
    state->readback_recorded = 0;
    return state;
}

void simplevk_printf_command_destroy(SimplevkPrintfCommand* state)
{
    if (!state)
        return;

    const VulkanDevice* vkdev = state->vkdev;

    if (state->descriptor_set)
    {
        vkFreeDescriptorSets(vkdev->vkdevice(), state->descriptor_pool, 1, &state->descriptor_set);
        state->descriptor_set = 0;
    }
    if (state->descriptor_pool)
    {
        vkDestroyDescriptorPool(vkdev->vkdevice(), state->descriptor_pool, 0);
        state->descriptor_pool = 0;
    }
    if (state->mapped_ptr)
    {
        vkUnmapMemory(vkdev->vkdevice(), state->memory);
        state->mapped_ptr = 0;
    }
    if (state->buffer)
    {
        vkDestroyBuffer(vkdev->vkdevice(), state->buffer, 0);
        state->buffer = 0;
    }
    if (state->memory)
    {
        vkFreeMemory(vkdev->vkdevice(), state->memory, 0);
        state->memory = 0;
    }

    delete state;
}

void simplevk_printf_command_reset(SimplevkPrintfCommand* state)
{
    if (!state)
        return;

    state->used = 0;
    state->reset_recorded = 0;
    state->readback_recorded = 0;
}

int simplevk_printf_command_used(const SimplevkPrintfCommand* state)
{
    return state && state->used;
}

int simplevk_printf_command_ensure(SimplevkPrintfCommand* state)
{
    if (!state)
        return -1;

    if (state->buffer && state->descriptor_set)
        return 0;

    const VulkanDevice* vkdev = state->vkdev;
    const size_t buffer_size = 1024 * 1024;

    if (!state->buffer)
    {
        VkBufferCreateInfo bufferCreateInfo;
        bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCreateInfo.pNext = 0;
        bufferCreateInfo.flags = 0;
        bufferCreateInfo.size = buffer_size;
        bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        bufferCreateInfo.queueFamilyIndexCount = 0;
        bufferCreateInfo.pQueueFamilyIndices = 0;

        VkResult ret = vkCreateBuffer(vkdev->vkdevice(), &bufferCreateInfo, 0, &state->buffer);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkCreateBuffer failed %d", ret);
            return -1;
        }

        VkMemoryRequirements memoryRequirements;
        vkGetBufferMemoryRequirements(vkdev->vkdevice(), state->buffer, &memoryRequirements);

        uint32_t memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (memory_type_index == (uint32_t)-1)
        {
            NCNN_LOGE("simplevk shader printf could not find host coherent memory");
            return -1;
        }

        VkMemoryAllocateInfo memoryAllocateInfo;
        memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        memoryAllocateInfo.pNext = 0;
        memoryAllocateInfo.allocationSize = memoryRequirements.size;
        memoryAllocateInfo.memoryTypeIndex = memory_type_index;

        ret = vkAllocateMemory(vkdev->vkdevice(), &memoryAllocateInfo, 0, &state->memory);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkAllocateMemory failed %d", ret);
            return -1;
        }

        ret = vkBindBufferMemory(vkdev->vkdevice(), state->buffer, state->memory, 0);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkBindBufferMemory failed %d", ret);
            return -1;
        }

        ret = vkMapMemory(vkdev->vkdevice(), state->memory, 0, buffer_size, 0, &state->mapped_ptr);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkMapMemory failed %d", ret);
            return -1;
        }

        state->buffer_size = buffer_size;
        state->data_words = (uint32_t)(buffer_size / 4 - 4);
    }

    if (!state->descriptor_pool)
    {
        VkDescriptorPoolSize poolSize;
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 1;

        VkDescriptorPoolCreateInfo descriptorPoolCreateInfo;
        descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptorPoolCreateInfo.pNext = 0;
        descriptorPoolCreateInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        descriptorPoolCreateInfo.maxSets = 1;
        descriptorPoolCreateInfo.poolSizeCount = 1;
        descriptorPoolCreateInfo.pPoolSizes = &poolSize;

        VkResult ret = vkCreateDescriptorPool(vkdev->vkdevice(), &descriptorPoolCreateInfo, 0, &state->descriptor_pool);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkCreateDescriptorPool failed %d", ret);
            return -1;
        }
    }

    if (!state->descriptor_set)
    {
        VkDescriptorSetLayout descriptorset_layout = vkdev->simplevk_printf_descriptorset_layout();
        if (!descriptorset_layout)
            return -1;

        VkDescriptorSetAllocateInfo descriptorSetAllocateInfo;
        descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocateInfo.pNext = 0;
        descriptorSetAllocateInfo.descriptorPool = state->descriptor_pool;
        descriptorSetAllocateInfo.descriptorSetCount = 1;
        descriptorSetAllocateInfo.pSetLayouts = &descriptorset_layout;

        VkResult ret = vkAllocateDescriptorSets(vkdev->vkdevice(), &descriptorSetAllocateInfo, &state->descriptor_set);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkAllocateDescriptorSets failed %d", ret);
            return -1;
        }

        VkDescriptorBufferInfo descriptorBufferInfo;
        descriptorBufferInfo.buffer = state->buffer;
        descriptorBufferInfo.offset = 0;
        descriptorBufferInfo.range = state->buffer_size;

        VkWriteDescriptorSet writeDescriptorSet;
        writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSet.pNext = 0;
        writeDescriptorSet.dstSet = state->descriptor_set;
        writeDescriptorSet.dstBinding = NCNN_SIMPLEVK_PRINTF_BINDING;
        writeDescriptorSet.dstArrayElement = 0;
        writeDescriptorSet.descriptorCount = 1;
        writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writeDescriptorSet.pImageInfo = 0;
        writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
        writeDescriptorSet.pTexelBufferView = 0;

        vkUpdateDescriptorSets(vkdev->vkdevice(), 1, &writeDescriptorSet, 0, 0);
    }

    return 0;
}

VkDescriptorSet simplevk_printf_command_descriptorset(SimplevkPrintfCommand* state)
{
    if (simplevk_printf_command_ensure(state) != 0)
        return 0;

    state->used = 1;
    return state->descriptor_set;
}

int simplevk_printf_command_record_reset(SimplevkPrintfCommand* state, VkCommandBuffer command_buffer)
{
    if (simplevk_printf_command_ensure(state) != 0)
        return -1;

    if (state->reset_recorded)
        return 0;

    uint32_t* p = (uint32_t*)state->mapped_ptr;
    p[0] = 0;
    p[1] = state->data_words;
    p[2] = 0;
    p[3] = 0;

    VkBufferMemoryBarrier barrier;
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.pNext = 0;
    barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = state->buffer;
    barrier.offset = 0;
    barrier.size = state->buffer_size;

    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, 0, 1, &barrier, 0, 0);

    state->used = 1;
    state->reset_recorded = 1;

    return 0;
}

int simplevk_printf_command_record_bind(SimplevkPrintfCommand* state, VkCommandBuffer command_buffer, VkPipelineLayout pipeline_layout)
{
    if (simplevk_printf_command_ensure(state) != 0)
        return -1;

    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, NCNN_SIMPLEVK_PRINTF_SET, 1, &state->descriptor_set, 0, 0);

    state->used = 1;

    return 0;
}

int simplevk_printf_command_record_readback(SimplevkPrintfCommand* state, VkCommandBuffer command_buffer)
{
    if (!state || !state->used || state->readback_recorded)
        return 0;

    VkBufferMemoryBarrier barrier;
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.pNext = 0;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = state->buffer;
    barrier.offset = 0;
    barrier.size = state->buffer_size;

    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, 0, 1, &barrier, 0, 0);

    state->readback_recorded = 1;

    return 0;
}

static float word_to_float(uint32_t v)
{
    union
    {
        uint32_t u;
        float f;
    } x;
    x.u = v;
    return x.f;
}

static std::string format_scalar(char type, uint32_t word)
{
    char tmp[128];
    if (type == 'd' || type == 'i')
        sprintf(tmp, "%d", (int)word);
    else if (type == 'u')
        sprintf(tmp, "%u", word);
    else if (type == 'x')
        sprintf(tmp, "%x", word);
    else if (type == 'X')
        sprintf(tmp, "%X", word);
    else if (type == 'e')
        sprintf(tmp, "%e", word_to_float(word));
    else if (type == 'g')
        sprintf(tmp, "%g", word_to_float(word));
    else
        sprintf(tmp, "%f", word_to_float(word));

    return std::string(tmp);
}

static std::string format_record(const SimplevkPrintfFormat& fmt, const uint32_t* payload, uint32_t payload_words)
{
    std::string line;
    size_t arg_index = 0;
    uint32_t word_index = 0;

    for (size_t i = 0; i < fmt.format.size(); i++)
    {
        char ch = fmt.format[i];
        if (ch != '%')
        {
            line += ch;
            continue;
        }

        if (i + 1 < fmt.format.size() && fmt.format[i + 1] == '%')
        {
            line += '%';
            i++;
            continue;
        }

        if (arg_index >= fmt.args.size())
            break;

        const SimplevkPrintfArg& arg = fmt.args[arg_index++];
        if (word_index + arg.lanes > payload_words)
            break;

        if (arg.lanes == 1)
        {
            line += format_scalar(arg.type, payload[word_index]);
            word_index++;
            i += 1;
            continue;
        }

        line += "(";
        for (int j = 0; j < arg.lanes; j++)
        {
            if (j != 0)
                line += ",";
            line += format_scalar(arg.type, payload[word_index++]);
        }
        line += ")";
        i += 3;
    }

    return line;
}

int simplevk_printf_command_readback(SimplevkPrintfCommand* state)
{
    if (!state || !state->used || !state->mapped_ptr)
        return 0;

    MutexLockGuard lock(g_simplevk_printf_lock);

    const uint32_t* words = (const uint32_t*)state->mapped_ptr;
    uint32_t cursor = words[0];
    uint32_t cap = words[1];
    uint32_t dropped = words[2];
    if (cursor > cap)
        cursor = cap;

    const uint32_t* data = words + 4;
    uint32_t pos = 0;
    while (pos + 2 <= cursor)
    {
        uint32_t fmt_id = data[pos];
        uint32_t payload_words = data[pos + 1];
        pos += 2;

        if (pos + payload_words > cursor)
            break;

        const SimplevkPrintfFormat* fmt = find_format(fmt_id);
        if (fmt)
        {
            std::string line = format_record(*fmt, data + pos, payload_words);
            NCNN_LOGE("%s", line.c_str());
        }

        pos += payload_words;
    }

    if (dropped != 0 && !g_simplevk_printf_warned_overflow)
    {
        NCNN_LOGE("simplevk shader printf dropped %u records", dropped);
        g_simplevk_printf_warned_overflow = 1;
    }

    simplevk_printf_command_reset(state);

    return 0;
}

} // namespace ncnn

#endif // NCNN_VULKAN
