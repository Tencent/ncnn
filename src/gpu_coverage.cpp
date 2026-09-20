// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gpu.h"

#if NCNN_COVERAGE && NCNN_VULKAN
#include "glslang/SPIRV/spirv.hpp11"

#include <errno.h>
#include <fcntl.h>
#include <initializer_list>
#include <map>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/stat.h>
#if _WIN32
#include <io.h>
#include <process.h>
#else
#include <unistd.h>
#endif

namespace ncnn {

struct ShaderCoverageSource
{
    const char* path;
    uint32_t base;
    uint32_t lines;
};

#include "shader_coverage_data.h"

static const size_t shader_coverage_source_count = sizeof(shader_coverage_sources) / sizeof(ShaderCoverageSource);
static const uint32_t shader_coverage_word_count = (shader_coverage_line_count + 31) / 32;

const char* shader_coverage_source_name(int shader_type_index)
{
    return shader_type_index >= 0 && (size_t)shader_type_index < shader_coverage_source_count ? shader_coverage_sources[shader_type_index].path : "ncnn-user-shader";
}

const char* shader_coverage_map_hash()
{
    return shader_coverage_source_hash;
}

struct ShaderCoverageState
{
    ShaderCoverageState() : executable(shader_coverage_word_count, 0), serial(0)
    {
    }

    Mutex mutex;
    std::vector<uint32_t> executable;
    unsigned int serial;
};

static ShaderCoverageState& coverage_state()
{
    // keep the registry alive for GPU instance destruction in other translation units
    static ShaderCoverageState* state = new ShaderCoverageState;
    return *state;
}

static void append_spirv_instruction(std::vector<uint32_t>& out, spv::Op op, std::initializer_list<uint32_t> args)
{
    out.push_back(((uint32_t)args.size() + 1) << 16 | (uint32_t)op);
    out.insert(out.end(), args.begin(), args.end());
}

static uint32_t get_spirv_constant(std::vector<uint32_t>& definitions, std::map<uint32_t, uint32_t>& constants, uint32_t uint_type, uint32_t value, uint32_t& next_id)
{
    std::map<uint32_t, uint32_t>::const_iterator it = constants.find(value);
    if (it != constants.end())
        return it->second;

    const uint32_t id = next_id++;
    constants[value] = id;
    append_spirv_instruction(definitions, spv::Op::OpConstant, {uint_type, id, value});

    return id;
}

int instrument_shader_coverage(std::vector<uint32_t>& spirv)
{
    if (spirv.size() < 5 || spirv[0] != spv::MagicNumber)
        return -1;

    std::map<uint32_t, const ShaderCoverageSource*> files;
    uint32_t uint_type = 0;
    uint32_t uint_ptr = 0;
    size_t types_begin = 0;
    size_t functions_begin = 0;
    bool buffer_block = false;
    bool vulkan_memory_model = false;
    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t* p = &spirv[i];
        const uint32_t wordcount = p[0] >> 16;
        const spv::Op op = (spv::Op)(p[0] & 0xffff);
        if (!wordcount || wordcount > spirv.size() - i)
            return -1;
        if (op == spv::Op::OpString && wordcount >= 3)
        {
            const char* name = (const char*)(p + 2);
            if (!memchr(name, 0, (wordcount - 2) * sizeof(uint32_t)))
                return -1;
            for (size_t f = 0; f < shader_coverage_source_count; f++)
            {
                if (strcmp(name, shader_coverage_sources[f].path) == 0)
                    files[p[1]] = &shader_coverage_sources[f];
            }
        }
        if (op == spv::Op::OpTypeInt && wordcount == 4 && p[2] == 32 && p[3] == 0)
            uint_type = p[1];
        if (op == spv::Op::OpMemoryModel && wordcount == 3)
            vulkan_memory_model = p[2] == (uint32_t)spv::MemoryModel::VulkanKHR;
        if (!types_begin && op >= spv::Op::OpTypeVoid && op <= spv::Op::OpTypeForwardPointer)
            types_begin = i;
        if (!functions_begin && op == spv::Op::OpFunction)
            functions_begin = i;
        if (op == spv::Op::OpDecorate && wordcount >= 3)
        {
            if (p[2] == (uint32_t)spv::Decoration::BufferBlock)
                buffer_block = true;
            if (wordcount == 4 && p[2] == (uint32_t)spv::Decoration::DescriptorSet && p[3] == 1)
            {
                NCNN_LOGE("shader coverage reserves descriptor set 1");
                return -1;
            }
        }
        i += wordcount;
    }
    if (!types_begin || !functions_begin)
        return 0;

    const uint32_t storage = spirv[1] < 0x00010300 || buffer_block ? (uint32_t)spv::StorageClass::Uniform : (uint32_t)spv::StorageClass::StorageBuffer;
    for (size_t i = types_begin; i < functions_begin; i += spirv[i] >> 16)
    {
        const uint32_t* p = &spirv[i];
        if ((spv::Op)(p[0] & 0xffff) == spv::Op::OpTypePointer && p[2] == storage && p[3] == uint_type)
            uint_ptr = p[1];
    }

    uint32_t next_id = spirv[3];
    std::vector<uint32_t> definitions;
    if (!uint_type)
    {
        uint_type = next_id++;
        append_spirv_instruction(definitions, spv::Op::OpTypeInt, {uint_type, 32, 0});
    }
    if (!uint_ptr)
    {
        uint_ptr = next_id++;
        append_spirv_instruction(definitions, spv::Op::OpTypePointer, {uint_ptr, storage, uint_type});
    }
    const uint32_t array_type = next_id++;
    const uint32_t struct_type = next_id++;
    const uint32_t struct_ptr = next_id++;
    const uint32_t coverage_var = next_id++;
    append_spirv_instruction(definitions, spv::Op::OpTypeRuntimeArray, {array_type, uint_type});
    append_spirv_instruction(definitions, spv::Op::OpTypeStruct, {struct_type, array_type});
    append_spirv_instruction(definitions, spv::Op::OpTypePointer, {struct_ptr, storage, struct_type});
    append_spirv_instruction(definitions, spv::Op::OpVariable, {struct_ptr, coverage_var, storage});

    std::vector<uint32_t> decorations;
    append_spirv_instruction(decorations, spv::Op::OpDecorate, {array_type, (uint32_t)spv::Decoration::ArrayStride, 4});
    append_spirv_instruction(decorations, spv::Op::OpDecorate, {struct_type, storage == (uint32_t)spv::StorageClass::Uniform ? (uint32_t)spv::Decoration::BufferBlock : (uint32_t)spv::Decoration::Block});
    append_spirv_instruction(decorations, spv::Op::OpMemberDecorate, {struct_type, 0, (uint32_t)spv::Decoration::Offset, 0});
    append_spirv_instruction(decorations, spv::Op::OpDecorate, {coverage_var, (uint32_t)spv::Decoration::DescriptorSet, 1});
    append_spirv_instruction(decorations, spv::Op::OpDecorate, {coverage_var, (uint32_t)spv::Decoration::Binding, 0});

    std::map<uint32_t, uint32_t> constants;
    const uint32_t zero = get_spirv_constant(definitions, constants, uint_type, 0, next_id);
    // use the compute queue family scope to avoid requiring the optional DeviceScope feature
    const uint32_t scope = get_spirv_constant(definitions, constants, uint_type, vulkan_memory_model ? (uint32_t)spv::Scope::QueueFamilyKHR : (uint32_t)spv::Scope::Device, next_id);
    std::vector<uint32_t> functions;
    std::vector<uint32_t> executable(shader_coverage_word_count, 0);
    std::set<uint32_t> seen;
    std::set<uint32_t> pending;
    uint32_t file = 0;
    uint32_t line = 0;
    bool in_block = false;
    bool after_merge = false;
    for (size_t i = functions_begin; i < spirv.size();)
    {
        const uint32_t* p = &spirv[i];
        const uint32_t wordcount = p[0] >> 16;
        const spv::Op op = (spv::Op)(p[0] & 0xffff);
        if (op == spv::Op::OpLine)
        {
            file = p[1];
            line = p[2];
        }
        else if (op == spv::Op::OpNoLine)
        {
            file = line = 0;
        }
        else if (op == spv::Op::OpLabel)
        {
            in_block = true;
            after_merge = false;
            seen.clear();
            pending.clear();
        }
        else if (in_block && op != spv::Op::OpVariable && !after_merge)
        {
            std::map<uint32_t, const ShaderCoverageSource*>::const_iterator it = files.find(file);
            if (it != files.end() && line > 0 && line <= it->second->lines)
            {
                const uint32_t id = it->second->base + line - 1;
                if (seen.insert(id).second)
                    pending.insert(id);
            }
            // insert probes after phi nodes and entry block variables
            if (op != spv::Op::OpPhi)
            {
                for (std::set<uint32_t>::const_iterator j = pending.begin(); j != pending.end(); j++)
                {
                    const uint32_t id = *j;
                    const uint32_t ptr = next_id++;
                    const uint32_t result = next_id++;
                    const uint32_t word_index = get_spirv_constant(definitions, constants, uint_type, id >> 5, next_id);
                    append_spirv_instruction(functions, spv::Op::OpAccessChain, {uint_ptr, ptr, coverage_var, zero, word_index});
                    const uint32_t mask = get_spirv_constant(definitions, constants, uint_type, 1u << (id & 31), next_id);
                    append_spirv_instruction(functions, spv::Op::OpAtomicOr, {uint_type, result, ptr, scope, zero, mask});
                    executable[id >> 5] |= 1u << (id & 31);
                }
                pending.clear();
            }
        }
        functions.insert(functions.end(), p, p + wordcount);
        if (op == spv::Op::OpLoopMerge || op == spv::Op::OpSelectionMerge)
            after_merge = true;
        if (op == spv::Op::OpBranch || op == spv::Op::OpBranchConditional || op == spv::Op::OpSwitch || op == spv::Op::OpReturn || op == spv::Op::OpReturnValue || op == spv::Op::OpKill || op == spv::Op::OpUnreachable || op == spv::Op::OpFunctionEnd)
        {
            // reset source locations at block terminators
            in_block = false;
            file = line = 0;
        }
        i += wordcount;
    }

    std::vector<uint32_t> result(spirv.begin(), spirv.begin() + 5);
    for (size_t i = 5; i < functions_begin;)
    {
        const uint32_t* p = &spirv[i];
        const uint32_t wordcount = p[0] >> 16;
        if (i == types_begin)
            result.insert(result.end(), decorations.begin(), decorations.end());
        // include all statically used globals in the SPIR-V 1.4+ entry point interface
        if ((spv::Op)(p[0] & 0xffff) == spv::Op::OpEntryPoint && spirv[1] >= 0x00010400)
        {
            result.push_back(((wordcount + 1) << 16) | (uint32_t)spv::Op::OpEntryPoint);
            result.insert(result.end(), p + 1, p + wordcount);
            result.push_back(coverage_var);
        }
        else
            result.insert(result.end(), p, p + wordcount);
        i += wordcount;
    }
    result.insert(result.end(), definitions.begin(), definitions.end());
    result.insert(result.end(), functions.begin(), functions.end());
    result[3] = next_id;
    spirv.swap(result);

    ShaderCoverageState& state = coverage_state();
    MutexLockGuard lock(state.mutex);
    for (uint32_t i = 0; i < shader_coverage_word_count; i++)
        state.executable[i] |= executable[i];
    return 0;
}

void register_shader_coverage(const uint32_t* spirv, size_t spv_word_count)
{
    // recover executable bits from probes in newly compiled or cached shaders
    std::set<uint32_t> coverage_vars;
    std::map<uint32_t, uint32_t> constants;
    std::map<uint32_t, uint32_t> pointers;
    ShaderCoverageState& state = coverage_state();
    MutexLockGuard lock(state.mutex);
    for (size_t i = 5; i < spv_word_count;)
    {
        const uint32_t* p = spirv + i;
        const uint32_t wordcount = p[0] >> 16;
        const spv::Op op = (spv::Op)(p[0] & 0xffff);
        if (!wordcount || wordcount > spv_word_count - i)
            return;
        if (op == spv::Op::OpDecorate && wordcount == 4 && p[2] == (uint32_t)spv::Decoration::DescriptorSet && p[3] == 1)
            coverage_vars.insert(p[1]);
        if (op == spv::Op::OpConstant && wordcount == 4)
            constants[p[2]] = p[3];
        if (op == spv::Op::OpAccessChain && wordcount == 6 && coverage_vars.count(p[3]) && constants.count(p[5]))
            pointers[p[2]] = constants[p[5]];
        if (op == spv::Op::OpAtomicOr && wordcount == 7 && pointers.count(p[3]) && constants.count(p[6]))
        {
            const uint32_t word = pointers[p[3]];
            if (word < shader_coverage_word_count)
                state.executable[word] |= constants[p[6]];
        }
        i += wordcount;
    }
}

// one persistent bitset per VulkanDevice, updated atomically by shaders
class VulkanShaderCoverage
{
public:
    explicit VulkanShaderCoverage(const VulkanDevice* vkdev);
    ~VulkanShaderCoverage();
    int create();
    void host_read_barrier(VkCommandBuffer command_buffer) const;

    VkDescriptorSetLayout descriptorset_layout;
    VkDescriptorSetLayout empty_descriptorset_layout;
    VkDescriptorSet descriptorset;

private:
    void write_report() const;

    const VulkanDevice* vkdev;
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDescriptorPool descriptor_pool;
    void* mapped_ptr;
};

VulkanShaderCoverage::VulkanShaderCoverage(const VulkanDevice* _vkdev)
    : descriptorset_layout(0), empty_descriptorset_layout(0), descriptorset(0), vkdev(_vkdev), buffer(0), memory(0), descriptor_pool(0), mapped_ptr(0)
{
}

int VulkanShaderCoverage::create()
{
    const VkDevice device = vkdev->vkdevice();
    const VkDeviceSize size = shader_coverage_word_count * sizeof(uint32_t);
    if (size > vkdev->info.physicalDeviceProperties().limits.maxStorageBufferRange)
        return -1;

    VkBufferCreateInfo bufferCreateInfo;
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.pNext = 0;
    bufferCreateInfo.flags = 0;
    bufferCreateInfo.size = size;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufferCreateInfo.queueFamilyIndexCount = 0;
    bufferCreateInfo.pQueueFamilyIndices = 0;

    VkResult ret = vkCreateBuffer(device, &bufferCreateInfo, 0, &buffer);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateBuffer failed %d", ret);
        return -1;
    }

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);

    VkMemoryAllocateInfo memoryAllocateInfo;
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.pNext = 0;
    memoryAllocateInfo.allocationSize = memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
    if (memoryAllocateInfo.memoryTypeIndex == (uint32_t)-1)
        return -1;

    ret = vkAllocateMemory(device, &memoryAllocateInfo, 0, &memory);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkAllocateMemory failed %d", ret);
        return -1;
    }

    ret = vkBindBufferMemory(device, buffer, memory, 0);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkBindBufferMemory failed %d", ret);
        return -1;
    }

    ret = vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &mapped_ptr);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkMapMemory failed %d", ret);
        return -1;
    }

    memset(mapped_ptr, 0, (size_t)size);

    VkMappedMemoryRange mappedMemoryRange;
    mappedMemoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    mappedMemoryRange.pNext = 0;
    mappedMemoryRange.memory = memory;
    mappedMemoryRange.offset = 0;
    mappedMemoryRange.size = VK_WHOLE_SIZE;

    ret = vkFlushMappedMemoryRanges(device, 1, &mappedMemoryRange);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkFlushMappedMemoryRanges failed %d", ret);
        return -1;
    }

    VkDescriptorSetLayoutBinding descriptorSetLayoutBinding;
    descriptorSetLayoutBinding.binding = 0;
    descriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorSetLayoutBinding.descriptorCount = 1;
    descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    descriptorSetLayoutBinding.pImmutableSamplers = 0;

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo;
    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.pNext = 0;
    descriptorSetLayoutCreateInfo.flags = 0;
    descriptorSetLayoutCreateInfo.bindingCount = 0;
    descriptorSetLayoutCreateInfo.pBindings = 0;

    ret = vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, 0, &empty_descriptorset_layout);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateDescriptorSetLayout failed %d", ret);
        return -1;
    }

    descriptorSetLayoutCreateInfo.bindingCount = 1;
    descriptorSetLayoutCreateInfo.pBindings = &descriptorSetLayoutBinding;

    ret = vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, 0, &descriptorset_layout);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateDescriptorSetLayout failed %d", ret);
        return -1;
    }

    VkDescriptorPoolSize descriptorPoolSize;
    descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorPoolSize.descriptorCount = 1;

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo;
    descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.pNext = 0;
    descriptorPoolCreateInfo.flags = 0;
    descriptorPoolCreateInfo.maxSets = 1;
    descriptorPoolCreateInfo.poolSizeCount = 1;
    descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize;

    ret = vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, 0, &descriptor_pool);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateDescriptorPool failed %d", ret);
        return -1;
    }

    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo;
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.pNext = 0;
    descriptorSetAllocateInfo.descriptorPool = descriptor_pool;
    descriptorSetAllocateInfo.descriptorSetCount = 1;
    descriptorSetAllocateInfo.pSetLayouts = &descriptorset_layout;

    ret = vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, &descriptorset);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkAllocateDescriptorSets failed %d", ret);
        return -1;
    }

    VkDescriptorBufferInfo descriptorBufferInfo;
    descriptorBufferInfo.buffer = buffer;
    descriptorBufferInfo.offset = 0;
    descriptorBufferInfo.range = size;

    VkWriteDescriptorSet writeDescriptorSet;
    writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet.pNext = 0;
    writeDescriptorSet.dstSet = descriptorset;
    writeDescriptorSet.dstBinding = 0;
    writeDescriptorSet.dstArrayElement = 0;
    writeDescriptorSet.descriptorCount = 1;
    writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet.pImageInfo = 0;
    writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
    writeDescriptorSet.pTexelBufferView = 0;

    vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, 0);

    return 0;
}

void VulkanShaderCoverage::host_read_barrier(VkCommandBuffer command_buffer) const
{
    VkBufferMemoryBarrier bufferMemoryBarrier;
    bufferMemoryBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bufferMemoryBarrier.pNext = 0;
    bufferMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bufferMemoryBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    bufferMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferMemoryBarrier.buffer = buffer;
    bufferMemoryBarrier.offset = 0;
    bufferMemoryBarrier.size = VK_WHOLE_SIZE;

    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, 0, 1, &bufferMemoryBarrier, 0, 0);
}

void VulkanShaderCoverage::write_report() const
{
    const char* directory = getenv("NCNN_SHADER_COVERAGE_DIR");
    if (!directory || !directory[0])
        directory = ".";
    ShaderCoverageState& state = coverage_state();
    MutexLockGuard lock(state.mutex);

    bool has_executable_lines = false;
    for (uint32_t i = 0; i < shader_coverage_word_count; i++)
    {
        if (state.executable[i])
        {
            has_executable_lines = true;
            break;
        }
    }
    if (!has_executable_lines)
        return;

    FILE* fp = 0;
    std::string path;
    // exclusive creation handles pid reuse and multiple device lifetimes
    for (;;)
    {
        char filename[96];
#if _WIN32
        const int pid = _getpid();
#else
        const int pid = (int)getpid();
#endif
        snprintf(filename, sizeof(filename), "/shader-coverage-%d-%u.info", pid, state.serial++);
        path = std::string(directory) + filename;
#if _WIN32
        const int fd = _open(path.c_str(), _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY, _S_IREAD | _S_IWRITE);
#else
        const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0666);
#endif
        if (fd < 0)
        {
            if (errno == EEXIST)
                continue;
            NCNN_LOGE("cannot create shader coverage report %s: %s", path.c_str(), strerror(errno));
            return;
        }
#if _WIN32
        fp = _fdopen(fd, "wb");
        if (!fp)
            _close(fd);
#else
        fp = fdopen(fd, "wb");
        if (!fp)
            close(fd);
#endif
        break;
    }
    bool ok = fp != 0;
    if (ok)
    {
        const uint32_t* hits = (const uint32_t*)mapped_ptr;
        for (size_t i = 0; ok && i < shader_coverage_source_count; i++)
        {
            const ShaderCoverageSource& source = shader_coverage_sources[i];
            uint32_t found = 0;
            uint32_t covered = 0;
            for (uint32_t line = 0; ok && line < source.lines; line++)
            {
                const uint32_t id = source.base + line;
                const uint32_t mask = 1u << (id & 31);
                if (!(state.executable[id >> 5] & mask))
                    continue;
                if (found == 0 && fprintf(fp, "TN:\nSF:%s/%s\n", shader_coverage_source_root, source.path) < 0)
                {
                    ok = false;
                    break;
                }
                const unsigned int hit = (hits[id >> 5] & mask) ? 1 : 0;
                ok = fprintf(fp, "DA:%u,%u\n", line + 1, hit) >= 0;
                found++;
                covered += hit;
            }
            if (ok && found)
                ok = fprintf(fp, "LF:%u\nLH:%u\nend_of_record\n", found, covered) >= 0;
        }
        if (fclose(fp) != 0)
            ok = false;
    }
    if (!ok)
    {
        NCNN_LOGE("failed writing shader coverage report %s", path.c_str());
        remove(path.c_str());
    }
}

VulkanShaderCoverage::~VulkanShaderCoverage()
{
    const VkDevice device = vkdev->vkdevice();
    if (descriptorset && vkDeviceWaitIdle(device) == VK_SUCCESS)
    {
        VkMappedMemoryRange mappedMemoryRange;
        mappedMemoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        mappedMemoryRange.pNext = 0;
        mappedMemoryRange.memory = memory;
        mappedMemoryRange.offset = 0;
        mappedMemoryRange.size = VK_WHOLE_SIZE;

        VkResult ret = vkInvalidateMappedMemoryRanges(device, 1, &mappedMemoryRange);
        if (ret == VK_SUCCESS)
            write_report();
        else
            NCNN_LOGE("vkInvalidateMappedMemoryRanges failed %d", ret);
    }

    if (descriptor_pool)
    {
        vkDestroyDescriptorPool(device, descriptor_pool, 0);
    }
    if (descriptorset_layout)
    {
        vkDestroyDescriptorSetLayout(device, descriptorset_layout, 0);
    }
    if (empty_descriptorset_layout)
    {
        vkDestroyDescriptorSetLayout(device, empty_descriptorset_layout, 0);
    }
    if (mapped_ptr)
    {
        vkUnmapMemory(device, memory);
    }
    if (buffer)
    {
        vkDestroyBuffer(device, buffer, 0);
    }
    if (memory)
    {
        vkFreeMemory(device, memory, 0);
    }
}

VulkanShaderCoverage* create_shader_coverage(const VulkanDevice* vkdev)
{
    VulkanShaderCoverage* coverage = new VulkanShaderCoverage(vkdev);
    if (coverage->create() != 0)
    {
        delete coverage;
        return 0;
    }

    return coverage;
}

void destroy_shader_coverage(VulkanShaderCoverage* coverage)
{
    delete coverage;
}

void get_shader_coverage_descriptorset_layouts(const VulkanShaderCoverage* coverage, VkDescriptorSetLayout descriptorset_layout, VkDescriptorSetLayout* layouts)
{
    layouts[0] = descriptorset_layout ? descriptorset_layout : coverage->empty_descriptorset_layout;
    layouts[1] = coverage->descriptorset_layout;
}

void bind_shader_coverage_descriptorset(const VulkanShaderCoverage* coverage, VkCommandBuffer command_buffer, VkPipelineLayout pipeline_layout)
{
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 1, 1, &coverage->descriptorset, 0, 0);
}

void record_shader_coverage_barrier(const VulkanShaderCoverage* coverage, VkCommandBuffer command_buffer)
{
    coverage->host_read_barrier(command_buffer);
}

} // namespace ncnn
#endif // NCNN_COVERAGE && NCNN_VULKAN
