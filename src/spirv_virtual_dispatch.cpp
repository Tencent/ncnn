// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "spirv_virtual_dispatch.h"

#if NCNN_VULKAN

#include "glslang/SPIRV/spirv.hpp11"

#include <string.h>

namespace ncnn {

static void append_instruction(std::vector<uint32_t>& out, spv::Op op, uint32_t wc,
                               uint32_t a = 0, uint32_t b = 0, uint32_t c = 0, uint32_t d = 0,
                               uint32_t e = 0, uint32_t f = 0, uint32_t g = 0)
{
    const uint32_t operands[7] = {a, b, c, d, e, f, g};
    out.push_back((wc << 16) | (uint32_t)op);
    for (uint32_t i = 0; i + 1 < wc; i++)
        out.push_back(operands[i]);
}

static void append_words(std::vector<uint32_t>& out, const uint32_t* p, uint32_t wc)
{
    for (uint32_t i = 0; i < wc; i++)
        out.push_back(p[i]);
}

enum DispatchBuiltin
{
    DISPATCH_NONE = 0,
    DISPATCH_WORKGROUP_ID,
    DISPATCH_GLOBAL_ID,
    DISPATCH_NUM_WORKGROUPS
};

static uint32_t load_dispatch_constant(std::vector<uint32_t>& out, uint32_t& next_id,
                                       uint32_t uint_type, uint32_t ptr_type, uint32_t pc_var,
                                       uint32_t member_index, uint32_t result_id = 0)
{
    const uint32_t ptr = next_id++;
    if (!result_id)
        result_id = next_id++;
    append_instruction(out, spv::Op::OpAccessChain, 5, ptr_type, ptr, pc_var, member_index);
    append_instruction(out, spv::Op::OpLoad, 4, uint_type, result_id, ptr);
    return result_id;
}

// Rewrite loads rather than the Input variables themselves. In particular, keep
// the original storage classes, entry point, local size and subgroup semantics.
int inject_virtual_dispatch(const uint32_t* code, size_t size, int push_constant_count,
                            uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z,
                            std::vector<uint32_t>& out)
{
    if (size < 5 * sizeof(uint32_t) || size % sizeof(uint32_t) || push_constant_count < 0)
        return -1;

    const size_t words = size / sizeof(uint32_t);
    const uint32_t bound = code[3];
    if (!bound)
        return -1;

    uint32_t next_id = bound;
    uint32_t uint_type = 0;
    uint32_t vec_type = 0;
    uint32_t pc_var = 0;
    uint32_t pc_struct = 0;
    uint32_t pc_uint_ptr = 0;
    uint32_t pc_var_ptr = 0;
    size_t first_type = words;
    size_t first_function = words;

    std::vector<uint32_t> pointer_pointee(bound, 0);
    std::vector<uint32_t> constant_value(bound, 0);
    std::vector<unsigned char> is_constant(bound, 0);
    std::vector<unsigned char> is_scalar32(bound, 0);
    std::vector<int> builtin(bound, DISPATCH_NONE);
    std::vector<uint32_t> builtin_var(bound, 0);
    std::vector<uint32_t> component_index(bound, 0);

    // Gather the scalar types, push constant variable, and decorated input IDs.
    for (size_t i = 5; i < words;)
    {
        const uint32_t* p = code + i;
        const uint32_t wc = p[0] >> 16;
        const spv::Op op = (spv::Op)(p[0] & 0xffff);
        if (!wc || wc > words - i)
            return -1;

        if (op >= spv::Op::OpTypeVoid && op <= spv::Op::OpTypeForwardPointer && first_type == words)
            first_type = i;
        if (op == spv::Op::OpFunction && first_function == words)
            first_function = i;

        if (op == spv::Op::OpTypeInt && wc == 4 && p[2] == 32)
        {
            is_scalar32[p[1]] = 1;
            if (p[3] == 0)
                uint_type = p[1];
        }
        if (op == spv::Op::OpTypeFloat && wc == 3 && p[2] == 32)
            is_scalar32[p[1]] = 1;
        if (op == spv::Op::OpTypePointer && wc == 4)
            pointer_pointee[p[1]] = p[3];
        if (op == spv::Op::OpConstant && wc == 4)
        {
            is_constant[p[2]] = 1;
            constant_value[p[2]] = p[3];
        }
        if (op == spv::Op::OpVariable && wc >= 4 && p[3] == (uint32_t)spv::StorageClass::PushConstant)
        {
            if (pc_var)
                return -1;
            pc_var = p[2];
            pc_var_ptr = p[1];
            pc_struct = pointer_pointee[pc_var_ptr];
        }
        if (op == spv::Op::OpDecorate && wc == 4 && p[2] == (uint32_t)spv::Decoration::BuiltIn)
        {
            int kind = DISPATCH_NONE;
            if (p[3] == (uint32_t)spv::BuiltIn::WorkgroupId)
                kind = DISPATCH_WORKGROUP_ID;
            if (p[3] == (uint32_t)spv::BuiltIn::GlobalInvocationId)
                kind = DISPATCH_GLOBAL_ID;
            if (p[3] == (uint32_t)spv::BuiltIn::NumWorkgroups)
                kind = DISPATCH_NUM_WORKGROUPS;
            if (kind != DISPATCH_NONE)
            {
                builtin[p[1]] = kind;
                builtin_var[p[1]] = p[1];
            }
        }
        i += wc;
    }

    if (first_type == words || first_function == words || (!pc_var && push_constant_count != 0))
        return -1;

    const bool new_pc = pc_var == 0;
    if (!uint_type)
        uint_type = next_id++;

    std::vector<unsigned char> member_offsets(push_constant_count, 0);
    bool found_pc_struct = false;
    // Check the ncnn scalar push constant ABI, and recognize component pointers.
    for (size_t i = 5; i < words;)
    {
        const uint32_t* p = code + i;
        const uint32_t wc = p[0] >> 16;
        const spv::Op op = (spv::Op)(p[0] & 0xffff);
        if (op == spv::Op::OpTypeVector && wc == 4 && p[2] == uint_type && p[3] == 3)
            vec_type = p[1];
        if (op == spv::Op::OpTypePointer && wc == 4 && p[2] == (uint32_t)spv::StorageClass::PushConstant && p[3] == uint_type)
            pc_uint_ptr = p[1];

        if (op == spv::Op::OpTypeStruct && p[1] == pc_struct)
        {
            if (wc != (uint32_t)push_constant_count + 2)
                return -1;
            for (uint32_t j = 2; j < wc; j++)
                if (!is_scalar32[p[j]])
                    return -1;
            found_pc_struct = true;
        }
        if (op == spv::Op::OpMemberDecorate && wc == 5 && p[1] == pc_struct && p[3] == (uint32_t)spv::Decoration::Offset)
        {
            if (p[2] >= (uint32_t)push_constant_count || p[4] != p[2] * 4)
                return -1;
            member_offsets[p[2]] = 1;
        }
        if ((op == spv::Op::OpAccessChain || op == spv::Op::OpInBoundsAccessChain) && wc >= 4 && builtin[p[3]] != DISPATCH_NONE)
        {
            if (wc != 5 || component_index[p[3]])
                return -1;
            builtin[p[2]] = builtin[p[3]];
            builtin_var[p[2]] = builtin_var[p[3]];
            component_index[p[2]] = p[4];
        }
        i += wc;
    }

    if (!new_pc && !found_pc_struct)
        return -1;
    for (int i = 0; i < push_constant_count; i++)
        if (!member_offsets[i])
            return -1;

    if (new_pc)
    {
        pc_struct = next_id++;
        pc_var_ptr = next_id++;
        pc_var = next_id++;
    }
    const bool new_pc_uint_ptr = pc_uint_ptr == 0;
    if (new_pc_uint_ptr)
        pc_uint_ptr = next_id++;

    const uint32_t local_sizes[3] = {local_size_x, local_size_y, local_size_z};
    uint32_t member_ids[VIRTUAL_DISPATCH_CONSTANT_COUNT];
    uint32_t local_size_ids[3];
    for (int i = 0; i < VIRTUAL_DISPATCH_CONSTANT_COUNT; i++)
        member_ids[i] = next_id++;
    for (int i = 0; i < 3; i++)
        local_size_ids[i] = next_id++;

    out.clear();
    append_words(out, code, 5);

    for (size_t i = 5; i < words;)
    {
        const uint32_t* p = code + i;
        const uint32_t wc = p[0] >> 16;
        const spv::Op op = (spv::Op)(p[0] & 0xffff);

        if (i == first_type)
        {
            if (new_pc)
                append_instruction(out, spv::Op::OpDecorate, 3, pc_struct, (uint32_t)spv::Decoration::Block);
            for (int j = 0; j < VIRTUAL_DISPATCH_CONSTANT_COUNT; j++)
            {
                const uint32_t member = push_constant_count + j;
                append_instruction(out, spv::Op::OpMemberDecorate, 5, pc_struct, member, (uint32_t)spv::Decoration::Offset, member * 4);
            }

            // The existing push constant struct may precede the original uint type.
            // Move that independent scalar type to the beginning of the types.
            append_instruction(out, spv::Op::OpTypeInt, 4, uint_type, 32, 0);
        }

        if (i == first_function)
        {
            if (new_pc)
            {
                append_instruction(out, spv::Op::OpTypeStruct, 8, pc_struct, uint_type, uint_type, uint_type, uint_type, uint_type, uint_type);
                append_instruction(out, spv::Op::OpTypePointer, 4, pc_var_ptr, (uint32_t)spv::StorageClass::PushConstant, pc_struct);
            }
            if (new_pc_uint_ptr)
                append_instruction(out, spv::Op::OpTypePointer, 4, pc_uint_ptr, (uint32_t)spv::StorageClass::PushConstant, uint_type);
            for (int j = 0; j < VIRTUAL_DISPATCH_CONSTANT_COUNT; j++)
                append_instruction(out, spv::Op::OpConstant, 4, uint_type, member_ids[j], push_constant_count + j);
            for (int j = 0; j < 3; j++)
                append_instruction(out, spv::Op::OpConstant, 4, uint_type, local_size_ids[j], local_sizes[j]);
            if (new_pc)
                append_instruction(out, spv::Op::OpVariable, 4, pc_var_ptr, pc_var, (uint32_t)spv::StorageClass::PushConstant);
        }

        // SPIR-V 1.4+ lists all statically used global variables in OpEntryPoint.
        if (op == spv::Op::OpEntryPoint && code[1] >= 0x00010400)
        {
            bool listed = false;
            const size_t name_words = (strlen((const char*)(p + 3)) + 4) / 4;
            for (size_t j = 3 + name_words; j < wc; j++)
                if (p[j] == pc_var)
                    listed = true;
            if (!listed)
            {
                out.push_back(((wc + 1) << 16) | (uint32_t)op);
                append_words(out, p + 1, wc - 1);
                out.push_back(pc_var);
                i += wc;
                continue;
            }
        }

        if (op == spv::Op::OpTypeInt && p[1] == uint_type)
        {
            i += wc;
            continue;
        }

        if (op == spv::Op::OpTypeStruct && p[1] == pc_struct)
        {
            out.push_back(((wc + VIRTUAL_DISPATCH_CONSTANT_COUNT) << 16) | (uint32_t)op);
            append_words(out, p + 1, wc - 1);
            for (int j = 0; j < VIRTUAL_DISPATCH_CONSTANT_COUNT; j++)
                out.push_back(uint_type);
            i += wc;
            continue;
        }

        if (op == spv::Op::OpLoad && wc >= 4 && builtin[p[3]] != DISPATCH_NONE)
        {
            const uint32_t result_type = p[1];
            const uint32_t result = p[2];
            const uint32_t ptr = p[3];
            const uint32_t index = component_index[ptr];
            const int kind = builtin[ptr];
            const bool count = kind == DISPATCH_NUM_WORKGROUPS;
            const bool global = kind == DISPATCH_GLOBAL_ID;
            const int member_base = count ? 3 : 0;

            // Most GLSL accesses are individual constant components. Avoid
            // loading the other two components or adding vector operations.
            if (index && is_constant[index] && constant_value[index] < 3)
            {
                if (result_type != uint_type)
                    return -1;
                const uint32_t axis = constant_value[index];
                uint32_t offset = load_dispatch_constant(out, next_id, uint_type, pc_uint_ptr, pc_var, member_ids[member_base + axis], count ? result : 0);
                if (!count)
                {
                    if (global && local_sizes[axis] != 1)
                    {
                        const uint32_t scaled = next_id++;
                        append_instruction(out, spv::Op::OpIMul, 5, uint_type, scaled, offset, local_size_ids[axis]);
                        offset = scaled;
                    }
                    const uint32_t physical = next_id++;
                    out.push_back(p[0]);
                    out.push_back(result_type);
                    out.push_back(physical);
                    append_words(out, p + 3, wc - 3);
                    append_instruction(out, spv::Op::OpIAdd, 5, uint_type, result, physical, offset);
                }
            }
            else
            {
                if (!vec_type || (index ? result_type != uint_type : result_type != vec_type))
                    return -1;

                uint32_t components[3];
                for (int j = 0; j < 3; j++)
                {
                    components[j] = load_dispatch_constant(out, next_id, uint_type, pc_uint_ptr, pc_var, member_ids[member_base + j]);
                    if (global && local_sizes[j] != 1)
                    {
                        const uint32_t scaled = next_id++;
                        append_instruction(out, spv::Op::OpIMul, 5, uint_type, scaled, components[j], local_size_ids[j]);
                        components[j] = scaled;
                    }
                }
                const uint32_t logical = index ? next_id++ : result;
                const uint32_t offset = count ? logical : next_id++;
                append_instruction(out, spv::Op::OpCompositeConstruct, 6, vec_type, offset, components[0], components[1], components[2]);
                if (!count)
                {
                    const uint32_t physical = next_id++;
                    append_instruction(out, spv::Op::OpLoad, 4, vec_type, physical, builtin_var[ptr]);
                    append_instruction(out, spv::Op::OpIAdd, 5, vec_type, logical, physical, offset);
                }
                if (index)
                    append_instruction(out, spv::Op::OpVectorExtractDynamic, 5, uint_type, result, logical, index);
            }

            i += wc;
            continue;
        }

        append_words(out, p, wc);
        i += wc;
    }

    out[3] = next_id;
    return 0;
}

} // namespace ncnn

#endif // NCNN_VULKAN
