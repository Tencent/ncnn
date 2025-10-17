// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Corsix <https://github.com/corsix>
// Copyright (c) 2025 Molly Sophia <mollysophia@gmail.com>

#ifndef AMX_USABILITY_H
#define AMX_USABILITY_H

// From https://github.com/corsix/amx/blob/main/aarch64.h
#define AMX_NOP_OP_IMM5(op, imm5)                            \
    __asm("nop\nnop\nnop\n.word (0x201000 + (%0 << 5) + %1)" \
          :                                                  \
          : "i"(op), "i"(imm5)                               \
          : "memory")

#define AMX_OP_GPR(op, gpr)                                       \
    __asm(".word (0x201000 + (%0 << 5) + 0%1 - ((0%1 >> 4) * 6))" \
          :                                                       \
          : "i"(op), "r"((uint64_t)(gpr))                         \
          : "memory")

#define AMX_LDX(gpr)                   AMX_OP_GPR(0, gpr)
#define AMX_LDY(gpr)                   AMX_OP_GPR(1, gpr)
#define AMX_STX(gpr)                   AMX_OP_GPR(2, gpr)
#define AMX_STY(gpr)                   AMX_OP_GPR(3, gpr)
#define AMX_LDZ(gpr)                   AMX_OP_GPR(4, gpr)
#define AMX_STZ(gpr)                   AMX_OP_GPR(5, gpr)
#define AMX_LDZI(gpr)                  AMX_OP_GPR(6, gpr)
#define AMX_STZI(gpr)                  AMX_OP_GPR(7, gpr)
#define AMX_EXTRX(gpr)                 AMX_OP_GPR(8, gpr)
#define AMX_EXTRY(gpr)                 AMX_OP_GPR(9, gpr)
#define AMX_FMA64(gpr)                 AMX_OP_GPR(10, gpr)
#define AMX_FMS64(gpr)                 AMX_OP_GPR(11, gpr)
#define AMX_FMA32(gpr)                 AMX_OP_GPR(12, gpr)
#define AMX_FMS32(gpr)                 AMX_OP_GPR(13, gpr)
#define AMX_MAC16(gpr)                 AMX_OP_GPR(14, gpr)
#define AMX_FMA16(gpr)                 AMX_OP_GPR(15, gpr)
#define AMX_FMS16(gpr)                 AMX_OP_GPR(16, gpr)
#define AMX_VECINT(gpr)                AMX_OP_GPR(18, gpr)
#define AMX_VECFP(gpr)                 AMX_OP_GPR(19, gpr)
#define AMX_MATINT(gpr)                AMX_OP_GPR(20, gpr)
#define AMX_MATFP(gpr)                 AMX_OP_GPR(21, gpr)
#define AMX_GENLUT(gpr)                AMX_OP_GPR(22, gpr)
#define PTR_ROW_FLAGS(ptr, row, flags) (((uint64_t) & *(ptr)) + (((uint64_t)((row) + (flags)*64)) << 56))
static void amx_set()
{
    AMX_NOP_OP_IMM5(17, 0);
}

static void amx_clr()
{
    AMX_NOP_OP_IMM5(17, 1);
}

static void amx_ldx(bool pair, unsigned int x_row, const void* ptr)
{
    if (x_row >= 8)
        return;

    uint64_t oprand = (uint64_t)ptr + ((uint64_t)x_row << 56);
    if (pair)
        oprand |= 1ULL << 62;

    AMX_LDX(oprand);
}

static void amx_ldy(bool pair, unsigned int y_row, const void* ptr)
{
    if (y_row >= 8)
        return;

    uint64_t oprand = (uint64_t)ptr + ((uint64_t)y_row << 56);
    if (pair)
        oprand |= 1ULL << 62;

    AMX_LDY(oprand);
}

static void amx_ldz(bool pair, unsigned int z_row, const void* ptr)
{
    if (z_row >= 64)
        return;

    uint64_t oprand = (uint64_t)ptr + ((uint64_t)z_row << 56);
    if (pair)
        oprand |= 1ULL << 62;

    AMX_LDZ(oprand);
}

static void amx_stz(bool pair, unsigned int z_row, const void* ptr)
{
    if (z_row >= 64)
        return;

    uint64_t oprand = (uint64_t)ptr + ((uint64_t)z_row << 56);
    if (pair)
        oprand |= 1ULL << 62;

    AMX_STZ(oprand);
}

static void amx_fma16_masked(bool vector, unsigned int x_offset, unsigned int y_offset, int z_row,
                             uint8_t x_mode, uint8_t x_mask, uint8_t y_mode, uint8_t y_mask,
                             bool skip_x, bool skip_y, bool skip_z)
{
    uint64_t oprand = 0;
    if (vector)
        oprand |= 1ULL << 63;

    if (skip_x)
        oprand |= 1ULL << 29;
    if (skip_y)
        oprand |= 1ULL << 28;
    if (skip_z)
        oprand |= 1ULL << 27;

    oprand |= (uint64_t)y_offset & 0x1FF;
    oprand |= ((uint64_t)x_offset & 0x1FF) << 10;
    oprand |= ((uint64_t)z_row & 0x3F) << 20;
    oprand |= ((uint64_t)y_mask & 0x1F) << 32;
    oprand |= ((uint64_t)y_mode & 0x3) << 37;
    oprand |= ((uint64_t)x_mask & 0x1F) << 41;
    oprand |= ((uint64_t)x_mode & 0x3) << 46;

    AMX_FMA16(oprand);
}

static void amx_fma16(bool vector, unsigned int x_offset, unsigned int y_offset, int z_row)
{
    amx_fma16_masked(vector, x_offset, y_offset, z_row, 0, 0, 0, 0, false, false, false);
}

static void amx_fma32_masked(bool vector, unsigned int x_offset, unsigned int y_offset, int z_row, uint8_t x_mode, uint8_t x_mask, uint8_t y_mode, uint8_t y_mask)
{
    uint64_t oprand = 0;
    if (vector)
        oprand |= 1ULL << 63;

    oprand |= (uint64_t)y_offset & 0x1FF;
    oprand |= ((uint64_t)x_offset & 0x1FF) << 10;
    oprand |= ((uint64_t)z_row & 0x3F) << 20;
    oprand |= ((uint64_t)y_mask & 0x1F) << 32;
    oprand |= ((uint64_t)y_mode & 0x3) << 37;
    oprand |= ((uint64_t)x_mask & 0x1F) << 41;
    oprand |= ((uint64_t)x_mode & 0x3) << 46;

    AMX_FMA32(oprand);
}

static void amx_fma32(bool vector, unsigned int x_offset, unsigned int y_offset, int z_row)
{
    amx_fma32_masked(vector, x_offset, y_offset, z_row, 0, 0, 0, 0);
}

#endif // AMX_USABILITY_H