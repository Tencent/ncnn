#include "cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Improved version of is_option_disabled function that doesn't modify the original string
bool is_option_disabled(const char* options, const char* option)
{
    char* options_copy = strdup(options);
    char* token = strtok(options_copy, ",");
    bool disabled = false;

    while (token)
    {
        if (strcmp(token, option) == 0)
        {
            disabled = true;
            break;
        }
        token = strtok(NULL, ",");
    }

    free(options_copy);
    return disabled;
}

// Helper function to check and report instruction set support
bool check_instruction_disabled(const char* options, const char* option, bool cpu_support, const char* instruction_name)
{
    if (is_option_disabled(options, option) && cpu_support)
    {
        fprintf(stderr, "Error: %s should be disabled but it is enabled!\n", instruction_name);
        return true;
    }
    return false;
}

int main()
{
    const char* ncnn_isa = getenv("NCNN_ISA");

    // Check if NCNN_ISA is set to disable certain options
    if (ncnn_isa)
    {
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
        if (check_instruction_disabled(ncnn_isa, "-avx", ncnn::cpu_support_x86_avx(), "avx")) return 1;
        if (check_instruction_disabled(ncnn_isa, "-xop", ncnn::cpu_support_x86_xop(), "xop")) return 1;
        if (check_instruction_disabled(ncnn_isa, "-fma", ncnn::cpu_support_x86_fma(), "fma")) return 1;
        if (check_instruction_disabled(ncnn_isa, "-f16c", ncnn::cpu_support_x86_f16c(), "f16c")) return 1;
        if (check_instruction_disabled(ncnn_isa, "-avx2", ncnn::cpu_support_x86_avx2(), "avx2")) return 1;
        if (check_instruction_disabled(ncnn_isa, "-avx512", ncnn::cpu_support_x86_avx512(), "avx512")) return 1;
        if (check_instruction_disabled(ncnn_isa, "-avx_vnni", ncnn::cpu_support_x86_avx_vnni(), "avx_vnni")) return 1;
        if (check_instruction_disabled(ncnn_isa, "-avx512_vnni", ncnn::cpu_support_x86_avx512_vnni(), "avx512_vnni")) return 1;
        if (check_instruction_disabled(ncnn_isa, "-avx512_bf16", ncnn::cpu_support_x86_avx512_bf16(), "avx512_bf16")) return 1;
        if (check_instruction_disabled(ncnn_isa, "-avx512_fp16", ncnn::cpu_support_x86_avx512_fp16(), "avx512_fp16")) return 1;
#endif

#if defined(__aarch64__) || defined(__arm__)
        if (check_instruction_disabled(ncnn_isa, "-cpuid", ncnn::cpu_support_arm_cpuid(), "cpuid")) return 1;
        if (check_instruction_disabled(ncnn_isa, "-asimdhp", ncnn::cpu_support_arm_asimdhp(), "asimdhp")) return 1;
        if (check_instruction_disabled(ncnn_isa, "-asimddp", ncnn::cpu_support_arm_asimddp(), "asimddp")) return 1;
        if (check_instruction_disabled(ncnn_isa, "-asimdfhm", ncnn::cpu_support_arm_asimdfhm(), "asimdfhm")) return 1;
        if (check_instruction_disabled(ncnn_isa, "-bf16", ncnn::cpu_support_arm_bf16(), "bf16")) return 1;
        if (check_instruction_disabled(ncnn_isa, "-i8mm", ncnn::cpu_support_arm_i8mm(), "i8mm")) return 1;
        if (check_instruction_disabled(ncnn_isa, "-sve", ncnn::cpu_support_arm_sve(), "sve")) return 1;
        if (check_instruction_disabled(ncnn_isa, "-sve2", ncnn::cpu_support_arm_sve2(), "sve2")) return 1;
        if (check_instruction_disabled(ncnn_isa, "-svebf16", ncnn::cpu_support_arm_svebf16(), "svebf16")) return 1;
        if (check_instruction_disabled(ncnn_isa, "-svei8mm", ncnn::cpu_support_arm_svei8mm(), "svei8mm")) return 1;
        if (check_instruction_disabled(ncnn_isa, "-svef32mm", ncnn::cpu_support_arm_svef32mm(), "svef32mm")) return 1;
        if (check_instruction_disabled(ncnn_isa, "-edsp", ncnn::cpu_support_arm_edsp(), "edsp")) return 1;
        if (check_instruction_disabled(ncnn_isa, "-vfpv4", ncnn::cpu_support_arm_vfpv4(), "vfpv4")) return 1;
        if (check_instruction_disabled(ncnn_isa, "-neon", ncnn::cpu_support_arm_neon(), "neon")) return 1;
#endif

#if defined(__loongarch64)
        if (check_instruction_disabled(ncnn_isa, "-lsx", ncnn::cpu_support_loongarch_lsx(), "lsx")) return 1;
        if (check_instruction_disabled(ncnn_isa, "-lasx", ncnn::cpu_support_loongarch_lasx(), "lasx")) return 1;
#endif

#if defined(__mips__)
        if (check_instruction_disabled(ncnn_isa, "-msa", ncnn::cpu_support_mips_msa(), "msa")) return 1;
        if (check_instruction_disabled(ncnn_isa, "-mmi", ncnn::cpu_support_loongson_mmi(), "mmi")) return 1;
#endif

#if defined(__riscv)
        if (check_instruction_disabled(ncnn_isa, "-rvv", ncnn::cpu_support_riscv_v(), "rvv")) return 1;
        if (check_instruction_disabled(ncnn_isa, "-zfh", ncnn::cpu_support_riscv_zfh(), "zfh")) return 1;
#endif
    }

    return 0;
}
