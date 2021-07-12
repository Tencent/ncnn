// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-none-freebsd %s -o %t
// RUN: llvm-mc -filetype=obj -triple=aarch64-none-freebsd %S/Inputs/abs256.s -o %t256.o
// RUN: ld.lld %t %t256.o -o %t2
// RUN: llvm-objdump -s %t2 | FileCheck %s

.globl _start
_start:
.section .R_AARCH64_ABS64, "ax",@progbits
  .xword foo + 0x24

// S = 0x100, A = 0x24
// S + A = 0x124
// CHECK: Contents of section .R_AARCH64_ABS64:
// CHECK-NEXT: 210120 24010000 00000000

.section .R_AARCH64_PREL64, "ax",@progbits
  .xword foo - . + 0x24

// S + A - P = 0x100 + 0x24 - 0x210128 = 0xffffffffffdefffc
// CHECK: Contents of section .R_AARCH64_PREL64:
// CHECK-NEXT: 210128 fcffdeff ffffffff
