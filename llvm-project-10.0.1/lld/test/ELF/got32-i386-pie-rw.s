# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=i686-pc-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t -pie
# RUN: llvm-readelf -r -S %t | FileCheck %s

# Unlike bfd and gold we accept this.

# CHECK: .foobar           PROGBITS        00002180
# CHECK: .got              PROGBITS        [[GOT:[0-9a-z]*]]
# CHECK-DAG: 00002182  00000008 R_386_RELATIVE
# CHECK-DAG: [[GOT]]   00000008 R_386_RELATIVE
foo:

.section .foobar, "awx"
.global _start
_start:
 movl foo@GOT, %ebx
