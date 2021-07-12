# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: echo "SECTIONS { .rela.dyn : { *(.rela.data) } }" > %t.script
# RUN: ld.lld --hash-style=sysv -T %t.script --emit-relocs %t.o -o %t.so -shared
# RUN: llvm-readobj -r %t.so | FileCheck %s

.data
.quad .foo

# CHECK:      Relocations [
# CHECK-NEXT:   Section ({{.*}}) .rela.dyn {
# CHECK-NEXT:     0xF8 R_X86_64_64 .foo 0x0
# CHECK-NEXT:   }
# CHECK-NEXT:   Section ({{.*}}) .rela.data {
# CHECK-NEXT:     0xF8 R_X86_64_64 .foo 0x0
# CHECK-NEXT:   }
# CHECK-NEXT: ]
