; RUN: llc -mtriple=aarch64-none-eabi -code-model=tiny < %s | FileCheck %s

@var8 = global i8 0
@var16 = global i16 0
@var32 = global i32 0
@var64 = global i64 0

define i8* @global_addr() {
; CHECK-LABEL: global_addr:
  ret i8* @var8
  ; The adr calculation should end up returned directly in x0.
; CHECK: adr x0, var8
; CHECK-NEXT: ret
}

define i8 @global_i8() {
; CHECK-LABEL: global_i8:
  %val = load i8, i8* @var8
  ret i8 %val
; CHECK: adr x[[ADDR_REG:[0-9]+]], var8
; CHECK: ldrb w0, [x[[ADDR_REG]]]
}

define i16 @global_i16() {
; CHECK-LABEL: global_i16:
  %val = load i16, i16* @var16
  ret i16 %val
; CHECK: adr x[[ADDR_REG:[0-9]+]], var16
; CHECK: ldrh w0, [x[[ADDR_REG]]]
}

define i32 @global_i32() {
; CHECK-LABEL: global_i32:
  %val = load i32, i32* @var32
  ret i32 %val
; CHECK: ldr w0, var32
}

define i64 @global_i64() {
; CHECK-LABEL: global_i64:
  %val = load i64, i64* @var64
  ret i64 %val
; CHECK: ldr x0, var64
}

define <2 x i64> @constpool() {
; CHECK-LABEL: constpool:
  ret <2 x i64> <i64 123456789, i64 987654321100>

; CHECK: adr x[[ADDR_REG:[0-9]+]], {{.LCPI[0-9]+_[0-9]+}}
; CHECK: ldr q0, [x[[ADDR_REG]]]
}
