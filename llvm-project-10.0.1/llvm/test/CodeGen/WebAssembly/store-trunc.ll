; RUN: llc < %s -asm-verbose=false -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s

; Test that truncating stores are assembled properly.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: trunc_i8_i32:
; CHECK: i32.store8 0($0), $1{{$}}
define void @trunc_i8_i32(i8 *%p, i32 %v) {
  %t = trunc i32 %v to i8
  store i8 %t, i8* %p
  ret void
}

; CHECK-LABEL: trunc_i16_i32:
; CHECK: i32.store16 0($0), $1{{$}}
define void @trunc_i16_i32(i16 *%p, i32 %v) {
  %t = trunc i32 %v to i16
  store i16 %t, i16* %p
  ret void
}

; CHECK-LABEL: trunc_i8_i64:
; CHECK: i64.store8 0($0), $1{{$}}
define void @trunc_i8_i64(i8 *%p, i64 %v) {
  %t = trunc i64 %v to i8
  store i8 %t, i8* %p
  ret void
}

; CHECK-LABEL: trunc_i16_i64:
; CHECK: i64.store16 0($0), $1{{$}}
define void @trunc_i16_i64(i16 *%p, i64 %v) {
  %t = trunc i64 %v to i16
  store i16 %t, i16* %p
  ret void
}

; CHECK-LABEL: trunc_i32_i64:
; CHECK: i64.store32 0($0), $1{{$}}
define void @trunc_i32_i64(i32 *%p, i64 %v) {
  %t = trunc i64 %v to i32
  store i32 %t, i32* %p
  ret void
}
