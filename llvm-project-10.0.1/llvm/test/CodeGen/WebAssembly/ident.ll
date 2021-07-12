; RUN: llc < %s -asm-verbose=false -wasm-keep-registers | FileCheck %s

; Test llvm.ident.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK: .ident "hello world"

!llvm.ident = !{!0}

!0 = !{!"hello world"}
