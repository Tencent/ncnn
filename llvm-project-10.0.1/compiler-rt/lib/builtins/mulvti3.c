//===-- mulvti3.c - Implement __mulvti3 -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements __mulvti3 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#ifdef CRT_HAS_128BIT

// Returns: a * b

// Effects: aborts if a * b overflows

COMPILER_RT_ABI ti_int __mulvti3(ti_int a, ti_int b) {
  const int N = (int)(sizeof(ti_int) * CHAR_BIT);
  const ti_int MIN = (ti_int)1 << (N - 1);
  const ti_int MAX = ~MIN;
  if (a == MIN) {
    if (b == 0 || b == 1)
      return a * b;
    compilerrt_abort();
  }
  if (b == MIN) {
    if (a == 0 || a == 1)
      return a * b;
    compilerrt_abort();
  }
  ti_int sa = a >> (N - 1);
  ti_int abs_a = (a ^ sa) - sa;
  ti_int sb = b >> (N - 1);
  ti_int abs_b = (b ^ sb) - sb;
  if (abs_a < 2 || abs_b < 2)
    return a * b;
  if (sa == sb) {
    if (abs_a > MAX / abs_b)
      compilerrt_abort();
  } else {
    if (abs_a > MIN / -abs_b)
      compilerrt_abort();
  }
  return a * b;
}

#endif // CRT_HAS_128BIT
