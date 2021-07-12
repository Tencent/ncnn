//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iostream>

// istream clog;

#include <iostream>

#include "test_macros.h"

int main(int, char**)
{
#if 0
    std::clog << "Hello World!\n";
#else
    (void)std::clog;
#endif

  return 0;
}
