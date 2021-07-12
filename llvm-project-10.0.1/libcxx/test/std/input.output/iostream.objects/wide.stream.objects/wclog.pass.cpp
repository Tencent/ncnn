//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iostream>

// istream wclog;

#include <iostream>

#include "test_macros.h"

int main(int, char**)
{
#if 0
    std::wclog << L"Hello World!\n";
#else
    (void)std::wclog;
#endif

  return 0;
}
