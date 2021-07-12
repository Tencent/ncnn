// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef SUPPORT_DEMANGLE_H
#define SUPPORT_DEMANGLE_H

#include "test_macros.h"
#include <string>
#include <cstdlib>

#if !defined(TEST_HAS_NO_DEMANGLE)
# if defined(__GNUC__) || defined(__clang__)
#   if __has_include("cxxabi.h") && !defined(_LIBCPP_ABI_MICROSOFT)
#     include "cxxabi.h"
#   else
#     define TEST_HAS_NO_DEMANGLE
#   endif
# else
#   define TEST_HAS_NO_DEMANGLE
# endif
#endif

#if defined(TEST_HAS_NO_DEMANGLE)
inline std::string demangle(const char* mangled_name) {
  return mangled_name;
}
#else
template <size_t N> struct Printer;
inline std::string demangle(const char* mangled_name) {
  int status = 0;
  char* out = __cxxabiv1::__cxa_demangle(mangled_name, nullptr, nullptr, &status);
  if (out != nullptr) {
    std::string res(out);
    std::free(out);
    return res;
  }
  return mangled_name;
}
#endif

#endif // SUPPORT_DEMANGLE_H
