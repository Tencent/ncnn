// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>

namespace std {

exception_ptr::~exception_ptr() _NOEXCEPT
{
#  warning exception_ptr not yet implemented
  fprintf(stderr, "exception_ptr not yet implemented\n");
  ::abort();
}

exception_ptr::exception_ptr(const exception_ptr& other) _NOEXCEPT
    : __ptr_(other.__ptr_)
{
#  warning exception_ptr not yet implemented
  fprintf(stderr, "exception_ptr not yet implemented\n");
  ::abort();
}

exception_ptr& exception_ptr::operator=(const exception_ptr& other) _NOEXCEPT
{
#  warning exception_ptr not yet implemented
  fprintf(stderr, "exception_ptr not yet implemented\n");
  ::abort();
}

nested_exception::nested_exception() _NOEXCEPT
    : __ptr_(current_exception())
{
}

#if !defined(__GLIBCXX__)

nested_exception::~nested_exception() _NOEXCEPT
{
}

#endif

_LIBCPP_NORETURN
void
nested_exception::rethrow_nested() const
{
#  warning exception_ptr not yet implemented
  fprintf(stderr, "exception_ptr not yet implemented\n");
  ::abort();
#if 0
  if (__ptr_ == nullptr)
      terminate();
  rethrow_exception(__ptr_);
#endif // FIXME
}

exception_ptr current_exception() _NOEXCEPT
{
#  warning exception_ptr not yet implemented
  fprintf(stderr, "exception_ptr not yet implemented\n");
  ::abort();
}

_LIBCPP_NORETURN
void rethrow_exception(exception_ptr p)
{
#  warning exception_ptr not yet implemented
  fprintf(stderr, "exception_ptr not yet implemented\n");
  ::abort();
}

} // namespace std
