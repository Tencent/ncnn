//===-- hwasan_new_delete.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of HWAddressSanitizer.
//
// Interceptors for operators new and delete.
//===----------------------------------------------------------------------===//

#include "hwasan.h"
#include "interception/interception.h"
#include "sanitizer_common/sanitizer_allocator.h"
#include "sanitizer_common/sanitizer_allocator_report.h"

#if HWASAN_REPLACE_OPERATORS_NEW_AND_DELETE

#include <stddef.h>

using namespace __hwasan;

// Fake std::nothrow_t to avoid including <new>.
namespace std {
  struct nothrow_t {};
}  // namespace std


// TODO(alekseys): throw std::bad_alloc instead of dying on OOM.
#define OPERATOR_NEW_BODY(nothrow) \
  GET_MALLOC_STACK_TRACE; \
  void *res = hwasan_malloc(size, &stack);\
  if (!nothrow && UNLIKELY(!res)) ReportOutOfMemory(size, &stack);\
  return res

INTERCEPTOR_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
void *operator new(size_t size) { OPERATOR_NEW_BODY(false /*nothrow*/); }
INTERCEPTOR_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
void *operator new[](size_t size) { OPERATOR_NEW_BODY(false /*nothrow*/); }
INTERCEPTOR_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
void *operator new(size_t size, std::nothrow_t const&) {
  OPERATOR_NEW_BODY(true /*nothrow*/);
}
INTERCEPTOR_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
void *operator new[](size_t size, std::nothrow_t const&) {
  OPERATOR_NEW_BODY(true /*nothrow*/);
}

#define OPERATOR_DELETE_BODY \
  GET_MALLOC_STACK_TRACE; \
  if (ptr) hwasan_free(ptr, &stack)

INTERCEPTOR_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
void operator delete(void *ptr) NOEXCEPT { OPERATOR_DELETE_BODY; }
INTERCEPTOR_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
void operator delete[](void *ptr) NOEXCEPT { OPERATOR_DELETE_BODY; }
INTERCEPTOR_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
void operator delete(void *ptr, std::nothrow_t const&) { OPERATOR_DELETE_BODY; }
INTERCEPTOR_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
void operator delete[](void *ptr, std::nothrow_t const&) {
  OPERATOR_DELETE_BODY;
}

#endif // HWASAN_REPLACE_OPERATORS_NEW_AND_DELETE
