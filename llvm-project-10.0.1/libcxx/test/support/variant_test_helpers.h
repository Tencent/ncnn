// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef SUPPORT_VARIANT_TEST_HELPERS_H
#define SUPPORT_VARIANT_TEST_HELPERS_H

#include <type_traits>
#include <utility>
#include <cassert>

#include "test_macros.h"

#if TEST_STD_VER <= 14
#error This file requires C++17
#endif

// FIXME: Currently the variant<T&> tests are disabled using this macro.
#define TEST_VARIANT_HAS_NO_REFERENCES
#ifdef _LIBCPP_ENABLE_NARROWING_CONVERSIONS_IN_VARIANT
# define TEST_VARIANT_ALLOWS_NARROWING_CONVERSIONS
#endif

#ifdef TEST_VARIANT_ALLOWS_NARROWING_CONVERSIONS
constexpr bool VariantAllowsNarrowingConversions = true;
#else
constexpr bool VariantAllowsNarrowingConversions = false;
#endif

#ifndef TEST_HAS_NO_EXCEPTIONS
struct CopyThrows {
  CopyThrows() = default;
  CopyThrows(CopyThrows const&) { throw 42; }
  CopyThrows& operator=(CopyThrows const&) { throw 42; }
};

struct MoveThrows {
  static int alive;
  MoveThrows() { ++alive; }
  MoveThrows(MoveThrows const&) {++alive;}
  MoveThrows(MoveThrows&&) {  throw 42; }
  MoveThrows& operator=(MoveThrows const&) { return *this; }
  MoveThrows& operator=(MoveThrows&&) { throw 42; }
  ~MoveThrows() { --alive; }
};

int MoveThrows::alive = 0;

struct MakeEmptyT {
  static int alive;
  MakeEmptyT() { ++alive; }
  MakeEmptyT(MakeEmptyT const&) {
      ++alive;
      // Don't throw from the copy constructor since variant's assignment
      // operator performs a copy before committing to the assignment.
  }
  MakeEmptyT(MakeEmptyT &&) {
      throw 42;
  }
  MakeEmptyT& operator=(MakeEmptyT const&) {
      throw 42;
  }
  MakeEmptyT& operator=(MakeEmptyT&&) {
      throw 42;
  }
   ~MakeEmptyT() { --alive; }
};
static_assert(std::is_swappable_v<MakeEmptyT>, ""); // required for test

int MakeEmptyT::alive = 0;

template <class Variant>
void makeEmpty(Variant& v) {
    Variant v2(std::in_place_type<MakeEmptyT>);
    try {
        v = std::move(v2);
        assert(false);
    } catch (...) {
        assert(v.valueless_by_exception());
    }
}
#endif // TEST_HAS_NO_EXCEPTIONS


#endif // SUPPORT_VARIANT_TEST_HELPERS_H
