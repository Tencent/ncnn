//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// reference operator[] (size_type)
// const_reference operator[] (size_type); // constexpr in C++14
// reference at (size_type)
// const_reference at (size_type); // constexpr in C++14
// Libc++ marks these as noexcept

#include <array>
#include <cassert>

#include "test_macros.h"

// std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

#if TEST_STD_VER > 14
constexpr bool check_idx( size_t idx, double val )
{
    std::array<double, 3> arr = {1, 2, 3.5};
    return arr[idx] == val;
}
#endif

int main(int, char**)
{
    {
        typedef double T;
        typedef std::array<T, 3> C;
        C c = {1, 2, 3.5};
        LIBCPP_ASSERT_NOEXCEPT(c[0]);
        ASSERT_SAME_TYPE(C::reference, decltype(c[0]));
        C::reference r1 = c[0];
        assert(r1 == 1);
        r1 = 5.5;
        assert(c.front() == 5.5);

        C::reference r2 = c[2];
        assert(r2 == 3.5);
        r2 = 7.5;
        assert(c.back() == 7.5);
    }
    {
        typedef double T;
        typedef std::array<T, 3> C;
        const C c = {1, 2, 3.5};
        LIBCPP_ASSERT_NOEXCEPT(c[0]);
        ASSERT_SAME_TYPE(C::const_reference, decltype(c[0]));
        C::const_reference r1 = c[0];
        assert(r1 == 1);
        C::const_reference r2 = c[2];
        assert(r2 == 3.5);
    }
    { // Test operator[] "works" on zero sized arrays
        typedef double T;
        typedef std::array<T, 0> C;
        C c = {};
        C const& cc = c;
        LIBCPP_ASSERT_NOEXCEPT(c[0]);
        LIBCPP_ASSERT_NOEXCEPT(cc[0]);
        ASSERT_SAME_TYPE(C::reference, decltype(c[0]));
        ASSERT_SAME_TYPE(C::const_reference, decltype(cc[0]));
        if (c.size() > (0)) { // always false
          C::reference r1 = c[0];
          C::const_reference r2 = cc[0];
          ((void)r1);
          ((void)r2);
        }
    }
    { // Test operator[] "works" on zero sized arrays
        typedef double T;
        typedef std::array<const T, 0> C;
        C c = {{}};
        C const& cc = c;
        LIBCPP_ASSERT_NOEXCEPT(c[0]);
        LIBCPP_ASSERT_NOEXCEPT(cc[0]);
        ASSERT_SAME_TYPE(C::reference, decltype(c[0]));
        ASSERT_SAME_TYPE(C::const_reference, decltype(cc[0]));
        if (c.size() > (0)) { // always false
          C::reference r1 = c[0];
          C::const_reference r2 = cc[0];
          ((void)r1);
          ((void)r2);
        }
    }
#if TEST_STD_VER > 11
    {
        typedef double T;
        typedef std::array<T, 3> C;
        constexpr C c = {1, 2, 3.5};
        LIBCPP_ASSERT_NOEXCEPT(c[0]);
        ASSERT_SAME_TYPE(C::const_reference, decltype(c[0]));

        constexpr T t1 = c[0];
        static_assert (t1 == 1, "");

        constexpr T t2 = c[2];
        static_assert (t2 == 3.5, "");
    }
#endif

#if TEST_STD_VER > 14
    {
        static_assert (check_idx(0, 1), "");
        static_assert (check_idx(1, 2), "");
        static_assert (check_idx(2, 3.5), "");
    }
#endif

  return 0;
}
