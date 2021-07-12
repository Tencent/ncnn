//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test constexpr bool test(size_t pos) const;

#include <bitset>
#include <cstdlib>
#include <cassert>
#include <stdexcept>

#include "test_macros.h"

#if defined(TEST_COMPILER_C1XX)
#pragma warning(disable: 6294) // Ill-defined for-loop:  initial condition does not satisfy test.  Loop body not executed.
#endif

template <std::size_t N>
std::bitset<N>
make_bitset()
{
    std::bitset<N> v;
    for (std::size_t i = 0; i < N; ++i)
        v[i] = static_cast<bool>(std::rand() & 1);
    return v;
}

template <std::size_t N>
void test_test(bool test_throws)
{
    const std::bitset<N> v1 = make_bitset<N>();
#ifdef TEST_HAS_NO_EXCEPTIONS
    if (test_throws) return;
#else
    try
    {
#endif
        bool b = v1.test(50);
        if (50 >= v1.size())
            assert(false);
        assert(b == v1[50]);
        assert(!test_throws);
#ifndef TEST_HAS_NO_EXCEPTIONS
    }
    catch (std::out_of_range&)
    {
        assert(test_throws);
    }
#endif
}

int main(int, char**)
{
    test_test<0>(true);
    test_test<1>(true);
    test_test<31>(true);
    test_test<32>(true);
    test_test<33>(true);
    test_test<63>(false);
    test_test<64>(false);
    test_test<65>(false);
    test_test<1000>(false);

  return 0;
}
