//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test bitset<N>& set(size_t pos, bool val = true);

#include <bitset>
#include <cassert>
#include <stdexcept>

#include "test_macros.h"

template <std::size_t N>
void test_set_one(bool test_throws)
{
    std::bitset<N> v;
#ifdef TEST_HAS_NO_EXCEPTIONS
    if (test_throws) return;
#else
    try
#endif
    {
        v.set(50);
        if (50 >= v.size())
            assert(false);
        assert(v[50]);
        assert(!test_throws);
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    catch (std::out_of_range&)
    {
        assert(test_throws);
    }
    try
#endif
    {
        v.set(50, false);
        if (50 >= v.size())
            assert(false);
        assert(!v[50]);
        assert(!test_throws);
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    catch (std::out_of_range&)
    {
        assert(test_throws);
    }
#endif
}

int main(int, char**)
{
    test_set_one<0>(true);
    test_set_one<1>(true);
    test_set_one<31>(true);
    test_set_one<32>(true);
    test_set_one<33>(true);
    test_set_one<63>(false);
    test_set_one<64>(false);
    test_set_one<65>(false);
    test_set_one<1000>(false);

  return 0;
}
