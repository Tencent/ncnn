//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test bitset<N>& reset(size_t pos);

#include <bitset>
#include <cassert>
#include <stdexcept>

#include "test_macros.h"

template <std::size_t N>
void test_reset_one(bool test_throws)
{
    std::bitset<N> v;
#ifdef TEST_HAS_NO_EXCEPTIONS
    if (test_throws) return;
#else
    try
    {
#endif
        v.set();
        v.reset(50);
        if (50 >= v.size())
            assert(false);
        for (unsigned i = 0; i < v.size(); ++i)
            if (i == 50)
                assert(!v[i]);
            else
                assert(v[i]);
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
    test_reset_one<0>(true);
    test_reset_one<1>(true);
    test_reset_one<31>(true);
    test_reset_one<32>(true);
    test_reset_one<33>(true);
    test_reset_one<63>(false);
    test_reset_one<64>(false);
    test_reset_one<65>(false);
    test_reset_one<1000>(false);

  return 0;
}
