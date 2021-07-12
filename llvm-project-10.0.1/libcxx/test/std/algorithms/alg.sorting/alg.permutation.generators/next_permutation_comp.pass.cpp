//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<BidirectionalIterator Iter, StrictWeakOrder<auto, Iter::value_type> Compare>
//   requires ShuffleIterator<Iter>
//         && CopyConstructible<Compare>
//   bool
//   next_permutation(Iter first, Iter last, Compare comp);

#include <algorithm>
#include <functional>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

#include <cstdio>

int factorial(int x)
{
    int r = 1;
    for (; x; --x)
        r *= x;
    return r;
}

template <class Iter>
void
test()
{
    typedef std::greater<int> C;
    int ia[] = {6, 5, 4, 3, 2, 1};
    const int sa = sizeof(ia)/sizeof(ia[0]);
    int prev[sa];
    for (int e = 0; e <= sa; ++e)
    {
        int count = 0;
        bool x;
        do
        {
            std::copy(ia, ia+e, prev);
            x = std::next_permutation(Iter(ia), Iter(ia+e), C());
            if (e > 1)
            {
                if (x)
                    assert(std::lexicographical_compare(prev, prev+e, ia, ia+e, C()));
                else
                    assert(std::lexicographical_compare(ia, ia+e, prev, prev+e, C()));
            }
            ++count;
        } while (x);
        assert(count == factorial(e));
    }
}

int main(int, char**)
{
    test<bidirectional_iterator<int*> >();
    test<random_access_iterator<int*> >();
    test<int*>();

  return 0;
}
