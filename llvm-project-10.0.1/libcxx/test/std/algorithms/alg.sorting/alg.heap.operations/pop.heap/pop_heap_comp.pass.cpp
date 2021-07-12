//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<RandomAccessIterator Iter, StrictWeakOrder<auto, Iter::value_type> Compare>
//   requires ShuffleIterator<Iter> && CopyConstructible<Compare>
//   void
//   pop_heap(Iter first, Iter last, Compare comp);

#include <algorithm>
#include <functional>
#include <random>
#include <cassert>
#include <memory>

#include "test_macros.h"
#include "test_iterators.h"

struct indirect_less
{
    template <class P>
    bool operator()(const P& x, const P& y)
        {return *x < *y;}
};


std::mt19937 randomness;

void test(int N)
{
    int* ia = new int [N];
    for (int i = 0; i < N; ++i)
        ia[i] = i;
    std::shuffle(ia, ia+N, randomness);
    std::make_heap(ia, ia+N, std::greater<int>());
    for (int i = N; i > 0; --i)
    {
        std::pop_heap(ia, ia+i, std::greater<int>());
        assert(std::is_heap(ia, ia+i-1, std::greater<int>()));
    }
    std::pop_heap(ia, ia, std::greater<int>());

    typedef random_access_iterator<int *> RI;
    std::shuffle(RI(ia), RI(ia+N), randomness);
    std::make_heap(RI(ia), RI(ia+N), std::greater<int>());
    for (int i = N; i > 0; --i)
    {
        std::pop_heap(RI(ia), RI(ia+i), std::greater<int>());
        assert(std::is_heap(RI(ia), RI(ia+i-1), std::greater<int>()));
    }
    std::pop_heap(RI(ia), RI(ia), std::greater<int>());

    delete [] ia;
}

int main(int, char**)
{
    test(1000);

#if TEST_STD_VER >= 11
    {
    const int N = 1000;
    std::unique_ptr<int>* ia = new std::unique_ptr<int> [N];
    for (int i = 0; i < N; ++i)
        ia[i].reset(new int(i));
    std::shuffle(ia, ia+N, randomness);
    std::make_heap(ia, ia+N, indirect_less());
    for (int i = N; i > 0; --i)
    {
        std::pop_heap(ia, ia+i, indirect_less());
        assert(std::is_heap(ia, ia+i-1, indirect_less()));
    }
    delete [] ia;
    }
#endif

  return 0;
}
