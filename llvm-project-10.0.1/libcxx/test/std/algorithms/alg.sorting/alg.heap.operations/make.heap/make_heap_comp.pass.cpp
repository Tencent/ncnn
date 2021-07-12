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
//   make_heap(Iter first, Iter last, Compare comp);

#include <algorithm>
#include <functional>
#include <memory>
#include <random>
#include <cassert>

#include "test_macros.h"
#include "counting_predicates.h"
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
    {
    for (int i = 0; i < N; ++i)
        ia[i] = i;
    std::shuffle(ia, ia+N, randomness);
    std::make_heap(ia, ia+N, std::greater<int>());
    assert(std::is_heap(ia, ia+N, std::greater<int>()));

    std::shuffle(ia, ia+N, randomness);
    std::make_heap(random_access_iterator<int *>(ia),
                   random_access_iterator<int *>(ia+N), std::greater<int>());
    assert(std::is_heap(ia, ia+N, std::greater<int>()));
    }

//  Ascending
    {
    binary_counting_predicate<std::greater<int>, int, int> pred ((std::greater<int>()));
    for (int i = 0; i < N; ++i)
        ia[i] = i;
    std::make_heap(ia, ia+N, std::ref(pred));
    assert(pred.count() <= 3u*N);
    assert(std::is_heap(ia, ia+N, pred));
    }

//  Descending
    {
    binary_counting_predicate<std::greater<int>, int, int> pred ((std::greater<int>()));
    for (int i = 0; i < N; ++i)
        ia[N-1-i] = i;
    std::make_heap(ia, ia+N, std::ref(pred));
    assert(pred.count() <= 3u*N);
    assert(std::is_heap(ia, ia+N, pred));
    }

//  Random
    {
    binary_counting_predicate<std::greater<int>, int, int> pred ((std::greater<int>()));
    std::shuffle(ia, ia+N, randomness);
    std::make_heap(ia, ia+N, std::ref(pred));
    assert(pred.count() <= 3u*N);
    assert(std::is_heap(ia, ia+N, pred));
    }

    delete [] ia;
}

int main(int, char**)
{
    test(0);
    test(1);
    test(2);
    test(3);
    test(10);
    test(1000);
    test(10000);
    test(100000);

#if TEST_STD_VER >= 11
    {
    const int N = 1000;
    std::unique_ptr<int>* ia = new std::unique_ptr<int> [N];
    for (int i = 0; i < N; ++i)
        ia[i].reset(new int(i));
    std::shuffle(ia, ia+N, randomness);
    std::make_heap(ia, ia+N, indirect_less());
    assert(std::is_heap(ia, ia+N, indirect_less()));
    delete [] ia;
    }
#endif

  return 0;
}
