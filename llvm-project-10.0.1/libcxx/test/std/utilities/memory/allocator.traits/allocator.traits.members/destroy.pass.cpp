//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class Alloc>
// struct allocator_traits
// {
//     template <class Ptr>
//         static void destroy(allocator_type& a, Ptr p);
//     ...
// };

#include <memory>
#include <new>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "incomplete_type_helper.h"

template <class T>
struct A
{
    typedef T value_type;

};

int b_destroy = 0;

template <class T>
struct B
{
    typedef T value_type;

    template <class U>
    void destroy(U* p)
    {
        ++b_destroy;
        p->~U();
    }
};

struct A0
{
    static int count;
    ~A0() {++count;}
};

int A0::count = 0;

int main(int, char**)
{
    {
        A0::count = 0;
        A<int> a;
        std::aligned_storage<sizeof(A0)>::type a0;
        std::allocator_traits<A<int> >::construct(a, (A0*)&a0);
        assert(A0::count == 0);
        std::allocator_traits<A<int> >::destroy(a, (A0*)&a0);
        assert(A0::count == 1);
    }
    {
      typedef IncompleteHolder* VT;
      typedef A<VT> Alloc;
      Alloc a;
      std::aligned_storage<sizeof(VT)>::type store;
      std::allocator_traits<Alloc>::destroy(a, (VT*)&store);
    }
#if defined(_LIBCPP_VERSION) || TEST_STD_VER >= 11
    {
        A0::count = 0;
        b_destroy = 0;
        B<int> b;
        std::aligned_storage<sizeof(A0)>::type a0;
        std::allocator_traits<B<int> >::construct(b, (A0*)&a0);
        assert(A0::count == 0);
        assert(b_destroy == 0);
        std::allocator_traits<B<int> >::destroy(b, (A0*)&a0);
        assert(A0::count == 1);
        assert(b_destroy == 1);
    }
#endif

  return 0;
}
