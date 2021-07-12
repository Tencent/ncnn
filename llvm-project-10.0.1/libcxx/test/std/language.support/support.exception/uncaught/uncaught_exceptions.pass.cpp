//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-no-exceptions
// XFAIL: libcpp-no-exceptions

// XFAIL: macosx10.7
// XFAIL: macosx10.8
// XFAIL: macosx10.9
// XFAIL: macosx10.10
// XFAIL: macosx10.11
// XFAIL: with_system_cxx_lib=macosx10.12
// XFAIL: with_system_cxx_lib=macosx10.13

// test uncaught_exceptions

#include <exception>
#include <cassert>

#include "test_macros.h"

struct Uncaught {
    Uncaught(int depth) : d_(depth) {}
    ~Uncaught() { assert(std::uncaught_exceptions() == d_); }
    int d_;
    };

struct Outer {
    Outer(int depth) : d_(depth) {}
    ~Outer() {
    try {
        assert(std::uncaught_exceptions() == d_);
        Uncaught u(d_+1);
        throw 2;
    }
    catch (int) {}
    }
    int d_;
};

int main(int, char**) {
    assert(std::uncaught_exceptions() == 0);
    {
    Outer o(0);
    }

    assert(std::uncaught_exceptions() == 0);
    {
    try {
        Outer o(1);
        throw 1;
        }
    catch (int) {
        assert(std::uncaught_exceptions() == 0);
        }
    }
    assert(std::uncaught_exceptions() == 0);

  return 0;
}
