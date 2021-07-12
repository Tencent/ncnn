//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// <condition_variable>

// class condition_variable;

// void notify_one();

#include <condition_variable>
#include <atomic>
#include <mutex>
#include <thread>
#include <cassert>

#include "test_macros.h"


std::condition_variable cv;
std::mutex mut;

std::atomic_int test1(0);
std::atomic_int test2(0);

void f1()
{
    std::unique_lock<std::mutex> lk(mut);
    assert(test1 == 0);
    while (test1 == 0)
        cv.wait(lk);
    assert(test1 == 1);
    test1 = 2;
}

void f2()
{
    std::unique_lock<std::mutex> lk(mut);
    assert(test2 == 0);
    while (test2 == 0)
        cv.wait(lk);
    assert(test2 == 1);
    test2 = 2;
}

int main(int, char**)
{
    std::thread t1(f1);
    std::thread t2(f2);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    {
        std::unique_lock<std::mutex>lk(mut);
        test1 = 1;
        test2 = 1;
    }
    cv.notify_one();
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::unique_lock<std::mutex>lk(mut);
    }
    if (test1 == 2)
    {
        assert(test2 == 1);
        t1.join();
        test1 = 0;
    }
    else if (test2 == 2)
    {
        assert(test1 == 1);
        t2.join();
        test2 = 0;
    }
    else
        assert(false);
    cv.notify_one();
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::unique_lock<std::mutex>lk(mut);
    }
    if (test1 == 2)
    {
        assert(test2 == 0);
        t1.join();
        test1 = 0;
    }
    else if (test2 == 2)
    {
        assert(test1 == 0);
        t2.join();
        test2 = 0;
    }
    else
        assert(false);

  return 0;
}
