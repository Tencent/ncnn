// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cpu.h"

// Test CpuSet with >64 CPUs
static int test_cpuset_large()
{
    printf("Testing CpuSet with >64 CPUs...\n");
    
    ncnn::CpuSet set;
    
    // Test basic operations with large CPU IDs
    const int test_cpus[] = {0, 63, 64, 65, 127, 128, 255, 256, 511, 512, 1023};
    const int num_test_cpus = sizeof(test_cpus) / sizeof(test_cpus[0]);
    
    // Initially all should be disabled
    for (int i = 0; i < num_test_cpus; i++)
    {
        if (set.is_enabled(test_cpus[i]))
        {
            fprintf(stderr, "CPU %d should be disabled initially\n", test_cpus[i]);
            return 1;
        }
    }
    
    if (set.num_enabled() != 0)
    {
        fprintf(stderr, "Initially no CPUs should be enabled\n");
        return 1;
    }
    
    if (!set.is_empty())
    {
        fprintf(stderr, "Initially CpuSet should be empty\n");
        return 1;
    }
    
    // Enable all test CPUs
    for (int i = 0; i < num_test_cpus; i++)
    {
        set.enable(test_cpus[i]);
    }
    
    // Verify they are enabled
    for (int i = 0; i < num_test_cpus; i++)
    {
        if (!set.is_enabled(test_cpus[i]))
        {
            fprintf(stderr, "CPU %d should be enabled\n", test_cpus[i]);
            return 1;
        }
    }
    
    if (set.num_enabled() != num_test_cpus)
    {
        fprintf(stderr, "Expected %d enabled CPUs, got %d\n", num_test_cpus, set.num_enabled());
        return 1;
    }
    
    if (set.is_empty())
    {
        fprintf(stderr, "CpuSet should not be empty after enabling CPUs\n");
        return 1;
    }
    
    // Test max_cpu_id
    int max_cpu = set.max_cpu_id();
    if (max_cpu != 1023)
    {
        fprintf(stderr, "Expected max CPU ID 1023, got %d\n", max_cpu);
        return 1;
    }
    
    // Test disable
    set.disable(test_cpus[0]);
    if (set.is_enabled(test_cpus[0]))
    {
        fprintf(stderr, "CPU %d should be disabled after disable()\n", test_cpus[0]);
        return 1;
    }
    
    if (set.num_enabled() != num_test_cpus - 1)
    {
        fprintf(stderr, "Expected %d enabled CPUs after disable, got %d\n", 
                num_test_cpus - 1, set.num_enabled());
        return 1;
    }
    
    // Test set_range
    set.disable_all();
    set.set_range(100, 200, true);
    
    int expected_range_count = 200 - 100 + 1;
    if (set.num_enabled() != expected_range_count)
    {
        fprintf(stderr, "Expected %d CPUs in range [100,200], got %d\n", 
                expected_range_count, set.num_enabled());
        return 1;
    }
    
    for (int i = 100; i <= 200; i++)
    {
        if (!set.is_enabled(i))
        {
            fprintf(stderr, "CPU %d should be enabled in range [100,200]\n", i);
            return 1;
        }
    }
    
    // Test copy constructor
    ncnn::CpuSet set_copy(set);
    if (set_copy.num_enabled() != set.num_enabled())
    {
        fprintf(stderr, "Copy constructor failed: different num_enabled\n");
        return 1;
    }
    
    for (int i = 0; i <= 1023; i++)
    {
        if (set_copy.is_enabled(i) != set.is_enabled(i))
        {
            fprintf(stderr, "Copy constructor failed: CPU %d state differs\n", i);
            return 1;
        }
    }
    
    // Test assignment operator
    ncnn::CpuSet set_assigned;
    set_assigned.enable(999);
    set_assigned = set;
    
    if (set_assigned.num_enabled() != set.num_enabled())
    {
        fprintf(stderr, "Assignment operator failed: different num_enabled\n");
        return 1;
    }
    
    for (int i = 0; i <= 1023; i++)
    {
        if (set_assigned.is_enabled(i) != set.is_enabled(i))
        {
            fprintf(stderr, "Assignment operator failed: CPU %d state differs\n", i);
            return 1;
        }
    }
    
    printf("CpuSet large CPU test passed!\n");
    return 0;
}

// Test boundary conditions
static int test_cpuset_boundary()
{
    printf("Testing CpuSet boundary conditions...\n");
    
    ncnn::CpuSet set;
    
    // Test CPU ID 0
    set.enable(0);
    if (!set.is_enabled(0))
    {
        fprintf(stderr, "CPU 0 should be enabled\n");
        return 1;
    }
    
    // Test exactly 64 CPUs (boundary between fast and extended path)
    set.disable_all();
    for (int i = 0; i < 64; i++)
    {
        set.enable(i);
    }
    
    if (set.num_enabled() != 64)
    {
        fprintf(stderr, "Expected 64 enabled CPUs, got %d\n", set.num_enabled());
        return 1;
    }
    
    // Test 65th CPU (should trigger extended mode)
    set.enable(64);
    if (set.num_enabled() != 65)
    {
        fprintf(stderr, "Expected 65 enabled CPUs, got %d\n", set.num_enabled());
        return 1;
    }
    
    // Test negative CPU ID (should be ignored)
    set.enable(-1);
    set.disable(-1);
    // Should not crash
    
    // Test very large CPU ID
    set.enable(10000);
    if (!set.is_enabled(10000))
    {
        fprintf(stderr, "CPU 10000 should be enabled\n");
        return 1;
    }
    
    printf("CpuSet boundary test passed!\n");
    return 0;
}

// Test performance with large CPU sets
static int test_cpuset_performance()
{
    printf("Testing CpuSet performance with large CPU sets...\n");
    
    ncnn::CpuSet set;
    
    // Enable many CPUs
    const int max_cpu = 2048;
    for (int i = 0; i < max_cpu; i += 2)  // Enable every other CPU
    {
        set.enable(i);
    }
    
    // Verify count
    int expected_count = max_cpu / 2;
    if (set.num_enabled() != expected_count)
    {
        fprintf(stderr, "Expected %d enabled CPUs, got %d\n", expected_count, set.num_enabled());
        return 1;
    }
    
    // Test copy performance
    ncnn::CpuSet set_copy(set);
    if (set_copy.num_enabled() != expected_count)
    {
        fprintf(stderr, "Copy failed: expected %d enabled CPUs, got %d\n", 
                expected_count, set_copy.num_enabled());
        return 1;
    }
    
    printf("CpuSet performance test passed!\n");
    return 0;
}

int main()
{
    return 0
           || test_cpuset_large()
           || test_cpuset_boundary()
           || test_cpuset_performance();
}
