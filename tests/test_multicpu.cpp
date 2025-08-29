#include "../src/cpu.h"
#include <iostream>
#include <cassert>
#include <vector>

// Test CpuSet - a class that simulates multi-CPU functionality
void test_cpuset_basic_functionality()
{
    std::cout << "=== Testing CpuSet Basic Functionality ===" << std::endl;
    
    ncnn::CpuSet cpuset;
    
    // Test enabling and disabling CPUs
    assert(!cpuset.is_enabled(0));
    cpuset.enable(0);
    assert(cpuset.is_enabled(0));
    assert(cpuset.num_enabled() == 1);
    
    cpuset.disable(0);
    assert(!cpuset.is_enabled(0));
    assert(cpuset.num_enabled() == 0);
    
    std::cout << "Basic enable/disable tests passed" << std::endl;
}

void test_cpuset_multi_group_simulation()
{
    std::cout << "=== Testing Multi-Group CPU Simulation ===" << std::endl;
    
    ncnn::CpuSet cpuset;
    
    // Simulate enabling CPUs across multiple groups
    std::vector<int> test_cpus = {0, 1, 32, 63, 64, 65, 96, 127, 128, 200, 256};
    
    for (int cpu : test_cpus) {
        cpuset.enable(cpu);
        assert(cpuset.is_enabled(cpu));
    }
    
    std::cout << "Total enabled CPUs: " << cpuset.num_enabled() << std::endl;
    assert(cpuset.num_enabled() == (int)test_cpus.size());
    
    // Test group masks
#if defined (_WIN32)
    assert(cpuset.get_group_mask(0) != 0);
    assert(cpuset.get_group_mask(1) != 0);
    assert(cpuset.get_group_mask(2) != 0);
    assert(cpuset.get_group_mask(3) != 0);
    assert(cpuset.get_group_mask(5) == 0);
#endif 

    std::cout << "Multi-group CPU simulation tests passed" << std::endl;
}

void test_cpuset_boundary_conditions()
{
    std::cout << "=== Testing Boundary Conditions ===" << std::endl;
    
    ncnn::CpuSet cpuset;
    
    // Test enabling CPUs at the boundary of groups
    std::vector<int> boundary_cpus = {63, 64, 127, 128, 191, 192, 255, 256};
    
    for (int cpu : boundary_cpus) {
        cpuset.enable(cpu);
        bool enabled = cpuset.is_enabled(cpu);
        std::cout << "CPU " << cpu << " (group boundary): " 
                  << (enabled ? "enabled" : "not enabled") << std::endl;
    }
    
    // Test disabling all CPUs
    cpuset.disable_all();
    assert(cpuset.num_enabled() == 0);
    std::cout << "disable_all() works correctly" << std::endl;
}

void test_cpuset_edge_cases()
{
    std::cout << "=== Testing Edge Cases ===" << std::endl;
    
    ncnn::CpuSet cpuset;
    
    // Test enabling out-of-bounds CPUs
    cpuset.enable(-1);  // Should be ignored
    cpuset.enable(5000); // Should be ignored
    assert(cpuset.num_enabled() == 0);
    std::cout << "Invalid CPU numbers handled correctly" << std::endl;
    
    // Test duplicate enables
    cpuset.enable(10);
    cpuset.enable(10);
    assert(cpuset.num_enabled() == 1);
    std::cout << "Duplicate enable handled correctly" << std::endl;
    
    // Test disabling a non-enabled CPU
    cpuset.disable(20);
    assert(cpuset.num_enabled() == 1);
    std::cout << "Disabling non-enabled CPU handled correctly" << std::endl;
}

void test_real_system_integration()
{
    std::cout << "=== Testing Integration with Real System ===" << std::endl;
    
    int real_cpu_count = ncnn::get_cpu_count();
    std::cout << "Real system CPU count: " << real_cpu_count << std::endl;
    
    ncnn::CpuSet cpuset;
    
    // Enable all real CPUs
    for (int i = 0; i < real_cpu_count; i++) {
        cpuset.enable(i);
    }
    
    assert(cpuset.num_enabled() == real_cpu_count);
    std::cout << "All real CPUs enabled successfully" << std::endl;
    
    // Test powersave modes
    const ncnn::CpuSet& all_mask = ncnn::get_cpu_thread_affinity_mask(0);
    const ncnn::CpuSet& little_mask = ncnn::get_cpu_thread_affinity_mask(1);
    const ncnn::CpuSet& big_mask = ncnn::get_cpu_thread_affinity_mask(2);
    
    std::cout << "All cores enabled: " << all_mask.num_enabled() << std::endl;
    std::cout << "Little cores enabled: " << little_mask.num_enabled() << std::endl;
    std::cout << "Big cores enabled: " << big_mask.num_enabled() << std::endl;
    
}

void test_processor_group_simulation()
{
    std::cout << "=== Testing Processor Group Simulation ===" << std::endl;
    
    // Simulate a system with multiple processor groups
    ncnn::CpuSet group0_cpus, group1_cpus, group2_cpus;
    
    // First group: CPU 0-63
    for (int i = 0; i < 64; i++) {
        group0_cpus.enable(i);
    }
    
    // Second group：CPU 64-127
    for (int i = 64; i < 128; i++) {
        group1_cpus.enable(i);
    }
    
    // Third group: CPU 128-191
    for (int i = 128; i < 192; i++) {
        group2_cpus.enable(i);
    }
    
    std::cout << "Group 0 CPUs: " << group0_cpus.num_enabled() << std::endl;
    std::cout << "Group 1 CPUs: " << group1_cpus.num_enabled() << std::endl; 
    std::cout << "Group 2 CPUs: " << group2_cpus.num_enabled() << std::endl;
    
    #if defined _WIN32
    for (int group = 0; group < 3; group++) {
        ncnn::CpuSet* test_set = (group == 0) ? &group0_cpus : 
                                (group == 1) ? &group1_cpus : &group2_cpus;
        ULONG_PTR mask = test_set->get_group_mask(group);
        std::cout << "Group " << group << " mask: 0x" << std::hex << mask << std::dec << std::endl;
    }
    #endif
}

int main()
{
    std::cout << "Starting CpuSet Multi-CPU Simulation Tests..." << std::endl;
    std::cout << "Current system CPU count: " << ncnn::get_cpu_count() << std::endl;
#if defined _WIN32
    std::cout << "NCNN_MAX_CPU_COUNT: " << NCNN_MAX_CPU_COUNT << std::endl;
    std::cout << "NCNN_CPU_MASK_GROUPS: " << NCNN_CPU_MASK_GROUPS << std::endl;
#endif
    std::cout << std::endl;
    
    try {
        test_cpuset_basic_functionality();
        std::cout << std::endl;
        
        test_cpuset_multi_group_simulation();
        std::cout << std::endl;
        
        test_cpuset_boundary_conditions();
        std::cout << std::endl;
        
        test_cpuset_edge_cases();
        std::cout << std::endl;
        
        test_real_system_integration();
        std::cout << std::endl;
        
        test_processor_group_simulation();
        std::cout << std::endl;
        
        std::cout << "All simulation tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}