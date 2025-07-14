#include <stdio.h>
#include <windows.h>
#include <vector>
#include <thread>
#include <chrono>
#include "cpu.h"

static void print_separator(const char* title) {
    printf("\n=== %s ===\n", title);
}

static int test_basic_cpu_info() {
    print_separator("Basic CPU Information Test");
    
    int cpu_count = ncnn::get_cpu_count();
    int big_cpu_count = ncnn::get_big_cpu_count();
    int little_cpu_count = ncnn::get_little_cpu_count();
    int physical_cpu_count = ncnn::get_physical_cpu_count();
    
    printf("CPU Count: %d\n", cpu_count);
    printf("Big CPU Count: %d\n", big_cpu_count);
    printf("Little CPU Count: %d\n", little_cpu_count);
    printf("Physical CPU Count: %d\n", physical_cpu_count);
    
    if (cpu_count <= 0) {
        printf("ERROR: Invalid CPU count\n");
        return -1;
    }
    
    return 0;
}

static int test_windows_api_comparison() {
    print_separator("Windows API Comparison Test");
    
    // Get ncnn detected CPU count
    int ncnn_cpu_count = ncnn::get_cpu_count();
    
    // Get Windows API CPU count
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    DWORD win_cpu_count = sysinfo.dwNumberOfProcessors;
    
    printf("NCNN detected CPUs: %d\n", ncnn_cpu_count);
    printf("Windows GetSystemInfo CPUs: %d\n", win_cpu_count);
    
    // Test GetLogicalProcessorInformationEx for >64 core support
    DWORD buffer_size = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &buffer_size);
    
    if (buffer_size > 0) {
        std::vector<char> buffer(buffer_size);
        if (GetLogicalProcessorInformationEx(RelationProcessorCore, 
                (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buffer.data(), &buffer_size)) {
            
            int core_count = 0;
            int group_count = 0;
            PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX current = 
                (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buffer.data();
            
            while ((char*)current < buffer.data() + buffer_size) {
                if (current->Relationship == RelationProcessorCore) {
                    core_count++;
                    group_count = max(group_count, (int)current->Processor.GroupCount);
                }
                current = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)
                    ((char*)current + current->Size);
            }
            
            printf("GetLogicalProcessorInformationEx cores: %d\n", core_count);
            printf("Processor groups detected: %d\n", group_count);
            
            if (core_count > 64) {
                printf("SUCCESS: Detected >64 core system\n");
            }
        }
    }
    
    return 0;
}

static int test_cpuset_basic_operations() {
    print_separator("CpuSet Basic Operations Test");
    
    ncnn::CpuSet cpuset;
    
    // Test initial state
    int initial_enabled = cpuset.num_enabled();
    printf("Initial enabled CPUs: %d\n", initial_enabled);
    
    // Test enabling specific CPUs
    int cpu_count = ncnn::get_cpu_count();
    for (int i = 0; i < min(cpu_count, 8); i++) {
        cpuset.enable(i);
        if (!cpuset.is_enabled(i)) {
            printf("ERROR: Failed to enable CPU %d\n", i);
            return -1;
        }
    }
    
    printf("Enabled first 8 CPUs, total enabled: %d\n", cpuset.num_enabled());
    
    // Test disabling
    cpuset.disable(0);
    if (cpuset.is_enabled(0)) {
        printf("ERROR: Failed to disable CPU 0\n");
        return -1;
    }
    
    printf("Disabled CPU 0, total enabled: %d\n", cpuset.num_enabled());
    
    // Test disable_all
    cpuset.disable_all();
    if (cpuset.num_enabled() != 0) {
        printf("ERROR: disable_all failed\n");
        return -1;
    }
    
    printf("After disable_all, enabled CPUs: %d\n", cpuset.num_enabled());
    
    return 0;
}

static int test_cpuset_large_core_numbers() {
    print_separator("CpuSet Large Core Numbers Test");
    
    ncnn::CpuSet cpuset;
    int cpu_count = ncnn::get_cpu_count();
    
    // Test enabling all available CPUs
    for (int i = 0; i < cpu_count; i++) {
        cpuset.enable(i);
    }
    
    int enabled_count = cpuset.num_enabled();
    printf("Enabled all %d CPUs, actual enabled: %d\n", cpu_count, enabled_count);
    
    if (enabled_count != cpu_count) {
        printf("WARNING: Mismatch between expected and actual enabled CPUs\n");
    }
    
    // Test boundary conditions
    if (cpu_count > 64) {
        printf("Testing >64 core boundary...\n");
        
        cpuset.disable_all();
        
        // Enable CPUs around the 64-core boundary
        for (int i = 60; i < min(cpu_count, 68); i++) {
            cpuset.enable(i);
            if (!cpuset.is_enabled(i)) {
                printf("ERROR: Failed to enable CPU %d (around 64-core boundary)\n", i);
                return -1;
            }
        }
        
        printf("Successfully enabled CPUs around 64-core boundary\n");
    }
    
    return 0;
}

#ifdef _WIN32
static int test_windows_specific_features() {
    print_separator("Windows Specific Features Test");
    
    ncnn::CpuSet cpuset;
    
    // Test Windows-specific methods
    int max_cpus = cpuset.get_max_cpus();
    int active_groups = cpuset.get_active_group_count();
    
    printf("Max CPUs: %d\n", max_cpus);
    printf("Active processor groups: %d\n", active_groups);
    
    // Test group masks
    for (int group = 0; group < active_groups && group < 4; group++) {
        ULONG_PTR mask = cpuset.get_group_mask(group);
        printf("Group %d mask: 0x%llx\n", group, (unsigned long long)mask);
    }
    
    // Test enabling CPUs in different groups
    if (active_groups > 1) {
        printf("Testing multi-group CPU enabling...\n");
        
        cpuset.disable_all();
        
        // Enable some CPUs in group 0
        for (int i = 0; i < min(4, max_cpus); i++) {
            cpuset.enable(i);
        }
        
        // Enable some CPUs in group 1 (if exists)
        if (max_cpus > 64) {
            for (int i = 64; i < min(68, max_cpus); i++) {
                cpuset.enable(i);
            }
        }
        
        printf("Multi-group test completed, enabled CPUs: %d\n", cpuset.num_enabled());
    }
    
    return 0;
}
#endif

static int test_thread_affinity() {
    print_separator("Thread Affinity Test");
    
    // Test getting thread affinity masks
    const ncnn::CpuSet& mask_all = ncnn::get_cpu_thread_affinity_mask(0);
    const ncnn::CpuSet& mask_little = ncnn::get_cpu_thread_affinity_mask(1);
    const ncnn::CpuSet& mask_big = ncnn::get_cpu_thread_affinity_mask(2);
    
    printf("All cores mask enabled CPUs: %d\n", mask_all.num_enabled());
    printf("Little cores mask enabled CPUs: %d\n", mask_little.num_enabled());
    printf("Big cores mask enabled CPUs: %d\n", mask_big.num_enabled());
    
    // Test setting thread affinity
    ncnn::CpuSet custom_mask;
    int cpu_count = ncnn::get_cpu_count();
    
    // Enable every other CPU
    for (int i = 0; i < cpu_count; i += 2) {
        custom_mask.enable(i);
    }
    
    printf("Setting custom affinity with %d CPUs...\n", custom_mask.num_enabled());
    int result = ncnn::set_cpu_thread_affinity(custom_mask);
    
    if (result == 0) {
        printf("Thread affinity set successfully\n");
    } else {
        printf("Thread affinity setting failed with code: %d\n", result);
    }
    
    return 0;
}

int main() {
    printf("NCNN CPU Core Support Test for Windows 64+ Cores\n");
    printf("================================================\n");
    
    int result = 0;
    
    result |= test_basic_cpu_info();
    result |= test_windows_api_comparison();
    result |= test_cpuset_basic_operations();
    result |= test_cpuset_large_core_numbers();
    
#ifdef _WIN32
    result |= test_windows_specific_features();
#endif
    
    result |= test_thread_affinity();
    
    print_separator("Test Summary");
    if (result == 0) {
        printf("All tests PASSED\n");
    } else {
        printf("Some tests FAILED (return code: %d)\n", result);
    }
    
    return result;
}