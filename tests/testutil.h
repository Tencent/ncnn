// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TESTUTIL_H
#define TESTUTIL_H

#include "cpu.h"
#include "layer.h"
#include "mat.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#if NCNN_VULKAN && defined(NCNN_WEIGHT_TEST_INTERNAL)
#include "gpu.h"
#if NCNN_BENCHMARK
#include <map>
#include <string>
#endif
#endif

#define TEST_LAYER_DISABLE_AUTO_INPUT_PACKING (1 << 0)
#define TEST_LAYER_DISABLE_AUTO_INPUT_CASTING (1 << 1)
#define TEST_LAYER_DISABLE_GPU_TESTING        (1 << 2)
#define TEST_LAYER_ENABLE_FORCE_INPUT_PACK8   (1 << 3)
#define TEST_LAYER_ENABLE_THREADING           (1 << 4)
#define TEST_LAYER_DISABLE_CPU_TESTING        (1 << 5)

void SRAND(int seed);

uint64_t RAND();

float RandomFloat(float a = -1.2f, float b = 1.2f);

int RandomInt(int a = -10000, int b = 10000);

signed char RandomS8();

void Randomize(ncnn::Mat& m, float a = -1.2f, float b = 1.2f);

void RandomizeInt(ncnn::Mat& m, int a = -10000, int b = 10000);

void RandomizeS8(ncnn::Mat& m);

ncnn::Mat RandomMat(int w, float a = -1.2f, float b = 1.2f);

ncnn::Mat RandomMat(int w, int h, float a = -1.2f, float b = 1.2f);

ncnn::Mat RandomMat(int w, int h, int c, float a = -1.2f, float b = 1.2f);

ncnn::Mat RandomMat(int w, int h, int d, int c, float a = -1.2f, float b = 1.2f);

ncnn::Mat RandomIntMat(int w);

ncnn::Mat RandomIntMat(int w, int h);

ncnn::Mat RandomIntMat(int w, int h, int c);

ncnn::Mat RandomIntMat(int w, int h, int d, int c);

ncnn::Mat RandomS8Mat(int w);

ncnn::Mat RandomS8Mat(int w, int h);

ncnn::Mat RandomS8Mat(int w, int h, int c);

ncnn::Mat RandomS8Mat(int w, int h, int d, int c);

ncnn::Mat scales_mat(const ncnn::Mat& mat, int m, int k, int ldx);

bool NearlyEqual(float a, float b, float epsilon);

int Compare(const ncnn::Mat& a, const ncnn::Mat& b, float epsilon = 0.001);

int CompareMat(const ncnn::Mat& a, const ncnn::Mat& b, float epsilon = 0.001);

int CompareMat(const std::vector<ncnn::Mat>& a, const std::vector<ncnn::Mat>& b, float epsilon = 0.001);

int test_layer_naive(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const std::vector<ncnn::Mat>& a, int top_blob_count, std::vector<ncnn::Mat>& b, int flag);

int test_layer_cpu(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const std::vector<ncnn::Mat>& a, int top_blob_count, std::vector<ncnn::Mat>& c, const std::vector<ncnn::Mat>& top_shapes, int flag);

#if NCNN_VULKAN
int test_layer_gpu(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const std::vector<ncnn::Mat>& a, int top_blob_count, std::vector<ncnn::Mat>& d, const std::vector<ncnn::Mat>& top_shapes, int flag);
#endif // NCNN_VULKAN

int test_layer(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const std::vector<ncnn::Mat>& a, int top_blob_count, const std::vector<ncnn::Mat>& top_shapes = std::vector<ncnn::Mat>(), float epsilon = 0.001, int flag = 0);

int test_layer_naive(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Mat& a, ncnn::Mat& b, int flag);

int test_layer_cpu(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const ncnn::Mat& a, ncnn::Mat& c, const ncnn::Mat& top_shape, int flag);

#if NCNN_VULKAN
int test_layer_gpu(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const ncnn::Mat& a, ncnn::Mat& d, const ncnn::Mat& top_shape, int flag);
#endif // NCNN_VULKAN

int test_layer(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const ncnn::Mat& a, const ncnn::Mat& top_shape = ncnn::Mat(), float epsilon = 0.001, int flag = 0);

int test_layer_opt(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& opt, const std::vector<ncnn::Mat>& a, int top_blob_count = 1, float epsilon = 0.001, int flag = 0);

int test_layer_opt(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& opt, const ncnn::Mat& a, float epsilon = 0.001, int flag = 0);

int test_layer(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const std::vector<ncnn::Mat>& a, int top_blob_count = 1, float epsilon = 0.001, int flag = 0);

int test_layer(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Mat& a, float epsilon = 0.001, int flag = 0);

// oom test

int test_layer_oom_opt(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& opt, const std::vector<ncnn::Mat>& a, int top_blob_count = 1, int flag = 0);

int test_layer_oom_opt(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& opt, const ncnn::Mat& a, int flag = 0);

int test_layer_oom(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const std::vector<ncnn::Mat>& a, int top_blob_count = 1, int flag = 0);

int test_layer_oom(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Mat& a, int flag = 0);

class TestParamDict : public ncnn::ParamDict
{
public:
#if NCNN_STRING
    using ncnn::ParamDict::load_param;
    int load_param(const char* str);
#endif
    using ncnn::ParamDict::load_param_bin;
    int load_param_bin(const unsigned char* mem);
};

#if NCNN_VALIDATION
int test_layer_param(int typeindex, const ncnn::ParamDict& pd, int expected_ret = 0);

int test_layer_param(int typeindex, const ncnn::ParamDict& base, int id, int value, int expected_ret);
int test_layer_param(int typeindex, const ncnn::ParamDict& base, int id, float value, int expected_ret);
int test_layer_param(int typeindex, const ncnn::ParamDict& base, int id, const ncnn::Mat& value, int expected_ret);
#endif // NCNN_VALIDATION

#if NCNN_VULKAN && defined(NCNN_WEIGHT_TEST_INTERNAL)
#if NCNN_BENCHMARK
std::map<std::string, uint64_t>& weight_benchmark_counts();
std::map<std::string, double>& weight_benchmark_times();
#endif

// Reset counters and install the collector for the calling thread.
void reset_weight_benchmark();

// Vulkan entry points are intercepted process-wide; audits must not overlap.
class WeightAllocationAudit
{
public:
    enum Failure
    {
        FAIL_NONE,
        FAIL_ALLOCATE,
        FAIL_BIND,
        FAIL_MAP,
        FAIL_SUBMIT,
        FAIL_SECOND_SUBMIT
    };

    WeightAllocationAudit(const ncnn::VulkanDevice* vkdev, Failure failure = FAIL_NONE);
    ~WeightAllocationAudit();

    bool failed;
    bool queue_held_while_waiting;
    uint32_t submissions;
    uint32_t waited_submissions;
    size_t current[2];
    size_t peak[2];

private:
    WeightAllocationAudit(const WeightAllocationAudit&);
    WeightAllocationAudit& operator=(const WeightAllocationAudit&);

    struct Allocation
    {
        VkDeviceMemory memory;
        size_t size;
        int storage;
    };

    static VKAPI_ATTR VkResult VKAPI_CALL allocate_memory(VkDevice device, const VkMemoryAllocateInfo* info, const VkAllocationCallbacks* callbacks, VkDeviceMemory* memory);
    static VKAPI_ATTR void VKAPI_CALL free_memory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* callbacks);
    static VKAPI_ATTR VkResult VKAPI_CALL bind_memory(VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize offset);
    static VKAPI_ATTR VkResult VKAPI_CALL map_memory(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size, VkMemoryMapFlags flags, void** data);
    static VKAPI_ATTR VkResult VKAPI_CALL submit_queue(VkQueue queue, uint32_t count, const VkSubmitInfo* submits, VkFence fence);

    static VKAPI_ATTR VkResult VKAPI_CALL wait_fences(VkDevice device, uint32_t count, const VkFence* fences, VkBool32 all, uint64_t timeout);
    static VKAPI_ATTR VkResult VKAPI_CALL fence_status(VkDevice device, VkFence fence);
    void check_queue_leases();

    const ncnn::VulkanDevice* vkdev;
    Failure failure;
    std::vector<VkFence> submitted_fences;
    std::vector<Allocation> allocations;
    PFN_vkAllocateMemory allocate;
    PFN_vkFreeMemory free;
    PFN_vkBindBufferMemory bind;
    PFN_vkMapMemory map;
    PFN_vkQueueSubmit submit;
    PFN_vkWaitForFences wait;
    PFN_vkGetFenceStatus status;
};
#endif // NCNN_VULKAN && NCNN_WEIGHT_TEST_INTERNAL

#endif // TESTUTIL_H
