// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_SIMPLEVK_PRINTF_H
#define NCNN_SIMPLEVK_PRINTF_H

#include "gpu.h"

#if NCNN_VULKAN

#include <string>

namespace ncnn {

#define NCNN_SIMPLEVK_PRINTF_SET              3
#define NCNN_SIMPLEVK_PRINTF_BINDING          0
#define NCNN_SIMPLEVK_PRINTF_SHADER_INFO_FLAG 1

struct SimplevkPrintfCommand;

int simplevk_printf_device_supported(const GpuInfo& info);
int simplevk_printf_shader_info_supported(const GpuInfo& info, const ShaderInfo& shader_info);

int simplevk_printf_source_contains_log(const char* data, int size);
int simplevk_printf_source_conflicts(const char* data, int size);
int simplevk_printf_rewrite_shader(const char* data, int size, int enable, std::string& output, int* log_count);
std::string simplevk_printf_glsl_header();

int simplevk_printf_shader_info_has_printf(const ShaderInfo& shader_info);
void simplevk_printf_shader_info_set_has_printf(ShaderInfo& shader_info);

SimplevkPrintfCommand* simplevk_printf_command_create(const VulkanDevice* vkdev);
void simplevk_printf_command_destroy(SimplevkPrintfCommand* state);
void simplevk_printf_command_reset(SimplevkPrintfCommand* state);
int simplevk_printf_command_used(const SimplevkPrintfCommand* state);
int simplevk_printf_command_ensure(SimplevkPrintfCommand* state);
VkDescriptorSet simplevk_printf_command_descriptorset(SimplevkPrintfCommand* state);
int simplevk_printf_command_record_reset(SimplevkPrintfCommand* state, VkCommandBuffer command_buffer);
int simplevk_printf_command_record_bind(SimplevkPrintfCommand* state, VkCommandBuffer command_buffer, VkPipelineLayout pipeline_layout);
int simplevk_printf_command_record_readback(SimplevkPrintfCommand* state, VkCommandBuffer command_buffer);
int simplevk_printf_command_readback(SimplevkPrintfCommand* state);

} // namespace ncnn

#endif // NCNN_VULKAN

#endif // NCNN_SIMPLEVK_PRINTF_H
