# Practical Vulkan Debug Printf in ncnn

A multi-platform guide to debugging ncnn Vulkan shaders with `debugPrintfEXT`.

## Overview

Vulkan debug printf allows shader code to emit debug messages to the validation
layer output. In ncnn, this is accessible through `NCNN_LOGE` in shader code.

## Prerequisites

- ncnn built with Vulkan support AND validation layers enabled:
  `cmake .. -DNCNN_VULKAN=ON -DNCNN_ENABLE_VALIDATION_LAYER=ON`
- Vulkan SDK installed (1.3+)
- Validation layers enabled
- GPU supporting `VK_KHR_shader_non_semantic_info`

## Platform Setup

### Linux

```bash
# Set validation layer
export VK_LAYER_PATH=/usr/share/vulkan/explicit_layer.d
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
export VK_LAYER_SETTINGS_PATH=/path/to/vk_layer_settings.txt
```

### Windows

```powershell
$env:VK_LAYER_PATH="C:\VulkanSDK\1.3.x.x\Bin"
$env:VK_INSTANCE_LAYERS="VK_LAYER_KHRONOS_validation"
$env:VK_LAYER_SETTINGS_PATH="C:\tmp\vk_layer_settings.txt"
```

### macOS

```bash
export VK_LAYER_PATH=/usr/local/share/vulkan/explicit_layer.d
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
```

### Android

```bash
# Push validation layers to device
adb push libVkLayer_khronos_validation.so /data/local/tmp
adb shell setprop debug.vulkan.layers VK_LAYER_KHRONOS_validation
adb logcat -s ncnn
```

## Configuration

Create `vk_layer_settings.txt`:

```ini
khronos_validation.debug_action = VK_DBG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT
khronos_validation.debug_printf_to_stdout = true
khronos_validation.debug_printf_preserve = true
```

## Using NCNN_LOGE in Shaders

### Example: Debug a MatMul Pipeline

```glsl
#version 450
#extension GL_EXT_debug_printf : enable

layout(local_size_x_id = 0) in;

void main() {
    // Debug input dimensions (only active when ncnn is built with validation layers)
#if ncnn_enable_validation_layer
    NCNN_LOGE("matmul layer: M=%d N=%d K=%d", M, N, K);
#endif

    float sum = 0.0;
    for (int k = 0; k < K; k++) {
        sum += A[...] * B[...];
    }

#if ncnn_enable_validation_layer
    NCNN_LOGE("partial sum[%d][%d] = %f", row, col, sum);
#endif
    // ...
}
```

### Filter Debug Output

```bash
# Run ncnn benchmark with Vulkan, filter debug printf output.
# NCNN_LOGE expands to debugPrintfEXT; the validation layer emits the
# format strings directly (e.g. "matmul layer..."), not the macro name.
./benchncnn 1 1 0 0 resnet 0 2>&1 | grep "matmul layer"
```

## Verifying Output

### Expected Output

```
[VULKAN DEBUG] RESIDUAL_PREFILL: matmul layer: M=1 N=512 K=256
[VULKAN DEBUG] RESIDUAL_PREFILL: partial sum[0][0] = 0.734512
[VULKAN DEBUG] RESIDUAL_PREFILL: matmul layer: M=1 N=256 K=512
[VULKAN DEBUG] RESIDUAL_PREFILL: partial sum[0][1] = -0.123456
```

### Common Debug Scenarios

| Use Case | Shader Code |
|----------|------------|
| Check tensor shape | `NCNN_LOGE("shape: (%d,%d,%d)", w, h, c);` |
| Verify weight values | `NCNN_LOGE("weight[%d]=%f", i, w[i]);` |
| Trace execution path | `NCNN_LOGE("reached branch A");` |
| Profile section timing | `NCNN_LOGE("section took %d us", dt);` |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| No debug output | Verify validation layers are loaded: `vulkaninfo \| grep debugPrintf` |
| `VK_ERROR_EXTENSION_NOT_PRESENT` | GPU doesn't support `VK_KHR_shader_non_semantic_info` |
| Performance too slow | Debug printf adds overhead; disable in production builds |
| logcat flooding | Filter: `adb logcat -s ncnn:V *:S` |

## References

- [Khronos Debug Printf](https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/main/docs/debug_printf.md)
- [ncnn Vulkan Documentation](https://github.com/Tencent/ncnn/wiki/vulkan)
