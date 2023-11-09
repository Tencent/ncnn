# ncnn vulkan driver loader

ncnn turns on the ```NCNN_SIMPLEVK``` cmake option by default, when ```NCNN_VULKAN``` is enabled

simplevk is ncnn's built-in vulkan loader. It provides vulkan function declarations and function entries that meet ncnn's needs. It allows the use and compilation of vulkan-related codes without relying on vulkan-sdk. It can dynamically load the vulkan runtime library at runtime or directly load the graphics card driver. vulkan driver. When distributing ncnn applications, it is not required that the target system has a vulkan driver.

Usually you don't need to care about how simplevk loads the vulkan driver, because ncnn will automatically load and initialize when using vulkan related functions. It is sufficient to set the `Option` switch before loading the model.

Typical code

```cpp
ncnn::Net net;
net.opt.use_vulkan_compute = true;
net.load_param("model.param");
net.load_param("model.bin");
```

Using the in-house vulkan loader instead of the standard libvulkan has the following benefits

- Can compile ncnn vulkan code without installing vulkan-sdk
- Can deploy and distribute applications without libvulkan linkage
- Can load external vulkan driver instead of system driver
- Can directly load android hal module
- Can directly load graphics card driver files via NCNN_VULKAN_DRIVER env
- Able to actively search for graphics card driver files in the system and load them
- Can compile android libraries supporting vulkan under the platform of android-api<24

## Create and manage gpu context

```cpp
int create_gpu_instance(const char* driver_path = 0);

void destroy_gpu_instance();

VkInstance get_gpu_instance();
```

## Loading order

```
If driver_path == 0
  1a from env ```VK_ICD_FILENAMES```
  1b from env ```NCNN_VULKAN_DRIVER```

If driver_path != 0
  1 from specified driver_path

2 from vulkan-1.dll / libvulkan.so / libvulkan.dylib in system

3 search driver by name nvoglv64.dll / amdvlk64.dll / libGLX_nvidia.so.0 .... and load it
```

## Load from system vulkan library or graphics driver

This is the default behavior and it should work on most systems

sample usage
```cpp
int ret = create_gpu_instance();
```

Load from system-installed libvulkan

#### Windows
vulkan-1.dll

#### Linux Android
libvulkan.so

#### macOS iOS and other APPLE platforms
Requires static meltvk driver linking and should always succeed

If failed, it will try to find graphics driver object and load it

#### Windows
for 64bit applications. search in ```%SystemRoot%\System32\DriverStore\FileRepository```
- nvoglv64.dll
- amdvlk64.dll
- igvk64.dll

for 32bit applications. search in ```%SystemRoot%\System32\DriverStore\FileRepository```
- nvoglv32.dll
- amdvlk32.dll
- igvk32.dll

#### Linux
`dlopen()` search for
- libGLX_nvidia.so.0
- libvulkan_radeon.so
- libvulkan_intel.so
- libMaliVulkan.so.1
- libVK_IMG.so

#### Android
for 64bit applications
- /vendor/lib64/hw/vulkan.adreno.so
- /vendor/lib64/egl/libGLES_mali.so

for 32bit applications
- /vendor/lib/hw/vulkan.adreno.so
- /vendor/lib/egl/libGLES_mali.so

## Load from driver_path

for advanced developer

sample usage
```cpp
int ret = create_gpu_instance("libvulkan.so");
int ret = create_gpu_instance("/usr/lib64/libvulkan_radeon.so");
int ret = create_gpu_instance("/vendor/lib64/hw/vulkan.adreno.so");
int ret = create_gpu_instance("/data/local/tmp/vulkan.ad07XX.so");
```

## Load from env VK_ICD_FILENAMES

for debug purpose

sample usage
```sh
export VK_ICD_FILENAMES=./vk_swiftshader_icd.json
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
```

## Load from env NCNN_VULKAN_DRIVER

for debug purpose

sample usage
```sh
export NCNN_VULKAN_DRIVER=/data/local/tmp/vulkan.ad07XX.so
```
