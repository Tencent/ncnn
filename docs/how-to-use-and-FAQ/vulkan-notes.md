## supported platform

* Y = known work
* ? = shall work, not confirmed
* / = not applied

|    |windows|linux|android|mac|ios|
|---|---|---|---|---|---|
|intel|Y|Y|?|?|/|
|amd|Y|Y|/|?|/|
|nvidia|Y|Y|?|/|/|
|qcom|/|/|Y|/|/|
|apple|/|/|/|?|Y|
|arm|/|?|?|/|/|

## enable vulkan compute support
```
$ sudo dnf install vulkan-devel
$ cmake -DNCNN_VULKAN=ON ..
```

## enable vulkan compute inference
```cpp
ncnn::Net net;
net.opt.use_vulkan_compute = 1;
```

## proper allocator usage
```cpp
ncnn::VkAllocator* blob_vkallocator = vkdev.acquire_blob_allocator();
ncnn::VkAllocator* staging_vkallocator = vkdev.acquire_blob_allocator();

net.opt.blob_vkallocator = blob_vkallocator;
net.opt.workspace_vkallocator = blob_vkallocator;
net.opt.staging_vkallocator = staging_vkallocator;

// ....

// after inference
vkdev.reclaim_blob_allocator(blob_vkallocator);
vkdev.reclaim_staging_allocator(staging_vkallocator);
```

## select gpu device
```cpp
// get gpu count
int gpu_count = ncnn::get_gpu_count();

// set specified vulkan device before loading param and model
net.set_vulkan_device(0); // use device-0
net.set_vulkan_device(1); // use device-1
```

## zero-copy on unified memory device
```cpp
ncnn::VkMat blob_gpu;
ncnn::Mat mapped = blob_gpu.mapped();

// use mapped.data directly
```

## hybrid cpu/gpu inference
```cpp
ncnn::Extractor ex_cpu = net.create_extractor();
ncnn::Extractor ex_gpu = net.create_extractor();
ex_cpu.set_vulkan_compute(false);
ex_gpu.set_vulkan_compute(true);

#pragma omp parallel sections
{
    #pragma omp section
    {
        ex_cpu.input();
        ex_cpu.extract();
    }
    #pragma omp section
    {
        ex_gpu.input();
        ex_gpu.extract();
    }
}
```

## zero-copy gpu inference chaining
```cpp
ncnn::Extractor ex1 = net1.create_extractor();
ncnn::Extractor ex2 = net2.create_extractor();

ncnn::VkCompute cmd(&vkdev);

ncnn::VkMat conv1;
ncnn::VkMat conv2;
ncnn::VkMat conv3;

ex1.input("conv1", conv1);
ex1.extract("conv2", conv2, cmd);

ex2.input("conv2", conv2);
ex2.extract("conv3", conv3, cmd);

cmd.submit();

cmd.wait();

```

## batch inference
```cpp
int max_batch_size = vkdev->info.compute_queue_count;

ncnn::Mat inputs[1000];
ncnn::Mat outputs[1000];

#pragma omp parallel for num_threads(max_batch_size)
for (int i=0; i<1000; i++)
{
    ncnn::Extractor ex = net1.create_extractor();
    ex.input("data", inputs[i]);
    ex.extract("prob", outputs[i]);
}
```

## control storage and arithmetic precision

disable all lower-precision optimzations, get full fp32 precision

```cpp
ncnn::Net net;
net.opt.use_fp16_packed = false;
net.opt.use_fp16_storage = false;
net.opt.use_fp16_arithmetic = false;
net.opt.use_int8_storage = false;
net.opt.use_int8_arithmetic = false;
```

## debugging tips
```cpp
#define ENABLE_VALIDATION_LAYER 1 // modify to 1 in gpu.cpp
```

## add vulkan compute support to layer
1. add vulkan shader in src/layer/shader/

2. upload model weight data in Layer::upload_model()

3. setup pipeline in Layer::create_pipeline()

4. destroy pipeline in Layer::destroy_pipeline()

5. record command in Layer::forward()

## add optimized shader path
1. add vulkan shader in src/layer/shader/ named XXX_abc.comp

2. create pipeline with "XXX_abc"

3. record command using XXX_abc pipeline

## low-level op api
1. create layer

2. load param and load model

3. upload model

4. create pipeline

5. new command

6. record

7. submit and wait

