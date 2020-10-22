NCNN openmp best practice

### CPU loadaverage is too high with NCNN.

   When inference the neural network with NCNN, the cpu occupancy is very high even all CPU cores occupancy
   close to 100%.

   If there are other threads or processes that require more cpu resources, the running speed of the program
   will drop severely.

### The root cause of high CPU usage

1. NCNN uses openmp API to speed up the inference compute. the thread count equals to the cpu core count.
   If the computing work need to run frequently, it must consume many cpu resources.

2. There is a thread pool managed by openmp, the pool size is equal to the cpu core size.
   (the max vulue is 15 if there is much more cpu core?)
   Openmp need to sync the thread when acquiring and returning threads to the pool.
   In order to improve efficiency, almost all omp implementations use spinlock synchronization (except for simpleomp). 
   The default spin time of the spin lock is 200ms. So after a thread is scheduled, the thread need to busy-wait up to 200ms.

### Why the CPU usage is still high even using vulkan acceleration.

1. Openmp is also used when loading the param bin file, and this part runs on cpu.

2. The fp32 to fp16 conversion before and after the GPU memory upload is executed on the cpu,
   and this part of the logic also uses openmp.

### Solution
```
1. Bind to the specific cpu core.
```
   If you use a device with large and small core CPUs, it is recommended to bind large or small cores through
   ncnn::set_cpu_powersave(int). Note that Windows does not support binding cores.
```
2. Use fewer threads.
```
   Set the number of threads to half of the cpu cores count or less through the ncnn::set_omp_num_threads(int) 
   or net.opt.num_threads field.
   If you are coding with clang libomp, it's recommended that the number of threads does not exceed 8.
   If you use other omp libraries, it is recommended that the number of threads does not exceed 4.
```
3. Reduce openmp blocktime.
```
   You can modify openmp blocktime by call ncnn::set_kmp_blocktime(int) method or modify net.opt.openmp_blocktime field.
   This argument is the spin time set by the ncnn API, and the default is 20ms.You can set a smaller value according to
   the situation, or directly change it to 0.

   Limitations: At present, only the libomp library of clang is implemented. Neither vcomp nor libgomp have corresponding interfaces.
   If it is not compiled with clang, this value is still 200ms by default.
   If you use vcomp or libgomp, you can use the environment variable OMP_WAIT_POLICY=PASSIVE to disable spin time. If you use simpleomp,
   It's no need to set this parameter.
```
4. Limit the number of threads available in the openmp thread pool.
```
   Even if the number of openmp threads is reduced, the CPU occupancy rate may still be high. This is more common on servers with
   particularly many CPU cores. 
   This is because the waiting threads in the thread pool use a spin lock to busy-wait, which can be reducedby limiting the number of
   threads available in the thread pool.

   Generally, you can set the OMP_THREAD_LIMIT environment variable. simpleomp currently does not support this feature so it's no need to be set.
   Note that this environment variable is only valid if it is set before the program starts.
```
5. Disable openmp completely
```
   If there is only one cpu core, or use the vulkan gpu acceleration, it is recommended to disable openmp, just specify -DNCNN_OPENMP=OFF
   when compiling with cmake.