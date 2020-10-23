ncnn openmp 最佳实践

### ncnn占用过多cpu资源

   使用ncnn推理运算，cpu占用非常高甚至所有核心占用都接近100%。

   如果还有其它线程或进程需要较多的cpu资源，运行速度下降严重。

### cpu占用高的根本原因

1. ncnn使用openmp API控制多线程加速推理计算。默认情况下，线程数等于cpu内核数。如果推理需要高频率运行，必然占用大部分
   cpu资源。

2. openmp内部维护一个线程池，线程池最大可用线程数等于cpu内核数。(核心过多时最大限制是15？）获取和归还线程时需要同步。

   为了提高效率，几乎所有omp实现都使用了自旋锁同步(simpleomp除外)。自旋锁默认的spin time是200ms。因此一个线程被调度后，
   需要忙等待最多200ms。

### 为什么使用vulkan加速后cpu占用依然很高。

1. 加载参数文件时也使用了openmp，这部分是在cpu上运行的。

2. 显存上传前和下载后的 fp32 fp16转换是在cpu上执行的，这部分逻辑也使用了openmp。

### 解决方法

```
1. 绑核
```
   如果使用有大小核cpu的设备，建议通过ncnn::set_cpu_powersave(int)绑定大核或小核，注意windows系统不支持绑核。顺便说一下，ncnn支持不同的模型运行在不同的核心。假设硬件平台有2个大核，4个小核，你想把netA运行在大核，netB运行在小核。
   可以通过std::thread or pthread创建两个线程，运行如下代码：
   
   ```
   void thread_1()
   {
      ncnn::set_cpu_powersave(2); // bind to big cores
      netA.opt.num_threads = 2;
   }

   void thread_2()
   {
      ncnn::set_cpu_powersave(1); // bind to little cores
      netB.opt.num_threads = 4;
   }
   ```

```
2. 使用更少的线程数。
```
   通过ncnn::set_omp_num_threads(int)或者net.opt.num_threads字段设置线程数为cpu内核数的一半或更小。如果使用clang的libomp，
   建议线程数不超过8，如果使用其它omp库，建议线程数不超过4。
```
3. 减小openmp blocktime。
```
   可以修改ncnn::set_kmp_blocktime(int)或者修改net.opt.openmp_blocktime，这个参数是ncnn API设置的spin time，默认是20ms。
   可以根据情况设置更小的值，或者直接改为0。

   局限：目前只有clang的libomp库有实现，vcomp和libgomp都没有相应接口，如果不是使用clang编译的，这个值默认还是200ms。
   如果使用vcomp或libgomp, 可以使用环境变量OMP_WAIT_POLICY=PASSIVE禁用spin time，如果使用simpleomp,不需要设置这个参数。
```
4. 限制openmp线程池可用线程数量。
```
   即使减小了openmp线程数量，cpu占用率仍然可能会很高。这在cpu核心特别多的服务器上比较常见。这是因为线程池中的等待线程使用
   自旋锁忙等待，可以通过限制线程池可用线程数量减轻这种影响。

   一般可以通过设置OMP_THREAD_LIMIT环境变量。simpleomp目前不支持这一特性，不需要设置。注意这个环境变量仅在程序启动前设置才有效。
```
5. 完全禁用openmp
```
   如果只有一个cpu核心，或者使用vulkan加速，建议关闭openmp, cmake编译时指定-DNCNN_OPENMP=OFF即可。